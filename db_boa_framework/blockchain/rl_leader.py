"""
blockchain/rl_leader.py
=======================
Reinforcement-Learning Leader Selection  (title's "Reinforcement Learning")

The original leader selection (leader_block.py::select_leader) solves a *myopic,
stateless* optimisation every round — argmin (CT + CC + MS) — with DB-BOA.  That
ignores the fact that leader election in a consortium is a *sequential* decision:
the same nodes are elected round after round, their reputation and token balances
evolve, and over-using one node is short-sighted.

This module models leader selection as a Markov Decision Process and learns a
leader *policy* with reinforcement learning:

    state  s_t — per-node resource (CT/CC/MS) + reputation + token + load profile
    action a_t — which node is elected leader for round t
    reward r_t — the on-chain incentive payout to the elected leader
                 (+leader_success_reward [+latency_bonus] on success,
                  −consensus_fail_penalty on failure — INCENTIVE_CONFIG §6.2),
                 i.e. the SAME reward the consensus/incentive mechanism already
                 pays.  No reward is hand-engineered for the agent.
    transition — simulate_consensus_round() endorsement / reputation dynamics

Algorithm — linear function-approximation Q-learning (Sutton & Barto, 2018,
§9-10).  The action-value of electing node a in state s is linear in that node's
feature vector φ(s, a):

        Q(s, a) = θ · φ(s, a)

and θ is updated with the temporal-difference rule (bootstrapped, γ > 0):

        δ   = r + γ · max_{a'} Q(s', a') − Q(s, a)
        θ  ← θ + α · δ · φ(s, a)

Because γ > 0 the agent assigns credit across rounds — it can learn to rotate
leaders for load-balancing and to avoid nodes whose reliability is degrading,
neither of which the single-round DB-BOA objective can express.  ε-greedy
exploration (with decay) supplies the exploration/exploitation trade-off that a
pure optimiser also lacks.

A linear agent (not deep RL) is deliberate: with ~10-20 nodes the policy is
small, trains in milliseconds, is fully reproducible from a seed, and is simple
to defend in a thesis viva — DQN/PPO would be unjustified machinery here.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RL_LEADER_CONFIG

# Number of features in φ(s, a) — keep in sync with node_features().
N_FEATURES = 8


def node_features(node, nodes, round_idx: int) -> np.ndarray:
    """
    Featurise candidate ``node`` (electing it as leader) in the current
    consortium state ``nodes`` at round ``round_idx``.

    All features are normalised to roughly [0, 1] so the single shared θ vector
    is well-conditioned.  The agent learns the sign/magnitude of each weight; we
    do not bake in "lower cost is better".

        [0] bias
        [1] 1 − CT          (compute headroom; higher = faster node)
        [2] 1 − CC          (bandwidth headroom)
        [3] 1 − MS          (memory headroom)
        [4] reputation, scaled from [0.5, 2.0] → [0, 1]
        [5] token balance relative to the consortium mean (squashed)
        [6] 1 − leadership load (rounds_as_leader / total so far); higher = under-used
        [7] 1 − recent failure rate (successes / (successes+failures))
    """
    tokens = np.array([nd.tokens for nd in nodes], dtype=float)
    tok_mean, tok_std = tokens.mean(), tokens.std() + 1e-6
    tok_rel = np.tanh((node.tokens - tok_mean) / tok_std)          # ∈ (−1, 1)

    total_leaderships = sum(nd.rounds_as_leader for nd in nodes) + 1
    load = node.rounds_as_leader / total_leaderships               # ∈ [0, 1]

    trials = node.consensus_successes + node.consensus_failures
    fail_rate = node.consensus_failures / trials if trials else 0.0

    return np.array([
        1.0,
        1.0 - node.ct,
        1.0 - node.cc,
        1.0 - node.ms,
        (node.reputation - 0.5) / 1.5,
        0.5 * (tok_rel + 1.0),
        1.0 - load,
        1.0 - fail_rate,
    ], dtype=float)


class RLLeaderSelector:
    """
    Linear-FA Q-learning agent that elects a consortium leader each round.

    Typical driving loop (see ConsortiumBlockchain.simulate_rl_round and
    experiments/rl_leader_sweep.py)::

        agent = RLLeaderSelector(seed=7)
        for r in range(1, R + 1):
            idx, phi = agent.select(bc.nodes, r)        # ε-greedy election
            bc.current_leader = bc.nodes[idx]
            before = bc.nodes[idx].tokens
            bc.simulate_consensus_round(round_num=r, ...)
            reward = bc.nodes[idx].tokens - before      # incentive payout
            agent.learn(phi, reward, bc.nodes, r + 1)   # TD update
    """

    def __init__(self, n_features: int = N_FEATURES,
                 cfg: dict = None, seed: int = None):
        self.cfg     = cfg or RL_LEADER_CONFIG
        self.theta   = np.zeros(n_features, dtype=float)
        self.alpha   = self.cfg["alpha"]
        self.gamma   = self.cfg["gamma"]
        self.epsilon = self.cfg["epsilon_start"]
        self.eps_min = self.cfg["epsilon_min"]
        self.eps_dec = self.cfg["epsilon_decay"]
        self.r_scale = self.cfg["reward_scale"]
        self.rng     = np.random.RandomState(
            self.cfg["random_state"] if seed is None else seed)
        # Telemetry for plots / ledger
        self.history = {
            "epsilon"   : [],   # ε used at each election
            "reward"    : [],   # scaled reward observed
            "td_error"  : [],   # |δ| each update
            "explored"  : [],   # bool: was the election a random explore?
            "elected"   : [],   # node_id elected
        }

    # ── value function ────────────────────────────────────────────────────────

    def q(self, phi: np.ndarray) -> float:
        """Action value Q(s, a) = θ · φ(s, a)."""
        return float(self.theta @ phi)

    def _all_features(self, nodes, round_idx: int) -> np.ndarray:
        """Stack φ(s, a) for every candidate node → (n_nodes, n_features)."""
        return np.stack([node_features(nd, nodes, round_idx) for nd in nodes])

    def _max_q(self, nodes, round_idx: int) -> float:
        """max_{a'} Q(s', a') used for the TD bootstrap target."""
        if not nodes:
            return 0.0
        return float((self._all_features(nodes, round_idx) @ self.theta).max())

    # ── policy ────────────────────────────────────────────────────────────────

    def select(self, nodes, round_idx: int, greedy: bool = False):
        """
        ε-greedy leader election.

        Returns ``(idx, phi)`` where ``idx`` is the elected node's position in
        ``nodes`` and ``phi`` is φ(s, a) for that election (needed by learn()).
        With probability ε (0 if ``greedy``) a random node is explored;
        otherwise the argmax-Q node is exploited.
        """
        phis = self._all_features(nodes, round_idx)
        q_vals = phis @ self.theta

        explore = (not greedy) and (self.rng.rand() < self.epsilon)
        if explore:
            idx = int(self.rng.randint(len(nodes)))
        else:
            # random tie-break among argmax to avoid a fixed bias toward index 0
            best = np.flatnonzero(q_vals == q_vals.max())
            idx  = int(self.rng.choice(best))

        self.history["epsilon"].append(self.epsilon)
        self.history["explored"].append(bool(explore))
        self.history["elected"].append(int(nodes[idx].node_id))
        return idx, phis[idx]

    # ── learning ──────────────────────────────────────────────────────────────

    def learn(self, phi_sa: np.ndarray, reward: float,
              next_nodes, next_round_idx: int, done: bool = False):
        """
        One temporal-difference update for the elected action.

        ``reward`` is the raw on-chain token payout; it is multiplied by
        ``reward_scale`` so the TD target stays on the same scale as Q (token
        rewards are ±10-25, which would otherwise make θ explode).
        """
        r = reward * self.r_scale
        target = r if done else r + self.gamma * self._max_q(next_nodes, next_round_idx)
        td_error = target - self.q(phi_sa)
        self.theta += self.alpha * td_error * phi_sa

        self.history["reward"].append(float(r))
        self.history["td_error"].append(float(abs(td_error)))

    def decay_epsilon(self):
        """Anneal exploration toward eps_min after each round."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)
