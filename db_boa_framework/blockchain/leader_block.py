"""
blockchain/leader_block.py
==========================
Leader Block Selection using DB-BOA  (paper §V)

In consortium blockchain, one node per consensus round acts as the
"leader" that:
  • Collects and verifies incoming transactions
  • Broadcasts the new block with its signature
  • Earns rewards via the incentive mechanism

DB-BOA selects the optimal leader by minimising the combined cost:

    Obf1 = argmin { CT + CC + MS }           (Eq.10)
             {Lb_bc}

    CT — Computation Time   : how fast the node can process transactions
    CC — Communication Cost : bandwidth cost to broadcast the block
    MS — Memory Size        : current storage usage

This module also implements the incentive mechanism:
  • Leader earns base_reward tokens each round
  • Performance bonus if latency < threshold
  • Penalty if consensus fails
"""

import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config          import LEADER_BLOCK_CONFIG, INCENTIVE_CONFIG, LOG_WIDTH
from algorithms.db_boa import DBBOA


# ─── helpers ──────────────────────────────────────────────────────────────────

def _print(msg: str):
    print(f"[CHAIN] {msg}", flush=True)

def _sep(c="─"):
    print(c * LOG_WIDTH, flush=True)


# ─── Real Fabric consensus measurements (B2) ───────────────────────────────────

_MEASURED_CACHE = {}

def load_measured_consensus(path: str = None):
    """
    Load the REAL latency/throughput numbers measured against the live
    Hyperledger Fabric test-network by db_boa_fabric/api-server/measure_consensus.js
    (results/fabric_consensus_measured.json).  Returns the dict or None if the
    file is absent (caller falls back to the legacy simulated arithmetic).

    Cached so repeated rounds do not re-read the file.
    """
    import json
    path = path or LEADER_BLOCK_CONFIG.get("measured_consensus_file")
    if not path:
        return None
    if path in _MEASURED_CACHE:
        return _MEASURED_CACHE[path]
    data = None
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError):
        data = None
    _MEASURED_CACHE[path] = data
    return data


# ─── Node model ───────────────────────────────────────────────────────────────

class BlockchainNode:
    """
    Represents a single peer node in the consortium blockchain.

    Attributes
    ----------
    node_id       : int
    org           : str      — Org1 / Org2 / … (Hyperledger Fabric org)
    ct            : float    — normalised computation time  [0, 1]
    cc            : float    — normalised communication cost [0, 1]
    ms            : float    — normalised memory usage       [0, 1]
    reputation    : float    — accumulated over rounds
    tokens        : int      — incentive balance
    rounds_as_leader : int
    """

    def __init__(self, node_id: int, ct: float, cc: float, ms: float):
        self.node_id          = node_id
        self.org              = f"Org{(node_id % 2) + 1}"   # alternate Org1/Org2
        self.ct               = float(ct)
        self.cc               = float(cc)
        self.ms               = float(ms)
        self.reputation       = 1.0    # initialised at 1.0, bounded [0.5, 2.0] (§6.3)
        self.reliability      = 1.0    # P(block is well-formed) as leader; 1.0 = identity.
                                       # NOT part of the DB-BOA cost objective (CT+CC+MS),
                                       # so a drop here is invisible to the myopic optimiser
                                       # but is felt by the RL agent through reward.
        self.tokens           = 100    # starting balance
        self.rounds_as_leader = 0
        self.consensus_successes = 0
        self.consensus_failures  = 0

    @property
    def cost(self):
        return self.ct + self.cc + self.ms

    def __repr__(self):
        return (f"Node-{self.node_id:02d}({self.org})  "
                f"CT={self.ct:.3f}  CC={self.cc:.3f}  MS={self.ms:.3f}  "
                f"cost={self.cost:.3f}  tokens={self.tokens}")


# ─── Consortium Blockchain ────────────────────────────────────────────────────

class ConsortiumBlockchain:
    """
    Simulates the consortium blockchain network for the thesis demo.

    Key functions:
      • initialise_nodes()  — spin up N peer nodes
      • select_leader()     — run DB-BOA to pick the optimal leader
      • simulate_round()    — one full consensus round (propose → endorse → commit)
      • apply_incentive()   — reward / penalise nodes
    """

    def __init__(self, cfg: dict = None, seed: int = 42):
        self.cfg  = cfg or LEADER_BLOCK_CONFIG
        self.inc  = INCENTIVE_CONFIG
        self.rng  = np.random.RandomState(seed)
        self.seed = seed
        self.nodes: list[BlockchainNode] = []
        self.rounds: list[dict] = []
        self.current_leader: BlockchainNode = None
        self.rl_agent = None          # set via attach_rl_agent() for RL election

    # ── public API ────────────────────────────────────────────────────────────

    def initialise_nodes(self, verbose: bool = True):
        """Create N blockchain nodes with random resource profiles."""
        n = self.cfg["n_nodes"]
        ct_lo, ct_hi = self.cfg["ct_bounds"]
        cc_lo, cc_hi = self.cfg["cc_bounds"]
        ms_lo, ms_hi = self.cfg["ms_bounds"]

        if verbose:
            _sep("═")
            _print(f"Initialising consortium blockchain with {n} nodes …")
            _sep()
            _print(f"{'Node':<10} {'Org':<8} {'CT':>8} {'CC':>8} "
                   f"{'MS':>8} {'Cost':>10} {'Tokens':>8}")
            _sep()

        for i in range(n):
            ct  = self.rng.uniform(ct_lo, ct_hi)
            cc  = self.rng.uniform(cc_lo, cc_hi)
            ms  = self.rng.uniform(ms_lo, ms_hi)
            node = BlockchainNode(i, ct, cc, ms)
            self.nodes.append(node)

            if verbose:
                print(f"  Node-{i:02d}     {node.org:<8} "
                      f"{ct:>8.4f} {cc:>8.4f} {ms:>8.4f} "
                      f"{node.cost:>10.4f} {node.tokens:>8}", flush=True)

        if verbose:
            _sep()

    def select_leader(self, verbose: bool = True):
        """
        Use DB-BOA to select the optimal leader block.

        The search space is the N-dimensional space of node indices.
        The objective encodes Eq.10: minimise  CT + CC + MS  of the
        selected node (fractional selection → round to nearest node).
        """
        if verbose:
            _sep("═")
            _print("Running DB-BOA for Leader Block Selection …")
            _print(f"Objective: minimise  CT + CC + MS  (Eq.10)")
            _sep()

        n_nodes = len(self.nodes)
        costs   = np.array([nd.cost for nd in self.nodes])

        # ── objective function ────────────────────────────────────────────────
        def objective(x: np.ndarray) -> float:
            """
            x is a continuous 1-D vector in [0, n_nodes).
            We map it to the nearest node index and return that node's cost.
            Reputation is used as a tie-breaker (lower cost / higher rep wins).
            """
            idx = int(np.clip(round(x[0]), 0, n_nodes - 1))
            nd  = self.nodes[idx]
            # Small reputation bonus: lowers cost for high-reputation nodes
            return nd.cost - 0.05 * nd.reputation

        optimizer = DBBOA(
            objective_fn = objective,
            lb           = np.array([0.0]),
            ub           = np.array([float(n_nodes - 1)]),
            n_pop        = self.cfg["population_size"],
            max_iter     = self.cfg["max_iterations"],
            task_name    = "Leader Block Selection",
            seed         = self.cfg["random_state"],
        )

        best_pos, best_fit, history = optimizer.optimise(verbose=verbose)

        leader_idx = int(np.clip(round(best_pos[0]), 0, n_nodes - 1))
        self.current_leader = self.nodes[leader_idx]
        self.current_leader.rounds_as_leader += 1
        self._last_leader_history = history
        self._last_opt_stats      = optimizer.summary_stats()

        if verbose:
            _sep()
            _print(f"✔ LEADER SELECTED: Node-{leader_idx:02d} ({self.current_leader.org})")
            _print(f"  CT  = {self.current_leader.ct:.4f}")
            _print(f"  CC  = {self.current_leader.cc:.4f}")
            _print(f"  MS  = {self.current_leader.ms:.4f}")
            _print(f"  Total cost = {self.current_leader.cost:.4f}")
            _print(f"  Reputation = {self.current_leader.reputation:.4f}")
            _sep("═")

        return self.current_leader, leader_idx, history

    # ── Reinforcement-Learning leader election (paper title) ──────────────────

    def attach_rl_agent(self, agent=None, seed: int = None):
        """
        Attach (or lazily create) an RLLeaderSelector for RL-based election.

        Call once after initialise_nodes().  With no ``agent`` a fresh agent is
        built from RL_LEADER_CONFIG.  Returns the agent so the caller can read
        its learned θ / telemetry afterwards.
        """
        if agent is None:
            from blockchain.rl_leader import RLLeaderSelector
            agent = RLLeaderSelector(seed=self.seed if seed is None else seed)
        self.rl_agent = agent
        return agent

    def select_leader_rl(self, round_num: int = 1, greedy: bool = False,
                         verbose: bool = True):
        """
        Elect the leader for this round with the RL policy (ε-greedy over Q).

        Unlike select_leader() (which re-solves an optimisation from scratch),
        this consults the agent's learned value function.  The TD update happens
        later in run_rl_round() once the round's reward is known.
        """
        if self.rl_agent is None:
            self.attach_rl_agent()

        idx, phi = self.rl_agent.select(self.nodes, round_num, greedy=greedy)
        self.current_leader = self.nodes[idx]
        self.current_leader.rounds_as_leader += 1
        self._last_rl_phi   = phi
        self._last_rl_round = round_num

        if verbose:
            explored = self.rl_agent.history["explored"][-1]
            mode = "explore" if explored else "exploit"
            _print(f"✔ RL LEADER : Node-{idx:02d} ({self.current_leader.org})  "
                   f"Q={self.rl_agent.q(phi):.4f}  ε={self.rl_agent.epsilon:.3f} "
                   f"[{mode}]")
        return self.current_leader, idx

    def run_rl_round(self, round_num: int = 1, n_transactions: int = 50,
                     greedy: bool = False, verbose: bool = True):
        """
        One full RL consensus round: elect → simulate → observe reward → learn.

        Reward = the incentive payout to the elected leader (token delta applied
        by apply_incentives), i.e. the consensus mechanism's own signal — the
        agent is never given a separately engineered reward.

        Returns the simulate_consensus_round() result dict, augmented with the
        RL bookkeeping fields (reward, td_error, epsilon, explored).
        """
        leader, idx = self.select_leader_rl(round_num, greedy=greedy, verbose=verbose)

        tokens_before = leader.tokens
        result = self.simulate_consensus_round(
            round_num=round_num, n_transactions=n_transactions, verbose=verbose)
        reward = leader.tokens - tokens_before     # on-chain incentive payout

        # TD update against the post-round consortium state (s')
        self.rl_agent.learn(self._last_rl_phi, reward, self.nodes, round_num + 1)
        self.rl_agent.decay_epsilon()

        result.update({
            "leader_method" : "rl",
            "rl_reward"     : float(reward),
            "rl_td_error"   : self.rl_agent.history["td_error"][-1],
            "rl_epsilon"    : self.rl_agent.history["epsilon"][-1],
            "rl_explored"   : self.rl_agent.history["explored"][-1],
        })
        return result

    def simulate_consensus_round(self, round_num: int = 1,
                                  n_transactions: int = 50,
                                  verbose: bool = True):
        """
        Simulate one full consensus round:
          1. Leader proposes a block of transactions
          2. Endorser peers (all other nodes) validate
          3. Ordering service sequences the block
          4. Block is committed to all peers

        Returns round metrics (latency, throughput, success flag).
        """
        if self.current_leader is None:
            raise RuntimeError("Call select_leader() before simulating a round.")

        t0 = time.time()

        if verbose:
            _sep()
            _print(f"Consensus Round {round_num:03d}  |  "
                   f"Leader: Node-{self.current_leader.node_id:02d}  |  "
                   f"Transactions: {n_transactions}")

        # ── Phase 1: Leader proposes block ───────────────────────────────────
        # The propose/endorse/order/commit *phasing* below is illustrative (ct is
        # a normalised resource score, not a network measurement).  The reported
        # round latency/throughput, however, are REAL wall-clock numbers measured
        # against the live Fabric test-network when use_measured_consensus=True
        # (see the elapsed/latency block below and load_measured_consensus); the
        # resource-score arithmetic is only the fallback when no measurement file
        # is present.
        proposal_latency = self.current_leader.ct * 0.1  # seconds (scaled)
        time.sleep(min(proposal_latency, 0.01))  # tiny sleep for demo

        if verbose:
            _print(f"  ► [PROPOSE ]  Node-{self.current_leader.node_id:02d} "
                   f"broadcasts block with {n_transactions} txns …")

        # ── Phase 2: Endorsement by other peers ──────────────────────────────
        endorsers = [nd for nd in self.nodes if nd != self.current_leader]
        endorse_latencies = []
        n_endorsed = 0

        for nd in endorsers[:4]:   # require 4 endorsements (realistic quorum)
            # Simulate endorsement: fail if node has very high memory usage.
            # Endorsers also reject a malformed proposal, so the leader's
            # reliability (1.0 by default → no effect) gates endorsement too.
            success_prob = (1.0 - 0.1 * nd.ms) * self.current_leader.reliability
            endorsed = self.rng.rand() < success_prob
            if endorsed:
                n_endorsed += 1
                endorse_latencies.append(nd.ct * 0.05)
                if verbose:
                    _print(f"  ► [ENDORSE ]  Node-{nd.node_id:02d}({nd.org}) "
                           f"✔ endorsed  (latency={nd.ct*0.05*1000:.1f} ms)")
            else:
                if verbose:
                    _print(f"  ► [ENDORSE ]  Node-{nd.node_id:02d}({nd.org}) "
                           f"✗ failed to endorse")

        consensus_ok = n_endorsed >= 2   # require ≥2 endorsements

        # ── Phase 3: Ordering ─────────────────────────────────────────────────
        order_latency = 0.02
        if verbose:
            _print(f"  ► [ORDER   ]  Orderer sequences the block …")

        # ── Phase 4: Commit ───────────────────────────────────────────────────
        commit_latency = np.mean(endorse_latencies) if endorse_latencies else 0.05
        if verbose:
            _print(f"  ► [COMMIT  ]  Block committed to all {len(self.nodes)} peers.")

        elapsed   = time.time() - t0

        # Latency / throughput: REAL measured (B2) when available, else simulated.
        measured = (load_measured_consensus()
                    if LEADER_BLOCK_CONFIG.get("use_measured_consensus", False)
                    else None)
        if measured:
            # Draw this round's latency from the measured Fabric distribution
            # (mean ± observed spread), and report the measured sustained TPS.
            m_mean = measured["latency_ms"]["mean_ms"]
            m_p95  = measured["latency_ms"]["p95_ms"]
            m_p50  = measured["latency_ms"]["p50_ms"]
            spread = max(1.0, (m_p95 - m_p50))
            latency    = float(max(1.0, self.rng.normal(m_mean, spread * 0.5)))
            throughput = float(measured["peak_tps"])
            self._latency_source = "measured-fabric"
        else:
            latency   = (proposal_latency + max(endorse_latencies or [0.02]) +
                         order_latency + commit_latency) * 1000  # ms
            throughput = n_transactions / max(elapsed, 0.001)     # tps
            self._latency_source = "simulated"

        # ── Incentive mechanism ───────────────────────────────────────────────
        self.apply_incentives(consensus_ok, latency)

        result = {
            "round"        : round_num,
            "leader"       : self.current_leader.node_id,
            "n_endorsed"   : n_endorsed,
            "consensus_ok" : consensus_ok,
            "latency_ms"   : latency,
            "throughput"   : throughput,
            "latency_source": getattr(self, "_latency_source", "simulated"),
            "n_txns"       : n_transactions,
        }
        self.rounds.append(result)

        if verbose:
            status = "✔ CONSENSUS OK" if consensus_ok else "✗ CONSENSUS FAILED"
            _print(f"  {status}  |  latency={latency:.1f} ms  |  "
                   f"throughput={throughput:.1f} tps")

        return result

    def apply_incentives(self, success: bool, latency_ms: float):
        """
        Apply the incentive mechanism to the current leader (§6.2 table).

        success=True :  +leader_success_reward (10)
                        +latency_bonus (15) if latency < latency_threshold_ms
                        reputation += 0.02, bounded at 2.0
        success=False:  −consensus_fail_penalty (2)
                        reputation -= 0.05, bounded at 0.5
        """
        threshold = self.inc.get("latency_threshold_ms", 300)
        if success:
            reward = self.inc["leader_success_reward"]
            if latency_ms < threshold:
                reward += self.inc["latency_bonus"]
            self.current_leader.tokens += reward
            self.current_leader.consensus_successes += 1
            self.current_leader.reputation = min(
                2.0, self.current_leader.reputation + 0.02)
        else:
            self.current_leader.tokens -= self.inc["consensus_fail_penalty"]
            self.current_leader.consensus_failures += 1
            self.current_leader.reputation = max(
                0.5, self.current_leader.reputation - 0.05)

    def print_node_status(self):
        """Print current token balance and reputation of all nodes."""
        _sep("═")
        _print("Node Status after Incentive Application:")
        _sep()
        _print(f"{'Node':<10} {'Org':<8} {'Tokens':>8} {'Rep':>8} "
               f"{'Leader?':>8} {'Wins':>6} {'Fails':>6}")
        _sep()
        for nd in self.nodes:
            is_leader = "★" if nd == self.current_leader else " "
            print(f"  Node-{nd.node_id:02d}  {nd.org:<8} "
                  f"{nd.tokens:>8}  {nd.reputation:>8.3f}  "
                  f"{is_leader:>8}  {nd.consensus_successes:>6}  "
                  f"{nd.consensus_failures:>6}", flush=True)
        _sep("═")


# ─── RL vs DB-BOA head-to-head ────────────────────────────────────────────────

def _gini(counts) -> float:
    """Gini coefficient of a non-negative array (0 = perfectly equal share)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def _run_method(method: str, n_rounds: int, seed: int, cfg: dict = None,
                degrade_node: int = None, degrade_round: int = None):
    """
    Replay ``n_rounds`` consensus rounds under one leader-election ``method``
    ('db_boa' or 'rl') on an identically-seeded node population.

    Optional non-stationarity: from ``degrade_round`` onward, ``degrade_node``'s
    *reliability* collapses (its blocks start getting rejected), so electing it
    yields failed rounds and negative reward.  Crucially, reliability is NOT in
    the DB-BOA cost objective (CT+CC+MS), so the myopic optimiser keeps electing
    the now-bad node on its stale low-cost profile, while the RL agent feels the
    negative reward and learns to avoid it.  ``degrade_node="auto"`` targets the
    lowest-cost node — exactly the one DB-BOA monopolises — making the contrast
    sharp.

    Returns per-round rewards, the leadership histogram, and summary stats.
    """
    bc = ConsortiumBlockchain(cfg=cfg, seed=seed)
    bc.initialise_nodes(verbose=False)
    agent = bc.attach_rl_agent(seed=seed) if method == "rl" else None

    if degrade_node == "auto":
        degrade_node = int(np.argmin([nd.cost for nd in bc.nodes]))

    rewards, successes, leaders = [], [], []
    for r in range(1, n_rounds + 1):
        if degrade_node is not None and degrade_round is not None and r >= degrade_round:
            bc.nodes[degrade_node].reliability = 0.15   # blocks mostly rejected

        if method == "rl":
            res = bc.run_rl_round(round_num=r, n_transactions=50, verbose=False)
            rewards.append(res["rl_reward"])
        else:
            # DB-BOA re-solves the single-round optimiser each round (myopic).
            leader, idx, _ = bc.select_leader(verbose=False)
            before = leader.tokens
            res = bc.simulate_consensus_round(round_num=r, n_transactions=50,
                                              verbose=False)
            rewards.append(leader.tokens - before)
        successes.append(bool(res["consensus_ok"]))
        leaders.append(bc.current_leader.node_id)

    counts = np.bincount(leaders, minlength=len(bc.nodes))
    out = {
        "rewards"       : [float(x) for x in rewards],
        "cum_reward"    : np.cumsum(rewards).tolist(),
        "total_reward"  : float(np.sum(rewards)),
        "success_rate"  : float(np.mean(successes)),
        "leader_counts" : counts.tolist(),
        "leader_gini"   : _gini(counts),
        "leaders"       : leaders,
        "degrade_node"  : degrade_node,
    }
    if method == "rl":
        out["theta"]   = agent.theta.tolist()
        out["epsilon"] = agent.history["epsilon"]
    return out


def compare_leader_methods(n_rounds: int = 20, seed: int = 42, cfg: dict = None,
                           degrade_node: int = None, degrade_round: int = None):
    """
    Head-to-head DB-BOA vs RL leader selection on the same node population.

    Returns ``{"db_boa": {...}, "rl": {...}, "n_rounds": n}`` — see _run_method
    for per-method fields (cumulative reward, success rate, leadership Gini).
    """
    return {
        "n_rounds": n_rounds,
        "db_boa"  : _run_method("db_boa", n_rounds, seed, cfg,
                                degrade_node, degrade_round),
        "rl"      : _run_method("rl",     n_rounds, seed, cfg,
                                degrade_node, degrade_round),
    }
