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
            # Simulate endorsement: fail if node has very high memory usage
            success_prob = 1.0 - 0.1 * nd.ms
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
        latency   = (proposal_latency + max(endorse_latencies or [0.02]) +
                     order_latency + commit_latency) * 1000  # ms
        throughput = n_transactions / max(elapsed, 0.001)     # tps

        # ── Incentive mechanism ───────────────────────────────────────────────
        self.apply_incentives(consensus_ok, latency)

        result = {
            "round"        : round_num,
            "leader"       : self.current_leader.node_id,
            "n_endorsed"   : n_endorsed,
            "consensus_ok" : consensus_ok,
            "latency_ms"   : latency,
            "throughput"   : throughput,
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
