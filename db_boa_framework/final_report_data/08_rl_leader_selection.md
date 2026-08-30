# Chapter 4/5 — Reinforcement-Learning Leader Selection (the title's "RL")

**Status:** draft for review. Code is implemented, run, and verified
(`blockchain/rl_leader.py`, `blockchain/leader_block.py`,
`experiments/rl_leader_sweep.py`). Apply to `.tex` only after approval.

**Why this section exists.** The thesis title names *Reinforcement Learning*. Before
this work there was **no RL anywhere in the code** (see `title_issue.md`). Rather than
remove the term (title is fixed/approved), RL was integrated where it is genuinely the
right tool: **sequential leader selection** in the consortium. This note is the
truthful, code-traceable description — write the report from this, and **do not
overclaim beyond it**.

---

## 4.x Motivation — why leader selection is an RL problem

The original leader selector (`leader_block.py::select_leader`) uses DB-BOA to solve a
**single-round, stateless** optimisation each round: `argmin (CT + CC + MS)` (Eq. 10).
But in a running consortium the *same* nodes are elected round after round, and their
reputation, token balances and leadership load evolve. Electing purely on a static cost
profile is myopic: it monopolises one node and cannot react to a node whose behaviour
degrades over time.

We therefore model leader selection as a **Markov Decision Process** and learn a leader
*policy* with reinforcement learning. This is a **secondary, consensus-layer
contribution** — the primary novelty remains the privacy↔incentive characterisation
(Chapters on DP / Shapley / economic isolation).

## 4.x MDP formulation (honest mapping)

| MDP element | Implementation |
|---|---|
| state `s_t` | per-node feature vector: CT, CC, MS, reputation (→[0,1]), token balance (relative to mean, tanh-squashed), leadership load, recent failure rate (`rl_leader.py::node_features`, 8-dim φ) |
| action `a_t`| which node is elected leader for round *t* |
| reward `r_t`| the **on-chain incentive payout** to the elected leader — `+leader_success_reward (+latency_bonus)` on success, `−consensus_fail_penalty` on failure (`INCENTIVE_CONFIG`). **No reward is hand-engineered for the agent** — it is exactly the token the consensus mechanism already pays. |
| transition  | `simulate_consensus_round()` endorsement / reputation dynamics |

## 4.x Algorithm — linear-FA Q-learning

Action value is **linear** in the node feature vector:  Q(s,a) = θ·φ(s,a). θ is updated
by the temporal-difference rule (bootstrapped, γ = 0.9):

> δ = r + γ·maxₐ′ Q(s′,a′) − Q(s,a),  θ ← θ + α·δ·φ(s,a)

ε-greedy exploration with decay (ε: 0.30 → 0.02). Because γ > 0, the agent performs
**long-horizon credit assignment** the single-round DB-BOA objective cannot express.

**Defend the design choice:** a *linear* agent (Sutton & Barto 2018, §9–10) — not DQN/PPO
— is deliberate. With 10–20 nodes the policy is small, trains in milliseconds, is fully
reproducible from a seed, and is simple to justify. Deep RL would be unjustified
machinery here; say so explicitly.

**Relationship to DB-BOA (not a replacement):** DB-BOA still performs the one-shot
**cold-start** pick (Phase 1) — a single selection the agent cannot have learned yet. RL
governs the **sequential** multi-round election (Phase 5). They are complementary.

## 5.x Results (verified — full run, 40 rounds × 5 seeds)

Source: `results/rl_leader_sweep.json`, figures `rl_leader_reward_fairness.png`,
`rl_leader_adaptivity.png`. Reproduce with `python3 experiments/rl_leader_sweep.py`.

**Regime 1 — stationary:**

| Metric | DB-BOA (myopic) | RL (Q-learning) |
|---|---|---|
| Cumulative reward | 1000 ± 0 | 1000 ± 0 |
| Consensus success rate | 100% | 100% |
| **Leadership Gini** (lower = fairer rotation) | **0.90 ± 0.00** | **0.78 ± 0.02** |

Reading: when every node can succeed, both earn the same reward, but DB-BOA monopolises
the single lowest-cost node (Gini 0.90) while RL rotates leadership more evenly (0.78).

**Regime 2 — non-stationary (a node goes bad mid-run):** from round 20 the lowest-cost
node's *reliability* collapses (its blocks start getting rejected).

| Metric | DB-BOA | RL |
|---|---|---|
| **Re-electing the bad node after it degrades** | **100%** | **15%** |

Reading: reliability is **not** part of DB-BOA's CT+CC+MS objective, so it keeps electing
the now-bad node every round on its stale low-cost profile. The RL agent feels the
negative reward and learns to avoid it.

## 5.x Threats to validity — STATE THESE, do not hide them

An examiner will push on exactly these; pre-empt them in the text:

1. **Constructed advantage.** The Regime-2 adaptivity gap depends on the `reliability`
   fault we *added* to the node model (`BlockchainNode.reliability`, default 1.0 = no-op).
   It is a deliberately chosen non-stationarity, not an emergent property. Frame the
   claim as *conditional*: "RL helps **when** node behaviour drifts in a way the static
   cost objective cannot observe." Do not claim RL beats DB-BOA universally — on
   stationary reward it does not.
2. **Simulation, not deployment.** The whole consensus/latency layer is simulated
   (resource-score arithmetic + `time.sleep`), not a live Hyperledger Fabric testnet
   (already stated in `leader_block.py` and the Limitations section). RL leader selection
   is therefore a **simulation result**; carry that caveat into this section.
3. **Scope.** This is a single, small RL component, framed as characterisation of *where*
   sequential election helps — consistent with the thesis's "characterisation, not new
   algorithms" novelty stance. It is **secondary** to the privacy↔incentive contribution.

## Honest one-line summary for the abstract/conclusion

> Leader selection is cast as a Markov decision process and solved with linear-FA
> Q-learning; in simulation it matches the DB-BOA optimiser on reward and consensus
> success while improving leadership fairness, and — unlike the myopic optimiser —
> adapts to nodes whose reliability degrades at runtime.
