# Title Issues — Claim-by-Claim Audit (current verdicts)

How well the **current thesis title** matches what the code actually does, as of 2026-06-06.

> **Current title** (`FINAL YEAR THESIS REPORT/core/titlepage.tex:7`):
> *"A Blockchain-Integrated Framework for Secure, Incentivized, and Scalable Machine
> Learning: Integrating Consensus Mechanisms and Reinforcement Learning"*

| Title claim | Backed by code? | Verdict |
|---|---|---|
| Blockchain-Integrated | Real Hyperledger Fabric test-network + chaincode | ✅ Supported |
| Incentivized | On-chain token/reputation chaincode | ✅ Supported (not novel) |
| Reinforcement Learning | RL leader selection: linear-FA Q-learning over consensus rounds (`blockchain/rl_leader.py`), default leader method | ✅ Supported (qualify: simulated, secondary) |
| Secure | DP + **weight-level Krum BFT run at f≥1** (Task D: n=5/f=1 & n=7/f=2, theorem n≥2f+3 holds; rejects 4 poisoning attacks 8/8). f=0 only in the default n=3 pipeline | ✅ Supported (qualify: ≤f orgs, simulated) |
| Consensus Mechanisms | Stock Fabric Raft; our "consensus round" is simulated | ⚠️ Overstated — soften |
| Scalable Machine Learning | Scalable *contribution attribution* only (MC-Shapley); throughput still simulated | ⚠️ Overstated — qualify |

## Does the title fully support the research? — **No, not fully (but closeable in writing)**

Six claims: **two** solidly true, **two** (RL, Secure) supported-but-qualified, **two** still overstated.
The **two weakest words are now "Scalable Machine Learning" and "Consensus Mechanisms"** — RL and
Secure are no longer the danger lines (Secure is now backed in code by Task D, not just softened).

1. **"Scalable Machine Learning"** read literally promises system/throughput scalability; we only have
   *algorithmic* scalability of the *contribution-attribution* layer. Overclaims in bare form (§3).
2. **"Consensus Mechanisms"** reads as "we built a consensus protocol"; we use Fabric's stock Raft and
   simulate the consensus round. Contribution is leader-selection + incentives, not consensus (§1).
3. **"Secure"** is now **backed in code** — Task D runs Krum where its theorem holds (n≥2f+3, f≥1)
   and it rejects 4 weight-poisoning attacks 8/8; the only remaining qualifier is the standard Krum
   scope (≤f colluding orgs) and that it is a single-process simulation. See §5.

**The remaining gap is closeable in the writing, not the code.** Two qualifying sentences
(abstract/intro + Limitations) make the prose match the title (Secure is closed in §5):
- *Scalable* → scope to **scalable contribution attribution**.
- *Consensus* → "we use Fabric's stock Raft; our contribution is RL-governed leader selection and the
  incentive layer, with the consensus round simulated."
- *Secure* → "DP + Krum Byzantine-robust aggregation, demonstrated at f≥1 (n≥2f+3); the f=0 default is
  the n=3 deployment, not a limit of the method."

---

## 1. Consensus Mechanisms — ⚠️ overstated

The word "consensus" does three different jobs; only one is a real blockchain protocol.

- **Real consensus that runs:** `db_boa_fabric` brings up the upstream Fabric **test-network**
  (`./network.sh up`) with the default **Raft (etcdraft) orderer** at `orderer.example.com:7050`.
  Genuine but **stock and unmodified** — not our contribution.
- **Our "consensus mechanism" is a simulation.** `leader_block.py:simulate_consensus_round()` fakes
  propose → endorse → order → commit with `time.sleep` + resource-score arithmetic; the code discloses
  this (`leader_block.py:220-225`). Latency/throughput are arithmetic, not measured.
- **ML "consensus"** (Krum "consensus-aligned org", majority-vote fraud verdict `(v_a+v_b+v_c)>=2`) is
  *agreement*, not a blockchain consensus protocol.

**Viva line:** *"We use Fabric's default Raft orderer; our contribution is the RL/DB-BOA
leader-selection + incentive layer, with the consensus round simulated."*

## 2. Incentive Mechanism — ✅ real, but NOT novel

- Implemented **on-chain**: `db_boa_chaincode.js:updateIncentive()` mutates token/reputation state,
  clamps reputation to [0.5, 2.0], writes world state, emits `IncentiveApplied`. Driven by
  `recordFraudResult`, `recordConsensusRound`, `recordFederationRound` (20-token pool split by Shapley
  weight). Mirrored in Python (`config.py INCENTIVE_CONFIG`, `leader_block.py:apply_incentives()`).
- **Not novel:** "Shapley contribution → on-chain token reward" is prior art (**FedCoin 2020**, **πFL**).
  Do **not** claim "first to bind Shapley to blockchain incentives."
- Reward values (+10/+15/−2/+20) are hand-set constants. The +15 latency bonus is keyed off the
  *simulated* consensus latency.
- The mechanism is the **object of study**, not the novelty. The novelty is *characterising when it
  breaks* (privacy-noise → wrong on-chain rewards; economic attacker isolation).

## 3. Scalable Machine Learning — ⚠️ qualify (attribution layer only)

Satisfied for the *contribution-attribution* layer; the system/throughput sense is not.

- `n_orgs` is parameterised end-to-end: `config.make_org_splits(n)`, `data_loader.split_for_orgs()`
  (+ equal-shard `samples_per_org` mode), and `experiments/scalability_sweep.py` sweeps n ∈ {3 … 20}.
  Default stays n=3, so the existing pipeline is byte-for-byte unchanged.
- `federation_manager._shapley_weights_mc()` adds a Monte-Carlo permutation estimator (TMC-Shapley,
  Ghorbani & Zou 2019) at O(samples·n); `_shapley_weights_exact()` keeps the exact baseline; a
  `shapley_method` flag dispatches between them.
- **Throughput/latency remain simulated arithmetic** (`leader_block.py`), not measured. Large dataset
  (284,807 rows) is *data* size, not system scalability.

**What the sweep measures (real wall-clock, single process; full run n_orgs 3→20, 8 000 samples/org,
mc_samples=200, fraud-stratified val set):**
- **Runtime (feasibility):** exact Shapley ≈0.14 s at n=3 → **113 s at n=12** (4095 coalitions),
  infeasible beyond (n=20 ⇒ 2ⁿ ≈ 1.05 M). MC runs every size up to **n=20 in 95 s**. Honest framing is
  a **crossover**: below n≈10 MC ≈1× (exact is cheap anyway); above it MC touches a small fraction of 2ⁿ
  (≈3.3× at n=12) and is the *only* estimator that still terminates. **Claim = tractability at scale,
  not a large constant-factor win.**
- **Fidelity:** MC weights track exact — **mean Spearman ρ≈+0.53** (up to +1.0), L1 ≤ 0.26, L∞ ≤ 0.10,
  so the token split (`federation_pool × Shapley weight`) is broadly preserved. Top contributor 5/10
  (flips on sub-noise ties). **Caveat:** at a fixed sample budget fidelity degrades as n grows (ρ≈0 at
  n=12) → samples must scale ≈O(n log n). TMC truncation left OFF.
- **Accuracy under scaling (confound-controlled):** *equal-shard* stays ~**79–87 %** balanced accuracy
  across n (size alone doesn't hurt); *fixed-pool* shows realistic data-dilution **92.3 % → 85.2 %**
  (n=3→12) as fraud positives thin out — a data-budget limit of the ULB set, disclosed.

Artifacts: `experiments/scalability_sweep.py`, `results/scalability_sweep.json`,
`results/scalability_shapley_runtime.png`, `results/scalability_fidelity_accuracy.png`,
`final_report_data/TASKC_scalability_results.md`, paste-ready LaTeX in
`final_report_data/REWRITE_06_results.md` §6.8.

**Action:** claim **"scalable contribution attribution"**, NOT bare "Scalable Machine Learning." Keep
throughput/distributed/Byzantine (Krum f=0) senses in Limitations.

## 4. Reinforcement Learning — ✅ supported (simulated, secondary)

RL is integrated where it is genuinely the right tool: **sequential leader selection**.

- `blockchain/rl_leader.py` implements `RLLeaderSelector`: linear function-approximation **Q-learning**
  (Sutton & Barto 2018, §9–10), ε-greedy, γ=0.9, explicit **MDP** (state = node CT/CC/MS + reputation +
  tokens + load + fail-rate; action = elected leader; reward = on-chain incentive payout; TD update
  `θ ← θ + α[r + γ·maxₐ′Q(s′,a′) − Q(s,a)]·φ`). The agent's reward **is** the token payout
  (`leader.tokens` delta from `apply_incentives`) — no separately engineered reward.
- `leader_block.py` adds `attach_rl_agent` / `select_leader_rl` / `run_rl_round` + a
  `compare_leader_methods` head-to-head; `config.py LEADER_BLOCK_CONFIG["leader_method"]` defaults to
  **"rl"** (DB-BOA still does the Phase-1 one-shot cold-start pick).

**Results (verified full run, 40 rounds × 5 seeds; `results/rl_leader_sweep.json`,
`results/rl_leader_reward_fairness.png`, `results/rl_leader_adaptivity.png`):**
- **Stationary:** RL matches DB-BOA on reward (1000) and success (100%) but rotates leadership more
  fairly — **leadership Gini 0.78 vs 0.90** (lower = fairer).
- **Non-stationary (a node's reliability collapses mid-run):** DB-BOA keeps electing the now-bad node
  **100%** of post-degrade rounds (reliability is invisible to its CT+CC+MS objective); RL learns to
  avoid it, electing it only **~15%**.

**Caveats to keep in the report:**
- The non-stationary win depends on the **added** `reliability` fault (default 1.0 = no-op), so the
  claim is **conditional** — "RL helps when node behaviour drifts in a way the cost objective can't
  see," not universal. On stationary reward RL only **ties** DB-BOA.
- A **simulation** result, not a live Fabric testnet.
- A **secondary** consensus-layer contribution; the primary novelty is the privacy↔incentive
  characterisation.

**Action:** keep the RL term; write up from `final_report_data/08_rl_leader_selection.md`. A *linear*
agent is the deliberate, defensible choice — do not claim DQN/PPO/deep RL.

## 5. Secure — ✅ now backed in code (Task D), not just softened

The earlier verdict ("⚠️ overstated — soften in prose") was driven by the default pipeline running Krum
at **n=3, f=0**, where Krum's robustness theorem (n ≥ 2f+3, Blanchard et al. NeurIPS 2017) reduces to
plain outlier rejection — "no adversary assumed." Rather than only reword, we **closed it in code**.

- `experiments/byzantine_robustness_sweep.py` runs the project's own Krum aggregator
  (`FederationManager._krum_aggregate`, already general in f) in the regimes where the theorem holds:
  **n=5, f=1** (5 ≥ 5) and **n=7, f=2** (7 ≥ 7), against four *weight-level* poisoning attacks —
  **sign-flip**, **scaled** (λ=50 norm-boosting), **Gaussian** junk, and a genuinely retrained
  **label-flip** model. This is distinct from Task B, which attacks at the *prediction* layer and
  characterises the complementary *economic* defence against a coordinated **majority** (> f).
- **Result (verified full run; `results/byzantine_robustness_sweep.json`,
  `results/byzantine_robustness_krum_vs_fedavg.png`, `final_report_data/TASKD_byzantine_robustness_results.md`):**
  Krum's score flags the poisoned org as the cluster outlier and **rejects it 8/8** (both regimes × all
  four attacks); the Krum-selected global model stays at **≈99.9%** balanced accuracy throughout. The
  outlier margin is graded — **≈10⁶** (scaled) → **≈10³** (sign-flip / Gaussian) → **≈10⁰** (subtle
  label-flip, which sits only just outside the honest spread).
- **Honest FedAvg comparison:** plain averaging only *collapses* under the magnitude-dominant **scaled**
  attack (**≈87.5% vs Krum's ≈99.9%, +12.5%**); for the other three a single (≤f) poisoned vector is
  diluted by the honest majority, so FedAvg happens to survive too. The claim is therefore **not** "FedAvg
  always fails" but "**Krum gives a *uniform* guarantee** — constant ≈99.9% and a provably-rejected
  attacker — whereas FedAvg's safety is attack-dependent and fails catastrophically under norm-boosting."

**Caveats to keep (the honest boundary):** (i) standard Krum scope — tolerates **≤ f** colluding orgs; a
coordinated **majority** (> f) still defeats it (that is exactly where Task B's economic loop takes over —
the two are complementary); (ii) the subtle **label-flip** margin is thin (~10⁰), so a sufficiently subtle
attack would eventually slip under the honest cluster's own spread; (iii) run with **DP off** to isolate
robustness (DP+Krum interaction is the privacy sweep), single-process **simulation**, not a live testnet.

**Action:** claim **"Secure"** in the qualified BFT sense — *"DP for privacy + Krum Byzantine-robust
aggregation, demonstrated tolerating f≥1 poisoned orgs where n≥2f+3"* — and keep the ≤f / majority /
label-flip-margin caveats in Limitations. The f=0 default is the n=3 deployment choice, **not** a limit of
the method.

---

## Net

The title is **fixed/approved and kept**. **RL** is now supported (simulated, secondary) — the
previously fatal "D15" line is resolved. **Secure** is now **backed in code** (Task D: Krum BFT at f≥1,
rejects 4 poisoning attacks 8/8) — qualify to ≤f orgs, no longer "soften only." **Consensus** remains the
one genuinely overstated word (stock Raft + simulated round — soften in prose). **Scalable ML** holds
only in the qualified "scalable contribution attribution" sense, not the bare system-throughput sense.
The honest core of the work is **a blockchain-integrated federated fraud-detection system whose research
contribution is the *characterisation* of its privacy↔incentive and economic-Byzantine behaviour, plus a
demonstrated statistical-BFT (Krum f≥1) + economic dual defence, a scalable fidelity-preserving
contribution-attribution layer, and an RL-governed leader-selection policy** — not new learning
algorithms. See `NOVELTY_TIPS.md`, `final_report_data/08_rl_leader_selection.md`,
`final_report_data/TASKD_byzantine_robustness_results.md`,
`final_report_data/TASKC_scalability_results.md`, `final_report_data/00_report_vs_code_divergences.md`.
