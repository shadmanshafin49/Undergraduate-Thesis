# Report vs. Code — Divergence Audit (READ FIRST)

This is the authoritative list of claims in `FINAL YEAR THESIS REPORT/` that the
actual code does **not** support. Each item: what the report says, what the code
actually does (with file reference), and the recommended action.

Severity: 🔴 = must fix (factual misrepresentation / fabricated result), 🟠 = should
fix (overstated / inconsistent), 🟢 = minor wording.

---

## 🔴 D1 — "DB-BOA Job 3" is NOT the federation method actually used

- **Report claims:** DB-BOA Job 3 (federated aggregation weight optimisation) is the
  *primary novel contribution*; weights converge to `[0.52, 0.32, 0.16]`; reward pool
  is split by DB-BOA weights. (chapter_5.tex §5.1.5, §5.2.4; chapter_6.tex §"DB-BOA
  Aggregation Weights"; chapter_9.tex.)
- **Code actually does:** `FEDERATION_CONFIG['use_shapley'] = True`
  (`db_boa_framework/config.py`). `federation_manager.py` computes **exact Shapley
  values** over all `2^3-1 = 7` coalitions for contribution weights, uses **Krum** to
  pick the global model, and **DB-BOA Job 3 is the disabled `else` fallback** only
  (`run_federation_round()` → `_shapley_weights()` is the live path;
  `_run_db_boa_job3()` runs only when `use_shapley=False`). The chaincode comment and
  `recordFederationRound` also state "shared by Shapley contribution weight."
- **Action:** Decide the actual novelty story and make report + code agree. Two options:
  - **(A) Re-frame the novelty around what is built:** "Shapley-value contribution
    attribution + Krum robust selection + DP weight sharing on a live Fabric ledger,
    coupled to on-chain incentives." This is defensible and *real*. DB-BOA then has two
    roles (hyperparameters, leader selection), not three.
  - **(B) Switch code back to DB-BOA Job 3** (`use_shapley=False`, `use_krum=False`) and
    re-run — but note the saved DB-BOA Job-3 weights do **not** converge (see D11) and
    its fitness is degenerate, so (A) is the stronger path.
- This single decision drives the title, abstract, Ch.1 objectives, Ch.5 §5.1.5, Ch.6,
  and the Conclusion.

---

## 🔴 D2 — ADTCN architecture is described as something it is not

- **Report claims** (chapter_5.tex §5.1.4, §5.2.8, §5.2.9, lines 816–818, Table 4.5):
  - MJE = linear projection of **29-dim → 64-dim** latent space with BatchNorm + Dropout
  - TCL = **dilated** temporal conv backbone, dilation `{1,2,4,8}`, receptive field ~15
  - MTTA = **softmax attention** pooling
- **Code actually is** (`db_boa_framework/models/adtcn.py`, class `_Conv1dClassifier`):
  - Input = **30 raw features** (33 with optional recurrence features), as a 10-step
    sequence — **no 64-dim embedding, no BatchNorm**.
  - `Conv1d(30, F, k=3, pad=1) → ReLU → Conv1d(F, 2F, k=3, pad=1) → ReLU → GlobalMaxPool
    → Linear(2F, 2)`. **No dilation** (all kernels dilation=1). "TCL" = these two conv
    layers over the window.
  - "MTTA" = `x.amax(dim=-1)` **global max-pool, not attention**. The source comment
    says so explicitly: *"pooling, not attention; the paper's MTTA label is re-used."*
- **Action:** Rewrite the architecture section to describe the real 2-layer 1D-CNN +
  global max-pool + linear head. You can still *name* the conceptual blocks (MJE/TCL/MTTA
  from the base paper) but must state how each is realised: MJE = raw multi-feature input,
  TCL = stacked 1D convolutions over a 10-step window, MTTA = global max-pooling. Remove
  "dilated", "64-dimensional", "BatchNorm", "softmax attention".

---

## 🔴 D3 — FedProx is claimed but not implemented

- **Report claims:** FedProx proximal regularisation with `μ = 0.05` is applied for
  non-IID; includes a **FedProx μ-sensitivity table** (chapter_5.tex lines 449, 470, 498,
  536, 545, 665–690, 837; chapter_6.tex line 144).
- **Code actually does:** No FedProx anywhere. `grep -ri fedprox db_boa_framework`
  returns **only a comment** in `config.py` saying FedProx *"would be more appropriate"*
  for severe heterogeneity. Local training is plain weighted cross-entropy; non-IID is
  handled (if at all) only by **stratified `split_for_orgs()`**, which keeps the 0.17%
  fraud rate identical in each shard.
- **Action:** Remove every FedProx claim and the μ-sensitivity table. Replace with the
  truth: heterogeneity is addressed by **stratified partitioning** (each org keeps the
  global fraud rate). List FedProx as *future work*, not as done.

---

## 🔴 D4 — The 8-model comparison table is copied from the base paper / fabricated

- **Report claims** (chapter_6.tex Table "Comprehensive Model Performance Comparison"):
  a full table for DTCN, EfficientNet, ResNet, DenseNet, MBO-ADTCN, WSA-ADTCN, BOA-ADTCN,
  DBOA-ADTCN, and "FL-ADTCN (Ours) 97.38% / MCC 0.966." Chapters 3 and 5 also say the
  framework is "evaluated against baselines from the base paper (MBO-ADTCN, WSA-ADTCN,
  … EfficientNet, ResNet, DenseNet, DTCN)."
- **Code actually does:** `utils/metrics.py::baseline_metrics()` returns **empty dicts**;
  its docstring states the base-paper numbers *"were produced on synthetic data and
  cannot be validly compared against ULB results. They have been removed."* The real
  baselines are defined in `run_baselines.py` / `config.py`: **FedAvg, FedAvg+Krum,
  FedAvg+DP, DB-BOA-ADTCN (proposed = Krum+DP+Shapley)** — and they are **not yet
  populated** (must run `run_baselines.py`).
- **Action:** Delete the 8-model base-paper comparison table (or clearly label it as
  "values quoted from base paper [prabanand2025] on their data, not re-run here" — but it
  is cleaner to remove). Replace with the **FedAvg / FedAvg+Krum / FedAvg+DP / proposed**
  comparison produced by `run_baselines.py`.

---

## 🔴 D5 — Headline result "97.38% accuracy, MCC 0.966" is not produced by any run

- **Report claims:** 97.38% accuracy, MCC 0.966 as the FL-ADTCN headline (abstract-level
  claim in chapter_5.tex line 174, 443, 545; chapter_6.tex Table comparative + t-test;
  chapter_9.tex).
- **Code / saved results actually show:**
  - `results/db_boa_results.json` (centralised run): **Accuracy 99.45%, MCC 0.941**,
    FPR 0.21%, TP 186 / TN 3792 / FP 8 / FN 14 (test n=4000).
  - chapter_5.tex's *own* Table 4.8 lists "FL-ADTCN + DB-BOA (Ours)" as **0.9786 / MCC
    0.7812** — which **contradicts** the 97.38 / 0.966 figure used elsewhere in the same
    report.
- **Action:** Pick **one** real number from a real run and use it everywhere. Re-run the
  pipeline (see D12), record the actual federated metrics, and report those. Remove
  97.38 / 0.966 unless a real run reproduces it.

---

## 🔴 D6 — Leader-selection statistics (28/18/4 over 50 rounds, 3 banks) are fabricated

- **Report claims** (chapter_6.tex Table "Leader Node Selection Frequency (50 Rounds)"):
  peer0.org1 28×, peer0.org2 18×, peer0.org3 4×; avg latencies 164/189/231 ms.
- **Code actually does** (`blockchain/leader_block.py`): a **10-node** simulation where
  `node.org = f"Org{(node_id % 2) + 1}"` — i.e. nodes alternate **Org1/Org2 only** (no
  Org3/BankC in leader selection). The saved run did **5 consensus rounds**, and the
  **leader was Node 7 every round** (`results/db_boa_results.json`: `leader_node: 7`,
  `consensus_rounds` length 5). Latencies are simulated arithmetic, ~94–104 ms.
- **Action:** Either remove this table, or re-run with a real multi-round leader rotation
  and report the actual frequencies. Do not present 50-round 3-bank numbers that the code
  did not generate.

---

## 🔴 D7 — Throughput/latency figures (85 TPS, 180 ms, 28.4% cut 252→180) are not real

- **Report claims:** 85 TPS, 180 ms average block latency, "DB-BOA reduces latency 28.4%
  from 252 ms to 180 ms" (chapter_5.tex §5.1.6, §5.2.3; chapter_6.tex §"Throughput and
  Latency"; chapter_9.tex).
- **Code actually does:** latency/throughput are computed from **resource-score
  arithmetic and `time.sleep`**, not a live Fabric network. Source comment in
  `simulate_consensus_round()`: *"Simulated latency … not a real network measurement …
  not from a live Fabric testnet."* Saved `consensus_rounds` show latency ≈ 94–104 ms and
  "throughput" ≈ 5,000–9,400 (a meaningless `n_txns / elapsed` ratio).
- **Action:** State clearly that consensus latency/throughput are from a **simulation**,
  report the actual simulated values, and remove the "252→180 ms / 85 TPS / 28.4%" claims
  unless measured on a real deployed network.

---

## 🔴 D8 — Statistical-significance section (paired t-tests, 3 seeds, p<0.001) is fabricated

- **Report claims** (chapter_6.tex §"Statistical Significance"): paired t-tests across
  three independent seeded runs, all p < 0.001, with ± std values.
- **Code actually does:** No multi-seed experiment, no t-test, no std collection anywhere
  in the repo. All runs use a single fixed `random_state` (42 / 7).
- **Action:** Remove this section, or actually run ≥3 seeds and compute the tests before
  claiming them.

---

## 🔴 D9 — Token balances (458/312/178 over 50/10 rounds) and convergence are fabricated

- **Report claims** (chapter_6.tex §"Token Balance Evolution"): BankA→458, BankB→312,
  BankC→178 over 50 consensus / 10 federation rounds; weights converge to
  `[0.52,0.32,0.16]` after ~4 rounds.
- **Code / saved results:** only **3 federation rounds** were saved; aggregation weights
  are **[0.19,0.24,0.56] → [0.48,0.09,0.43] → [0.52,0.31,0.17]** (not converging), and the
  DB-BOA Job-3 fitness is the degenerate constant `-1.0000000397e8` across all iterations
  (objective never improved). No 50-round token series exists.
- **Action:** Re-run to produce a real token series and report the actual values, or
  remove the specific numbers. Note that the live path is Shapley, not DB-BOA (see D1).

---

## 🟠 D10 — Dataset described as synthetic 20,000 samples in Ch.1 & Ch.3

- **Report claims:** chapter_1.tex line 118 and chapter_3.tex line 284: *"synthetically
  generated … 20,000 samples, 30 features, 5% fraud rate."*
- **Code actually does:** loads the **real Kaggle ULB Credit Card Fraud dataset**
  (`creditcard.csv`, **284,807 rows, 0.17% fraud**) — `data/data_loader.py`,
  `config.py::DATASET_PATH`. Chapter 5 *correctly* says ULB 284,807. So Ch.1/Ch.3
  contradict Ch.5.
- **Action:** Fix Ch.1 and Ch.3 to say the real ULB dataset (284,807 tx, 0.17% fraud).
  Remove "synthetic / 20,000 / 5%."

---

## 🟠 D11 — Hyperparameter results and search dimensionality are wrong

- **Report claims:** DB-BOA Job 1 is a **3-D** search over `(H_nD, E_pD, S_eD)`; optimal
  = **128 / 30 / 150**, composite fitness **5.82** (vs BOA 4.91, DBOA 5.21); converges in
  25 iterations from 3.4 (chapter_5.tex §5.1.4, §5.2.5; chapter_6.tex §"Job 1").
- **Code actually does:** search is **2-D** — `(n_filters, steps_per_epoch)`; **epoch
  count is fixed, not searched** (`adtcn.py::_ADTCNObjective`, docstring: *"epoch count is
  NOT part of the search space"*). Saved optimum: **hidden_neurons 98, epoch 23, steps 96**
  (`results/db_boa_results.json`) — note epoch 23 means this JSON predates the
  epoch-fixing change, i.e. it is **stale**. The 5.82 / 4.91 / 5.21 comparison is not in
  any results file.
- **Action:** State the real 2-D search and the real optimum from a fresh run. Remove the
  invented fitness comparison unless you actually run pure-BOA / pure-DBOA ablations.

---

## 🟠 D12 — Saved results JSON is STALE and internally from the old code path

- `results/db_boa_results.json` was generated by an **older version**: it ran **DB-BOA
  Job 3** (the `fed_rounds` contain `db_boa_history` / `dboa_iters` / `boa_iters`), not
  Shapley; and `epoch_count = 23` predates the epoch-fix. The current code defaults to
  Shapley+Krum+DP and fixes epochs.
- **Action:** **Re-run `python3 main.py` (and `--attack`) and `run_baselines.py` with the
  current code**, then build all Chapter 6 tables/plots from the fresh JSON. Do not cite
  the existing JSON's federation numbers — they are from a disabled path.

---

## 🟠 D13 — Activation-function comparison (ReLU/LeakyReLU/ELU/SELU) not run

- **Report claims** (chapter_6.tex §"Activation Function Comparison", Fig
  `activation_accuracy.png`): four activations compared, ReLU best (MCC 0.9249).
- **Code actually does:** ReLU is **hardcoded** in `_Conv1dClassifier`; source comment:
  *"TanH was the paper's claimed best activation but was not tested in this
  implementation; no activation ablation was run."*
- **Action:** Remove the activation comparison, or actually implement and run the ablation.

---

## 🟠 D14 — Byzantine-attack global-model degradation table is fabricated

- **Report claims** (chapter_6.tex Table "With and Without Byzantine Attack"): global
  model Acc 0.9121 (rounds 1–5) → 0.9612 (rounds 6–10); BankC isolated within 12 rounds,
  depletes to negative within 8.
- **Code actually does** (`main.py` `--attack`, 15 rounds): BankC always predicts fraud;
  **token balance and reputation are tracked in Python only** (not on-chain in the attack
  loop). It computes #disputed rounds, final tokens, final reputation, and re-runs Shapley
  with the attacker. The per-window (1–5 / 6–10) **global-model accuracy breakdown is not
  computed** by the code.
- **Action:** Report only what the attack script outputs (disputed-round count, token
  depletion curve, reputation floor, Shapley weight of attacker). Remove the invented
  accuracy-by-round-window table unless you add code to measure it.

---

## 🟢 D15 — Title still references Reinforcement Learning

- `core/titlepage.tex`: *"…Integrating Consensus Mechanisms and Reinforcement Learning."*
  There is **no RL** anywhere in the code (`grep -ri reinforcement db_boa_framework` →
  nothing). The abstract and chapters were already rewritten to drop RL.
- **Action:** Change the title to match (e.g. *"A Blockchain-Integrated Federated Learning
  Framework for Secure, Incentivised Financial Fraud Detection on Hyperledger Fabric"*).

---

## 🟢 D16 — Fabric SDK and environment specifics are slightly off

- **Report claims:** `@hyperledger/fabric-gateway v1.4.0`, PyTorch 2.1.0 + CUDA 11.8,
  CouchDB 3.3.3, Fabric 2.5.4, Ubuntu 22.04 (chapter_5.tex Dev Environment table).
- **Code actually shows:** API server uses **`fabric-network` v2.2.20 + `fabric-ca-client`
  v2.2.20** (the legacy wallet-based SDK, with `enrollAdmin.js`/`registerUser.js`), **not**
  the newer `fabric-gateway`. Chaincode deps are `fabric-contract-api`/`fabric-shim`
  `^2.5.4` (Fabric 2.5.x ✓). `requirements.txt` lists numpy/pandas/sklearn/scipy/matplotlib
  and marks **torch as optional/commented** (though `adtcn.py` requires it). GPU/CUDA is
  unverified in this environment (WSL2).
- **Action:** Correct the SDK name/version to `fabric-network` 2.2.x; pin `torch` in
  `requirements.txt`; only state CUDA/GPU if you actually trained on GPU. Keep Fabric 2.5.x.

---

## 🟢 D17 — DP is implemented but framed as "future work"

- chapter_1.tex line 147 and chapter_5.tex limitations list differential privacy as a
  *future* enhancement, but `FEDERATION_CONFIG['use_dp'] = True` and
  `federated_adtcn.py::extract_weights_with_dp()` **already apply** the Gaussian mechanism
  (ε=1.0, δ=1e-5, σ≈4.84, L2-clip C=1) before weight sharing.
- **Action:** Move DP into the *implemented methods* (with the honest caveat that ε=1.0 is
  a very tight budget that heavily degrades the shared weights — see the source docstring;
  this is itself a good, honest discussion point). Keep "stronger DP / DP-SGD" as future
  work.

---

## Quick map: report section → divergences to fix

| Report file | Divergences |
|-------------|-------------|
| `core/titlepage.tex` | D15 |
| `core/abstract.tex` | D1 (novelty), D5 |
| `chapters/chapter_1.tex` | D1, D10, D17 |
| `chapters/chapter_3.tex` | D4, D10, D16 |
| `chapters/chapter_5.tex` | D1, D2, D3, D11, D16, D17 |
| `chapters/chapter_6.tex` | D1, D3, D4, D5, D6, D7, D8, D9, D11, D12, D13, D14 |
| `chapters/chapter_9.tex` | D1, D5, D7 |
