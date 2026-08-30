# Ground-Truth Implementation Reference

The single source of truth for what the system *actually is*. Every figure or claim in
the report should be checkable against this file (and the cited source files).

## 1. Repository layout (what really exists)

```
db_boa_framework/            # Python: ML + optimisation + simulation
  config.py                  # ALL parameters (read this to know real settings)
  main.py                    # 8-phase pipeline orchestrator (--quick, --no-plots, --attack)
  run_baselines.py           # FedAvg / FedAvg+Krum / FedAvg+DP / proposed comparison
  algorithms/                # dboa.py, boa.py, db_boa.py (hybrid optimiser)
  models/
    adtcn.py                 # 1D-CNN fraud detector + DB-BOA hyperparameter wrapper
    federated_adtcn.py       # adds weight extract/load + DP weight sharing
    federation_manager.py    # DP → Krum → Shapley federation round
  blockchain/leader_block.py # 10-node consortium SIMULATION + leader selection + incentives
  data/
    data_loader.py           # loads real ULB creditcard.csv, temporal feature engineering
    graph_features.py        # 3 amount-recurrence features (NOT true graph features)
  utils/metrics.py           # all metrics, obf2_value(), baseline_metrics() (EMPTY)
  utils/visualizer.py        # plots
  results/db_boa_results.json # STALE saved run (old DB-BOA Job 3 path) + 12 PNGs
datasets/creditcard.csv      # ULB dataset (note: lives at repo root as creditcard.csv)
db_boa_fabric/               # Node.js: Hyperledger Fabric layer
  chaincode/lib/db_boa_chaincode.js  # DBBOAContract smart contract
  api-server/                # Express API (fabric-network 2.2.20) + web dashboard
fabric/fabric-samples/       # Fabric test-network base
build_log.md                 # ~30 documented fixes/enhancements
```

## 2. Dataset (real)

- **Kaggle ULB Credit Card Fraud Detection**: 284,807 transactions, **492 fraud
  (0.17%)**, features V1–V28 (PCA), Amount, Time, Class. `data/data_loader.py`.
- Split: **80% train / 10% val / 10% test**, stratified, `StandardScaler` fit on train.
- **Feature engineering** (`_engineer_temporal_features`): PTC rolling mean+std over
  windows {5,10,20}; NTC diffs of order {1,2}; MJE interactions Amount×V1..V4; plus 3
  optional recurrence features → **~300 engineered columns**.
- **Important caveat:** the ADTCN consumes only the **leading 30 (or 33 with recurrence)
  raw columns** as 10-step sequences. The ~270 PTC/NTC columns are **computed but
  discarded** from the model input (documented in `adtcn.py::_make_sequences`). Report
  them as "engineered for completeness; the CNN's temporal context comes from the
  10-step sliding window, not the PTC/NTC columns."
- **Single-bank caveat:** ULB is one bank's data; the 3 "orgs" are volume splits
  (50/30/20) of the same population — a controlled simulation, not a true cross-bank
  federation (`config.py` ORG_DATA_SPLITS comment, `data_loader.py::split_for_orgs`).

## 3. ADTCN model (real)

`models/adtcn.py`, `class _Conv1dClassifier`:

```
input  : (batch, seq_len=10, n_features=30 or 33)
Conv1d(n_features, F,   kernel=3, padding=1) → ReLU
Conv1d(F,          2F,  kernel=3, padding=1) → ReLU
GlobalMaxPool over time  →  (batch, 2F)
Linear(2F, 2)            →  logits [normal, fraud]
```

- Activation: **ReLU, hardcoded** (no TanH, no ablation).
- Class imbalance: **weighted cross-entropy** (`weight=[1, n_normal/n_fraud]`), no
  oversampling.
- Optimiser: Adam, lr 1e-3.
- Conceptual labels from base paper, as actually realised: MJE = raw multi-feature input;
  TCL = the two stacked 1D convolutions over the window; **MTTA = global max-pool (not
  attention)**.

## 4. DB-BOA optimiser (real)

`algorithms/db_boa.py` (+ `dboa.py`, `boa.py`). Hybrid of Dynamic Butterfly OA and
Billiards OA with **adaptive switching**: each iteration picks DBOA vs BOA by comparing a
uniform random draw to `bestfit/worstfit`. DBOA mutates the best solution via Lévy-flight
(LSAM); BOA does collision-style position updates. Used in **two live roles**:

- **Job 1 — Hyperparameter search (2-D):** searches `(n_filters ∈ [5,255], steps_per_epoch
  ∈ [50,250])`; **epoch count fixed** (not searched). pop=20, iter=30. Surrogate trains a
  small CNN on a 2,000-row stratified subsample for 5 epochs, maximises
  `Obf2 = Acc + Pre + NPV + MCC + 1/FPR`.
- **Job 2 — Leader selection:** over a **10-node** simulated consortium, minimises
  `cost = CT + CC + MS` (minus a small reputation bonus). pop=15, iter=25.
- **Job 3 — Federation weights:** **implemented but DISABLED by default**
  (`use_shapley=True` overrides it). Only runs if `use_shapley=False`.

## 5. Federated learning (real) — the actual aggregation pipeline

`models/federation_manager.py::run_federation_round()`:

1. **Extract weights with DP** (`use_dp=True`): Gaussian mechanism, ε=1.0, δ=1e-5,
   L2-clip C=1 → σ≈4.84 (`federated_adtcn.py::extract_weights_with_dp`). Honest caveat
   in source: at ε=1.0 the noise dwarfs the signal, so the DP-shared global model is
   near-random — a deliberate, documentable privacy/utility trade-off.
2. **Krum selection** (`use_krum=True`, `byzantine_f=0`, n=3): scores each org by sum of
   squared distances to nearest neighbours; selects the most consensus-aligned org's
   weights as the global model. With f=0 this is **outlier/consensus alignment, not
   Byzantine-tolerance** (source says so).
3. **Shapley contribution weights** (`use_shapley=True`): evaluates all `2^3−1 = 7`
   coalitions on a shared validation set, computes exact Shapley value per org, clips
   negatives, normalises to sum 1. These weights drive the **on-chain token split**.
4. Returns global weights + full metadata (Shapley values, coalition values, Krum scores,
   DP params) matching the ledger schema.

So the real federation method = **DP + Krum + exact Shapley**, three citable techniques
(Dwork 2006; Blanchard NeurIPS 2017; Wang/FedSV 2020), each with an honest scope caveat.

## 6. Blockchain layer (real)

- **Consortium:** 3 orgs — BankA (Org1MSP), BankB (Org2MSP), BankC (Org3MSP).
- **Chaincode** `DBBOAContract` (`db_boa_fabric/chaincode/lib/db_boa_chaincode.js`),
  Node.js, deterministic (timestamps from `ctx.stub.getTxTimestamp()`). Key functions:
  `initLedger`, `updateNodeMetrics`, `updateIncentive`, `submitTransaction`,
  `recordFraudResult`, `recordLeaderSelection`, `recordConsensusRound`,
  `recordHyperparams`, `submitModelMetrics`, `recordFederationRound`, plus
  `get*History` query functions.
- **Incentive rules (enforced on-chain, mirrored in Python):**
  | Event | Tokens | Function |
  |-------|--------|----------|
  | Fraud verdict confirmed by majority | +10 | `recordFraudResult` |
  | Confirmed verdict AND latency < 300 ms | +15 | `recordConsensusRound` |
  | Leader AND round succeeds | +10 | `recordConsensusRound` |
  | Verdict disputed by majority | −2 | `recordFraudResult` |
  | Round fails under leader | −2 | `recordConsensusRound` |
  | Federation pool, split by **Shapley** weight | +20·wᵢ | `recordFederationRound` |
- Reputation bounded **[0.5, 2.0]**, +0.02 on success / −0.05 on failure.
- **API server:** Express, **`fabric-network` 2.2.20** SDK (wallet + `enrollAdmin.js` /
  `registerUser.js`), `/api/plots/:filename`, `/api/submit-transaction`, etc. + web
  dashboard (`index.html`).
- **Reality caveat:** consensus latency/throughput in `leader_block.py` are a
  **simulation** (resource-score arithmetic + `time.sleep`), not measurements from a live
  running Fabric network.

## 7. Pipeline phases (`main.py`)

1. Leader block selection (DB-BOA Job 2) · 2. Hyperparameter optimisation (DB-BOA Job 1) ·
3. ADTCN training · 4. Evaluation · 5. Multi-round consensus simulation · 6. Plots ·
7. Federation rounds (DP→Krum→Shapley) · 8. (`--attack`) Byzantine BankC always-fraud, 15
rounds, Python-tracked token/reputation depletion + Shapley re-run with attacker.

## 8. Real numbers currently on disk (`results/db_boa_results.json`) — **STALE**

> Generated by the old DB-BOA-Job-3 path with epoch=23. Use only after re-running.
> **➡ VERIFIED replacements (use these instead, 2026-06-08 fresh fixed-objective run):** centralised
> **Acc 99.85% / MCC 0.677** (TP 83 / TN 56,794 / FP 70 / FN 15, test n=56,962); federated
> ablation (saved `results/baselines.json`) FedAvg 0.569 / +Krum 0.776 / both DP ε=1.0 rows collapse
> to MCC≈0 (degenerate single-class, direction varies by noise draw). See `05_results.md` §A/§B and
> REWRITE_06 §6.1/§6.2. The figures below are the old stale values.

- Centralised ADTCN test metrics: **Acc 99.45%, Precision 95.88%, Sensitivity 93.0%,
  Specificity 99.79%, NPV 99.63%, FPR 0.21%, F1 94.42%, MCC 0.941**; TP 186 / TN 3792 /
  FP 8 / FN 14 (test n = 4000).
- Optimal hyperparameters (stale): hidden_neurons 98, epoch 23, steps 96.
- Leader node: **Node 7**; 5 consensus rounds; latency ≈ 94–104 ms (simulated).
- Federation: 3 rounds; DB-BOA Job-3 weights non-converging (see D9/D11), degenerate
  fitness. **Shapley path was not the one saved.**

## 9. Build log

`build_log.md` documents ~30 dated, citation-backed fixes/enhancements (real ULB dataset,
Krum, DP, McMahan size-weighted FedAvg, path-traversal hardening, chaincode arg-order
bugs, iterator leak fixes). This is genuine, defensible engineering work and supports the
implementation chapter.
