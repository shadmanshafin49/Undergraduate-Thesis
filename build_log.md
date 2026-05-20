# Build Log — DB-BOA-FEL-ADTCN Framework

All changes applied to strengthen the thesis novelty and research defensibility.

---

## ✅ Tip 1 — Real benchmark dataset (2026-05-20)

**Files changed**
- `db_boa_framework/config.py` — added `DATASET_PATH`, replaced `n_samples`/`fraud_rate` with `dataset_path` in `DATA_CONFIG`
- `db_boa_framework/data/data_loader.py` — replaced `_generate_raw_transactions()` with `_load_real_transactions()` that reads `datasets/creditcard.csv`

**What changed**: Synthetic 20k-row self-generated data replaced with the ULB Credit Card Fraud Detection benchmark (284,807 rows, 0.17% fraud, 28 PCA features). Feature columns reordered to V1-V28, Amount, Time to match downstream temporal engineering assumptions.

**Verified**: Loader produces 284,807 samples, 99.83% normal / 0.17% fraud, 274 engineered features after PTC+NTC+MJE; train/val/test split confirmed.

---

## ✅ Tip 2 — Krum Byzantine-robust aggregation (2026-05-20)

**Files changed**
- `db_boa_framework/config.py` — added `use_krum: True`, `byzantine_f: 1` to `FEDERATION_CONFIG`
- `db_boa_framework/models/federation_manager.py` — added `_krum_aggregate()`, wired into `run_federation_round()` before DB-BOA

**What changed**: Before any aggregation, each org's weight vector is scored by the sum of squared L2 distances to its k = max(1, n−f−2) nearest neighbours (Blanchard et al., NeurIPS 2017). The org with the minimum score becomes the global model, preventing a Byzantine org from corrupting the aggregate before the token-penalty mechanism fires.

**Verified**: Smoke test with a simulated outlier (BankC score ~113 vs BankA/BankB ~0.03) confirmed correct rejection.

**Citation**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017.

---

## ✅ Tip 3 — Differential privacy for weight sharing (2026-05-20)

**Files changed**
- `db_boa_framework/config.py` — added `use_dp: True`, `dp_epsilon: 1.0`, `dp_delta: 1e-5` to `FEDERATION_CONFIG`
- `db_boa_framework/models/federated_adtcn.py` — added `extract_weights_with_dp(epsilon, delta)`; wired into `federation_manager.run_federation_round()`

**What changed**: Before sharing weights, each tensor is L2-clipped to norm ≤ 1.0 (bounding sensitivity C), then Gaussian noise N(0, σ²) is added where σ = C·√(2·ln(1.25/δ))/ε. For ε=1.0, δ=1e-5 this gives σ ≈ 4.84. The result dict carries `dp_enabled`, `dp_epsilon`, `dp_delta` for ledger audit.

**Verified**: DP extraction returns correct shapes; σ formula matches Dwork et al. (2006).

**Citation**: Dwork et al., "Calibrating Noise to Sensitivity in Private Data Analysis", TCC 2006.

---

## ✅ Tip 4 — Replace MLPClassifier with 1D-CNN (2026-05-20)

**Files changed**
- `db_boa_framework/models/adtcn.py` — full rewrite: replaced `sklearn.MLPClassifier` with PyTorch `_Conv1dClassifier`; added `_make_sequences()`
- `db_boa_framework/models/federated_adtcn.py` — updated `extract_weights()` / `load_weights()` to use `torch.state_dict()`

**What changed**: The classifier now processes SEQ_LEN=10 consecutive transactions as an ordered temporal sequence via `Conv1d(n_raw, F, k=3) → ReLU → Conv1d(F, 2F, k=3) → GlobalMaxPool → Linear(2F, 2)`. `_make_sequences()` pads the first row ×9, then slides a 10-step window over the first n_raw feature columns of the engineered 274-dim matrix. Class imbalance (0.17% fraud) handled with `CrossEntropyLoss(weight=[1.0, n_normal/n_fraud])`. DB-BOA still searches (n_filters, epochs, steps/epoch) using an SGD surrogate.

**Verified**: All 5 smoke tests pass — sequence shape (n,10,30), train, predict, weight round-trip (max diff 0.00), DP extraction.

---

## ✅ Tip 5 — Shapley-value contribution weights (2026-05-20)

**Files changed**
- `db_boa_framework/config.py` — added `use_shapley: True` to `FEDERATION_CONFIG`
- `db_boa_framework/models/federation_manager.py` — added `_shapley_weights()`; replaced DB-BOA Job 3 in `run_federation_round()` with Shapley computation; `_run_db_boa_job3()` kept as fallback

**What changed**: For n=3 orgs, all 7 non-empty coalitions are evaluated (equal-weight FedAvg within each coalition, Obf2 on validation set). Exact Shapley values computed via φ_i = Σ[|S|!(n−|S|−1)!/n!]·[v(S∪{i})−v(S)]. Negative contributions clipped to 0, normalised to sum=1. Result dict now carries `shapley_values` and `coalition_values` for on-chain incentive records.

**Verified**: 7 coalitions evaluated correctly; weights sum to 1.0; BankB negative Shapley value correctly clipped to weight=0.

**Citation**: Wang et al., "Measure Contribution of Participants in Federated Learning", IEEE BigData 2020 (FedSV).

---

## ✅ Tip 6 — Transaction graph features (2026-05-20)

**Files changed**
- `db_boa_framework/data/graph_features.py` — new file: `extract_graph_features(amounts, n_bins, window_size)`
- `db_boa_framework/data/data_loader.py` — appends 3 graph features to `X_raw` before temporal engineering when `use_graph_features=True`
- `db_boa_framework/config.py` — added `use_graph_features: True`, `graph_n_bins: 50`, `graph_window: 100` to `DATA_CONFIG`
- `db_boa_framework/models/adtcn.py` — `_make_sequences` promoted to instance method using `self._n_raw`; `fit()` detects actual raw feature count; `_Conv1dClassifier` input size is now dynamic

**What changed**: A temporal-amount similarity graph is constructed over the full 284,807-row dataset. Since the ULB dataset has no account IDs (all PII removed before PCA), edges are proxied via Amount-bucket co-occurrence within a rolling window of 100 rows: transaction i connects to transaction j if both fall in the same Amount percentile-bucket within the window. Three node-level features are extracted per transaction via fully-vectorised cumulative-sum operations (no Python loops): `in_degree_norm`, `out_degree_norm`, and `pagerank_norm` (in/total degree ratio). These 3 features are appended to X_raw (30→33 raw features) before temporal engineering, expanding the engineered feature matrix from 274 to 301 dimensions. The 1D-CNN input changes from `(n, 10, 30)` to `(n, 10, 33)` automatically.

**Verified**: Graph features produce correct shape (n, 3), values in [0,1]; full data pipeline outputs `X_train (199364, 301)`; `n_engineered_features=301` matches; sequence shape `(n, 10, 33)` confirmed.

**Citation**: Liu et al., "Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection", WWW 2021.

---
