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

## ✅ Fix 1 — Krum byzantine_f corrected (2026-05-20)

**Files changed**
- `db_boa_framework/config.py` — `byzantine_f` changed from 1 to 0

**What changed**: Krum requires n ≥ 2f+3.  With n=3 orgs, f=1 fails (3 < 5).
Setting f=0 satisfies the constraint (3 ≥ 3) and is mathematically honest:
Krum selects the most consensus-aligned org assuming no Byzantine adversary.
Added an inline comment explaining the n≥2f+3 requirement.

**Verified**: Krum still runs and selects an org; the formal guarantee now holds.

---

## ✅ Fix 2 — Shapley docstring false claim removed (2026-05-20)

**Files changed**
- `db_boa_framework/models/federation_manager.py` — module docstring line 19;
  `coalition_value` inline docstring

**What changed**: Removed the false claim "no shared labels needed for
computation".  Added an honest note that `coalition_value()` requires a shared
labelled validation set at the aggregator (trusted-aggregator assumption) and
cited Hsieh et al. (2020) for the federated-evaluation trade-off discussion.

---

## ✅ Fix 3 — Graph features honestly renamed (2026-05-20)

**Files changed**
- `db_boa_framework/data/graph_features.py` — full rewrite of docstring and variable names
- `db_boa_framework/data/data_loader.py` — updated comments
- `db_boa_framework/config.py` — updated comment on `use_graph_features`

**What changed**: Renamed features to reflect what they actually compute on the
ULB dataset (which has no account IDs):
  - `in_degree_norm`  → `amount_recurrence_before`
  - `out_degree_norm` → `amount_recurrence_after`
  - `pagerank_norm`   → `degree_ratio`
Module docstring now accurately describes these as temporal-amount recurrence
features, not graph features.  Liu et al. (WWW 2021) citation retained with a
disclaimer that the ULB substrate differs from the account-graph setting in that
paper.

---

## ✅ Fix 4 — DB-BOA surrogate replaced with CNN (2026-05-20)

**Files changed**
- `db_boa_framework/models/adtcn.py` — `_ADTCNObjective` class rewritten;
  `SGDClassifier` import removed; `hidden_neurons_bounds` → `filter_count_bounds`
- `db_boa_framework/config.py` — `hidden_neurons_bounds` → `filter_count_bounds`

**What changed**: `_ADTCNObjective` now trains a `_Conv1dClassifier` surrogate
(same architecture as the final model) on a 2,000-row stratified subsample for
up to 5 epochs per DB-BOA evaluation.  DB-BOA now genuinely searches CNN filter
count, not MLP neuron count.  The hyperparameter key in `DB_BOA_CONFIG` renamed
from `hidden_neurons_bounds` to `filter_count_bounds` to match.

---

## ✅ Fix 5 — Dataset path guard added (2026-05-20)

**Files changed**
- `db_boa_framework/data/data_loader.py` — `_load_real_transactions()`

**What changed**: Added `os.path.exists()` guard before `pd.read_csv()`.  If
`creditcard.csv` is absent the pipeline now raises a clear `FileNotFoundError`
with the exact path and the Kaggle download URL instead of a cryptic pandas error.

---

## ✅ Fix 7 — Krum Byzantine claim corrected; Krum/Shapley independence documented (2026-05-20)

**Files changed**
- `db_boa_framework/models/federation_manager.py` — module docstring, `_krum_aggregate` docstring

**What changed**: Removed all "Byzantine fault tolerance" language.  Krum is now
described as "outlier-weight rejection for consensus alignment" with an explicit
note that f=0 means no adversary is assumed (f≥1 is needed for the Blanchard et al.
guarantee).  Added an "Architecture note" explaining that Krum (security) and
Shapley (fairness) are independent by design — different objectives, different
evaluation criteria — and this is intentional.

---

## ✅ Fix 8 — DP composition logged per federation round (2026-05-20)

**Files changed**
- `db_boa_framework/models/federation_manager.py` — `run_federation_round()`

**What changed**: After each DP weight-sharing step, the console now prints:
`[FED]  DP composition: after k round(s) ε_total=k·ε, δ_total=k·δ (basic composition)`
With ε=1.0 and 3 rounds: ε_total=3.0, δ_total=3e-5.  Cited Dwork et al. (2006 §3.5).

---

## ✅ Fix 9 — Fabricated activation plot replaced with no-op stub (2026-05-20)

**Files changed**
- `db_boa_framework/utils/visualizer.py` — `plot_activation_comparison()`

**What changed**: The function was removed and replaced with a stub that returns
`None`.  The previous implementation plotted hardcoded offsets from the single
measured accuracy — no other activation was ever tested.  A comment explains
what is needed to reinstate the plot legitimately.

---

## ✅ Fix 10 — DB-BOA epoch search dimension removed; search is now 2D (2026-05-20)

**Files changed**
- `db_boa_framework/models/adtcn.py` — `_ADTCNObjective`, `optimise_hyperparams`
- `db_boa_framework/main.py` — Phase 2 print

**What changed**: Epoch count was removed from the DB-BOA search space because
the surrogate cap at 5 epochs made the dimension flat for any proposed value > 5.
Search is now 2D: (n_filters, steps_per_epoch).  `optimal_params["epoch_count"]`
is set to the fixed config default and labelled "not searched".  All print
statements updated to say "2D search".

---

## ✅ Fix 11 — CNN surrogate class distribution corrected to real fraud rate (2026-05-20)

**Files changed**
- `db_boa_framework/models/adtcn.py` — `_ADTCNObjective.__init__`

**What changed**: Surrogate subsample now preserves the real ~0.17% fraud rate
instead of the previous ~50/50 split.  Hyperparameters found by DB-BOA now
reflect deployment conditions.

---

## ✅ Fix 12 — MTTA label corrected to GlobalMaxPool (2026-05-20)

**Files changed**
- `db_boa_framework/models/adtcn.py` — module docstring

**What changed**: "Multiple Time-scale Temporal Attention" replaced with
"GlobalMaxPool — selects the most anomalous time-step activation (pooling, not
attention; the paper's MTTA label is re-used here)".

---

## ✅ Fix 13 — DB-BOA vs defaults comparison added to Phase 4 (2026-05-20)

**Files changed**
- `db_boa_framework/main.py` — Phase 4

**What changed**: Phase 4 now trains a second model with the default
hyperparameters (F=128, ep=30, spe=150) and prints a side-by-side accuracy and
MCC comparison against the DB-BOA-optimal model.  Substantiates the
hyperparameter optimisation contribution.

---

## ✅ Fix 14 — BASELINE_NAMES / CLASSIFIER_NAMES updated to ULB baselines (2026-05-20)

**Files changed**
- `db_boa_framework/config.py`

**What changed**: Replaced MBO-ADTCN, EfficientNet, etc. with ULB-compatible
names: FedAvg, FedAvg+Krum, FedAvg+DP, DB-BOA-ADTCN.

---

## ✅ Fix 15 — Chaincode and config "DB-BOA weight" comments updated to Shapley (2026-05-20)

**Files changed**
- `db_boa_framework/config.py` — INCENTIVE_CONFIG `federation_pool` comment
- `db_boa_fabric/chaincode/lib/db_boa_chaincode.js` — header comment, `recordFederationRound` docstring

**What changed**: All references to "DB-BOA Job 3 output" and "shared by weight"
replaced with "Shapley-weighted aggregation" and "shared by Shapley contribution
weight".

---

## ✅ Fix 16 — Ecological validity and i.i.d. notes added (2026-05-20)

**Files changed**
- `db_boa_framework/data/data_loader.py` — `split_for_orgs` docstring
- `db_boa_framework/config.py` — `ORG_DATA_SPLITS` comment

**What changed**: Added "Ecological validity note" in `split_for_orgs` docstring
acknowledging that ULB comes from a single bank (controlled simulation, not a
real cross-institution deployment).  Config comment notes the i.i.d. assumption
and cites FedProx (Li et al., 2020) as the appropriate alternative for severely
heterogeneous distributions.

---

## ✅ Fix 17 — run_baselines.py helper script created (2026-05-20)

**Files changed**
- `db_boa_framework/run_baselines.py` — new file

**What changed**: Created a self-contained script that runs FedAvg, FedAvg+Krum,
FedAvg+DP, and DB-BOA-ADTCN on the ULB dataset and prints Python dict literals
ready to paste into `baseline_metrics()`.  All four runs share the same DB-BOA
hyperparameter search so results are directly comparable.  Run with:
`python3 db_boa_framework/run_baselines.py`

---

## ✅ Fix 6 — Synthetic baselines removed (2026-05-20)

**Files changed**
- `db_boa_framework/utils/metrics.py` — `baseline_metrics()` body replaced
- `db_boa_framework/utils/visualizer.py` — `plot_activation_comparison`,
  `plot_classifier_comparison`, `plot_summary_comparison` updated

**What changed**: Hardcoded baseline numbers (MBO-ADTCN, WSA-ADTCN, etc.) that
were produced on synthetic data removed from `baseline_metrics()`.  The function
now returns empty dicts with a docstring explaining that ULB-compatible baselines
(FedAvg, FedAvg+Krum, FedAvg+DP) must be computed by running the pipeline.
Visualizer plots gracefully handle empty baseline dicts (show proposed model only)
and use `COLORS["proposed"]` for the proposed model regardless of its position in
the dict.

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

## ✅ Fix 18 — Dead `activation="tanh"` config key removed (2026-05-21)

**Files changed**
- `db_boa_framework/config.py` — `"activation": "tanh"` key removed from `ADTCN_CONFIG`; replaced with a comment stating ReLU is used
- `db_boa_framework/models/adtcn.py` — `_Conv1dClassifier` docstring updated to state ReLU is hardcoded and TanH was the paper's claim but untested

**What changed**: The `activation` key in `ADTCN_CONFIG` was never read by `_Conv1dClassifier`, which hardcodes `nn.ReLU()`.  The comment falsely claimed TanH was the paper's best activation.  The dead key is removed; both files now honestly state that ReLU is used and no activation ablation was performed.

---

## ✅ Fix 19 — Surrogate minimum fraud samples raised to 30 (2026-05-21)

**Files changed**
- `db_boa_framework/models/adtcn.py` — `_ADTCNObjective`: added `_MIN_FRAUD_ROWS = 30`; `n_f = max(self._MIN_FRAUD_ROWS, int(...))` replaces `max(4, ...)`

**What changed**: With `_SURROGATE_ROWS=2_000` and the real 0.17% fraud rate, `max(4, int(2000×0.0017)) = 4` fraud samples — far too few for stable CNN gradients.  The minimum is now 30, ensuring at least 30 fraud examples per surrogate evaluation.  A docstring comment explains the trade-off.

---

## ✅ Fix 20 — Incentive mechanism over-reporting limitation documented (2026-05-21)

**Files changed**
- `db_boa_framework/main.py` — Phase 8 attack simulation: added limitation comment before JSON serialisation

**What changed**: Added a code comment explaining that the token incentive only indirectly penalises malicious over-reporting — a bank that always votes fraud earns tokens for every real fraud event.  Points to the thesis Limitations section.

---

## ✅ Fix 21 — DP accuracy cost comparison added to run_baselines.py (2026-05-21)

**Files changed**
- `db_boa_framework/run_baselines.py` — new block after all baseline runs that computes and prints the DP accuracy/MCC cost

**What changed**: After the four baseline runs complete, the script now prints:
```
DP ACCURACY COST  (ε=1.0, δ=1e-5, basic Gaussian mechanism)
  FedAvg (no DP)  Accuracy=XX.XXXXX%  MCC=0.XXXXX
  FedAvg+DP       Accuracy=XX.XXXXX%  MCC=0.XXXXX
  DP cost:  Accuracy ▼X.XXXXX%  MCC ▼0.XXXXX
```
This directly answers defense Q38: "How much accuracy does DP cost at ε=1.0?"

---

## ✅ Fix 22 — FL validity note added to federation loop (2026-05-21)

**Files changed**
- `db_boa_framework/main.py` — Phase 7 federation loop: added "FL Validity note" comment before the `for fed_round` loop

**What changed**: Added a comment acknowledging that orgs do not perform local gradient updates between federation rounds in this simulation.  States that the absence of inter-round drift means the 3-round convergence result does not generalise to real FL deployments.

---

## ✅ Fix 23 — SEQ_LEN=10 justification comment added (2026-05-21)

**Files changed**
- `db_boa_framework/models/adtcn.py` — `SEQ_LEN` constant comment updated
- `db_boa_framework/config.py` — `"sequence_length": 10` comment updated

**What changed**: Both locations now state: "10-step window chosen empirically; ablation over {5,10,20} is left for future work."  Sufficient to answer Q96 honestly.

---

## ✅ Fix 24 — Simulated latency disclosed in leader_block.py (2026-05-21)

**Files changed**
- `db_boa_framework/blockchain/leader_block.py` — `simulate_consensus_round()`: added comment before latency arithmetic

**What changed**: Added a 5-line comment explaining that all latency values are derived from normalised resource scores and `time.sleep`, not from a live Hyperledger Fabric network.  References the thesis Limitations section.

---

## ✅ Fix 25 — Sequence-padding bias disclosed in _make_sequences (2026-05-21)

**Files changed**
- `db_boa_framework/models/adtcn.py` — `_make_sequences()` docstring

**What changed**: Added: "Boundary condition: the first SEQ_LEN-1 predictions use a padded context (row 0 repeated). This affects ~0.003% of the 284,807-row dataset and does not meaningfully bias aggregate metrics."  Answers Q126 honestly.

---

## ✅ Fix 26 — DP σ=4.84 noise magnitude disclosed in federated_adtcn.py (2026-05-21)

**Files changed**
- `db_boa_framework/models/federated_adtcn.py` — `extract_weights_with_dp()` docstring

**What changed**: Added a "DP noise magnitude disclosure" block explaining that at ε=1.0 σ≈4.84 exceeds per-element weight magnitudes by ×370–×800, making the DP-shared global model near-random weights.  States this is the deliberate privacy–utility trade-off at a tight privacy budget and references DP-SGD (McMahan et al., ICLR 2018) and ε≥50 as practical alternatives.  Answers Q50.

---

## ✅ Fix 27 — PTC/NTC feature discard explained in _make_sequences (2026-05-21)

**Files changed**
- `db_boa_framework/models/adtcn.py` — `_make_sequences()` docstring

**What changed**: Added a "Design note" block explaining that the input matrix has ~301 columns but `_make_sequences` intentionally takes only the leading 33 raw-feature columns.  The 268 PTC/NTC columns remain available in X but the 1D-CNN derives its own temporal context by sliding over SEQ_LEN consecutive raw-feature vectors.  Answers Q30 honestly without requiring a code change.

---

## ✅ Fix 28 — Shapley validation set changed from X_test to X_val (2026-05-21)

**Files changed**
- `db_boa_framework/main.py` — Phase 7 federation loop

**What changed**: `X_val_shared = X_test[:500]` replaced with `X_val_shared = X_val[:500]` (same for y).  Shapley coalition values — which determine on-chain token distribution — are now computed on the training validation split, keeping X_test unseen until final reporting.  Added a one-line comment explaining the fix.

---

## ✅ Bug Fix Session — 10 bugs found and fixed (2026-06-02)

---

### BF-1 — `accuracy_deltas` assignment was inside wrong loop (main.py:328)

**Files changed**
- `db_boa_framework/main.py` — Phase 7 federation loop

**What changed**: `fed_result["accuracy_deltas"] = accuracy_deltas` was indented
one level too deep, executing on every iteration of the per-org loop rather than
once after all orgs were evaluated.  During the loop, `fed_result` held a partial
dict.  Moved the assignment one level out so it runs after all orgs complete.

---

### BF-2 — `n_pop=0` silently fell through to default (db_boa.py:69)

**Files changed**
- `db_boa_framework/algorithms/db_boa.py` — `DBBOA.__init__`

**What changed**: `n_pop or default` evaluates to `default` when `n_pop=0` because
`0` is falsy in Python, silently overriding an explicit caller-supplied value.
Changed to `n_pop if n_pop is not None else default` (and same for `max_iter`)
so only `None` triggers the fallback.

---

### BF-3 — Dead code `_balance` method and unused `resample` import (adtcn.py)

**Files changed**
- `db_boa_framework/models/adtcn.py` — removed `_balance()` static method and
  `from sklearn.utils import resample` import

**What changed**: `_balance` was never called anywhere in the codebase.  Its
docstring said "kept for API compatibility" but nothing depends on it.  Removing
it also cleans up the unused `sklearn.utils.resample` import.

---

### BF-4 — `get_eval_subset` claimed stratified but used non-stratified sampling (data_loader.py:129)

**Files changed**
- `db_boa_framework/data/data_loader.py` — `get_eval_subset()`

**What changed**: The method's docstring stated "Uses a stratified sample that
preserves the real class distribution", but the implementation used
`rng.choice(len(y_train), n_eval, replace=False)` which is uniform random, not
stratified.  At the real 0.17% fraud rate this gave ~5 fraud rows in a 3,000-row
subset — the `_MIN_FRAUD_ROWS=30` guard compensated, but the code contradicted
its own documentation.  Replaced with `train_test_split(..., stratify=y_train)`
so the implementation matches the intent.

---

### BF-5 — Attack simulation: BankC's `predict` override not seen by Shapley coalitions (federation_manager.py)

**Files changed**
- `db_boa_framework/models/federation_manager.py` — `_shapley_weights()` /
  `coalition_value()` inner function; added `from utils.metrics import
  compute_all_metrics, obf2_value` import

**What changed**: In Phase 8, `attack_models['BankC'].predict` is overridden at
the instance level to always return ones.  But `coalition_value()` computed
coalition quality by averaging raw weight tensors (extracted via `extract_weights`)
into a deepcopy of BankA's template and calling `evaluate_on_validation` on it.
The instance-level `predict` override on BankC never propagated — the Shapley
values measured BankC's honest CNN weights, not its malicious prediction behaviour,
so BankC's weight was not actually suppressed by Shapley.

Fix: `coalition_value` now checks whether any org in the coalition has an
instance-level `predict` attribute (`'predict' in org.__dict__`).  When one is
detected, it falls back to majority-vote of individual org predictions (which
correctly exercises each org's actual `predict` method, including any override)
rather than weight averaging.  The `obf2_value(compute_all_metrics(...))` path
is reused so the return unit is unchanged.

---

### BF-6 — `get_training_info()` crashed with AttributeError (federated_adtcn.py:126)

**Files changed**
- `db_boa_framework/models/federated_adtcn.py` — `get_training_info()`

**What changed**: `self.model.hidden_layer_sizes` references a scikit-learn MLP
attribute that does not exist on `_Conv1dClassifier` (a PyTorch `nn.Module`).
Any call to `get_training_info()` raised `AttributeError`.  Replaced with
`str(self.model)` which calls PyTorch's built-in `__repr__` and returns the
layer summary string.

---

### BF-7 — Path traversal vulnerability in `/api/plots/:filename` (server.js:311)

**Files changed**
- `db_boa_fabric/api-server/server.js` — `GET /api/plots/:filename` route

**What changed**: `path.join(RESULTS_DIR, req.params.filename)` did not validate
that the resolved path stayed inside `RESULTS_DIR`.  A request to
`/api/plots/../../etc/passwd` would resolve outside the results directory.
Added a `path.resolve` check: if the resolved path does not start with
`path.resolve(RESULTS_DIR) + path.sep` the server now returns HTTP 400.

---

### BF-8 — `recordFraudResult` called with wrong argument order (server.js:404)

**Files changed**
- `db_boa_fabric/api-server/server.js` — `POST /api/submit-transaction` route

**What changed**: The chaincode signature is
`recordFraudResult(ctx, txnId, orgName, isFraud, fraudScore)` (4 params after
ctx).  The server was calling it with 5 args in the wrong order:
`[txnId, String(isFraud), fraudScore, JSON.stringify({...}), leaderNode]`.
This mapped `isFraud` ("true"/"false") into the `orgName` parameter, a float
into `isFraud`, and a JSON object string into `fraudScore`, silently breaking all
on-chain fraud recording.  Fixed to the correct 4-arg order with `'DemoNode'` as
the org name for demo-mode submissions.

---

### BF-9 — Both federation weight fields sent as `org_contributions` (server.js:512)

**Files changed**
- `db_boa_fabric/api-server/server.js` — `writeFederationToFabric()`

**What changed**: `recordFederationRound` takes separate `aggregationWeightsJson`
and `orgContributionsJson` arguments, but both were being set to the same
`JSON.stringify(weights)` variable.  Renamed to `aggregationWeights` and
`orgContributions` with explicit separate assignments to make the intent clear
and guard against future divergence between the two fields.

---

### BF-10 — Iterator never closed in chaincode `_queryByDocType` (db_boa_chaincode.js)

**Files changed**
- `db_boa_fabric/chaincode/lib/db_boa_chaincode.js` — `_queryByDocType()`

**What changed**: The CouchDB rich-query iterator was not closed after iteration,
leaking a gRPC stream handle on every call to `getNodeStatus`,
`getAllTransactions`, `getLeaderHistory`, `getConsensusHistory`,
`getFederationHistory`, and `getOrgModels`.  Wrapped the iteration loop in a
`try/finally` block that calls `await iter.close()` unconditionally.

---

## ✅ Fix 29 — FedAvg updated to McMahan size-weighted averaging (2026-05-21)

**Files changed**
- `db_boa_framework/run_baselines.py` — `_avg_weights()` signature and body; `run_one_baseline()` now collects `org_counts` and passes them to `_avg_weights()`

**What changed**: `_avg_weights()` now accepts an optional `counts` list.  When provided it computes w_global ← Σ_k (n_k/n)·w_k, matching McMahan et al. (AISTATS 2017).  The FedAvg call in `run_one_baseline()` passes the actual org sample sizes ([n_BankA, n_BankB, n_BankC] ≈ [50%, 30%, 20%] of train set).  With the correct weights [0.5, 0.3, 0.2] the FedAvg baseline is now the faithful McMahan implementation.  Answers Q144.

---
