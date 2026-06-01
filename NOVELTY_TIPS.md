# Open Issues — Must Fix Before Submission

Everything below is still open (or recently fixed). Issues are ordered by how fast they would end
a defense.

---

## SIGNIFICANT — Weakens a claimed contribution

### ✅ 1. DP noise (σ=4.84) completely overwhelms federated weight magnitudes

**File**: `db_boa_framework/models/federated_adtcn.py:69-81`
**Defense question**: Q50

σ = C·√(2·ln(1.25/δ))/ε = 1·√(2·ln(125000))/1 ≈ 4.84.  After L2 clipping
each weight tensor to norm ≤ 1, the per-element magnitude is 1/√dim:

  Conv1d(33, F, 3) weights : dim = F×33×3 ≈ 5940 → per-element ≈ 0.013
  Conv1d(F, 2F, 3) weights : dim = 2F×F×3 ≈ 24576 → per-element ≈ 0.006

Noise σ=4.84 is 370–800× the per-element signal.  Krum then selects ONE org's
noisy weights, so the global model at ε=1.0 is effectively random weights.

**Resolution**: Full disclosure added to `federated_adtcn.py:69-81` docstring
and thesis Limitations section.  The trade-off is explicitly described:
"At ε=1.0, σ≈4.84 exceeds per-element weight magnitudes by ×370–×800; the
global model post-DP is near-random weights."  A practical deployment would
use ε≥50 or DP-SGD (McMahan et al., ICLR 2018).

---

### ✅ 2. PTC/NTC feature engineering creates ~268 features never used by the CNN

**File**: `db_boa_framework/data/data_loader.py:156-200` (`_engineer_temporal_features`)
**File**: `db_boa_framework/models/adtcn.py:389-422` (`_make_sequences`)
**Defense question**: Q30

`_engineer_temporal_features()` computes PTC rolling mean/std over windows [5,10,20]
and NTC diff orders [1,2] across all 33 input columns → ~268 extra columns.
`_make_sequences()` extracts only the leading n_raw (≤33) columns.

**Resolution**: Explicit design note added to `_make_sequences` docstring
(adtcn.py:389-413): "PTC/NTC features are intentionally discarded (Q30).
The 1D-CNN receives its own temporal context implicitly by sliding over
SEQ_LEN consecutive raw-feature vectors."

---

## MINOR — Must disclose before defense

### ✅ 3. Shapley coalition values computed on X_test[:500] — test data used before evaluation

**File (Phase 7)**: `db_boa_framework/main.py:301-302`  → **FIXED** (uses `X_val[:500]`)
**File (Phase 8)**: `db_boa_framework/main.py:469-472` → **FIXED** (was `X_test[:500]`, now `X_val[:500]`)
**Defense question**: Q63, Q94

Phase 7 was previously fixed; Phase 8 (attack simulation) was overlooked.  Both
are now corrected: Shapley coalition values are evaluated on the training
validation split so X_test remains unseen until final reporting.

---

### ✅ 4. FedAvg baseline uses equal-weight average, not McMahan et al.'s size-weighted average

**File**: `db_boa_framework/run_baselines.py:61-79` (`_avg_weights`)
**Defense question**: Q144

**Resolution**: `_avg_weights(weights_list, counts)` now accepts org sample
counts and applies data-size-weighted averaging (n_k/n weights) when counts
are provided.  Equal-weight fallback is retained for backward compatibility.
Comment documents the McMahan et al. correct weights [0.5, 0.3, 0.2] for the
50/30/20 split.

---

## NEW ISSUES (found in loop iteration 2)

### ✅ 8. main.py Phase 8 crashed with KeyError on 'best_fitness' in Shapley mode

**File**: `db_boa_framework/main.py:484-485`
**Defense question**: Q89 (baselines), attack simulation

`atk_fed_result['best_fitness']` was accessed after `run_federation_round()`,
but that key exists only in the old DB-BOA Job 3 fallback path.  The Shapley
path (which is the active default) does not return `best_fitness`.  Running
`python main.py --attack` would raise `KeyError: 'best_fitness'`.

**Fix applied**: Replaced the `best_fitness` print with Shapley values display;
updated all Phase 8 labels from "DB-BOA Job 3" to "Shapley attribution."

---

### ✅ 9. visualizer.py contained fabricated baseline comparison data

**File**: `db_boa_framework/utils/visualizer.py`
**Defense question**: Any question asking you to defend a plot

Two functions generated plots with **made-up data**:

a) `plot_convergence` — simulated exponential-decay convergence curves for
   "MBO-ADTCN", "WSA-ADTCN", "DBOA-ADTCN", "BOA-ADTCN" using arithmetic
   offsets.  None of these baselines were ever run.

b) `plot_roc_curve` — hardcoded AUC values for "EfficientNet" (0.94),
   "ResNet" (0.97), "DenseNet" (0.95), "DTCN" (0.98).  These numbers
   come from no measurement on the ULB dataset.

Presenting fabricated comparison curves in a thesis defense is a serious
academic integrity issue if an examiner asks "where did the EfficientNet
AUC of 0.97 come from?"

**Fix applied**: Fabricated baseline curves removed from both functions.
Only actual DB-BOA-ADTCN measurements are plotted; real baselines will
appear automatically when `baseline_metrics()` is populated with true runs.

---

### ✅ 11. _ADTCNObjective crashed with ValueError when fraud pool < _MIN_FRAUD_ROWS

**File**: `db_boa_framework/models/adtcn.py:137-140`
**Defense question**: Q35 (surrogate fraud sampling)

After fixing `get_eval_subset` to use stratified sampling (Issue 6), the
eval subset now has ~5 fraud rows in 3,000 total.  `_ADTCNObjective` then
calls `rng.choice(fraud_idx, 30, replace=False)` — requesting 30 samples
from a pool of ~5, which raises:
  `ValueError: Cannot take a larger sample than population when 'replace' is False`

**Fix applied**: Added `replace_f = len(fraud_idx) < n_f` flag; fraud sampling
uses `replace=True` when the pool is smaller than `_MIN_FRAUD_ROWS`.  The
surrogate still receives exactly 30 fraud rows (with repetitions), keeping the
class-weight calculation stable.  The defence answer to Q35 must acknowledge
that the ~5 unique fraud rows are oversampled with replacement.

---

### ✅ 10. visualizer.py federation weights plot titled "DB-BOA Job 3"

**File**: `db_boa_framework/utils/visualizer.py:454`

The federation weights chart title still read "DB-BOA Job 3 — Federated
Aggregation Weights per Round" even after Shapley replaced Job 3.

**Fix applied**: Title changed to "Shapley-Weighted Federated Aggregation
Weights per Round."

---

## NEW ISSUES (found in loop iteration 1)

### ✅ 5. db_boa.py module docstring had wrong switching criterion formula

**File**: `db_boa_framework/algorithms/db_boa.py:14-19`
**Defense question**: Q7, Q8

The module-level docstring stated:
  "If rand < |best_fit| / |worst_fit|  →  DBOA"
but the actual implementation (line 127) computes:
  `threshold = max(0.0, 1.0 − |f_max − f_min| / max(|f_min|, |f_max|, ε))`

These are different formulas.  An examiner reading the docstring would quote a
different formula than the one implemented.

**Fix applied**: Module docstring updated to state the range-normalised formula
correctly, with a note that the raw abs-ratio form breaks for negated objectives.

---

### ✅ 6. get_eval_subset returns 50/50 balanced data — surrogate was NOT matching real 0.17% rate

**File**: `db_boa_framework/data/data_loader.py:115-129` (`get_eval_subset`)
**Defense question**: Q34, Q35, Q36

`get_eval_subset` previously used balanced 50/50 sampling (`n_eval//2` fraud,
`n_eval//2` normal).  `_ADTCNObjective` then computed `fraud_rate ≈ 0.5` from
this balanced input, producing a surrogate with ~50% fraud — directly contradicting
the defense answer to Q34: "now it matches the real 0.17% rate."

The `_MIN_FRAUD_ROWS=30` guard and real-distribution fix in `_ADTCNObjective` were
ineffective because the upstream `get_eval_subset` had already removed the class
imbalance.

**Fix applied**: `get_eval_subset` changed to stratified random sampling (no
class rebalancing).  With ~3,000 rows at 0.17% fraud, `_ADTCNObjective` now sees
~5 raw fraud rows, the `_MIN_FRAUD_ROWS=30` guard fires, and the effective
surrogate fraud rate is ~1.5% — which is the disclosed mismatch addressed by Q36.

---

### ✅ 7. dboa.py fragrance calculation breaks for negative fitness (raises NaN)

**File**: `db_boa_framework/algorithms/dboa.py:96-97`
**Defense question**: Q10, Q16 (standalone DBOA)

The standalone `DBOA` class computed:
  `g_j = self.d * (J_j ** self.b)`
Raising a negative float (e.g., `J_j = −Obf2 < 0`) to a fractional power
`b=0.1` returns `nan` in Python (domain error).  `db_boa.py` already used
`abs(float(fit[j]))` to avoid this; `dboa.py` did not.

**Fix applied**: Changed to `abs(float(J_j)) ** self.b` with an explanatory
comment.  Consistent with the db_boa.py implementation.

---

## What is genuinely solid (do not touch)

- **1D-CNN** (`adtcn.py:64-86`) — correct PyTorch layer sequence; docstring honest about ReLU.
- **DB-BOA 2D search** (`adtcn.py:113-200`) — epoch removed from search space; `_MIN_FRAUD_ROWS=30` guards against 4-sample surrogate.
- **DP formula** (`federated_adtcn.py:50-93`) — Gaussian mechanism σ is correct; composition logged after each round.
- **Shapley math** (`federation_manager.py`) — exact for n=3, formula correct; trusted-aggregator limitation disclosed.
- **Krum labelling** (`federation_manager.py`) — "outlier-weight rejection" (not "Byzantine fault tolerance"); f=0 correctly explained.
- **Weighted cross-entropy** (`adtcn.py:259-262`) — correct for 0.17% fraud rate.
- **Feature honesty** (`graph_features.py`) — recurrence features, not graph features.
- **Activation honesty** (`config.py:93-95`, `adtcn.py:74-75`) — ReLU use disclosed.
- **FL validity note** (`main.py:286-289`) — no local training between rounds acknowledged.
- **Incentive limitation** (`main.py:503-508`) — patient attacker vulnerability disclosed.
- **SEQ_LEN comment** (`adtcn.py:49`) — empirical choice, ablation is future work.
- **Latency disclosure** (`leader_block.py:221-225`) — simulated, not real Fabric.
- **Padding bias** (`adtcn.py:410-413`) — 0.003% boundary effect documented.
- **DP comparison** (`run_baselines.py:153-168`) — FedAvg vs FedAvg+DP printed.
- **Baseline script** (`run_baselines.py`) — four configurations, McMahan-weighted FedAvg.
- **Dataset path guard** — `FileNotFoundError` with Kaggle URL in place.
- **DB-BOA switching criterion** (`db_boa.py:114-130`) — correct implementation; docstring now matches code.
- **X_val used for Shapley** (`main.py:301-302, 469-472`) — both Phase 7 and Phase 8 now use training val set.
- **get_eval_subset** (`data_loader.py:115-129`) — now stratified (real distribution), consistent with surrogate design.
- **dboa.py fragrance** (`dboa.py:96-97`) — abs() added for negative fitness safety.

---

## Key citations

| Contribution | Citation |
|---|---|
| Krum | Blanchard et al., NeurIPS 2017 |
| Coordinate-wise median (alt. to Krum) | Yin et al., ICML 2018 |
| Differential privacy (Gaussian mechanism) | Dwork et al., TCC 2006 |
| DP-SGD (per-example DP) | McMahan et al., ICLR 2018 |
| DP composition | Dwork et al., TCC 2006 §3.5 (basic); Mironov (2017) for Rényi |
| FedAvg baseline | McMahan et al., AISTATS 2017 |
| FedProx (non-i.i.d. FL) | Li et al., MLSys 2020 |
| Shapley FL fairness | Wang et al. (FedSV), IEEE BigData 2020 |
| ULB dataset | Lopez-Rojas et al., 2016 |
| Federated evaluation trade-off | Hsieh et al., 2020 |
