# Open Issues — Must Fix Before Submission

Everything below is still open. Issues are ordered by how fast they would end
a defense.

---

## SIGNIFICANT — Weakens a claimed contribution

### ✅ 1. DP noise (σ=4.84) completely overwhelms federated weight magnitudes

**File**: `db_boa_framework/models/federated_adtcn.py:69-79`
**Defense question**: Q50

σ = C·√(2·ln(1.25/δ))/ε = 1·√(2·ln(125000))/1 ≈ 4.84.  After L2 clipping
each weight tensor to norm ≤ 1, the per-element magnitude is 1/√dim:

  Conv1d(30, F, 3) weights : dim = F×30×3 = 5760 → per-element ≈ 0.013
  Conv1d(F, 2F, 3) weights : dim = 2F×F×3 ≈ 24576 → per-element ≈ 0.006
  Bias tensors              : dim = F ≈ 64   → per-element ≈ 0.125

Noise σ=4.84 is 37–800× the per-element signal.  Krum then selects ONE org's
noisy weights (no averaging to reduce noise), so the global model is effectively
random weights.  `run_baselines.py` will reveal this as a catastrophic accuracy
drop for FedAvg+DP — the student must understand the cause or Q50 will catch
them off guard.

**Fix needed**: Either (a) increase ε to a less aggressive value (ε=50 or ε=100
are common in applied DP-FL) and report the privacy budget honestly, or (b)
apply DP-SGD (per-example gradient clipping + noise, McMahan et al. 2018) which
keeps C proportional to a single gradient rather than the entire weight tensor, or
(c) keep ε=1.0 but explicitly disclose in the thesis: "At ε=1.0, σ≈4.84 exceeds
per-element weight magnitudes by ×370; the global model post-DP is near-random
weights, and the DP contribution serves as a privacy mechanism at the cost of
near-complete model degradation.  Results in Section X confirm this trade-off."
Disclosure without a code change is the fastest fix before a defense.

---

### ✅ 2. PTC/NTC feature engineering creates ~268 features never used by the CNN

**File**: `db_boa_framework/data/data_loader.py:156-200` (`_engineer_temporal_features`)
**File**: `db_boa_framework/models/adtcn.py:391-399` (`_make_sequences`)
**Defense question**: Q30

`_engineer_temporal_features()` computes PTC rolling mean/std over windows [5,10,20]
and NTC diff orders [1,2] across all 33 input columns.  With 33 base features,
3 windows × 2 stats = 6 PTC passes → 198 extra columns; 2 NTC diffs → 66 extra
columns; plus MJE cross-features.  Total engineered matrix ≈ 301 columns.

`_make_sequences()` takes `X[:, :n_raw]` where `n_raw = min(X.shape[1], 33)`.
The entire PTC/NTC block (columns 33+) is silently discarded.  The CNN sees
exactly 33 features regardless of how much temporal engineering was applied.

A supervisor asking Q30 will note: "Your data loader engineers 301 features but
the model uses 33.  Why engineer 268 features you throw away?"  There is no good
answer other than "the feature pipeline was inherited and the CNN was designed to
use raw features only."

**Fix needed**: Either (a) pass the full engineered matrix into the CNN (change
`N_RAW_FEATURES + 3` to the full engineered dimension and update Conv1d input
channels), or (b) remove `_engineer_temporal_features()` and document that only
raw + recurrence features are used, or (c) add an explicit comment in
`_make_sequences` noting that PTC/NTC features are available in the input matrix
but the CNN uses only the leading 33 base columns, citing the design decision.
Option (c) is the fastest fix: one comment + one thesis sentence.

---

## MINOR — Must disclose before defense

### ✅ 3. Shapley coalition values computed on X_test[:500] — test data used before evaluation

**File**: `db_boa_framework/main.py:300-301`
**Defense question**: Q63, Q94

```python
X_val_shared = X_test[:500]
y_val_shared = y_test[:500]
```

The Shapley coalition values v(S) — which determine the on-chain token distribution
— are computed on a slice of the final test set.  Although Krum selects the global
model independently (no circularity in model accuracy), using test data for ANY
computation before final evaluation violates clean evaluation protocol.  Examiners
from a machine-learning background will immediately flag this when they see Q63.

A proper setup uses the training validation split (`X_val, y_val` returned by
`loader.load()`) for Shapley, keeping X_test untouched until final reporting.

**Fix needed**: Replace `X_test[:500]` / `y_test[:500]` with `X_val[:500]` /
`y_val[:500]` in the federation loop (and in `run_baselines.py` if applicable).
One-line change.  Add a thesis note: "Shapley coalition values are evaluated on
the held-out training validation set so the final test set remains unseen during
all intermediate computations."

---

### ✅ 4. FedAvg baseline uses equal-weight average, not McMahan et al.'s size-weighted average

**File**: `db_boa_framework/run_baselines.py:61-67` (`_avg_weights`)
**Defense question**: Q144

McMahan et al. (AISTATS 2017) defines FedAvg aggregation as:
  w_global ← Σ_k (n_k / n) · w_k
where n_k is the number of local samples for org k.  With the 50/30/20 split,
the correct FedAvg weights are [0.5, 0.3, 0.2].

`_avg_weights()` uses 1/K = 1/3 ≈ 0.333 for each org — equal weighting regardless
of data volume.  This is technically *unweighted FedAvg*, not McMahan et al.'s
original algorithm.  Q144 asks this directly.  An examiner who knows FedAvg will
catch the deviation.

**Fix needed**: Change `_avg_weights()` to data-size-weighted average:
```python
def _avg_weights(weights_list: list, counts: list) -> list:
    total = sum(counts)
    w = [c / total for c in counts]
    n_arrays = len(weights_list[0])
    return [
        sum(w[i] * weights_list[i][a] for i in range(len(weights_list)))
        for a in range(n_arrays)
    ]
```
Pass org sample counts as `counts`.  OR add a docstring note:
"This uses equal-weight averaging (unweighted FedAvg).  McMahan et al.'s original
uses data-size-weighted averaging; with 50/30/20 split the correct weights are
[0.5, 0.3, 0.2].  The difference is minor but acknowledged."

---

## What is genuinely solid (do not touch)

- **1D-CNN** (`adtcn.py:64-86`) — correct PyTorch layer sequence; docstring now
  honest about ReLU (activation dead-code fixed).
- **DB-BOA 2D search** (`adtcn.py:113-200`) — epoch removed from search space;
  `_MIN_FRAUD_ROWS=30` guards against 4-sample surrogate.
- **DP formula** (`federated_adtcn.py:50-80`) — Gaussian mechanism σ is correct;
  composition logged after each round.
- **Shapley math** (`federation_manager.py`) — exact for n=3, formula correct;
  trusted-aggregator limitation disclosed.
- **Krum labelling** (`federation_manager.py`) — "outlier-weight rejection" (not
  "Byzantine fault tolerance"); f=0 correctly explained.
- **Weighted cross-entropy** (`adtcn.py:259-262`) — correct for 0.17% fraud rate.
- **Feature honesty** (`graph_features.py`) — recurrence features, not graph features.
- **Activation honesty** (`config.py:93-95`, `adtcn.py:74-75`) — ReLU use disclosed.
- **FL validity note** (`main.py:286-289`) — no local training between rounds acknowledged.
- **Incentive limitation** (`main.py:503-508`) — patient attacker vulnerability disclosed.
- **SEQ_LEN comment** (`adtcn.py:49`) — empirical choice, ablation is future work.
- **Latency disclosure** (`leader_block.py:221-223`) — simulated, not real Fabric.
- **Padding bias** (`adtcn.py:398-400`) — 0.003% boundary effect documented.
- **DP comparison** (`run_baselines.py:138-153`) — FedAvg vs FedAvg+DP printed.
- **Baseline script** (`run_baselines.py`) — four configurations, comparable.
- **Dataset path guard** — `FileNotFoundError` with Kaggle URL in place.

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
