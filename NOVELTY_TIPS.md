# Open Issues — Must Fix Before Submission

The 1D-CNN, DP formula, Shapley math, and weighted cross-entropy are solid.
Everything below is still broken.

---

## CRITICAL — Will fail a supervisor question

### 1. Krum with f=0 claims "Byzantine-robust" but tolerates zero adversaries

**File**: `db_boa_framework/config.py:167` (`byzantine_f = 0`)
**File**: `db_boa_framework/models/federation_manager.py:18` (module docstring)

Setting `byzantine_f = 0` satisfies the n ≥ 2f+3 math (3 ≥ 3), but means the
algorithm is configured to assume **no adversaries exist**. With f=0, Krum simply
picks the org whose weights are closest to the others — a plain consensus-alignment
selector. The module docstring still says "Krum → security (Byzantine fault
tolerance)" which directly contradicts the config comment "no Byzantine adversary
is assumed". Citing Blanchard et al. (NeurIPS 2017) in this setting is still
misleading because their security guarantee requires f ≥ 1.

**Fix (choose one)**:
- Remove the Byzantine-robustness claim entirely. Describe Krum as "outlier-weight
  rejection for consensus alignment" — honest and still novel enough.
- Expand to 5 orgs (BankA–BankE) so f=1 satisfies n ≥ 2(1)+3 = 5. The federation
  simulation becomes more realistic and the Blanchard citation holds.

---

### 2. No ULB baseline runs exist — the comparison table is empty and unfalsifiable

**File**: `db_boa_framework/utils/metrics.py:150-152`

```python
algo_results: dict = {}   # populate with ULB-evaluated algorithm baselines
clf_results:  dict = {}   # populate with ULB-evaluated classifier baselines
```

The thesis narrative says *"Compared to vanilla FedAvg, we show X% accuracy
retention under Y% Byzantine nodes at ε=1.0."* X and Y are undefined. A thesis
with an empty comparison section cannot be defended.

**Fix**: Run the pipeline three times with these flag combinations and record the
test-set metrics:

| Run | `use_krum` | `use_dp` | `use_shapley` | Label |
|-----|-----------|---------|--------------|-------|
| 1   | False     | False   | False        | FedAvg baseline |
| 2   | True      | False   | False        | FedAvg + Krum |
| 3   | False     | True    | False        | FedAvg + DP |
| 4   | True      | True    | True         | Proposed (full) |

Paste the resulting numbers into `baseline_metrics()`. This is one afternoon of
compute, not a research problem.

---

## SIGNIFICANT — Weakens a claimed contribution

### 3. CNN surrogate optimises on 50/50 fraud rate; deployment is 0.17%

**File**: `db_boa_framework/models/adtcn.py:116-123` (`_ADTCNObjective.__init__`)

```python
n_f = min(len(fraud_idx),  self._SURROGATE_ROWS // 2)   # up to 1,000 fraud
n_n = min(len(normal_idx), self._SURROGATE_ROWS - n_f)  # remaining normal
```

The surrogate subsample has ~50% fraud. The final deployed model runs on 0.17%
fraud. The optimal number of filters and training epochs for a 50/50 dataset is
not the same as for a 0.17% dataset. DB-BOA is finding hyperparameters for the
wrong class distribution, then applying them to the right one.

**Fix**: Match the real fraud rate in the subsample and rely on weighted loss for
balance:

```python
fraud_rate = len(fraud_idx) / (len(fraud_idx) + len(normal_idx))
n_f = max(4, int(self._SURROGATE_ROWS * fraud_rate))
n_n = self._SURROGATE_ROWS - n_f
```

---

### 4. CNN surrogate runtime is undisclosed — 20–100 minutes on CPU

**File**: `db_boa_framework/models/adtcn.py:107` (`_ADTCNObjective`)
**File**: `db_boa_framework/main.py` (README / help text)

With `population_size=20` and `max_iterations=30`, DB-BOA makes 600 CNN training
calls. At 5–10 seconds each on a CPU laptop, that is 50–100 minutes for
hyperparameter search alone, before the final full-dataset training. The `--quick`
mode reduces this to 150 calls (~12–25 minutes) but this is still not disclosed
anywhere in the README, `--help` output, or `main.py` banner.

**Fix**: Add a runtime estimate to the `--quick` help string and the README quick-
start section. Consider adding a `--no-opt` flag that skips DB-BOA and uses
hardcoded default filter counts for fast runs.

---

## MINOR — Must fix before anyone reads the code

### 5. `BASELINE_NAMES` and `CLASSIFIER_NAMES` in config.py still list synthetic baselines

**File**: `db_boa_framework/config.py:106-121`

```python
BASELINE_NAMES   = ["MBO-ADTCN", "WSA-ADTCN", "DBOA-ADTCN", "BOA-ADTCN", "DB-BOA-ADTCN"]
CLASSIFIER_NAMES = ["EfficientNet", "ResNet", "DenseNet", "DTCN", "DB-BOA-ADTCN"]
```

These lists were never updated when synthetic baselines were removed from
`baseline_metrics()`. If `visualizer.py` still reads them, synthetic algorithm
names will appear on plots alongside real ULB results.

**Fix**: Replace with ULB-compatible names that match the four runs in issue #2:

```python
BASELINE_NAMES   = ["FedAvg", "FedAvg+Krum", "FedAvg+DP", "DB-BOA-ADTCN (proposed)"]
CLASSIFIER_NAMES = ["FedAvg", "FedAvg+Krum", "FedAvg+DP", "DB-BOA-ADTCN (proposed)"]
```

---

### 6. `INCENTIVE_CONFIG` comment and chaincode still say "DB-BOA weight" after Shapley replaced Job 3

**File**: `db_boa_framework/config.py:137`
**File**: `db_boa_fabric/chaincode/lib/db_boa_chaincode.js` (federation reward section)

```python
"federation_pool": 20,   # +20  shared by DB-BOA weight across orgs
```

DB-BOA Job 3 was replaced by Shapley. The incentive pool is now distributed by
Shapley weights. The comment is wrong. The chaincode likely still describes the
federation reward in DB-BOA terms — an examiner reading both files will see two
different mechanisms described for the same on-chain event.

**Fix**: Update the comment to "shared by Shapley contribution weight across orgs"
and verify the chaincode federation reward logic matches the Python Shapley output.

---

## What is genuinely solid (do not touch)

- **1D-CNN** (`adtcn.py:61-87`) — real temporal model, correct PyTorch sequences.
- **CNN surrogate** (`adtcn.py:91-192`) — architecture matches final model; only
  the class distribution (issue #3) and runtime disclosure (issue #4) need fixing.
- **DP formula** (`federated_adtcn.py:50-80`) — Gaussian mechanism is correct.
- **Shapley math** (`federation_manager.py:228-307`) — exact for n=3.
- **Weighted cross-entropy** (`adtcn.py:259-262`) — correct for 0.17% fraud rate.
- **graph_features.py docstring** — now honest about what the features are.
- **Shapley trusted-aggregator disclosure** — now correctly stated in docstring.

---

## Key citations

| Contribution | Citation |
|---|---|
| Krum | Blanchard et al., NeurIPS 2017 |
| Coordinate-wise median (alt. to Krum) | Yin et al., ICML 2018 |
| Differential privacy (Gaussian mechanism) | Dwork et al., TCC 2006 |
| FedAvg baseline | McMahan et al., AISTATS 2017 |
| Shapley FL fairness | Wang et al. (FedSV), IEEE BigData 2020 |
| ULB dataset | Lopez-Rojas et al., 2016 |
| Federated evaluation trade-off | Hsieh et al., 2020 |
