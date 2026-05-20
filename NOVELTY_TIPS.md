# Open Issues — Must Fix Before Submission

The 1D-CNN replacement is the only change that is solid. Everything else has bugs,
false claims in docstrings, or broken mathematical guarantees. Fix these in order.

---

## CRITICAL — Will fail a supervisor question

### 1. Krum node count violates its own mathematical requirement

**File**: `db_boa_framework/models/federation_manager.py:205-207`  
**File**: `db_boa_framework/config.py` (`byzantine_f = 1`)

Krum requires **n ≥ 2f + 3** to guarantee Byzantine fault tolerance. With n=3 orgs
and f=1: 3 ≥ 2(1)+3 = 5. This fails. The implementation runs without crashing but
provides zero formal guarantee. Citing Blanchard et al. (NeurIPS 2017) under these
conditions is actively misleading.

**Fix (choose one)**:
- Set `byzantine_f = 0` in config — Krum still runs, claim is "robust to 0 adversaries"
  which is honest (it just selects the most consensus-aligned org).
- Increase to 5+ orgs in the simulation (BankA–BankE) so n ≥ 2(1)+3 = 5 holds and
  the f=1 guarantee is mathematically valid.
- If staying at 3 orgs and f=1, switch to **coordinate-wise median** aggregation
  (Yin et al., ICML 2018) which tolerates f < n/2 = 1.5, i.e. f=1 out of 3.

---

### 2. Shapley docstring falsely claims no shared labels are needed

**File**: `db_boa_framework/models/federation_manager.py:19` (module docstring)  
**File**: `db_boa_framework/models/federation_manager.py:248` (`coalition_value`)

The docstring says *"no shared labels needed for computation"*. This is false.
`coalition_value()` calls `temp.evaluate_on_validation(X_val, y_val)` — a shared
labeled validation set at the aggregator. DP on weights does not remove this
requirement. The shared-validation-set privacy problem is unresolved.

**Fix**:
- Remove the false claim from the docstring immediately.
- Replace `coalition_value()` with a **loss-based** or **gradient-cosine-similarity**
  coalition value that uses only the org's own local data, OR
- Acknowledge honestly in the thesis that a trusted aggregator with a shared holdout
  set is assumed, and cite the federated evaluation literature that discusses this
  trade-off (e.g., Measuring the Effects of Non-Identical Data Distribution for
  Federated Visual Classification, Hsieh et al., 2020).

---

### 3. Graph features are not graph features — and "PageRank" is a degree ratio

**File**: `db_boa_framework/data/graph_features.py`

The ULB Credit Card Fraud dataset has no account IDs — they were removed during PCA
preprocessing. The "graph" built here connects transactions by amount-bucket
co-occurrence within a rolling window. This is a **sliding-window frequency counter**,
not a transaction graph. There are no nodes representing accounts, no edges
representing transfers.

Additionally, `pagerank_norm = in_deg / (in_deg + out_deg)` at line 98 is a
**degree ratio**, not PageRank. PageRank is an iterative eigenvector computation.
Calling it PageRank and citing Liu et al. (WWW 2021) — whose paper uses real
account-to-account graphs — is incorrect.

**Fix**:
- Rename the features honestly: `amount_recurrence_before`, `amount_recurrence_after`,
  `degree_ratio`. Remove the word "PageRank" entirely.
- Update the Liu et al. citation or remove it. The citation only applies if you have
  real account graphs.
- If you want real graph features, use the **PaySim dataset** instead of ULB — PaySim
  preserves account IDs, so genuine sender→receiver edges exist.

---

## SIGNIFICANT — Weakens the hyperparameter optimization claim

### 4. DB-BOA optimizes SGDClassifier but the deployed model is a 1D-CNN

**File**: `db_boa_framework/models/adtcn.py:107-148` (`_ADTCNObjective`)

The DB-BOA hyperparameter search evaluates candidate parameters using an
`SGDClassifier` surrogate for speed. The optimal `hidden_neurons` value found by
DB-BOA is then reused as `n_filters` for the CNN. MLP neuron count and CNN filter
count have completely different optimal ranges and sensitivities. The optimization
landscape of one has no predictable relationship to the other.

**Fix**:
- Use a **small CNN surrogate** for DB-BOA evaluation: same architecture but trained
  on a small subsample (e.g., 2,000 rows, 5 epochs) so each evaluation takes ~2s.
  This makes the hyperparameter search honest — DB-BOA is actually optimizing CNN
  filters, not MLP neurons.
- Rename `hidden_neurons_bounds` → `filter_count_bounds` in config to reflect what
  is actually being searched.

---

## MINOR — Must fix before anyone reads the code

### 5. Dataset path is not verified at startup — crash gives no useful error

**File**: `db_boa_framework/config.py:21`  
**File**: `db_boa_framework/data/data_loader.py:132-147`

`DATASET_PATH` resolves to `../datasets/creditcard.csv` relative to the project root.
If the file is absent the pipeline crashes with a generic pandas `FileNotFoundError`
with no guidance. The dataset does not ship with the repository and there is no
download script.

**Fix**: Add a guard at the top of `_load_real_transactions()`:

```python
if not os.path.exists(path):
    raise FileNotFoundError(
        f"ULB Credit Card Fraud dataset not found at:\n  {path}\n"
        "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "and place creditcard.csv in the datasets/ folder."
    )
```

---

### 6. Baseline comparisons are category-incompatible with current evaluation

**File**: `db_boa_framework/utils/metrics.py` (wherever `baseline_metrics()` is defined)

The hardcoded baseline numbers (MBO-ADTCN, WSA-ADTCN, etc.) come from the original
paper which was run on **synthetic data**. Your system now runs on the ULB real
dataset. These two result sets cannot be compared in the same table. Presenting them
together implies you re-ran those algorithms on the same data, which you did not.

**Fix**:
- Remove the synthetic baseline numbers from any table or plot.
- Compare only against methods also evaluated on ULB: vanilla FedAvg (McMahan 2017),
  FedAvg + Krum, FedAvg + DP. These are implementable in your existing codebase by
  toggling `use_krum` and `use_dp` flags.

---

## What is genuinely solid (do not touch)

- **1D-CNN** (`adtcn.py:61-87`) — real temporal model, defensible architecture,
  correctly implemented with PyTorch sequences.
- **DP formula** (`federated_adtcn.py:50-80`) — Gaussian mechanism is mathematically
  correct. The clipping and noise injection are right.
- **Shapley math** (`federation_manager.py:228-307`) — the marginal contribution loop
  and coefficient formula are exact for n=3.
- **Weighted cross-entropy for class imbalance** (`adtcn.py:259-262`) — correct
  approach for 0.17% fraud rate without oversampling.

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
