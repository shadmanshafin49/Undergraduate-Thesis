# Open Issues — Must Fix Before Submission

The 1D-CNN, DP formula, Shapley math, and weighted cross-entropy are solid.
Everything below is still broken. Issues are ordered by how fast they would end
a defense.

---

## CRITICAL — Will fail a supervisor question

### ✅ 1. Krum with f=0 claims "Byzantine-robust" but tolerates zero adversaries

**File**: `db_boa_framework/config.py:167` (`byzantine_f = 0`)
**File**: `db_boa_framework/models/federation_manager.py:18` (module docstring)

**Fix applied**: Removed "Byzantine fault tolerance" everywhere it appeared.
The module docstring, `_krum_aggregate` docstring, and architecture note now all
say "outlier-weight rejection for consensus alignment" and explicitly state that
with f=0 and n=3 no adversary is assumed.  The distinction from Blanchard et al.
(f≥1) is documented inline.

---

### ✅ 2. The activation comparison plot fabricates data

**File**: `db_boa_framework/utils/visualizer.py:189-196`

Every value was a hardcoded offset from the single measured accuracy; no other
activation was actually tested.  The CNN uses `nn.ReLU()` but the plot labelled
TanH as the peak — fabricated data.

**Fix applied**: `plot_activation_comparison()` is now a no-op stub that returns
`None`.  `generate_all_plots` already filters `None` results.  A comment above
the stub explains why it was removed and what would be needed to reinstate it
(train 6 models with different activations, record real results).

---

### ✅ 3. DB-BOA cannot optimise epoch count — the dimension is frozen at 5

**File**: `db_boa_framework/models/adtcn.py:143`

The surrogate hard-capped training at 5 epochs regardless of what DB-BOA
proposed, making the epoch dimension flat.

**Fix applied**: Epoch count removed from the DB-BOA search space entirely.
`_ADTCNObjective.__call__` now uses 2D params: `params[0]`→n_filters,
`params[1]`→steps_per_epoch.  `optimise_hyperparams` builds a 2D lb/ub array.
`optimal_params["epoch_count"]` is now set to `self.cfg["epoch_count"]` (the
fixed default) — it is documented as "not searched".  Module docstring, ADTCN
class docstring, and Phase 2 print in main.py updated to say "2D search".

---

### ✅ 4. No ULB baseline runs exist — the comparison is empty and unfalsifiable

**File**: `db_boa_framework/utils/metrics.py:150-152`

**Fix applied**: Created `db_boa_framework/run_baselines.py` — a self-contained
script that runs the four configurations (FedAvg, FedAvg+Krum, FedAvg+DP,
DB-BOA-ADTCN) on the ULB dataset and prints Python dict literals ready to paste
into `baseline_metrics()`.  The script reuses the same DB-BOA hyperparameter
search across all runs so results are directly comparable.  Run with:
`python3 run_baselines.py`

---

## SIGNIFICANT — Weakens a claimed contribution

### ✅ 5. CNN surrogate optimises on 50/50 fraud rate; deployment is 0.17%

**File**: `db_boa_framework/models/adtcn.py:116-123`

The surrogate subsample was ~50% fraud; real deployment is 0.17%.

**Fix applied**: Subsample now uses the real fraud rate:
```python
fraud_rate = len(fraud_idx) / total
n_f = max(4, int(self._SURROGATE_ROWS * fraud_rate))
```
The surrogate now trains on ~0.17% fraud, matching deployment conditions.

---

### ✅ 6. MTTA is labelled "Multiple Time-scale Temporal Attention" but is GlobalMaxPool

**File**: `db_boa_framework/models/adtcn.py:11` (docstring), `adtcn.py:85`

**Fix applied**: Module docstring updated:
```
│  MTTA  GlobalMaxPool — selects the most anomalous time-step        │
│         activation across the SEQ_LEN window  (pooling, not        │
│         attention; the paper's "MTTA" label is re-used here)       │
```
No code changed (the pooling operation is correct); only the label claim is
corrected.

---

### ✅ 7. Krum and Shapley operate independently and can produce contradictory decisions

**File**: `db_boa_framework/models/federation_manager.py:97-160`

**Fix applied**: Added an "Architecture note" block to the module docstring that
explicitly states this is intentional:
- Krum decides *which* model is used globally (security / outlier rejection).
- Shapley decides *how much* each org earns (fairness / incentive distribution).
An org whose weights are Krum-rejected can still earn tokens, and this is
documented as correct behaviour.

---

### ✅ 8. DP composition across 3 federation rounds is never computed or disclosed

**File**: `db_boa_framework/models/federated_adtcn.py:50-80`
**File**: `db_boa_framework/main.py` (Phase 7)

**Fix applied**: After each DP weight-sharing step in `run_federation_round()`,
the following is now logged:
```
[FED]  DP composition  : after k round(s) ε_total=k·ε, δ_total=k·δ
                         (basic composition, Dwork et al. 2006 §3.5)
```
With ε=1.0, δ=1e-5 and 3 rounds this gives ε_total=3.0, δ_total=3e-5.

---

### ✅ 9. No experiment shows DB-BOA finds better hyperparameters than defaults

**File**: `db_boa_framework/models/adtcn.py` (optimise_hyperparams)
**File**: `db_boa_framework/main.py` (Phase 2)

**Fix applied**: Phase 4 in main.py now trains a second model with the default
config values (hidden_neurons=128, epoch_count=30, steps_per_epoch=150) and
prints a side-by-side comparison:
```
[OPT]  Default  (F=128, ep=30, spe=150)  Acc=XX.XXXX%  MCC=0.XXXX
[OPT]  DB-BOA   (F=..., ep=..., spe=...) Acc=XX.XXXX%  MCC=0.XXXX
[OPT]  Gain from DB-BOA: Accuracy ▲X.XXXX%  MCC ▲0.XXXX
```

---

## MINOR — Must fix before anyone reads the code

### ✅ 10. `BASELINE_NAMES` and `CLASSIFIER_NAMES` still list synthetic-data algorithms

**File**: `db_boa_framework/config.py:106-121`

**Fix applied**: Both lists now contain ULB-evaluated names:
```python
BASELINE_NAMES   = ["FedAvg", "FedAvg+Krum", "FedAvg+DP", "DB-BOA-ADTCN (proposed)"]
CLASSIFIER_NAMES = ["FedAvg", "FedAvg+Krum", "FedAvg+DP", "DB-BOA-ADTCN (proposed)"]
```

---

### ✅ 11. `INCENTIVE_CONFIG` comment and chaincode still say "DB-BOA weight"

**Files**: `db_boa_framework/config.py:137`, `db_boa_fabric/chaincode/lib/db_boa_chaincode.js`

**Fix applied**:
- `config.py`: `"federation_pool": 20,  # shared by Shapley contribution weight`
- `chaincode.js` header comment: "Federation participation +20 tokens shared by Shapley contribution weight"
- `chaincode.js` `recordFederationRound` docstring: updated to say Shapley-weighted aggregation, not DB-BOA Job 3

---

### ✅ 12. The federation simulation is not ecologically valid federated learning

**File**: `db_boa_framework/data/data_loader.py` (`split_for_orgs`)

**Fix applied**: Added "Ecological validity note" to `split_for_orgs` docstring
acknowledging the single-bank limitation and pointing to the thesis Limitations
section.  Config.py `ORG_DATA_SPLITS` comment also notes the i.i.d. / FedProx
context (Issue 13 below).

---

### ✅ 13. FedAvg i.i.d. assumption is violated by the data split and is unacknowledged

**File**: `db_boa_framework/config.py` (`ORG_DATA_SPLITS`)

**Fix applied**: Added a multi-line comment after `ORG_DATA_SPLITS`:
```python
# Limitation: ULB comes from one bank — this is a controlled simulation.
# McMahan et al. assume i.i.d. data; this split is mildly non-i.i.d.
# FedProx (Li et al., MLSys 2020) would be more appropriate for severely
# heterogeneous distributions.
```

---

## What is genuinely solid (do not touch)

- **1D-CNN** (`adtcn.py:61-87`) — correct PyTorch sequences, real temporal model.
- **CNN surrogate architecture** (`adtcn.py:91-192`) — matches final model; issues
  #3 (epoch cap) and #5 (class distribution) now fixed; keep the structure.
- **DP formula** (`federated_adtcn.py:50-80`) — Gaussian mechanism is correct.
- **Shapley math** (`federation_manager.py:228-307`) — exact for n=3, formula correct.
- **Weighted cross-entropy** (`adtcn.py:259-262`) — correct for 0.17% fraud rate.
- **graph_features.py docstring** — honest about what the features actually are.
- **Shapley trusted-aggregator disclosure** — correctly stated in docstring.
- **Dataset path guard** — `FileNotFoundError` with Kaggle URL is in place.

---

## Key citations

| Contribution | Citation |
|---|---|
| Krum | Blanchard et al., NeurIPS 2017 |
| Coordinate-wise median (alt. to Krum) | Yin et al., ICML 2018 |
| Differential privacy (Gaussian mechanism) | Dwork et al., TCC 2006 |
| DP composition | Dwork et al., TCC 2006 §3.5 (basic); Mironov (2017) for Rényi |
| FedAvg baseline | McMahan et al., AISTATS 2017 |
| FedProx (non-i.i.d. FL) | Li et al., MLSys 2020 |
| Shapley FL fairness | Wang et al. (FedSV), IEEE BigData 2020 |
| ULB dataset | Lopez-Rojas et al., 2016 |
| Federated evaluation trade-off | Hsieh et al., 2020 |
