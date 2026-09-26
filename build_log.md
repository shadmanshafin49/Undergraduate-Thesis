# Build Log — DB-BOA-FEL-ADTCN Framework (FL-ADTCN)

All changes applied to strengthen the thesis novelty and research defensibility.

> **Coverage.** Every change set from the first commit (`b903bf2`, 2026-05-20) to the latest
> commit on branch `obj13-surrogate-repair` (`c66d1d1`, 2026-09-13). Nothing in the working tree is
> newer than that commit (checked 2026-09-26). `origin/main` still sits at `ccdd19f` (2026-08-30):
> **the whole paper extension (Phases 3–7) exists only on `obj13-surrogate-repair`.**
>
> **How to use this file.** It is a changelog: for each change set, the files touched, what changed,
> and how it was checked. The *why*, told as a story, is in `research_journey.md`. The live
> state of work, pre-registrations and scorecards are in `TASK.md`. Numbers live only in
> `db_boa_framework/results/*.json`.
>
> **Historical entries are kept verbatim.** Tips 1–6, Fixes 1–29 and BF-1–10 were written at the
> time. Some of their numbers and claims were later withdrawn (for example DP at ε=1.0 as the
> default, and the DB-BOA-vs-default MCC gap). The table at the end,
> *Which early entries were later superseded*, lists them. The early entries are not edited.

## Phase index

| Phase | Dates | Entries | Where the work was done |
|---|---|---|---|
| 0 — Starting point | 2026-05-20 | B0-1, B0-2 | first two commits, `Undergraduate-Thesis` |
| 1 — Novelty tips and honesty fixes | 2026-05-20 → 2026-06-02 | Tips 1–6, Fixes 1–29, BF-1–10, Fixes 30–36 | `Undergraduate-Thesis` (local) |
| 2 — Audit, characterisation experiments, report rebuild | 2026-06-02 → 2026-06-11 | J-1 … J-15 | `Undergraduate-Thesis-v2` (Linux/WSL2, commit author `root`) |
| — Thesis accepted | **2026-06-13** | — | — |
| 3 — Paper extension begins: sync, loose ends, second dataset | 2026-08-30 → 2026-08-31 | X-1 … X-9 | local Windows laptop |
| 4 — Taking the system claims off one dataset | 2026-09-01 | X-10 … X-13 | laptop |
| 5 — Controls, privacy budget, DB-BOA surrogate repair | 2026-09-04 → 2026-09-05 | X-14 … X-22 | laptop |
| 6 — Third-dataset loader and figures | 2026-09-08 → 2026-09-10 | X-23 … X-25 | laptop |
| 7 — All five datasets taken to measurement | 2026-09-11 → 2026-09-13 | X-26 … X-41 | laptop + Kaggle |

---

# PHASE 0 — STARTING POINT (2026-05-20)

## ✅ B0-1 — Initial framework import (2026-05-20, commit `b903bf2` "first commit")

**Files added**
- `db_boa_framework/main.py` — six-phase orchestrator: leader selection (DB-BOA, Eq. 10), ADTCN hyperparameter optimisation (DB-BOA, Eq. 11), training, evaluation, multi-round consensus simulation, plots. Flags `--quick`, `--no-plots`.
- `db_boa_framework/config.py` — all parameters: `DB_BOA_CONFIG`, `LEADER_BLOCK_CONFIG`, `ADTCN_CONFIG`, `FEDERATION_CONFIG`, `INCENTIVE_CONFIG`.
- `db_boa_framework/algorithms/boa.py`, `dboa.py`, `db_boa.py` — the Billiards Optimisation Algorithm (BOA, Givi & Hubálovská 2023), the Dynamic Butterfly Optimisation Algorithm (DBOA, Tubishat 2020), and their hybrid DB-BOA (Dynamic Butterfly–Billiards) from the base paper.
- `db_boa_framework/blockchain/leader_block.py` — simulated 10-node consortium with leader selection and a simulated consensus round.
- `db_boa_framework/data/data_loader.py` — a synthetic 20,000-row generator (30 features, 5 % fraud) plus PTC/NTC temporal feature engineering.
- `db_boa_framework/models/adtcn.py` — "ADTCN", in fact a scikit-learn `MLPClassifier`.
- `db_boa_framework/models/federated_adtcn.py`, `models/federation_manager.py` — three banks; aggregation weights chosen by DB-BOA "Job 3".
- `db_boa_framework/utils/metrics.py` (Obf2 = Acc + Pre + NPV + MCC + 1/FPR, plus hard-coded base-paper baselines) and `utils/visualizer.py` (12 plot functions).
- `db_boa_framework/requirements.txt`, and 12 result PNGs plus `results/db_boa_results.json`.
- `db_boa_fabric/` — Hyperledger Fabric layer: chaincode (`chaincode/index.js`, `chaincode/lib/db_boa_chaincode.js`, `package.json`), Express API server (`api-server/server.js`), web dashboard (`api-server/index.html`), wallet scripts (`enrollAdmin.js`, `registerUser.js`, `wallet/admin.id`, `wallet/appUser.id`), launcher (`start_db_boa.sh`), `README.md`, plus `package.json` and two duplicate downloads of it (`package (1).json`, `package (2).json`).
- `fabric/install-fabric.sh` — Fabric binaries and samples installer (a modified upstream bootstrap script).
- `hello-world/docker-compose.yml` — a two-line Docker smoke test.
- `.gitignore`, `.claude/settings.local.json`.

**What it was.** The pre-thesis (P2) design as working code. DB-BOA had three jobs: tune the detector, pick the block leader, and set federated aggregation weights. It ran end to end, on synthetic data.

**Known at the time: nothing.** Everything that follows in Phase 1 is the list of what was wrong with it.

---

## ✅ B0-2 — Thesis sources, reference papers, first novelty review (2026-05-20, commit `fe39499` "2nd commit")

**Files added**
- `FINAL YEAR THESIS REPORT/` — LaTeX thesis: `main.tex`; `chapters/chapter_1.tex`, `chapter_2.tex`, `chapter_3.tex`, `chapter_5.tex`, `chapter_6.tex`, `chapter_9.tex`; `core/abstract.tex`, `acknowledgement.tex`, `approval.tex`, `declaration.tex`, `titlepage.tex`; `bibliography/references.bib`; about 30 images under `images/` (including `images/DB-BOA Metrics/`).
- `all papers/` — the first nine reference PDFs: Blanchard 2017 (Krum), Dwork 2006 (DP), Givi & Hubálovská 2023 (BOA), Liu 2021 (Pick & Choose GNN), Lopez-Rojas 2016 (PaySim), McMahan 2017 (FedAvg), **Prabanand & Thanabal 2025 (DB-BOA-ADTCN, the base paper)**, Tubishat 2020 (DBOA), Wang 2020 (FedSV).
- `NOVELTY_TIPS.md` — the first external review: six tiers of "novelty tips" (real dataset, Krum, DP, a real temporal model, Shapley instead of Job 3, graph features).

**Files changed**
- `config.py`, `data/data_loader.py`, `models/federation_manager.py` — first preparatory edits for the tips.

---

# PHASE 1 — NOVELTY TIPS AND HONESTY FIXES (2026-05-20 → 2026-06-02)

_The entries below were written at the time and are unedited. They are listed in the order they were originally recorded, which is not strictly by date._

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

_End of the entries written at the time. Everything below was reconstructed on 2026-09-26 from
the commit history, `NOVELTY_TIPS.md` (all revisions), `TASK.md`, `WORK_REPORT.tex`, the
`final_report_data/` drafts and the result JSONs._

---

## ✅ Documentation written alongside Phase 1 (2026-05-20 → 2026-06-01)

**Files added / rewritten**
- `NOVELTY_TIPS.md` — rewritten in six later commits (`c7d3012`, `2b38904`, `69aadb0`, `6a2b4a3`, `50492f4`, `e1bb51a`). It began as "Novelty Tips" and became **"Open Issues — Must Fix Before Submission"**: a list, ordered by how fast each issue would end a defense, that was re-audited after every batch of fixes. Most Fix entries above answer one of its items.
- `defense_questions.md` (commit `6a2b4a3`, 2026-05-20; revised `50492f4`, `e1bb51a`) — 150 likely viva questions (Q1–Q150), grouped by topic and phrased against the *current* code. Many fixes cite the question they answer (Q30, Q35, Q50, Q144 …).
- `build_log.md` (commit `c7d3012`) — this file.
- `.gitignore` (commit `2b38904`) — stopped tracking `.claude/settings.local.json`.

---

## ✅ Fixes 30–36 — Second open-issues loop (2026-06-01, commit `e1bb51a` "some more issues resolved")

_Recorded retroactively. Numbered to follow Fix 29; the item numbers in brackets are those used in `NOVELTY_TIPS.md`._

**Files changed**
- `db_boa_framework/algorithms/db_boa.py` — **Fix 30 [#5]**: the module docstring gave the DB-BOA switching rule as `rand < |best|/|worst|`, but the code uses the range-normalised threshold `max(0, 1 − |f_max − f_min| / max(|f_min|, |f_max|, ε))`. The docstring now matches the code, with a note that the raw ratio breaks for negated objectives.
- `db_boa_framework/data/data_loader.py` — **Fix 31 [#6]**: `get_eval_subset` returned a 50/50 balanced sample, so the surrogate trained at ~50 % fraud, contradicting the claim that it matched the real 0.17 %. It now samples at the real rate. (BF-4 on 2026-06-02 then made the sampling genuinely stratified.)
- `db_boa_framework/algorithms/dboa.py` — **Fix 32 [#7]**: fragrance `g_j = d·J_j^b` returned NaN for negative fitness (a negative number raised to b = 0.1). Now uses `abs(J_j)`, as `db_boa.py` already did.
- `db_boa_framework/main.py` — **Fix 33 [#8, #3]**: Phase 8 (`--attack`) read `atk_fed_result['best_fitness']`, a key only the disabled Job-3 path returns, so it crashed with `KeyError`. It now prints the Shapley values, and every "DB-BOA Job 3" label in Phase 8 says "Shapley attribution". Phase 8 also moved from `X_test[:500]` to `X_val[:500]`, so the test set stays unseen.
- `db_boa_framework/utils/visualizer.py` — **Fix 34 [#9]**: `plot_convergence` drew *invented* convergence curves for MBO/WSA/DBOA/BOA-ADTCN (arithmetic offsets), and `plot_roc_curve` drew hard-coded AUCs for EfficientNet 0.94 / ResNet 0.97 / DenseNet 0.95 / DTCN 0.98. None had ever been run. Both fabricated series were removed.
- `db_boa_framework/utils/visualizer.py` — **Fix 35 [#10]**: the federation-weights plot was titled "DB-BOA Job 3"; it is now "Shapley-Weighted Federated Aggregation Weights per Round".
- `db_boa_framework/models/adtcn.py` — **Fix 36 [#11]**: once the eval subset kept the real fraud rate, it held only ~5 fraud rows, and `rng.choice(fraud_idx, 30, replace=False)` raised `ValueError`. It now samples with replacement when fewer than 30 are available. ⚠ **This is the line that three months later turned out to make the surrogate validate on memorised rows — see X-19.**
- `models/federated_adtcn.py`, `models/federation_manager.py`, `run_baselines.py`, `defense_questions.md` — small consistency edits.

---

# PHASE 2 — AUDIT, CHARACTERISATION EXPERIMENTS, REPORT REBUILD (2026-06-02 → 2026-06-11)

_This work was done in a separate Linux/WSL2 checkout and first committed to
`Undergraduate-Thesis-v2` (commits `c2c22f3` 2026-06-06, `0f02ce8` 2026-06-07, `cbe5035` /
`7d84bf4` / `81981f0` 2026-06-08, `94c071c` 2026-06-11; commit author `root`). It reached this
repository in one sync commit, `ccdd19f`, on 2026-08-30 (X-1)._

---

## ✅ J-1 — Report-vs-code divergence audit, D1–D17 (2026-06-02 → 06-06, first committed `c2c22f3`)

**Files added**
- `final_report_data/00_report_vs_code_divergences.md` — every claim in the LaTeX report that the code does not support: 🔴 D1 (the report's "primary novel contribution", DB-BOA Job 3, is dead code, and Shapley + Krum is the live path), D2 (the architecture is described as dilated TCN + 64-d embedding + softmax attention, but is a 2-layer 1-D CNN with global max-pool), D3 (FedProx claimed, never implemented), D4 (the 8-model comparison table was copied from the base paper), D5 (97.38 % / MCC 0.966 produced by no run), D6 (leader split 28/18/4 invented), D7 (85 TPS / 180 ms / "28.4 % cut" invented), D8 (paired t-tests over 3 seeds invented), D9 (token balances 458/312/178 invented); 🟠 D10–D14; 🟢 D15–D17.
- `final_report_data/00_ground_truth_implementation.md` — what the system actually is, as a single source of truth.
- `final_report_data/README.md`, `01_introduction.md`, `02_literature_review.md`, `03_requirements_impacts_constraints.md`, `04_methodology.md`, `05_results.md`, `06_conclusion.md`, `07_actions_checklist.md` — per-chapter correction notes and the to-do list.
- `final_report_data/REWRITE_00_title_abstract.md`, `REWRITE_01_introduction.md`, `REWRITE_02_literature_and_bib.md`, `REWRITE_03_requirements.md`, `REWRITE_05_methodology.md`, `REWRITE_06_results.md`, `REWRITE_09_conclusion.md` — paste-ready LaTeX drafts, under the rule *drafts first, `.tex` only after review*.

**Decision recorded: "Option A".** Re-frame the novelty around what is actually built (Shapley + Krum + DP on a live ledger), rather than switch the code back to DB-BOA Job 3. Job 3's saved fitness was the degenerate constant −1.0000000397e8, so it never optimised anything.

---

## ✅ J-2 — Planning and critique documents (2026-06-02 → 06-06)

**Files added**
- `novel_plan.md` — an execution prompt that fixes the novelty *type*: **"characterisation novelty, not new-algorithm novelty."** Research question: *when does a Shapley-weighted, blockchain-enforced incentive stay honest?* Task A (privacy ↔ incentive) is primary and Task B (economic Byzantine tolerance) secondary. It forbids any "first to bind Shapley to blockchain" claim (FedCoin 2020 has priority).
- `title_issue.md` — a claim-by-claim audit of the fixed title (Blockchain-Integrated, Incentivized, RL, Secure, Consensus, Scalable). It lands on: four claims supported once qualified, "Consensus Mechanisms" and "Scalable ML" overstated.
- `new_issues.md` (dated 2026-06-06; **removed** 2026-06-08 once its fixes were folded into the REWRITE drafts) — the "brutal research analysis". N1: the default pipeline (DP at ε = 1) is the worst row in its own ablation, *below* random. N2: incentive fidelity only recovers at ε ≈ 1000–3000, and ε\* is unstable (it had moved 1000 → 3000). N3: two of four novelty results rest on conditions we injected (`reliability=0.15`, `ORG_LABEL_NOISE`). N4: half the title is simulation. N5: there is no ML novelty.
- `DEMO_GUIDE.md` — supervisor demo walkthrough for the WSL2 machine (Docker 29 API fix, CCaaS chaincode, Fabric 2.5.10 / CA 1.5.13, symlinked dataset).

---

## ✅ J-3 — Task A: privacy ↔ incentive characterisation (2026-06-02 → 06-06)

**Files added**
- `db_boa_framework/experiments/privacy_incentive_sweep.py` — sweeps weight-channel DP ε and measures, per ε: global accuracy, Shapley fidelity (L1, cosine, Spearman ρ) against the ε = ∞ split, token error (tokens = 20 × Shapley weight), and the rank-inversion rate. It injects `ORG_LABEL_NOISE = {A: 0, B: 0.10, C: 0.25}` so that there is a non-degenerate honest ordering to invert (a constructed ground truth, disclosed).
- `results/privacy_incentive_sweep.json`, `results/privacy_incentive_tradeoff.png` (3-panel), `results/privacy_incentive_reward_bars.png`, `final_report_data/TASKA_privacy_incentive_results.md`.

**Result.** ε\* = 3000: rewards are mis-ordered at every budget up to 3000. Ground truth [0.619, 0.250, 0.131]. The global model sits near 50 % balanced accuracy until ε ≈ 300.

---

## ✅ J-4 — Task B: economic Byzantine tolerance (2026-06-02 → 06-06)

**Files added**
- `experiments/economic_byzantine_sweep.py` — always-fraud, label-flip and free-rider attackers (1 or 2 of 3 banks); reputation isolation driven by Shapley weight; accuracy with vs without the incentive coupling.
- `results/economic_byzantine_sweep.json`, `results/economic_isolation_trajectory.png`, `results/economic_accuracy_protection.png`, `final_report_data/TASKB_economic_byzantine_results.md`.

**Result (ULB).** A 2-of-3 colluding majority is caught: +40.78 pp (always-fraud), +79.52 pp (label-flip). A lone attacker and a free-rider are **not** isolated. Reported as a negative result.

---

## ✅ J-5 — Task C: scalable contribution attribution (2026-06-02 → 06-06)

**Files changed / added**
- `models/federation_manager.py` — `_build_coalition_value`, `_shapley_weights_exact` (all 2ⁿ − 1 coalitions), `_shapley_weights_mc` (Monte-Carlo permutation Shapley, TMC-style, Ghorbani & Zou 2019), `_normalise_shapley`, and a `shapley_method` dispatch.
- `config.py` — `make_org_splits(n)`, `shapley_method`, `shapley_mc_samples = 200`.
- `data/data_loader.py` — `split_for_orgs(..., samples_per_org)` equal-shard mode.
- `experiments/scalability_sweep.py` — n = 3 … 20 orgs, 8,000 rows per org, exact vs MC runtime and fidelity, equal-shard vs fixed-pool accuracy.
- `results/scalability_sweep.json`, `results/scalability_shapley_runtime.png`, `results/scalability_fidelity_accuracy.png`, `final_report_data/TASKC_scalability_results.md`.

**Result.** Exact Shapley takes ≈0.14 s at n = 3 and 113 s at n = 12 (4,095 coalitions); MC covers n = 20 in 95 s. MC gives almost no speed-up below n ≈ 10. MC top-1 agreement is 5/10 (later withdrawn as a threshold, see X-12).

---

## ✅ J-6 — Task D: Krum where its theorem holds (2026-06-02 → 06-06)

**Files added**
- `experiments/byzantine_robustness_sweep.py` — n = 5 / f = 1 and n = 7 / f = 2 (so n ≥ 2f + 3 holds); weight-level attacks: sign-flip, scaled (λ = 50), Gaussian, and a retrained label-flip model; Krum vs unprotected FedAvg.
- `results/byzantine_robustness_sweep.json`, `results/byzantine_robustness_krum_vs_fedavg.png`, `final_report_data/TASKD_byzantine_robustness_results.md`.

**Result (ULB).** The attacker was rejected 8/8, with the Krum model at ≈99.9 % balanced accuracy. FedAvg collapses only under the scaled attack (87.46 %). This closes the "Secure" title word in code, not just in wording. ⚠ The 8/8 was **later broken on the Handbook** (X-36) and shown to be hollow on AMLSim (X-38).

---

## ✅ J-7 — Balanced-accuracy coalition score for Shapley (2026-06-02 → 06-06)

**Files changed**
- `utils/metrics.py` — new `coalition_score()` = (Sensitivity + Specificity) / 2. Shapley coalition values no longer use Obf2, whose unbounded 1/FPR term blew up to ~1e8 and made aggregation weights meaningless.

---

## ✅ J-8 — Reinforcement-learning leader selection, to honour the fixed title (2026-06-02 → 06-06)

**Files added / changed**
- `db_boa_framework/blockchain/rl_leader.py` — `RLLeaderSelector`: linear function-approximation Q-learning (ε-greedy, γ = 0.9). State = node CT/CC/MS + reputation + tokens + load + fail-rate; action = elected leader; reward = the on-chain token payout itself.
- `blockchain/leader_block.py` — `attach_rl_agent`, `select_leader_rl`, `run_rl_round`, `_gini`, `_run_method`, `compare_leader_methods`, and a `reliability` fault hook (default 1.0, a no-op).
- `config.py` — `RL_LEADER_CONFIG`; `LEADER_BLOCK_CONFIG["leader_method"] = "rl"` (DB-BOA keeps the Phase-1 cold-start pick).
- `experiments/rl_leader_sweep.py`, `results/rl_leader_sweep.json`, `results/rl_leader_reward_fairness.png`, `results/rl_leader_adaptivity.png`.
- `db_boa_framework/final_report_data/08_rl_leader_selection.md` — the honest write-up.

**Result (40 rounds × 5 seeds).** In the stationary case RL ties DB-BOA on reward, and leadership Gini falls from 0.90 to 0.78. In the non-stationary case (a node's reliability collapses mid-run), DB-BOA keeps re-electing the bad node 100 % of the time and RL about 15 %. ⚠ That win depends on the injected fault, and is disclosed as conditional.

---

## ✅ J-9 — Bounded DB-BOA objective (first committed 2026-06-06, `c2c22f3`)

**Files changed**
- `models/adtcn.py` — the surrogate fitness became **Obf2 = 2·MCC + Specificity + Precision + NPV**, replacing Eq. 11's unbounded `Acc + Pre + NPV + MCC + 1/FPR`. The maximum is 5.0.
- ⚠ `utils/metrics.py::obf2_value` kept the old unbounded formula until OBJ-10 (X-5).

---

## ✅ J-10 — B1: output-channel DP, the central contribution (2026-06-06/07, commit `0f02ce8` "now title sticks word to word")

**Files added / changed**
- `models/federation_manager.py` — `_privatise_incentive()`: output perturbation (after Chaudhuri 2011) on the 3-dimensional contribution vector φ, clipped at C = ‖φ‖₂, instead of noising the ~111,874-dimensional weight vector.
- `config.py` — weight-channel `use_dp` **default changed True → False** (it becomes a swept knob); new `use_private_incentive`.
- `experiments/private_incentive_sweep.py` — head-to-head weight channel vs output channel over the same paired noise draws (`np.random.seed(3000 + rep)`).
- `results/private_incentive_sweep.json`, `results/private_incentive_channel.png`, `final_report_data/TASKB1_private_incentive_results.md`.

**Result (June version).** ε\* 3000 (weight) → **50 (output)**; at ε = 50, Spearman ρ −0.288 → **+0.950** over 100 draws. The privacy ↔ incentive failure is a property of the *channel*, not of DP-plus-Shapley in principle. ⚠ The "60×" factor was **retired** as a headline on 2026-09-04 (X-17).

---

## ✅ J-11 — B2: real Hyperledger Fabric measurement (2026-06-06/07, `0f02ce8`)

**Files added / changed**
- `db_boa_fabric/api-server/measure_consensus.js` — times `recordConsensusRound` against the **live** test network through the fabric-network gateway: 50 sequential rounds, plus a concurrency sweep of `updateNodeMetrics` over 10 conflict-free keys.
- `db_boa_framework/results/fabric_consensus_measured.json`.
- `experiments/write_b2_draft.py` → `final_report_data/B2_fabric_consensus_measured.md`.
- `blockchain/leader_block.py` — `load_measured_consensus()`; `config.py` — `use_measured_consensus`, `measured_consensus_file`.
- `db_boa_fabric/api-server/wallet/admin.id`, `appUser.id` — re-enrolled identities.

**Result.** Mean latency **2115.9 ms** (p95 2172, n = 50), which is the 2 s Raft `BatchTimeout` floor. Peak **40.3 TPS at concurrency 10**, then an MVCC knee (50 % fail at c = 20, 70 % at c = 40). These replace the invented 85 TPS / 180 ms (D7).

---

## ✅ J-12 — B3: temporal pipeline and report-faithful architecture (2026-06-06/07, `0f02ce8`)

**Files added / changed**
- `models/adtcn.py` — `_DilatedBlock`, `_DilatedAttnClassifier`, `make_temporal_model(architecture=…)`: the dilated-conv + attention ADTCN that the report described, now actually implemented.
- `experiments/temporal_pipeline_ablation.py` → `results/temporal_pipeline_ablation.json`, `final_report_data/TASKB3_temporal_pipeline.md`.
- `experiments/architecture_ablation.py`.

**Result.** CNN on random order 0.770; CNN on time order 0.474; dilated + attention on random order 0.714; on time order 0.459. **Temporal complexity did not pay on ULB.** The CNN stays deployed. (On 2026-08-31 BankSim showed this was about entity-less windows, not about time. See X-6.)

---

## ✅ J-13 — Federated ablation on ULB and the report merge (2026-06-08, commits `cbe5035`, `7d84bf4`, `81981f0` "report done")

**Files added / changed**
- `run_baselines.py` — writes `results/baselines.json` (FedAvg / +Krum / +DP / proposed, ULB test set). Result: FedAvg MCC 0.569, +Krum **0.776**, +DP(ε = 1) 0.000, proposed ≈ 0 (collapse).
- `FINAL YEAR THESIS REPORT/` — REWRITE drafts merged into `chapter_1/2/3/5/6/9.tex`, `core/abstract.tex`, `core/approval.tex`, `core/acknowledgement.tex`; `references.bib` grew to 71 entries; `main.pdf` compiled. LaTeX build artefacts (`main.aux/.bbl/.bcf/.log/...`) were committed once and then removed again.
- `final_report_data/REWRITE_08_limitations_disclosures.md` (threat model plus limitations), `02_literature_review_DRAFT.md`.
- `finalreport_checklist.md` — progress tracker for the draft → `.tex` merge.
- Removed: `new_issues.md`, the empty placeholders `NOVELTY_TIPS.md` / `issue_fixer.md` in v2, `hello-world/` in v2, and `*:Zone.Identifier` files. `.gitignore` extended.

---

## ✅ J-14 — DB-BOA vs default, figures, supervisor brief (2026-06-10 → 06-11, commit `94c071c`)

**Files added / changed**
- `experiments/dbboa_vs_default.py` → `results/dbboa_vs_default.json` — paired retraining of the default (128 filters / 150 steps) against the tuned configuration (142 / 76). ⚠ Its "tuned MCC 0.313, gap 0.47" was **withdrawn 2026-08-31** (X-3).
- `experiments/plot_federated_confusion.py` → `results/confusion_matrix_federated.png` — a pure visualisation of the saved counts; no re-run.
- `main.py` — Phase 4's default-vs-tuned comparison is now saved into the results JSON.
- `SUPERVISOR_BRIEF.md`, `SUPERVISOR_BRIEF.tex`, `SUPERVISOR_BRIEF.pdf` — the one-read walkthrough for the supervisor.
- `FINAL YEAR THESIS REPORT/core/nomenclature.tex`, `.latexmkrc`, `main.ilg`, `main.nls`; `images/methodology_overview.svg` + `.png`; `images/fabric_live_capture.png`; `images/ulb_class_distribution.png`; chapter restructure (Implementation moved to Chapter 5, a Comparison with Prior Work section, Discussion, Limitations and Future Work).
- `example_thesis/T2430427_P3 (1) (1).pdf`, `example_thesis/T2430427_P3_Slides (1).pptx` — an approved BRAC thesis used as the rubric benchmark.
- `finalreport_checklist.md` §§ 6–8.

---

## ✅ J-15 — Final audit fix batch (2026-06-11)

**Files added / changed**
- `final_report_data/REWRITE_FIX_2026-06-11.md` — changelog of 10 substantive report errors found in a full audit and fixed the same day. The DB-BOA table became three rows. The "3-org Fabric deployment" was rewritten as the real 2-org single-host CCaaS network. Features 29/30 → **33** (30 base + 3 recurrence). AdamW / weight decay / gradient clipping / LR scheduler → plain Adam lr 1e-3. The Threshold-Optimisation section was deleted (the code uses argmax). Rounds became 5 consensus / 3 federation / 15 attack. The statistical summary was corrected against the raw CSV. The DB-BOA maths now matches the implementation.
- `FINAL YEAR THESIS REPORT/chapters/chapter_7.tex` — deleted (empty, not included).

**2026-06-13 — the thesis was accepted.**

---

# PHASE 3 — PAPER EXTENSION BEGINS (2026-08-30 → 2026-08-31)

## ✅ X-1 — Repository sync and hygiene (2026-08-30, commit `ccdd19f`)

- The local checkout was a stale 2026-06-02 tree, missing every experiment the accepted report describes. A local commit `69ef481` ("refreshing") had tracked `creditcard.csv` (143.8 MB, over GitHub's 100 MB blob limit) and 17 `.pyc` files. It was dropped with `git reset --soft`, and the files were untracked.
- Branch `backup-before-v2-sync` preserves `69ef481`. A read-only `v2` remote was added.
- The tree was synced to v2's tip `94c071c` (all of Phase 2) and committed as `ccdd19f`, "Sync repository to the state described by the final thesis report". It also brought in 21 more reference PDFs in `all papers/`, the `FINAL YEAR THESIS REPORT/images/` results figures, `final_report_data/`, `reports/P2_REPORT_T2430460.pdf`, `reports/P2_POSTER_T2430460.pdf`, `reports/report.txt`, and the `fabric/fabric-samples/` test network.
- `.gitignore` (32 lines) — datasets, `__pycache__/`, `*.pyc`, LaTeX build files.
- `origin/main` still points at this commit.

## ✅ X-2 — The mission board and the dataset plan (2026-08-30 → 08-31)

- `TASK.md` created — "Operation: Honest Federation". Rules of engagement (1 novel **and** completely honest; 2 no number a script did not produce; 3 negative results stay negative; 4 no result-shopping; 5 the repo is the truth; 6 scope every claim; 7 no invention claims; 8 a dataset must be able to overturn a conclusion), plus objectives OBJ-1 … OBJ-12, an INTEL table of verified numbers, and a FIELD MAP. First committed in `ac606de` (2026-09-05).
- Plan agreed: add entity-linked datasets beside ULB, **BankSim first**, to test whether "CNN beats ADTCN" is a ULB artefact.

## ✅ X-3 — OBJ-2 / OBJ-4: the MCC contradiction and multi-seeding the detector (2026-08-31)

**Files added / changed**
- `experiments/detector_multiseed.py` — 5 seeds × 2 configurations × 30 epochs, a thread-count determinism probe, and a reproduction of the historical runs; restartable, one JSON per run in `results/_multiseed_runs/` (`_determinism.json`, `dbboa_tuned_seed{42..46}_t2.json`, `dbboa_tuned_seed42_t4/t8.json`, `hand_set_default_seed{42..46}_t2.json`, `hand_set_default_seed42_t8.json`).
- `experiments/check_thread_sensitivity.py`.
- `results/detector_multiseed.json`, `final_report_data/OBJ2_detector_multiseed.md`.
- `requirements.txt` — **torch pinned at 2.12.0** (it had been "optional, unpinned").
- Report: Table 6.4, `sec:centralised`, `sec:dbboa-hpo`, the abstract, Chapters 5 and 9 re-based.

**Result.** Training is bitwise deterministic for a fixed *(seed, torch version, thread count)*. `main.py` used 4 threads and `dbboa_vs_default.py` forced 8. The tuned configuration scores 0.708 / 0.746 / 0.800 at 4 / 2 / 8 threads, while the default is bitwise invariant. Over 5 seeds: tuned **0.753 ± 0.055**, default **0.706 ± 0.076**, difference **+0.047, p = 0.34**. **MCC 0.313 withdrawn** (the lowest in 7 runs is 0.6598).

## ✅ X-4 — OBJ-3: measured Fabric numbers into the report (2026-08-31)

- `FINAL YEAR THESIS REPORT/chapters/chapter_6.tex` — new `sec:fabric-measured` section. Six stale "throughput is simulated" statements in Chapters 5, 6 and 9 were corrected. `B2_fabric_consensus_measured.md` was stamped CONSUMED.
- New finding: the **+15-token bonus for latency < 300 ms can never be earned** on a 2.1 s floor.

## ✅ X-5 — OBJ-10: one objective definition (2026-08-31)

- `utils/metrics.py::obf2_value` now holds the bounded `2·MCC + Spec + Pre + NPV`, and `models/adtcn.py` calls it instead of keeping its own copy. Checked bit-identical over 2,000 random metric dicts.

## ✅ X-6 — OBJ-1: BankSim loader and the entity-linked window grid (2026-08-31)

**Files added / changed**
- `data/banksim_loader.py` — BankSim (`datasets/bs140513_032310.csv`, gitignored). `order_rows` breaks same-`step` ties with a **seeded permutation, never file order**. `split_for_orgs(partition="customer"|"stratified")`; the customer split shares 0 customers across banks (2,055 / 1,233 / 823).
- `config.py` — `BANKSIM_CONFIG`, `DATASETS`, `get_loader()`.
- `models/adtcn.py` — `build_sequences(groups=…)`, which builds entity-linked windows.
- `data/data_loader.py` — `raw_feature_count`.
- `main.py`, `run_baselines.py` — `--dataset`, `--partition`.
- `experiments/__init__.py`, `experiments/_seqtrain.py` (shared sequence-training helper), `experiments/banksim_temporal_grid.py` (7 architectures × {global, customer} windows × 3 seeds; checkpoints after each cell).
- `results/banksim_temporal_grid.json`, `final_report_data/OBJ1_banksim_temporal_grid.md`.

**Trap disarmed.** BankSim's raw file places frauds next to each other within a step: 3,635 adjacent fraud pairs, 84 after a within-step shuffle. File-order windows would have given the control arm P(fraud | prev fraud) = 0.50. With the seeded tie-break it is 0.0117 against a 0.0121 base rate.

**Result (5.3 h).** Linked windows help **every** architecture (+0.0999 to +0.1717 MCC). Yet ADTCN ranks **7th of 7** on linked windows (0.7089), behind its own no-attention ablation DTCN (0.7733). Best linked: LSTM 0.7805. No-window reference: 0.6257.

## ✅ X-7 — OBJ-1b: re-implement the base paper's comparison (2026-08-31)

**Files added**
- `models/basepaper_models.py` — EfficientNet-1D, ResNet-1D, DenseNet-1D, DTCN (= ADTCN minus attention), LSTM, for 1-D transaction windows.
- `algorithms/mbo.py` (Mine Blast, Sadollah 2013), `algorithms/wsa.py` (Water Strider, Kaveh 2020). All five optimisers were checked to converge on a sphere function.
- `experiments/basepaper_comparison.py` — classifier track and optimiser track; checkpoints each cell (temp file + rename).
- `results/basepaper_comparison_ulb.json`, `results/basepaper_optimisers_ulb.json`, `results/basepaper_optimisers_banksim.json`; `final_report_data/BASEPAPER_comparison_ulb.md`, `BASEPAPER_optimisers_ulb.md`, `BASEPAPER_optimisers_banksim.md`.

**Result.** ULB classifiers: ResNet-1D 0.5995 > CNN 0.4974 > DenseNet > LSTM > **ADTCN 0.3296 (5th of 7)** > EfficientNet > DTCN. The optimisers on BankSim are statistically indistinguishable. On ULB **all five hit Obf2 = 5.0000 exactly**, which opened OBJ-13. No number from the base paper is reproduced; their results are MATLAB runs on private data.

**Trap found.** With stdout redirected to a file on Windows, `─` separators crash cp1252. Always export `PYTHONIOENCODING=utf-8`.

## ✅ X-8 — Federated ablation on BankSim, both partitions (2026-08-31)

- `run_baselines.py --dataset banksim --partition {stratified,customer}` → `results/baselines_banksim_stratified.json`, `results/baselines_banksim_customer.json` (search skipped; pinned 32 / 150).
- `experiments/federated_cross_dataset.py` → `results/federated_cross_dataset.json`, `final_report_data/FEDERATED_cross_dataset.md`.

**Result.** The DP collapse **replicates** (MCC 0.018 / −0.000), and its **direction flips**. On ULB the DP model predicts no fraud and accuracy *rises*; on BankSim it predicts everything (119,784 FP) and accuracy falls 97.55 pp. Krum's cost is roughly constant (−0.037 / −0.033).

## ✅ X-9 — OBJ-13 opened: does the DB-BOA search select on noise? (2026-08-31)

- `experiments/objective_noise_audit.py` → `results/objective_noise_audit.json`, `final_report_data/OBJECTIVE_noise_audit.md`. A fixed configuration re-evaluated 25× on ULB spans Obf2 3.4497–5.0000, and noise ÷ signal is 0.74 (ULB) / 0.94 (BankSim). The fitness redrew its split and its torch seed on every call, so the search could not rank its own candidates.

---

# PHASE 4 — TAKING THE SYSTEM CLAIMS OFF ONE DATASET (2026-09-01)

## ✅ X-10 — Direction review, rev. A → rev. B (2026-09-01)

- `TASK.md` — rev. A judged the thesis by ADTCN's rank among classifiers and proposed a two-claim reframe. That was **withdrawn** once the team lead pointed out that the unit is FL-ADTCN, a system of six components making six title claims. Rev. B found the real gap: **five of six title claims rested on ULB alone**, because the four system sweeps hard-coded the ULB loader. **Rule 9** added ("judge the system, not a component"). Rule 10 (Krum = security, Shapley = fairness) and Rule 11 (name the metric you quote, first enforced in OBJ-14 the same day) date from the same days.

## ✅ X-11 — OBJ-14: retracted numbers purged from live drafts (2026-09-01, ~40 min, no compute)

- `experiments/objective_noise_audit.py --render-only` — rebuilds the draft from JSON in seconds. The generator was the bug: it still printed 0.313 and "loses to the default (0.785)", and quoted one noise ratio without naming it. The draft now prints both ratios with their names.
- 8 historical files stamped `SUPERSEDED 2026-08-31 by OBJ-2`, **unedited**: `01_introduction.md`, `04_methodology.md`, `05_results.md`, `06_conclusion.md`, `REWRITE_06_results.md`, `REWRITE_09_conclusion.md`, `REWRITE_FIX_2026-06-11.md`, `finalreport_checklist.md`.
- Three breaches outside the stated scope were fixed: `SUPERVISOR_BRIEF.md` / `.tex` (the retracted 0.785 stated as a live claim) and the `experiments/_seqtrain.py` docstring.

## ✅ X-12 — OBJ-15: four system sweeps made dataset-aware and run on BankSim/customer (2026-09-01, 3 h 15 m)

**Files added / changed**
- `experiments/_dataset.py` — `add_dataset_args`, `resolve`, `apply_to_model_cfg` (forwards `raw_feature_count`, so BankSim's 79 features are not silently cut to 33), `provenance`, `suffix`, `redraft`.
- `byzantine_robustness_sweep.py`, `economic_byzantine_sweep.py`, `private_incentive_sweep.py`, `scalability_sweep.py` — `--dataset {ulb,banksim} [--partition …]`, `--redraft`; every JSON records `dataset` / `partition` / `raw_features`; filenames suffixed (ULB keeps its bare name).
- **Two latent bugs fixed before the overnight run.** (1) `write_report()` used a bare `open(md, "w")`, which died on `→` under cp1252 even with `PYTHONIOENCODING` set; `encoding="utf-8"` was added to **all 15 scripts** in `experiments/`. (2) Fixed draft names would have overwritten the ULB drafts.
- **Plumbing holes found from sweep 1's output.** Figures were never suffixed, so the BankSim run overwrote two ULB figures; they were recovered byte-identical from `ccdd19f`, and the other six were backed up to `results/_ulb_figures_backup/`. The drafts also asserted ULB conclusions ("on this 0.17 %-fraud data") beside tables showing otherwise; all four generators now compute that prose from the run.
- `experiments/run_obj15_banksim.ps1` (use this), `run_obj15_banksim.sh` (fails here: no `python` on Git Bash's PATH), `experiments/check_obj15.ps1`.
- Outputs: `*_banksim_customer.{json,png}` for all four sweeps, and `TASKB_…`, `TASKB1_…`, `TASKC_…`, `TASKD_…_banksim_customer.md`.

**Pre-registered at 13:05, before any output; scored 16:20.** Krum 8/8 held. "Lone attacker not caught" did not replicate as worded (isolation fires 3/3 but helps 0/3). The ε\* ordering held, with the factor moving 60× → 10×, and both factors censored. "MC top-1 fails from n ≈ 6" was **withdrawn** (the failures are intermittent).

## ✅ X-13 — The confound control launched: BankSim/stratified (2026-09-01 20:20 → 23:41)

- The same four sweeps with `--partition stratified`, so that *dataset* and *partition* stop moving together. Logs split per run: `results/_obj15_logs/banksim_customer/`, `results/_obj15_logs/banksim_stratified/`.
- `experiments/sweeps_cross_condition.py` — reads every condition's JSON and prints the **dataset effect** and the **partition effect** as separate columns, naming the metric on each row. It reproduces every INTEL number.
- `db_boa_framework/scratchpad/resume_new.md` — session handoff written while the control ran.
- Six expectations pre-registered at 20:20.

---

# PHASE 5 — CONTROLS, PRIVACY BUDGET, SURROGATE REPAIR (2026-09-04 → 2026-09-05)

## ✅ X-14 — The control scored: the best new claim dies (2026-09-04)

- `final_report_data/OBJ15_two_factor_decomposition.md` (generated).
- **2 of 6 pre-registrations wrong, both the same way.** Krum is negative in **7 of 8** cells under the *stratified* partition too. The dataset effect is negative in all 8 rows (−1.06 to −17.27 pp); at n = 7 the partition effect is *positive* in all four. **"Krum pays 1.5–12.6 pp under entity-disjointness" is withdrawn**, and no replacement mechanism is offered. Lone-attacker isolation also tracks the dataset (0/3 ULB, 3/3 on both BankSim splits). MC rank fidelity ρ at n = 12 moves +0.783 with the dataset.
- `TASK.md` — **Rule 12** added: "A control that can overturn a finding you already have outranks a dataset that can add a new one." The Locked Plan (consolidate first, protect the Sep 16–19 writing block, Handbook go/no-go on Sep 12) was agreed with the team lead.

## ✅ X-15 — OBJ-18: propagate the withdrawal; map the contaminated cells (2026-09-04, ~2 h, no CPU)

- `byzantine_robustness_sweep.py` (≈line 374) and `federated_cross_dataset.py` (≈line 193): the generators still emitted the withdrawn mechanism, and "value shows up under attack". They now state the measurement and say no mechanism is established. Lesson recorded: *a generator may only assert what its own input can falsify.*
- `SUPERVISOR_BRIEF.md` / `.tex` / `.pdf` — "~60×" → "≥60×" with a censoring box; Krum's *rejection* split from Krum's *utility*; "single-source data" rewritten. The PDF had been unbuildable since 8/30: MiKTeX lacked scalable Type 1 CM fonts, so `microtype` aborted. Fixed with `miktex packages install cm-super`, and the PDF was rebuilt (9 pp).
- `sweeps_cross_condition.py` — marks **contaminated cells** `(!)` (attacked FedAvg more than 0.5 pp above its own no-attack reference: 0/8 ULB, 2/8 BankSim/strat, 5/8 BankSim/customer) and prints a clean-subset table. The quoted "−12.57 to −1.48 pp" was carried by contaminated cells; the clean range is −4.73 to −1.48. The withdrawal survives on the three cells that are clean in all conditions.

## ✅ X-16 — OBJ-17 part 1: de-censor ε\* on an extended grid (2026-09-04, ~78 min)

- `results/_obj17_pre_extension/` + `README.md` — snapshot of the 9-point-grid JSONs and figures, the only evidence for the factors as published.
- `private_incentive_sweep.py` — `--eps-grid` (the 11-point `DEFAULT_EPS_GRID` up to ε = 30000), inversions reported as *draws*, and a **fragility table** (exact Clopper–Pearson intervals; a threshold decided by ≤ 5 draws is flagged "do not quote").
- `experiments/run_obj17_epsgrid.ps1`; logs in `results/_obj17_logs/grid/`.
- `environment()` stamping (torch / numpy / python / threads) added to every sweep JSON.
- `experiments/check_obj17_nesting.py`; `results/private_incentive_sweep_rep2.json` + `.png` + `TASKB1_…_rep2.md` (same-day ULB re-run).
- **Result.** All three conditions de-censored: 60× (ULB), 30× (BankSim/strat), 33× (BankSim/customer). ⛔ **ULB failed reproduction.** Re-running the 30 August budgets at identical seeds moved 13 of 18 cells, and the ground truth moved [0.619, 0.250, 0.131] → [0.554, 0.336, 0.110]. Both BankSim conditions reproduced bitwise. **Recorded as unexplained.**

## ✅ X-17 — OBJ-17 part 2: the `--repeats 1000` pass; the factor retired (2026-09-04, landed 17:53)

- `results/private_incentive_sweep_r1000.json`, `…_banksim_stratified_r1000.json`, `…_banksim_customer_r1000.json`, their three `.png` files, the three `TASKB1_…_r1000.md` drafts, and logs in `results/_obj17_logs/r1000/`.
- **Result.** The single-draw thresholds were real (7, 11, 16 and 10 per 1000). But BankSim/customer re-censored to **≥ 100×** on 1 draw in 1000. ε\* = max{ε : rate > 0} can only grow with more draws or more grid points. Re-derived at rate cut-offs 0 / 0.01 / 0.05, the factor swings **20× to ≥ 100×**, and the ranking of conditions changes. **The factor is retired as a headline.** The claim now rests on the paired **ordering** and the **ρ gap at ε = 50** (ULB −0.199 → +0.995; BankSim/strat +0.017 → +0.945; BankSim/customer −0.070 → +0.805). The supervisor brief was rebuilt.

## ✅ X-18 — OBJ-13 part 1: the surrogate repair applied (2026-09-04)

**Files added / changed**
- `db_boa_framework/scratchpad/obj13_new_objective.py` (the staged objective, rescued from a session temp directory), `scratchpad/apply_obj13.py` (idempotent applier; `--check` writes nothing), `scratchpad/verify_obj13_legacy.py`.
- `models/adtcn.py.pre_obj13`, `config.py.pre_obj13` — ⛔ **the only copy of the pre-patch state** (`adtcn.py` had uncommitted edits). Never delete them or gitignore them.
- `models/adtcn.py`, `config.py` — `eval_mode ∈ {legacy, deterministic, averaged}`. `deterministic` freezes the draw; `averaged` scores every candidate on k = 3 **shared** stratified draws (common random numbers). Surrogate batch = `ceil(n_train / spe)`, replacing `max(32, n_train // spe)`, which had pinned every candidate to 43 steps per epoch — a dead search axis.
- **Three defects caught before they shipped.** (1) The patch silently re-pointed `objective_noise_audit.py` at the new default, where it would have reported within-seed std 0.0000; it now pins `eval_mode="legacy"`. (2) The same silent switch in `basepaper_comparison.py`; it gained `--eval-mode`, stamps the mode into its JSON, and refuses to overwrite a file recorded under another mode. (3) `write_draft` named drafts by dataset, so the classifier and optimiser tracks overwrote each other; they are now named by the JSON.
- **Verified:** `legacy` is bit-for-bit identical to `adtcn.py.pre_obj13`, agreeing to 10 decimals on 3 configurations.

## ✅ X-19 — OBJ-13 part 2: the shipped-path leak (2026-09-04)

- `experiments/obj13_shipped_path_check.py` → `results/obj13_shipped_path_check.json` (zero training). The deployed ULB surrogate got a 3,000-row pool holding **5 unique fraud transactions**. Fix 36's replacement branch repeated them 6.01×, and **9 of 9 validation positives were copies of training rows**. `Obf2 = 5.0000` was memorisation. The repair did **not** fix it (`deterministic` leaked 9/9 too).
- **Fix:** `eval_subset` 3,000 → **36,000** in `DATA_CONFIG` and `BANKSIM_CONFIG`, plus a `RuntimeWarning` guard (`fraud_sampled_with_replacement`, `unique_fraud_available`). Re-verified **0 of 11** leaked. Logs: `results/_obj13_logs/shipped_path.log`, `shipped_path_after_fix.log`.

## ✅ X-20 — OBJ-13 part 3: surrogate-size knee sweep (2026-09-04, landed 20:38)

- `objective_noise_audit.py --knee` / `--knee-redraft` / `--knee-seeds` / `--components`, and a `null_range_test` → `results/objective_size_knee.json`, `final_report_data/OBJECTIVE_size_knee.md`, `results/_obj13_logs/knee.log`.
- **Pre-registration 6 wrong on both halves.** ULB improved sharply while pinned at 30 fraud rows; BankSim did not improve with 8× the fraud rows. **The "n ≈ 9 positives" mechanism was withdrawn.** No knee is locatable at one seed, so do not quote 10,000 rows.

## ✅ X-21 — OBJ-13 part 4: the 16.3-hour side-by-side (2026-09-04 20:54 → 2026-09-05 13:13)

- `experiments/obj13_surrogate_repair.py` (`--scout`, `--redraft`, held-out yardstick on a row-disjoint pool, paired comparison) and `experiments/run_obj13_sidebyside.ps1` (detached, preflights the patch *and* the pool fix).
- `results/obj13_surrogate_repair_banksim.json`, `final_report_data/OBJ13_surrogate_repair_banksim.md`, `results/_multiseed_runs/extra_configs.json` (repaired arms for `detector_multiseed.py`), logs `results/_obj13_logs/scout_banksim.log`, `sidebyside_banksim_production*.log`.
- **Result (BankSim, pop 20 × 30, 3 modes × 3 seeds).** **0 of 9 searched configurations beat a hand-set 128 / 150**; mean paired Δ −0.1982, and the shipped 142/76 also loses (−0.1900). Common random numbers, not determinism, make the search reproducible (filter spread: legacy 66, deterministic **81**, averaged **12**). `averaged` generalises *worst* (3.4237), because the removed noise had been acting as regularisation. The repaired `spe` axis is alive but coarse (23 batch sizes across 201 values). ⛔ `ADTCN.fit:681` still uses the floored formula, so the dead axis was surrogate-only. **Still undecided.**

## ✅ X-22 — Everything committed; the first work report (2026-09-05)

- Commit `889d623` — about 20 h of Aug 31 – Sep 4 results that had existed only as untracked files (OBJ-1b, OBJ-15, OBJ-17, OBJ-18).
- Commit `ac606de` — OBJ-13, plus `TASK.md` (first committed, 2,870 lines).
- `.gitignore` — the LaTeX rule `*.log` had been swallowing **experiment run logs**; `!db_boa_framework/results/**/*.log` excepts them and recovered the OBJ-15 / OBJ-17 logs.
- Commit `33bb97f` "new datasets" — `WORK_REPORT.tex` + `WORK_REPORT.pdf`, the first version of the 30 Aug → 5 Sep work report for the supervisor.

---

# PHASE 6 — THIRD-DATASET LOADER AND FIGURES (2026-09-08 → 2026-09-10)

## ✅ X-23 — OBJ-5 / OBJ-16: the Fraud Detection Handbook loader (2026-09-08)

- `data/handbook_loader.py` — reads `datasets/handbook_raw/` (a clone of `Fraud-Detection-Handbook/simulated-data-raw`, 183 daily pickles, gitignored) and consolidates it into `datasets/handbook_transactions.csv` (105 MB, gitignored). 35 features (log/amount, hour and day-of-week one-hots, weekend, night); **deliberately no entity-derived aggregates**. Three orderings (global / customer / terminal) and three partitions.
- `config.py` — `HANDBOOK_CONFIG`, `DATASETS["handbook"]`, `ENTITY_PARTITIONS` (replaces `_dataset.resolve`'s `dataset != "banksim"` guard).
- `experiments/check_handbook_loader.py` — **42 acceptance checks, all passing**, no training.
- `db_boa_framework/scratchpad/verify_handbook_no_regression.py` — ULB and BankSim unchanged.
- **Measured:** 1,754,155 tx · 4,990 customers · 10,000 terminals · 183 days · 0.837 % fraud · 1-second resolution. The file-order trap does **not** fire (lift 1.02×). **Terminal linkage 71.65×** against customer 13.35×. Cost ~11 h, not the budgeted ~6 h.
- Housekeeping: `datasets/bsNET140513_032310.csv` was verified to be a byte-identical column projection of BankSim, **not a dataset**.

## ✅ X-24 — Report figures from stored results (2026-09-10)

- `experiments/make_report_figures.py` — draws every WORK_REPORT figure from the result JSONs only (no training, no sampling) and prints each panel's provenance → `results/figures/fig_architecture_scoreboard.png`, `fig_cross_dataset_comparison.png`, `fig_memorisation_before_after.png`, `fig_process_optimisation.png`, `fig_scalability.png`, `fig_score_index.png`, `fig_statistical_significance.png`, `fig_system_performance.png`, `score_index.csv`, `score_index_table.tex`.

## ✅ X-25 — Work report rebuilt (2026-09-10, commit `6d8e41e` "--")

- `WORK_REPORT.tex` / `.pdf` rebuilt as a story (where we stood → the rules → the work in order → the decision log → the six claims → what may be quoted). Also: `TASK.md`, `config.py`, `_dataset.py`, `.gitignore`.

---

# PHASE 7 — ALL FIVE DATASETS TAKEN TO MEASUREMENT (2026-09-11 → 2026-09-13)

## ✅ X-26 — Operator decisions: Rule 8 suspended; placement (2026-09-11)

- `TASK.md` — Rule 8 marked **SUSPENDED** (original text kept). All five datasets are run and compared. ULB and the Handbook run on the laptop (the same environment as BankSim); PaySim and AMLSim run on Kaggle CPU sessions. AMLSim means the **real IBM simulator built with Java**, not the pre-generated AMLworld data. Every pre-registration is drafted by the assistant and approved by the team lead before any run.

## ✅ X-27 — PaySim: reconnaissance, loader, pre-registration (2026-09-11)

- `experiments/paysim_recon.py` → `results/paysim_recon.json` (357 s). 6,362,620 tx, 0.129 % fraud, fraud only in TRANSFER / CASH_OUT; 99.7 % of senders appear once. ⛔ **The time order carries the label**: P(fraud | prev fraud) is 0.72 raw and still 0.44 after the within-hour shuffle (151.6× on the chosen scope).
- `data/paysim_loader.py` — scope TRANSFER + CASH_OUT (2,770,409 rows, all 8,213 frauds), with an all-rows sensitivity dataset `paysim_all`; row fields plus raw balances only; stratified only (no banks).
- `experiments/check_paysim_loader.py`.
- Pre-registration (a)–(e) approved as written.

## ✅ X-28 — IBM AMLSim: generate, discard, regenerate (2026-09-11)

- `data/make_amlsim_config.py` — the shipped `paramFiles/10K`, with accounts split across `bank_a` / `bank_b` / `bank_c` (50 / 30 / 20), plus three forced key repairs copied from upstream's root `conf.json`.
- `data/generate_amlsim.py` — end-to-end pipeline: IBM/AMLSim `7338a4bc`, MASON 20 built from source (tag `v20`, SHA-256 recorded), Maven 3.9.9, JBR 21, Python 3.8.20 + networkx 1.11 in `datasets/amlsim/py38`. It repairs upstream's Windows `\r\r\n` line endings and asserts that only CR bytes were removed. SHA-256 for every output.
- `data/amlsim_loader.py`, `experiments/amlsim_recon.py` → `results/amlsim_recon.json`, `experiments/check_amlsim_loader.py`.
- ⛔ **First derivation discarded.** Contiguous per-bank blocks gave bank volumes of 94.6 / 5.1 / 0.2 %, and 43 % of bank_c's transactions were laundering: the bank had become a proxy for the label. Interleaving every 10 positions gave 50.1 / 30.0 / 19.9 %.
- **Final:** 198,015 tx · 685 laundering (0.346 %) · 12,043 accounts · three native banks. Rows carry nothing (logistic MCC 0.0001); links carry a lot (sender 26.0×, receiver 41.5×). Uploaded as a private Kaggle dataset. Pre-registration (f), (g), (b), (c), (d) approved.

## ✅ X-29 — Kaggle job tooling (2026-09-11 → 09-12)

- `experiments/kaggle_jobs.py` — subcommands: `bundle` (code → private dataset `fl-adtcn-code`, manifest pins the laptop's library versions and a `bundle_sha256`), `push`, `status`, `fetch` (imports results only on exit 0), `queue --max 5`, `track`, `list`. Also `LAPTOP_ONLY_SWEEPS`, `FIT_POOL`, `NATIVE_ORG_CAP`, `NEEDS_MORE_ORGS`.
- **Traps fixed.** A smoke test found that Kaggle drops empty directories (the job now recreates `results/`) and that `OMP_NUM_THREADS` did not set torch's pool (jobs now start through a runpy wrapper calling `torch.set_num_threads(4)`). Every subprocess needs `encoding="utf-8", errors="replace"`. Status is now parsed from the `KernelWorkerStatus` token instead of substring-matching "error" (22:35 restart). `push()` now requires `successfully pushed`, because Kaggle refused a sixth session while the CLI exited 0, leaving a phantom job (re-pushed 02:58).
- `results/_kaggle/` — `_queue.json`, `_queue.log`, `_queue*.out/err.log`, `_queue3.pid`, `_track.*`, and one folder per job (`<job>/<job>.log`, `out/job.log`, `out/kaggle_job.json`, `out/db_boa_framework/results/*`, `out/final_report_data/*`): `paysim-smoke`, `paysim-{baselines,all-baselines,byzantine-robustness,economic-byzantine,private-incentive,scalability}-stratified`, `paysim-grid-{cnn,lstm,dtcn,dilated-attn,resnet,densenet,efficientnet}`, `amlsim-baselines-{stratified,bank}`, `amlsim-byzantine-robustness-stratified`, `amlsim-economic-byzantine-{stratified,bank}`, `amlsim-private-incentive-{stratified,bank}`, `amlsim-scalability-stratified` (+ `__failed_2335`), `amlsim-grid-{7 architectures}`, `handbook-grid-{7 architectures}`.
- `kaggle.md` (repo root, **gitignored**) holds the API token.

## ✅ X-30 — Entity window grids for the new datasets (2026-09-11 → 09-12)

- `experiments/entity_temporal_grid.py` — the PaySim / AMLSim grid, split per architecture for Kaggle. `--merge` refuses mixed protocols, duplicate cells and duplicate references.
- `experiments/handbook_temporal_grid.py` — the 7 × {global, customer, terminal} grid, resumable, with `--archs`, `--skip-reference`, `--merge`, `--allow-partial`. ⚠ `merge()` ignores `--out`, so never run a partial merge while the runner is live.
- `results/_smoke/paysim_temporal_grid_quick.json`, `_smoke/handbook_temporal_grid_quick.json` (smoke only; never results).
- Merged outputs: `results/paysim_temporal_grid.json`, `amlsim_temporal_grid.json`, `handbook_temporal_grid.json`, plus 21 per-architecture parts (`*_temporal_grid_{cnn,lstm,dtcn,dilated_attn,resnet,densenet,efficientnet}.json`); drafts `final_report_data/OBJ16_{paysim,amlsim,handbook}_temporal_grid.md`.

## ✅ X-31 — The windowing correction (2026-09-11)

- `experiments/check_partition_windowing.py` → `results/partition_windowing.json` (no training). With `groups=None`, an entity split leaves each org's rows grouped by entity, so its training windows are mostly single-entity: BankSim 0 % → **91.22 %**, Handbook 0 % → 96.37 %, AMLSim 0 % → 88.76 %. OBJ-15's claim that "the partition is the only factor moving" was wrong. Every partition effect is now quoted as "entity-disjoint split (which also changes the training windows)". The Krum withdrawal survives, because it rests on the dataset effect.

## ✅ X-32 — ULB regeneration on the fixed pool (2026-09-11 16:20 → 2026-09-12 04:03)

- `experiments/run_obj13_ulb_regen.ps1` (detector pool: 2 bitwise reproduction runs + the 2 repaired arms × seeds 42–46, then `--collect`, then `main.py --dataset ulb --attack`, then `run_baselines.py --dataset ulb`) and `experiments/run_main_ulb_rerun.ps1`. Logs in `results/_obj13_logs/ulb_regen/` (45 files, including `db_boa_results_phase1to7_2250.json` and `main_rerun_compare.txt`).
- `experiments/check_detector_repro.py` → `results/detector_repro_check.json`. Both 2026-08-31 detector records re-ran **bit for bit**, so the ULB reproduction failure does not reach the detector track.
- `results/_multiseed_runs/dbboa_repaired_{deterministic,averaged}_seed{42..46}_t2.json`; `detector_multiseed.json` extended. **OBJ-13 pre-registration 5 held**: deterministic 137/111 +0.0469 (p = 0.199), averaged 137/109 −0.1471 (p = 0.239). Neither is distinguishable from the default.
- `results/db_boa_results.json` regenerated (2026-09-11 22:50): search optimum 117 filters / 203 spe, best surrogate Obf2 **3.93** (no longer the 5.0 ceiling).
- `main.py` — Phase 8's attacker stand-in `predict` lambda lacked the `groups` argument. Fixed, re-run, and phases 1–7 reproduced bitwise.
- `results/baselines.json` regenerated (2026-09-12 01:48). FedAvg 0.5694 → **0.4247**, Krum 0.7762 → **0.7508** (+0.326 over FedAvg), DP now flags 43,089 transactions. **The DP collapse changed direction** between two ULB runs. The report had printed the old accuracy *rise* as a fall, and that sign error was corrected in place.

## ✅ X-33 — AMLSim scalability deviation (2026-09-11 23:35)

- `experiments/scalability_sweep.py --fit-pool` — AMLSim's training pool (138,514 rows) holds 17 of the 20 × 8,000-row shards, and the first job crashed. With the flag, the MC range stops at n = 16 and the exact range 3–12 is unchanged; `pool_fit` is recorded. Written down before any AMLSim scalability number existed. The first 16 shards are bitwise identical to the full protocol.

## ✅ X-34 — PaySim and AMLSim scored (2026-09-11 → 09-12 03:12; all 29 Kaggle jobs, 28.8 h)

- Results: `baselines_{paysim,paysim_all}_stratified.json`, `baselines_amlsim_{stratified,bank}.json`, `byzantine_robustness_sweep_{paysim,amlsim}_stratified.json`, `economic_byzantine_sweep_{paysim_stratified,amlsim_stratified,amlsim_bank}.json`, `private_incentive_sweep_{paysim_stratified,amlsim_stratified,amlsim_bank}.json`, `scalability_sweep_{paysim,amlsim}_stratified.json`, their PNGs, and the matching `TASKB_`, `TASKB1_`, `TASKC_`, `TASKD_` drafts.
- **PaySim:** (a) WRONG (Krum *protects* accuracy on clean cells, +3.8 to +19.2 pp); (b), (c) (11/11 strictly) and (e) held; (d) 1 wrong (receiver linking helps 6 of 7) and 2 held.
- **AMLSim:** (f) held (stratified at the floor, MCC −0.011, 0 of 106 caught); (g) held (native banks +0.158); (b) held; (c) **two scores, no verdict** (11/11 no more often, 8/11 strictly), by the team lead's decision; (d) 2 held, 3 wrong (ADTCN 1st on global windows, at the floor).

## ✅ X-35 — Collators extended to every condition (2026-09-12)

- `sweeps_cross_condition.py`, `federated_cross_dataset.py`, `make_report_figures.py` — a column appears only when its JSON exists; Kaggle runs and floor conditions (FedAvg MCC < 0.05) are marked. `results/figures/*` regenerated.

## ✅ X-36 — The Handbook suite and its scorecards (2026-09-12 04:03 → 18:02, commit `f92b6de`)

- `experiments/run_obj16_handbook.ps1` — waits for the ULB re-run, then for the pre-registration header to read APPROVED; runs byzantine / economic / private sweeps + ablation (`--filters 32`) × {stratified, customer}, then the grid (`--resume`), then scalability × 2. Logs: `results/_obj16_logs/handbook/` (27 files).
- The grid moved to Kaggle (7 jobs, `--archs`) and was merged at 15:09:30. The runner's grid step returned in 22 s, saving ~16 h. The laptop reproduced Kaggle's windowing statistics exactly.
- Results: `*_handbook_{stratified,customer}.{json,png}` for three sweeps, `baselines_handbook_{stratified,customer}.json`, and the drafts.
- **Scored: 6 of 8 predictions wrong.** (a) WRONG, and ⛔ **Krum selects a label-flipping attacker** (6/8 stratified, 7/8 customer; the poisoned model has the *lowest* Krum score). (b) the lone clause holds only *vacuously* (isolation never fires in 12 scenarios); collusion WRONG (gap 0.00); on `customer` the reputation ordering **inverts** (an honest bank ends 0.5068 against the 0.5 floor while the attacker rises). (c) WRONG (customer 9/11 under both readings). (d) 1 held, 3 wrong: linked windows help 3 of 14, and **the 72× terminal ordering trains the worst models for all seven architectures**.
- `baselines_customer` landed 16:01. FedAvg 0.0439 (stratified, just under the floor) → **0.1500** (customer). Entity *partitioning* lifts the Handbook; entity *windowing* does not. DB-BOA-ADTCN degenerates to constant-positive (351,280 FP, accuracy 0.95 %). **Decision 21:** items already scored stay scored; a floor borrowed from another dataset is not applied after the fact.
- `WORK_REPORT.tex` / `.pdf`, `TASK.md` updated; 395 files in the commit.

## ✅ X-37 — Suite complete; fidelity counting corrected (2026-09-12 23:53, commit `2efa8e7`)

- `results/scalability_sweep_handbook_{stratified,customer}.json` + 4 PNGs + drafts. The suite finished 11/11 OK in 19 h 50 m.
- **Correction.** Shapley fidelity is only meaningful where the 8,000-row shard models learn. PaySim, AMLSim and Handbook/stratified sit at 50 % balanced accuracy, where top-1 agreement of 3/10 is what guessing gives (expected 1.60). So it is **7 conditions, 4 testable**. The report stopped averaging "5/10, 5/10, 3/10, 3/10, 3/10". Handbook/customer is testable (56.9–58.3 %) and agrees at n = 12. "Fails from ~6 banks" stays withdrawn.

## ✅ X-38 — Krum separation margins across all seven conditions (2026-09-13, commit `13157f8`)

- `TASK.md`, `WORK_REPORT.tex` / `.pdf`. From each sweep's stored `score_margin` (lowest attacker score − highest honest score): BankSim +67 to +149, PaySim +48 to +134, ULB +7 to +8, **AMLSim −20 to −60**, Handbook −36 to −224. AMLSim's "8/8" is a pass by *ordering*, not by *detection*.

## ✅ X-39 — AMLSim cannot test Krum (2026-09-13, commit `7ef1ca8`)

- `TASK.md`, `WORK_REPORT.tex` / `.pdf`. `byzantine_robustness_sweep.py` fixes `REGIMES = [(5,1,1), (7,2,2)]`, and AMLSim has three native banks. The partition whose models work is too small to run the sweep; the partition big enough is inert. Krum is reported as **untestable on AMLSim**. Three alternatives are written down, none run.

## ✅ X-40 — Per-dataset journey section (2026-09-13, commit `4a8052e`)

- `WORK_REPORT.tex` — new § "The five datasets, one at a time" (`sec:datasetjourney`): per dataset, why it was brought in, what it can and cannot test, predicted vs found, what broke, net contribution. Also a coverage matrix (`sec:coverage`) and a cross-dataset synthesis ending *"a defence that cannot be observed failing is not a defence that passed."*
- Layout: the six-claims scoreboard moved from `tabularx` to `xltabular`; zero overfull vboxes and zero undefined references (65 pp).

## ✅ X-41 — Repository link on the cover (2026-09-13, commit `c66d1d1`) — **LATEST**

- `WORK_REPORT.tex` / `.pdf` — clickable link to `Undergraduate-Thesis`, with the branch named and a note that the work is not yet merged to `main`. Cover dates corrected to 30 Aug – 13 Sep.

---

## State at the latest commit (`c66d1d1`, 2026-09-13)

- **Nothing is running.** All five datasets are measured: 9 federated-ablation conditions (7 above the floor). WORK_REPORT counts **49 pre-registered predictions, 20 wrong**. Branch `obj13-surrogate-repair` is pushed to `origin`; `main` is still at `ccdd19f`.
- **Open (not done in any commit):** OBJ-7 (formal DP threat model and accountant), OBJ-8 (on-chain / off-chain execution map), OBJ-9 (scope "Scalable" and "Consensus" in the title), OBJ-6 (Shapley under a BankSim entity-disjoint split), the `ADTCN.fit:681` batch-formula decision, merging to `main`, and rotating the Kaggle token.
- **Small inconsistency to fix before quoting:** `detector_multiseed.json` stores tuned MCC **0.7531 ± 0.0553** (sample SD). WORK_REPORT's per-dataset section prints ± 0.0494 / ± 0.0676, which is the population SD. TASK.md and the brief use ± 0.055 / ± 0.076.

## Which early entries were later superseded

| Early entry | What replaced it | When |
|---|---|---|
| Tip 3 — DP ε = 1.0 on by default | Weight-channel DP is a swept knob, off by default; the privacy claim moved to the contribution channel (J-10) | 2026-06-06 |
| Tip 5 / Fix 2 — Shapley on Obf2 | Coalition value = balanced accuracy (J-7) | 2026-06-06 |
| Fix 1 / Fix 7 — Krum at f = 0 only | Krum also run at n = 5 / f = 1 and n = 7 / f = 2 (J-6); later broken on the Handbook (X-36) and hollow on AMLSim (X-38) | 2026-06-06, 09-12, 09-13 |
| Fix 4 / Fix 10 / Fix 11 / Fix 19 / Fix 36 — the CNN surrogate | Found to validate on memorised rows on ULB; repaired and the pool enlarged (X-18, X-19) | 2026-09-04 |
| Fix 12 — "MTTA" = global max-pool | The real dilated + attention ADTCN implemented and ablated (J-12) | 2026-06-07 |
| Fix 13 — DB-BOA vs default comparison | `dbboa_vs_default.py` (J-14), then 5-seed `detector_multiseed.py` (X-3), then the 9-run side-by-side (X-21) | 2026-06-11 → 09-05 |
| Fix 17 / Fix 21 — `run_baselines.py` | Writes `baselines*.json`; `--dataset` / `--partition` (X-6); ULB file regenerated (X-32) | 2026-06-08 → 09-12 |
| Fix 24 — "latency is simulated" | Real Fabric measurement (J-11, X-4) | 2026-06-07 / 08-31 |
| Tip 6 / Fix 3 — "graph features" | Renamed recurrence features; true entity linkage came from BankSim / Handbook / AMLSim IDs (X-6, X-23, X-28) | 2026-08-31 → 09-11 |

## Commit ledger

| Commit | Date | Branch / repo | Message | Entries |
|---|---|---|---|---|
| `b903bf2` | 2026-05-20 | local | first commit | B0-1 |
| `fe39499` | 2026-05-20 | local | 2nd commit | B0-2 |
| `c7d3012` | 2026-05-20 | local | novel approaches | Tips 1–6 |
| `2b38904` | 2026-05-20 | local | Update .gitignore and untrack ignored files | docs |
| `69aadb0` | 2026-05-20 | local | some more novelty approaches | Fixes 1–6 |
| `6a2b4a3` | 2026-05-20 | local | some more novelty approaches | Fixes 7–17 (+ `defense_questions.md`) |
| `50492f4` | 2026-05-21 | local | some issues resolved | Fixes 18–29 |
| `e1bb51a` | 2026-06-01 | local | some more issues resolved | Fixes 30–36 |
| `37b5d4a` | 2026-06-02 | local | bug fix | BF-1–10 |
| `c2c22f3` | 2026-06-06 | v2 | Initial commit: DB-BOA federated learning + Hyperledger Fabric thesis | J-1 … J-9 |
| `0f02ce8` | 2026-06-07 | v2 | now title sticks word to word | J-10 … J-12 |
| `cbe5035`, `7d84bf4`, `81981f0` | 2026-06-08 | v2 | report done | J-13 |
| `94c071c` | 2026-06-11 | v2 | report done | J-14, J-15 |
| `69ef481` | 2026-08-30 | local (dropped; kept on `backup-before-v2-sync`) | refreshing | X-1 |
| `ccdd19f` | 2026-08-30 | `main` = `obj13-surrogate-repair` base | Sync repository to the state described by the final thesis report | X-1 |
| `889d623` | 2026-09-05 | `obj13-surrogate-repair` | Land accumulated uncommitted work: OBJ-1b, OBJ-15, OBJ-17, OBJ-18 | X-3 … X-17 |
| `ac606de` | 2026-09-05 | `obj13-surrogate-repair` | OBJ-13: repair the DB-BOA surrogate, and measure that it changes nothing | X-18 … X-22 |
| `33bb97f` | 2026-09-05 | `obj13-surrogate-repair` | new datasets | X-22 |
| `6d8e41e` | 2026-09-10 | `obj13-surrogate-repair` | -- | X-23 … X-25 |
| `f92b6de` | 2026-09-12 | `obj13-surrogate-repair` | OBJ-16: score the Handbook's four pre-registered items; three of them fail | X-26 … X-36 |
| `2efa8e7` | 2026-09-12 | `obj13-surrogate-repair` | OBJ-16: finish the Handbook suite; correct how scalability fidelity is counted | X-37 |
| `13157f8` | 2026-09-13 | `obj13-surrogate-repair` | Krum's label-flip separation fails on AMLSim too, not just the Handbook | X-38 |
| `7ef1ca8` | 2026-09-13 | `obj13-surrogate-repair` | AMLSim cannot test Krum on either partition: record why, do not fill the cell | X-39 |
| `4a8052e` | 2026-09-13 | `obj13-surrogate-repair` | Add a per-dataset journey section, written as thesis source material | X-40 |
| `c66d1d1` | 2026-09-13 | `obj13-surrogate-repair` (**HEAD**) | Put the repository link on the cover page, and correct the cover dates | X-41 |

---
