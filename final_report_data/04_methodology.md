# Chapter 4/5 — Proposed Methodology (the real version to write)

Rewrite the methodology so each described component matches `00_ground_truth_
implementation.md`. Below is the accurate content to use, section by section.

## 4.1 System overview
Four layers (Table "Integrated Architectural Flow" — keep, but correct the rows):
- **Learning layer:** Federated ADTCN (1D-CNN) trained locally per bank.
- **Optimisation layer:** DB-BOA for (1) ADTCN hyperparameters and (2) leader selection.
- **Federation layer:** DP weight sharing → Krum selection → Shapley contribution weights.
- **Trust layer:** Hyperledger Fabric (3 orgs) + `DBBOAContract` chaincode + token incentives.
- **Ledger layer:** Fabric world state (CouchDB), immutable records.

## 4.2 ADTCN detector (write the REAL architecture — 🔴 D2)
> The detector is a temporal **1-D convolutional network** applied to a sliding window of
> the 10 most recent transactions. Each transaction is a 30-dimensional vector (V1–V28
> PCA components, standardised Amount and Time). The window passes through two stacked
> `Conv1d` layers (kernel 3, padding 1) with ReLU — first projecting to F channels then to
> 2F — followed by **global max-pooling over the time axis** and a linear classifier to two
> logits (normal / fraud). Class imbalance (0.17% fraud) is handled by **weighted
> cross-entropy** rather than resampling.

Map the base-paper block names honestly: MJE = the raw multi-feature input; TCL = the two
1-D convolutions over the window; MTTA = the global max-pool (a pooling operator, used in
place of the base paper's attention). **Do not** claim 64-dim embeddings, BatchNorm,
dilated convolutions, or softmax attention.

Feature engineering note: PTC (rolling mean/std over {5,10,20}) and NTC (diffs {1,2}) and
MJE interaction features are computed in `data_loader.py` for completeness, but the CNN
consumes only the leading 30/33 raw features — say this plainly.

## 4.3 DB-BOA optimiser
Describe the DBOA+BOA hybrid with the **adaptive switch** `rand vs bestfit/worstfit`, LSAM
Lévy mutation, and billiards collision update. Two live jobs:
- **Job 1 (hyperparameters, 2-D — 🔴 D11):** searches `(n_filters, steps_per_epoch)`;
  **epoch count is fixed, not searched**. pop=20, iter=30. Objective **fixed** to the bounded,
  MCC-dominated `Obf2 = 2·MCC + Spec + Pre + NPV` on a surrogate CNN — the old unbounded
  `1/FPR` term was removed because it diverges as FPR→0 and made the search degenerate (see
  REWRITE_05 §5.A). Honest finding (REWRITE_06 §6.5): even with the corrected objective DB-BOA
  **does not beat the hand-set default** (tuned MCC 0.677 vs default 0.785, single fixed-objective
  run) — it slightly trails it; frame DB-BOA's win as *automation* and *leader selection*, not
  accuracy. Verified fresh optimum: `{n_filters=142, steps_per_epoch=76}` (do not cite the old
  128/30/150 as DB-BOA's pick — that is the hand-set default).
- **Job 2 (leader selection):** 10-node simulated consortium, minimise `CT+CC+MS` minus a
  reputation bonus; pop=15, iter=25. Be explicit that this is a **simulation** and that
  the node→org mapping currently alternates Org1/Org2 (see D6 — fix or describe honestly).

## 4.4 Federation layer (the real method — 🔴 D1, D3)
Write three steps with their citations:
1. **Differentially-private weight sharing** (Dwork 2006): L2-clip C=1, Gaussian noise
   σ = C·√(2 ln(1.25/δ))/ε, ε=1.0, δ=1e-5 ⇒ σ≈4.84. Include the **honest caveat**: at
   ε=1.0 the noise dominates the clipped signal, so the DP-shared global model is heavily
   degraded — a deliberate privacy/utility trade-off; production would use larger ε or
   DP-SGD. The ablation confirms this empirically (REWRITE_06 §6.2: both DP rows collapse at
   ε=1.0 to a degenerate single-class predictor, MCC≈0). **One-DP-description note (Issue 4):**
   Contribution 1 (§6.6) uses an *adaptive per-tensor sensitivity* variant (C=‖w‖₂, Andrew et
   al. 2021); because the per-element noise-to-signal ratio is independent of C the transition
   budget is unchanged, so the deployed fixed-clip and the §6.6 adaptive descriptions are
   consistent — say so explicitly to pre-empt the "which DP?" viva question.
2. **Krum robust selection** (Blanchard 2017): with n=3, f=0, selects the most
   consensus-aligned org's weights as the global model. State this is **outlier/consensus
   alignment**, not full Byzantine tolerance (which needs n ≥ 2f+3 with f ≥ 1).
3. **Exact Shapley contribution weights** (Wang/FedSV 2020): evaluate all `2^3−1 = 7`
   coalitions on a shared validation set; Shapley value per org; clip negatives; normalise
   to sum 1. **These weights drive the on-chain federation token split.** This is the
   fairness/incentive contribution: reward = mathematically-derived contribution, on a
   live ledger. **Frame as characterisation, NOT priority (🔴):** FedCoin (2020) already does
   Shapley reward allocation on a blockchain — cite it (`fedcoin2020`), present this as
   *extending* it with a deployed Fabric implementation and a noise→fidelity→reward-error
   analysis; never claim "first." Note too that at the deployed ε=1.0 the per-round weights are
   noise-dominated (REWRITE_06 §6.6 places the incentive-fidelity threshold at ε\*=3000), so the
   ε=1.0 split illustrates the mechanism, not a trustworthy payout.

(If keeping a DB-BOA Job 3 subsection at all, label it clearly as an *alternative
aggregation mode available in the codebase* that is disabled by default.)

## 4.5 Blockchain + incentives
- 3 orgs (BankA/BankB/BankC), `DBBOAContract`, deterministic chaincode, CouchDB, Raft
  ordering, `fabric-network` 2.2.20 API server + dashboard.
- Token table (real — matches chaincode): +10 fraud-consensus, +15 latency<300ms bonus,
  +10 leader success, −2 dispute, −2 round fail, +20 federation pool split by **Shapley**
  weight. Reputation ∈ [0.5, 2.0], +0.02 / −0.05.
- **Off-chain vs on-chain boundary** (this section is good in the report — keep it): all
  optimisation/ML/randomness off-chain; only finalised results (leader id, hyperparams,
  Shapley weights, token deltas, metrics) written on-chain.

## 4.6 Experimental configuration (correct the table — D3, D16)
- Dataset: ULB 284,807 / 0.17% fraud; sequence length 10; 3 orgs 50/30/20.
- FL method: **DP + Krum + Shapley** (not "FedProx + DB-BOA Job 3").
- Federation interval: every 5 consensus rounds.
- Local epochs: the actual fixed value used (state the real number from config/run).
- Framework: PyTorch (CPU unless GPU verified), Hyperledger Fabric 2.5.x.
- Metrics: Accuracy, Precision, Sensitivity, Specificity, NPV, FPR, FNR, FDR, F1, MCC,
  ROC-AUC (all in `utils/metrics.py`).

## 4.7 Methodology process flow
Keep the 6/8-phase flow from `main.py` (it is accurate): leader selection → hyperparameter
search → local training → evaluation → consensus simulation → plots → federation (DP/Krum/
Shapley) → optional Byzantine attack.
