# Chapter 1 — Introduction (corrections to write)

The current `chapters/chapter_1.tex` is well written and mostly correct. Fix only the
points below so it matches the real work.

## What to keep (already accurate)
- Background framing: cross-institutional fraud, privacy/trust/incentive barriers.
- Rationale: builds on Prabanand & Thanabal (2025) DB-BOA-ADTCN; identifies the FL gap,
  the simulated-vs-real-blockchain gap, and the aggregation-weight gap.
- Six-phase methodology-in-brief and scopes/challenges structure.

## Fixes required

### 1. Dataset description (🔴 D10)
Current text (lines ~118 and Ch.3 ~284): *"synthetically generated … 20,000 samples,
30 features, 5% fraud rate."*

**Replace with the truth:**
> Experiments use the publicly available Kaggle **ULB Credit Card Fraud Detection**
> dataset — **284,807 anonymised transactions** (V1–V28 PCA components, Amount, Time),
> with **492 fraudulent cases (0.17%)** — partitioned across three simulated banks
> (BankA 50%, BankB 30%, BankC 20%) by stratified sampling that preserves the global
> fraud rate in each shard.

### 2. Novelty statement (🔴 D1)
Current text frames "DB-BOA in three coupled roles … third role = federated aggregation
weights" as the central novelty. The code uses **Shapley + Krum + DP** for federation,
not DB-BOA Job 3.

**DECIDED — Option A (matches code).** Use this framing consistently:
- **(A, matches code):**
  > The framework couples (i) a hybrid DB-BOA metaheuristic that tunes the ADTCN
  > detector and selects consensus leaders, with (ii) a privacy-preserving, fairness-aware
  > federation layer — differentially-private weight sharing, Krum robust model selection,
  > and **exact Shapley-value contribution attribution** — whose Shapley weights directly
  > drive on-chain token rewards, all on a real Hyperledger Fabric consortium.

> **Characterisation, NOT priority (🔴 do not claim "first").** FedCoin (2020) already does
> Shapley-value profit allocation for FL on a blockchain, so a "first to bind Shapley to
> blockchain incentives" claim is false and self-contradicting (the lit review cites FedCoin).
> Frame the contribution as *characterising* the privacy↔incentive coupling
> (noise→Shapley fidelity→reward error) of a deployed Fabric implementation, **extending**
> FedCoin — not claiming priority over it. Applied in REWRITE_01/05/09; `fedcoin2020` added to
> the bib in REWRITE_02.

> **DB-BOA framing (🟠 keep honest).** DB-BOA *automates* hyperparameter tuning and produces a
> working detector, but it *does not beat — and slightly trails — a hand-tuned default*
> (REWRITE_06 §6.5: tuned MCC 0.677 vs default 0.785); its clear win is automated leader selection.
> Say "automates selection without manual search," not "maximises fraud-detection performance."

- **(B, if you switch code to DB-BOA Job 3) — NOT chosen:** would require setting
  `use_shapley=False`/`use_krum=False`, re-running, and obtaining non-degenerate weights first
  (it does not converge — see divergence D9/D11). Left here for the record only.

### 3. Differential privacy is implemented, not future (🟢 D17)
Move DP from "future directions" into the contribution list (with the honest ε=1.0
trade-off caveat). Keep stronger DP / DP-SGD as future work.

### 4. Remove RL references downstream of the title (🟢 D15)
Ensure no Reinforcement Learning wording remains; the title must change too.

## Suggested objective list (rewritten to match code)
1. Deploy a Hyperledger Fabric 3-org consortium with the `DBBOAContract` chaincode
   enforcing consensus logging and token incentives.
2. Apply DB-BOA to **(a)** ADTCN hyperparameter search (2-D: filters, steps/epoch) and
   **(b)** consensus leader selection (minimising CT+CC+MS with reputation discount).
3. Implement a federated pipeline over BankA/BankB/BankC with **DP weight sharing + Krum
   selection + Shapley contribution weighting**.
4. Couple **Shapley** contribution weights to the on-chain federation token pool so reward
   tracks contribution automatically.
5. Evaluate fraud detection on the ULB dataset (centralised vs IID vs non-IID) against
   FedAvg / FedAvg+Krum / FedAvg+DP baselines.
6. Test Byzantine resilience with an always-fraud BankC and show token/reputation
   depletion isolates it.
