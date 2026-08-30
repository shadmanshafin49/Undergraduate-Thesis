# Chapter 2 — Literature Review (notes)

The current `chapters/chapter_2.tex` is in good shape and largely defensible. It cites
real, relevant work (McMahan FedAvg, FedProx, Hyperledger Fabric finance, BIT-FL, Zhao
Long-Term Proof-of-Contribution, Arora & Singh BOA, Prabanand & Thanabal DB-BOA, etc.).
Keep the structure. Two adjustments only.

## Fix 1 — Add the methods you actually use, so the methodology has a foundation
The implemented federation stack (DP + Krum + Shapley) needs literature scaffolding so
Chapter 4/5 can cite it honestly:
- **Differential privacy:** Dwork et al., *Calibrating Noise to Sensitivity*, TCC 2006
  (Gaussian mechanism). Already referenced in code docstrings.
- **Krum robust aggregation:** Blanchard et al., *Machine Learning with Adversaries:
  Byzantine-Tolerant Gradient Descent*, NeurIPS 2017.
- **Shapley contribution valuation in FL:** Wang et al., *Measure Contribution of
  Participants in Federated Learning*, IEEE BigData 2020 (FedSV); optionally Ghorbani &
  Zou, *Data Shapley*, ICML 2019.
Add a short subsection "Robust and Fair Federated Aggregation" covering these three, and
position the contribution as: *combining DP + Krum + Shapley and binding Shapley weights
to on-chain incentives on a real Fabric network*.

> **🔴 Cite FedCoin — and do NOT claim priority.** FedCoin (Liu et al. 2020, *FedCoin: A
> Peer-to-Peer Payment System for Federated Learning*) already does Shapley-value reward
> allocation for FL on a blockchain. Add it to the bib (`fedcoin2020`, done in
> REWRITE_02) and the subsection, and frame our work as *extending* it by **characterising**
> the privacy↔incentive coupling on a deployed Fabric stack — never as "first to bind Shapley
> to blockchain incentives" (that line is struck across REWRITE_01/05/09). Also add
> `andrew2021` (adaptive DP clipping) since the §6.6 contribution uses it.

## Fix 2 — FedProx mention must stay descriptive, not claimed as used (🔴 D3)
The FedProx paragraph (citing Li et al. 2020) is fine as **related work**, but the report
currently also says "this thesis employs … FedProx regularisation." Remove the claim that
*this thesis uses* FedProx (it does not). Either cite it purely as background, or move it
to future work.

## Research-gap table
The "Summary of Related Work and Identified Research Gap" table is good. Update the final
"This thesis" row to describe the **real** contribution (DP+Krum+Shapley federation with
incentive coupling on real Fabric + DB-BOA for hyperparameters/leader selection), not
"DB-BOA for all three roles."
