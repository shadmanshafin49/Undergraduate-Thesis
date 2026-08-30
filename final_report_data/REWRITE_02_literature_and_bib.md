# REWRITE — Chapter 2 Literature Review + Bibliography additions (Option A)

## A. Chapter 2 edits

### A1. FedProx must be background-only (not "employed")
In the `\subsection{Non-IID Data Heterogeneity}` paragraph, the sentence currently says
"This thesis employs both stratified sampling and FedProx regularisation...". FedProx is
**not** implemented. Replace that final sentence with:
```latex
This thesis addresses non-IID heterogeneity through stratified partitioning, which preserves
the global fraud rate in every institution's shard; proximal methods such as FedProx remain a
natural extension for more severely heterogeneous deployments.
```

### A2. Add a new subsection (after "Incentive Mechanisms") for the methods actually used
```latex
\subsection{Robust and Fair Federated Aggregation}

Three techniques underpin the aggregation layer of this thesis. \textbf{Differential privacy}
(DP), formalised by Dwork et al.~\cite{dwork2006}, calibrates noise to the sensitivity of a
computation; the Gaussian mechanism adds noise $\mathcal{N}(0,\sigma^2)$ with
$\sigma = C\sqrt{2\ln(1.25/\delta)}/\epsilon$ after clipping to an $L_2$ bound $C$, giving an
$(\epsilon,\delta)$ guarantee on shared model parameters. \textbf{Krum}, introduced by
Blanchard et al.~\cite{blanchard2017}, provides Byzantine-robust aggregation by scoring each
update by the summed distance to its nearest neighbours and selecting the most
consensus-aligned one, rejecting outliers. \textbf{Shapley values}, originating in
cooperative game theory and adapted to federated learning by Wang et al.~\cite{wang2020fedsv},
assign each participant its exact average marginal contribution across all coalitions,
providing a principled, axiomatic basis for contribution attribution. This thesis combines
all three -- DP for privacy, Krum for robustness, Shapley for fair attribution -- extending
prior Shapley-on-blockchain incentive work (FedCoin~\cite{fedcoin2020}) by \emph{characterising}
the privacy--incentive coupling of a Shapley-driven, chaincode-enforced reward mechanism
deployed on a real permissioned Fabric network.
```

### A3. Update the research-gap table's final row
Change the "This thesis (FL-ADTCN)" row "What It Achieves" cell to:
```latex
\textbf{Federated ADTCN on real Hyperledger Fabric with DP + Krum + Shapley aggregation and
Shapley-weighted on-chain incentives}
```

---

## B. Bibliography additions

Append these to `bibliography/references.bib`. (These cover the new Option-A citations.)

```bibtex
@inproceedings{dwork2006,
  author    = {Dwork, Cynthia and McSherry, Frank and Nissim, Kobbi and Smith, Adam},
  title     = {Calibrating Noise to Sensitivity in Private Data Analysis},
  booktitle = {Theory of Cryptography (TCC)},
  pages     = {265--284},
  year      = {2006},
  doi       = {10.1007/11681878_14}
}

@inproceedings{blanchard2017,
  author    = {Blanchard, Peva and El Mhamdi, El Mahdi and Guerraoui, Rachid and Stainer, Julien},
  title     = {Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {30},
  year      = {2017}
}

@inproceedings{wang2020fedsv,
  author    = {Wang, Tianhao and Rausch, Johannes and Zhang, Ce and Jia, Ruoxi and Song, Dawn},
  title     = {A Principled Approach to Data Valuation for Federated Learning},
  booktitle = {Federated Learning -- Privacy and Incentive},
  series    = {Lecture Notes in Computer Science},
  volume    = {12500},
  pages     = {153--167},
  year      = {2020},
  doi       = {10.1007/978-3-030-63076-8_11}
}

@inproceedings{fedcoin2020,
  author    = {Liu, Yuan and Ai, Zhengpeng and Sun, Shuai and Zhang, Shuangfeng and Liu, Zelei and Yu, Han},
  title     = {{FedCoin}: A Peer-to-Peer Payment System for Federated Learning},
  booktitle = {Federated Learning -- Privacy and Incentive},
  series    = {Lecture Notes in Computer Science},
  volume    = {12500},
  pages     = {125--138},
  year      = {2020},
  doi       = {10.1007/978-3-030-63076-8_8}
}

@inproceedings{andrew2021,
  author    = {Andrew, Galen and Thakkar, Om and McMahan, H. Brendan and Ramaswamy, Swaroop},
  title     = {Differentially Private Learning with Adaptive Clipping},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  pages     = {17455--17466},
  year      = {2021}
}
```

---

## C. PRE-EXISTING bibliography problem (flag for the team — not introduced by Option A)

Many keys already cited in the chapters are **missing** from `references.bib`, so the report
currently has undefined citations regardless of Option A:

`mcmahan2017, prabanand2025, li2020fedprox, abdallah2020, ahamad2022, arora2019,
hussain2024, li2022fdia, li2023, machhale2024, nourmohammadi2022, saveetha2024, truong2024,
tsoulias2020, wang2022, yang2024, ying2025, zhang2019, zhao2024, zhuang2019`

The bib presently uses numeric IEEE keys (e.g. `10044098`, `10746050`) plus a few named ones
(`acfe2022`, `ulb2018`). **Action:** either add `@article{...}` entries for each missing
named key, or change the `\cite{...}` calls to the numeric keys that already exist. This must
be fixed for the document to compile cleanly (otherwise citations render as `[?]`). It is
independent of the Option A content rewrite. The most important one to add is the base paper:

```bibtex
@article{prabanand2025,
  author  = {Prabanand, S. C. and Thanabal, M. S.},
  title   = {Advanced financial security system using smart contract in private Ethereum
             consortium blockchain with hybrid optimization strategy},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {6764},
  year    = {2025},
  doi     = {10.1038/s41598-025-XXXXX}
}
```
(Confirm the exact DOI/volume from the published paper before final submission.)
