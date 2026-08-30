# REWRITE — Title page & Abstract (paste-ready)

Framing: central contribution = **private-incentive characterisation (B1)** — the
privacy↔incentive coupling of Shapley-driven, chaincode-enforced rewards on a real Hyperledger
Fabric consortium. Supporting substrate = DP + Krum + exact/MC Shapley federated aggregation,
DB-BOA hyperparameter/leader tuning, RL sequential leader selection, and the ADTCN detector.
Each title word is earned in its qualified sense (see notes) — **the title is fixed and must not
be changed**.

---

## `core/titlepage.tex` — DO NOT change the title

> **The approved title is final and stays exactly as in the `.tex` (the build now "sticks word
> to word"). Do not replace it.** For reference, the title is:
>
> *"A Blockchain-Integrated Framework for Secure, Incentivized, and Scalable Machine Learning:
> Integrating Consensus Mechanisms and Reinforcement Learning."*
>
> Each title word is substantiated in its **qualified** sense and the body must say so explicitly
> (do not let the words imply more than the evidence): **Secure** = Krum statistical BFT in the
> $n\ge 2f+3$ regime (\S6.8) plus economic isolation of a colluding majority (\S6.7);
> **Scalable** = scalable contribution \emph{attribution} (MC-Shapley, $O(n\cdot\text{samples})$,
> faithful to $n\approx 5$ — \S6.9), \emph{not} blockchain throughput or distributed training;
> **Consensus Mechanisms** = Hyperledger Fabric Raft + DB-BOA/RL leader selection;
> **Reinforcement Learning** = a deliberately minimal linear-Q leader-selection component (\S on
> RL), secondary to the privacy↔incentive contribution. An earlier draft proposed a different,
> narrower title; that proposal is **superseded** — keep the approved title.

(Keep the rest of the title page unchanged: authors, dept, April 2026, © 2026.)

---

## `core/abstract.tex` — replace the abstract paragraph

```latex
\section*{Abstract}
The growth of digital financial services has made cross-institutional fraud faster to
spread than any single bank can detect, yet privacy regulation and competitive distrust
prevent institutions from pooling raw transaction data. This thesis presents FL-ADTCN, a
blockchain-integrated federated learning framework for collaborative financial fraud
detection that resolves the privacy, trust, and incentive barriers simultaneously. Each
institution trains a local Adaptive Deep Temporal Context Network (ADTCN) -- a temporal
one-dimensional convolutional detector -- on its own transactions, and only model
parameters, never raw data, are shared. The federated aggregation layer combines
\emph{differentially private} weight sharing (Gaussian mechanism), \emph{Krum} robust
selection to reject outlying updates, and \emph{Shapley-value} contribution attribution that
quantifies each institution's marginal value to the global model; the Shapley weights are
written to a Hyperledger Fabric ledger and drive an on-chain, chaincode-enforced token
incentive. The central contribution is a \textbf{characterisation of the
privacy--incentive coupling} of this mechanism: we show that differential privacy and
contribution-based incentives are in direct tension, and that applying the privacy noise on
the released contribution channel rather than on the shared weights restores rank-faithful,
DP-protected on-chain rewards at roughly two orders of magnitude lower privacy budget
($\epsilon^\star$: $3000\!\to\!50$). This is supported by a characterisation suite that bounds
the mechanism: the privacy--utility decoupling of accuracy versus reward fidelity; the economic
isolation of a colluding majority; statistical Byzantine fault tolerance; and the cost and
fidelity limits of Shapley attribution as the federation grows. The supporting substrate is a
Dynamic Butterfly--Billiards Optimisation Algorithm (DB-BOA) that automates ADTCN
hyperparameter tuning (without manual search, though it does not exceed a hand-tuned default) and
consensus leader selection, together with a deliberately minimal reinforcement-learning (linear
Q-learning) policy for \emph{sequential} leader rotation that, in simulation, matches the
optimiser on reward while adapting to nodes whose reliability degrades at runtime. The framework
is implemented end-to-end on a real Hyperledger Fabric consortium with the \texttt{DBBOAContract}
smart contract and evaluated on the public ULB Credit Card Fraud dataset (284{,}807 transactions,
0.17\% fraud) partitioned across three banks. Detection metrics are reported at the deployment
operating point, in which weight-channel differential privacy is disabled and privacy is instead
enforced on the incentive channel, the configuration the contribution above shows to be both
private and useful. \emph{Scalability} is established for contribution \emph{attribution} -- a
Monte-Carlo Shapley estimator that remains tractable to $n=20$ organisations, faithful to
$n\approx 5$ -- rather than for blockchain throughput or distributed training, which remain in the
limitations. \emph{Security} is evaluated on two complementary fronts: Krum is run in the regime
where its $n\ge 2f+3$ guarantee holds ($n=5,f=1$ and $n=7,f=2$), rejecting four weight-level
poisoning attacks, while the economic incentive layer isolates a behavioural colluding majority
that out-numbers any statistical defence; the default three-organisation pipeline runs at $f=0$
(consensus alignment, no adversary assumed).

\vspace{1cm}
\textbf{Keywords: }Federated Learning; FL-ADTCN; Adaptive Deep Temporal Context Network;
Hyperledger Fabric; Permissioned Consortium Blockchain; Smart Contracts; Financial Fraud
Detection; Differential Privacy; Krum; Shapley Value; Contribution Attribution; Token
Incentive Mechanism; Dynamic Butterfly--Billiards Optimisation (DB-BOA); Leader Selection;
Privacy Preservation; Byzantine Resilience
```

### Notes
- **Title unchanged** (fixed/approved, sticks word-for-word). The abstract substantiates each
  title word in its qualified sense rather than removing it: RL = minimal linear-Q leader
  rotation (secondary); Scalable = scalable contribution attribution; Secure = Krum BFT
  ($n\ge 2f+3$) + economic isolation of a majority; Consensus = Fabric Raft + DB-BOA/RL leader
  selection.
- Removed only the genuinely-absent terms: PoDL, FFL, AHE, GMM, "DB-BOA three roles", "DB-BOA
  Job 3 aggregation weights".
- B1 (private-incentive characterisation, $\epsilon^\star\,3000\!\to\!50$) is stated as the
  central contribution; the other tasks (privacy/utility decoupling, economic isolation,
  statistical BFT, Shapley scalability) are framed as the supporting characterisation.
- Detection numbers are explicitly tied to the **weight-DP-off** deployment operating point, with
  privacy carried by the incentive channel — closing the cherry-pick risk.
- Dataset stated correctly as ULB 284,807 / 0.17%.
- No specific accuracy number in the abstract (avoids the 97.38/0.966 problem); fill a real
  figure only after the full run, if you want one sentence of headline results.
