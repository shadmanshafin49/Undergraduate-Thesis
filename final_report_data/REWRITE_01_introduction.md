# REWRITE — Chapter 1 Introduction (Option A, paste-ready)

The current `chapter_1.tex` is well written. Apply these targeted replacements; keep the
Background paragraphs (privacy/trust/incentive barriers) as they are.

---

## 1.1 Replace the novelty paragraph (end of Background, ~line 80)

**Old:** "...its central algorithmic novelty is the use of the Dynamic Butterfly-Billiards
Optimisation Algorithm (DB-BOA) in three coupled roles that have never previously been
combined in a single system."

**New:**
```latex
The framework is called \textbf{FL-ADTCN} -- Federated Adaptive Deep Temporal Context
Network. Its \textbf{central contribution} is a \emph{characterisation of the
privacy--incentive coupling} of a Shapley-driven, chaincode-enforced reward mechanism on a real
permissioned blockchain, and a concrete fix for it: we show that differential privacy and
contribution-based incentives are in direct tension, and that applying the privacy noise to the
released contribution channel rather than to the shared model weights restores rank-faithful,
DP-protected on-chain rewards at roughly two orders of magnitude lower privacy budget
($\epsilon^\star$ from $\approx 3000$ down to $\approx 50$). Around this result we provide a
characterisation suite that motivates and bounds the mechanism: the decoupling of model accuracy
from reward fidelity under DP, the economic isolation of a colluding majority, statistical
Byzantine fault tolerance, and the cost--fidelity limits of Shapley attribution as the federation
grows. This contribution is built on, but is distinct from, the engineering substrate -- a
federated aggregation layer (differentially private weight sharing, Krum robust selection, and
exact/Monte-Carlo Shapley attribution) bound to an on-chain token incentive enforced by
Hyperledger Fabric chaincode; a hybrid Dynamic Butterfly--Billiards Optimisation Algorithm
(DB-BOA) that tunes the ADTCN detector's hyperparameters and selects the consensus leader; and a
deliberately minimal reinforcement-learning (linear Q-learning) policy for \emph{sequential}
leader rotation. Rather than claiming algorithmic priority, we extend prior Shapley-on-blockchain
incentive work (FedCoin~\cite{fedcoin2020}) with a deployed Hyperledger Fabric implementation and
a novel noise$\to$fidelity$\to$reward-error analysis and channel-placement result.
```

> **Contribution ordering (apply to the contributions list / Ch.1 enumerate too):** make the
> private-incentive characterisation **Contribution~1** (the one genuinely new, correctly-bounded
> result, \S6.6). List the characterisation tasks -- privacy/utility decoupling, economic
> isolation (\S6.7), statistical BFT (\S6.8), and scalable attribution (\S6.9) -- as Contributions
> 2--5 that bound it. Present DB-BOA, the ADTCN detector, the RL leader policy, and the Fabric
> deployment as the enabling \emph{substrate}, not as co-equal novelties. This matches what the
> evidence supports and stops five "equal" contributions from diluting the one new result.

---

## 1.2 Rationale — replace the "three gaps" paragraph (~line 85)

Keep the Prabanand \& Thanabal \cite{prabanand2025} base-paper sentence. Replace the gap
list with:
```latex
However, the base paper leaves three gaps that this thesis fills. First, it does not
implement federated learning: a single institution's model is optimised and deployed rather
than a global model trained collaboratively across organisations. Second, it uses a
simulated blockchain (MATLAB PEC-BC) rather than a real permissioned network, leaving
deployability unvalidated. Third, it provides no mechanism for fairly attributing
contribution or distributing incentives across multiple institutions. This thesis addresses
all three by implementing a federated ADTCN on a real Hyperledger Fabric consortium and by
adding a differential-privacy / Krum / Shapley aggregation layer whose Shapley weights drive
on-chain token rewards.
```

---

## 1.3 Problem Statement — fix the bulleted list (~lines 95-102)

Replace items 3--5 of the enumerated list with:
```latex
  \item Uses \textbf{differentially private} weight sharing (Gaussian mechanism) so that the
        parameters exchanged between institutions carry a formal privacy guarantee.
  \item Uses \textbf{Krum} robust aggregation to select the most consensus-aligned model and
        reject outlying (potentially poisoned) updates.
  \item Uses \textbf{exact Shapley values} to attribute each institution's marginal
        contribution to the global model, replacing naive equal weighting, and writes these
        weights to the Fabric ledger.
  \item Enforces a token-based incentive structure via \texttt{DBBOAContract} chaincode that
        distributes federation rewards proportionally to the Shapley contribution weights,
        creating a direct link between measured model contribution and economic reward.
  \item Demonstrates adversarial resilience by showing that the Shapley mechanism assigns a
        near-zero weight to a Byzantine participant that degrades the global model.
```

---

## 1.4 Objective — replace the enumerated objectives (~lines 107-114)

```latex
\begin{enumerate}
  \item To deploy a Hyperledger Fabric permissioned consortium blockchain with the
        \texttt{DBBOAContract} chaincode that records consensus decisions and enforces
        token-based incentives for a financial fraud-detection consortium
        (BankA, BankB, BankC).
  \item To implement DB-BOA for ADTCN hyperparameter optimisation -- automating selection of the
        convolutional filter count and steps-per-epoch without manual search (a working detector
        obtained automatically, though it does not exceed a hand-tuned default).
  \item To implement DB-BOA for consensus leader-node selection, minimising the composite
        resource objective $\text{Obf}_1 = \text{CT} + \text{CC} + \text{MS}$ with a
        reputation discount.
  \item To implement a privacy-preserving, fairness-aware federated learning pipeline across
        the three-bank consortium, combining differentially private weight sharing, Krum
        robust selection, and exact Shapley contribution attribution.
  \item To couple the Shapley contribution weights to an on-chain token incentive mechanism
        so that reward tracks contribution automatically and verifiably.
  \item To assess Byzantine resilience by simulating an organisation that always reports
        fraud and showing that the Shapley mechanism isolates it.
\end{enumerate}
```

---

## 1.5 Methodology in Brief — fix the dataset sentence (~line 118)

**Old:** "Financial transaction data was synthetically generated ... (20,000 samples, 30
features, 5\% fraud rate) ..."

**New:**
```latex
The framework is evaluated on the public Kaggle ULB Credit Card Fraud Detection dataset
(284{,}807 anonymised transactions; features V1--V28, Amount, Time; 492 fraudulent cases,
0.17\%)\cite{ulb2018}, partitioned across three simulated banks (BankA 50\%, BankB 30\%,
BankC 20\%) by stratified sampling that preserves the global fraud rate in each shard.
```

Then update the Phase descriptions: **Phase 4** becomes "Federated Aggregation: at each
federation interval, the framework applies differentially private weight sharing, Krum
selection, and exact Shapley contribution weighting; the Shapley weights determine the
on-chain token split." Drop the MBO/WSA/EfficientNet/ResNet/DenseNet baseline list in
**Phase 6**; replace with "evaluated against FedAvg, FedAvg+Krum, and FedAvg+DP baselines."

---

## 1.6 Scopes and Challenges — fix the DP line (~line 147)

DP is implemented, so move it out of "future directions". Replace the future-directions
sentence with:
```latex
Differential privacy (Gaussian mechanism) is already integrated into the weight-sharing
step; future directions include relaxing the privacy budget with DP-SGD, extending the
consortium beyond three organisations, deploying across live multi-host banking nodes, and
investigating quantum-resistant cryptographic primitives for long-term ledger integrity.
```
