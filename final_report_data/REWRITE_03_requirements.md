# REWRITE — Chapter 3 Requirements, Impacts, Constraints (Option A)

Keep the Societal / Environmental / Ethical / Standards / Economic sections (they are
sound). Apply these targeted edits.

## 3.1 Final Specifications — replace the DB-BOA-roles paragraph
**Old:** "...DB-BOA is applied across three distinct roles: (1) leader block selection ... (2)
ADTCN hyperparameter optimization ... (3) per-organization aggregation weight computation..."

**New:**
```latex
The Dynamic Butterfly--Billiards Optimisation Algorithm (DB-BOA) is applied to two tasks:
(1) ADTCN hyperparameter optimisation -- tuning the convolutional filter count and
steps-per-epoch; and (2) consensus leader-node selection -- minimising computation time,
communication cost, and memory size with a reputation discount. The federated aggregation
layer is handled separately by three dedicated mechanisms: differentially private weight
sharing (Gaussian mechanism), Krum robust model selection, and exact Shapley-value
contribution attribution, with the Shapley weights driving the on-chain token incentive.
```

## 3.2 ADTCN description — fix the architecture sentence
Replace "comprising Multi-modal Joint Embedding (MJE), Temporal Context Learning (TCL), and
Multiple Time-scale Temporal Attention (MTTA)" wherever it implies attention/embedding layers
with:
```latex
Adaptive Deep Temporal Context Networks (ADTCN) -- realised as a temporal one-dimensional
convolutional network over a sliding window of recent transactions, with global max-pooling
over time and a linear fraud/normal classifier -- serve as the core detection model.
```

## 3.3 Token incentive line — fix the federation pool
Replace "(+20 tokens shared by DB-BOA weight)" with
"(+20 tokens shared in proportion to the Shapley contribution weights)".

## 3.4 Software \& Resource Requirements — fix the bullets
```latex
\item \textbf{Machine Learning Implementation:} Python 3 with \textbf{PyTorch} for the ADTCN
      detector and \textbf{NumPy/scikit-learn} for DB-BOA, the DP/Krum/Shapley federation
      layer, and metrics.
\item \textbf{Blockchain Infrastructure:} \textbf{Hyperledger Fabric 2.5} (Docker test-network)
      with the \textbf{\texttt{fabric-network} 2.2 Node.js SDK} (wallet-based) for the
      application layer.
\item \textbf{Smart Contract Logic:} \textbf{Node.js chaincode} (\texttt{DBBOAContract})
      using \texttt{fabric-contract-api}/\texttt{fabric-shim} 2.5, deployed as a
      Chaincode-as-a-Service on the consortium peers.
\item \textbf{Dataset:} the public \textbf{ULB Credit Card Fraud} dataset
      (284{,}807 transactions; V1--V28, Amount, Time; 0.17\% fraud), partitioned 50/30/20
      across the three consortium banks by stratified sampling.
\item \textbf{Development Environment:} Python 3, Node.js 18, and Docker for the Fabric network.
```
(Remove "synthetically generated ... 20,000 samples ... 5\% fraud" and the
`@hyperledger/fabric-gateway` reference.)

## 3.5 Baseline list — wherever the evaluation baselines are listed
Replace "MBO-ADTCN, WSA-ADTCN, DBOA-ADTCN, BOA-ADTCN, EfficientNet, ResNet, DenseNet, DTCN"
with "FedAvg, FedAvg+Krum, and FedAvg+DP".

## 3.6 Risk Management table (referenced but currently missing a body — add it)
```latex
\begin{table}[!htbp]
\centering
\caption{Key risks and mitigations}
\label{tab:risk_management}
\begin{tabular}{p{4cm}p{4.5cm}p{5.5cm}}
\toprule
\textbf{Risk} & \textbf{Impact} & \textbf{Mitigation (as implemented)} \\
\midrule
Model poisoning by an organisation & Corrupted global model & Krum consensus-aligned selection
+ Shapley down-weighting + on-chain token/reputation penalty \\
Privacy leakage via shared weights & Re-identification risk & Differentially private
(Gaussian-mechanism) weight sharing \\
Non-deterministic chaincode & Endorsement mismatch / consensus failure & All optimisation and
randomness kept off-chain; only finalised results written on-chain \\
Extreme class imbalance (0.17\%) & Trivial all-normal classifier & Weighted cross-entropy
loss; MCC and ROC-AUC as primary metrics \\
Single-source dataset & Limited ecological validity & Stated as a limitation; future work uses
real multi-bank streams \\
\bottomrule
\end{tabular}
\end{table}
```
