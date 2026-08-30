# REWRITE — Chapter (Proposed Methodology) for Option A, paste-ready

This replaces the inaccurate parts of `chapter_5.tex`. Keep the high-level framing, the
Hyperledger Fabric infrastructure subsection, the off-chain/on-chain boundary discussion, and
the determinism subsection (all accurate). Replace the architecture, the "DB-BOA Job 3"
novelty box, the FedProx claims, and the experimental-config table with the blocks below.

---

## 5.A DB-BOA: two roles (replace "three optimisation roles")

```latex
\textbf{Two Optimisation Roles in This Framework.}
DB-BOA is a hybrid metaheuristic that alternates, via an adaptive switching rule, between the
exploration of the Dynamic Butterfly Optimisation Algorithm (DBOA) and the exploitation of the
Billiards Optimisation Algorithm (BOA). At each iteration $t$ a uniform random draw is compared
to the ratio $\mathrm{bestfit}^{(t)}/\mathrm{worstfit}^{(t)}$; DBOA (with L\'evy-flight LSAM
mutation) is used when the population has converged, BOA otherwise. In this framework DB-BOA
serves two roles:
\begin{itemize}
  \item \textit{Hyperparameter optimisation:} a two-dimensional search over the convolutional
        filter count and the steps-per-epoch that maximises a bounded ADTCN fitness on a
        validation subsample (the epoch count is fixed rather than searched). The fitness
        $\mathrm{Obf}_2 = 2\,\mathrm{MCC} + \mathrm{Spec} + \mathrm{Pre} + \mathrm{NPV}$ is a
        bounded variant of the base paper's objective: the unbounded $1/\mathrm{FPR}$ term is
        replaced by the bounded Specificity (a low-false-positive reward) and MCC is weighted
        to dominate, because the original $1/\mathrm{FPR}$ diverges as $\mathrm{FPR}\to 0$ and
        causes every false-positive-free candidate to tie at an unbounded best score, making
        the search degenerate.
  \item \textit{Leader-block selection:} per consensus round, selecting the validator node that
        minimises $\mathrm{Obf}_1 = \mathrm{CT}+\mathrm{CC}+\mathrm{MS}$ with a reputation
        discount, where CT, CC, MS are computation time, communication cost, and memory size.
\end{itemize}
The federated aggregation weights are \emph{not} computed by DB-BOA; they are obtained from an
exact Shapley-value analysis (Section~\ref{sec:fed-layer}).
```

---

## 5.B ADTCN architecture (replace the MJE/TCL/MTTA "dilated/64-dim/attention" text)

```latex
\subsection*{Adaptive Deep Temporal Context Network (ADTCN)}
The ADTCN detector is a temporal one-dimensional convolutional network applied to a sliding
window of the ten most recent transactions. Each transaction is represented by its 30 base
features (the PCA components V1--V28, the standardised Amount, and Time); an optional three
recurrence features may be appended. The window is processed by two stacked one-dimensional
convolution layers (kernel size 3, ReLU activations) that expand the channel dimension from
$F$ to $2F$, followed by \textbf{global max-pooling over the time axis} and a linear layer that
outputs the normal/fraud logits. The base-paper conceptual blocks map onto this realisation as
follows: \emph{Multi-Modal Joint Embedding} corresponds to the raw multi-feature input,
\emph{Temporal Context Learning} to the stacked convolutions over the ordered window, and
\emph{Multi-Timescale Temporal Attention} to the global max-pooling operator that selects the
most salient time step. Extreme class imbalance (0.17\% fraud) is handled with a class-weighted
cross-entropy loss rather than resampling. The convolutional filter count and steps-per-epoch
are set by DB-BOA; the optimiser is Adam with a learning rate of $10^{-3}$.

Temporal context features (rolling means/standard deviations over windows $\{5,10,20\}$ and
first/second differences) are engineered during preprocessing and retained for completeness;
the convolutional detector derives its temporal context from the ten-step sliding window
itself.
```

> Remove from the report: "64-dimensional latent space", "BatchNorm", "dilated ... rates
> $\{1,2,4,8\}$", "softmax attention", "receptive field $\approx 15$", and the
> Embedding/Attention mathematical-formulation lines that describe attention. The
> max-pool-over-time description above is the accurate one.

---

## 5.C Federated aggregation layer (replace the "DB-BOA Job 3 — PRIMARY NOVEL CONTRIBUTION" box)

```latex
\subsection{Privacy-Preserving, Fairness-Aware Federated Aggregation}
\label{sec:fed-layer}

\begin{center}
\fbox{\begin{minipage}{0.92\textwidth}\vspace{4pt}
\textbf{\textcolor{blue!70!black}{PRIMARY CONTRIBUTION}}\\[2pt]
Each federation round combines three complementary mechanisms -- differentially private weight
sharing, Krum robust selection, and exact Shapley-value contribution attribution -- and binds
the Shapley contribution weights to an on-chain token incentive enforced by Fabric chaincode.
Rather than claiming priority, we \emph{characterise} the privacy--incentive coupling of this
Shapley-driven, chaincode-enforced reward mechanism on a real permissioned blockchain,
extending prior Shapley-on-blockchain incentive work (FedCoin~\cite{fedcoin2020}) with a
deployed Hyperledger Fabric implementation and a noise$\to$fidelity$\to$reward-error analysis.
\vspace{4pt}\end{minipage}}
\end{center}

\paragraph{(1) Differentially private weight sharing.}
Before any sharing, each organisation's weight tensors are $L_2$-clipped to norm $C=1$ and
perturbed by Gaussian noise $\mathcal{N}(0,\sigma^2)$ with
$\sigma = C\sqrt{2\ln(1.25/\delta)}/\epsilon$ (Dwork et al.~\cite{dwork2006}). With
$\epsilon=1.0$, $\delta=10^{-5}$ this gives $\sigma\approx 4.84$ and an $(\epsilon,\delta)$
guarantee; privacy degrades gracefully under basic composition across rounds. As discussed in
the Limitations, $\epsilon=1.0$ is a deliberately tight budget. Contribution~1 (\S6.6) uses an
improved adaptive per-tensor sensitivity variant of this mechanism, setting
$C=\lVert w\rVert_2$ per tensor (Andrew et al.~\cite{andrew2021}) rather than the fixed
$C=1$ used in the deployed pipeline; because the per-element noise-to-signal ratio
$\approx\sqrt{2\ln(1.25/\delta)}\sqrt{\dim}/\epsilon$ is independent of $C$, the privacy/fidelity
transition budget is unchanged, so the two descriptions are consistent.

\paragraph{(2) Krum robust selection.}
Each organisation's update is scored by the sum of squared $L_2$ distances to its nearest
neighbours; the lowest-scoring (most consensus-aligned) update is chosen as the global model
(Blanchard et al.~\cite{blanchard2017}). With the default three-organisation deployment this
reduces to outlier rejection (the largest admissible $f$ under $n\ge 2f+3$ is $0$); we therefore
also evaluate the aggregator in the regime where the Byzantine-tolerance theorem applies --
$n=5,f=1$ and $n=7,f=2$ -- against four weight-level poisoning attacks, where it rejects the
adversarial update in every case (Section~\ref{sec:byzantine-robustness}, Task~D). Statistical
robustness for up to $f$ colluding organisations is thus demonstrated, not deferred; a
coordinated \emph{majority} ($>f$) is handled by the complementary economic mechanism.

\paragraph{(3) Exact Shapley contribution attribution.}
Each organisation $i$ is assigned its Shapley value
$\phi_i = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,[v(S\cup\{i\})-v(S)]$,
where the coalition value $v(S)$ is the balanced accuracy of the equally-averaged model of the
organisations in $S$ on a shared validation set (Wang et al.~\cite{wang2020fedsv}). For $n=3$
this evaluates all seven non-empty coalitions exactly. Negative values are clipped and the
weights normalised to sum to one. Balanced accuracy is used as the coalition value because it
is bounded and robust to the 0.17\% class imbalance, so a participant that catches fraud
without over-flagging scores near one while an always-fraud adversary scores near one half and
lowers any coalition it joins.

\paragraph{Economic coupling.}
The Shapley weights $\mathbf{w}^\ast$ are written to the Fabric ledger, and
\texttt{recordFederationRound} distributes the 20-token federation pool as
$\text{tokens}_i = \lfloor 20\, w_i^\ast \rfloor$. Because reward is the direct output of the
Shapley analysis, an organisation that contributes a stronger model earns more automatically,
and one that degrades the global model earns essentially nothing -- without human adjudication.
```

---

## 5.D Token incentive table (keep, fix one row)
The token table is accurate; only change the federation row to:
```latex
Federation participation (20-token pool, split by Shapley weight) & $+\lfloor 20 w_i^\ast\rfloor$ & \texttt{recordFederationRound} \\
```

---

## 5.E Experimental configuration table (replace the FedProx/Job-3 rows)
```latex
\begin{table}[H]\centering
\begin{tabular}{|p{4.2cm}|p{10.8cm}|}
\hline \textbf{Parameter} & \textbf{Description} \\ \hline
Dataset & ULB Credit Card Fraud (284{,}807 transactions; V1--V28, Amount, Time; 0.17\% fraud) \\ \hline
Train/val/test split & 80\% / 10\% / 10\%, stratified \\ \hline
Sequence length & 10 transactions (sliding window) \\ \hline
Consortium & 3 banks (BankA 50\%, BankB 30\%, BankC 20\%), stratified shards \\ \hline
Detector & ADTCN: 2-layer 1D-CNN ($F\!\to\!2F$, kernel 3) + global max-pool + linear head \\ \hline
Aggregation & DP (Gaussian, $\epsilon=1.0$, $\delta=10^{-5}$) + Krum + exact Shapley (7 coalitions) \\ \hline
Incentive split & 20-token federation pool by Shapley weight \\ \hline
DB-BOA roles & ADTCN hyperparameters (2-D) and consensus leader selection \\ \hline
Frameworks & PyTorch (CPU); Hyperledger Fabric 2.5 (CCaaS chaincode) \\ \hline
Metrics & Accuracy, Precision, Sensitivity, Specificity, NPV, FPR, FNR, F1, MCC, ROC-AUC \\ \hline
\end{tabular}
\caption{Experimental configuration for the FL-ADTCN consortium evaluation}
\end{table}
```

> Delete from `chapter_5.tex`: the "FedProx $\mu=0.05$" claims, the FedProx
> $\mu$-sensitivity table, and "FedProx + DB-BOA Job 3 weights" in the config table.
> Replace the Implementation subsection's "FedProx regularisation was incorporated" with
> "stratified partitioning preserves the global fraud rate in each shard."

---

## 5.F Data Collection subsection (keep — already accurate)
The "Data Collection" subsection that describes the ULB dataset (284,807 / 0.17%) and the
stratified 50/30/20 split is correct and should stay. Just ensure the development-environment
table lists `fabric-network` 2.2 (not `fabric-gateway` 1.4) and PyTorch on CPU unless a GPU
was actually used.
```
