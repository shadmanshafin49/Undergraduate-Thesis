# REWRITE — Chapter 6 Result Analysis (from the fresh, fixed-objective full run)

All §6.1/6.3/6.4/6.5 numbers come from a single real full-quality run produced by the **current,
fixed-objective code** (`python3 main.py --attack`, 7364 s, 2026-06-08; `results/db_boa_results.json`)
on the ULB dataset — NOT the fabricated tables in the current chapter_6.tex. (An earlier pre-fix
run that used the degenerate `1/FPR` objective has been removed from the repo as superseded; the
fixed-objective run is the single source of truth.) The §6.2 ablation is the saved
`results/baselines.json`. Delete the old fabricated
tables (8-model base-paper comparison, t-tests, 85 TPS, 28/18/4 leader split, 458/312/178 tokens,
activation comparison) per `00_report_vs_code_divergences.md`.

> Status: all sections below are filled from one consistent run. Headline detector MCC 0.677
> (weight-DP off); federation Shapley split is near-uniform (single-source data); the Byzantine
> attack penalises the attacker every round (tokens 100→70, reputation→0.5, Shapley weight 0).

---

## 6.1 Centralised ADTCN performance (VERIFIED — full test set, n = 56,962)

```latex
\begin{table}[H]\centering
\caption{Centralised ADTCN fraud-detection performance on the ULB test set (56,962 transactions, 98 fraud)}
\label{tab:centralized_results}
\begin{tabular}{|l|r|}
\hline \textbf{Metric} & \textbf{Value} \\ \hline
Accuracy        & 99.85\% \\ \hline
Precision       & 54.25\% \\ \hline
Sensitivity (Recall) & 84.69\% \\ \hline
Specificity     & 99.88\% \\ \hline
NPV             & 99.97\% \\ \hline
FPR             & 0.12\% \\ \hline
FNR             & 15.31\% \\ \hline
F1-score        & 66.14\% \\ \hline
MCC             & 0.677 \\ \hline
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Confusion matrix --- centralised ADTCN (test set)}
\begin{tabular}{|l|r|r|}
\hline & \textbf{Predicted Normal} & \textbf{Predicted Fraud} \\ \hline
\textbf{Actual Normal} & 56{,}794 (TN) & 70 (FP) \\ \hline
\textbf{Actual Fraud}  & 15 (FN)       & 83 (TP) \\ \hline
\end{tabular}
\end{table}
```

**Interpretation paragraph (paste):**
```latex
On the held-out test set of 56{,}962 transactions (98 fraudulent), the DB-BOA-tuned ADTCN
detects 83 of 98 frauds (84.7\% recall) at a false-positive rate of 0.12\%, yielding an MCC of
0.677. Because the data are extremely imbalanced (0.17\% fraud), accuracy alone is
uninformative -- a trivial ``predict normal'' classifier already reaches 99.83\% -- so MCC and
the confusion matrix are the meaningful indicators. The 70 false positives against 83 true
positives reflect the conservative operating point appropriate to a first-stage fraud screen,
where flagged transactions are passed to secondary review rather than blocked outright.

These detection metrics are reported at the \textbf{operating point used for deployment}, in
which differentially private \emph{weight} sharing is disabled ($\epsilon=\infty$ on the
parameter channel). This is a deliberate design decision, not a convenience: as the ablation in
\S6.2 and the characterisation in \S6.6 show, the deployment budget $\epsilon=1.0$ adds Gaussian
weight noise ($\sigma\approx 4.84$ at $C=1$) that exceeds the weight magnitudes and near-randomises
the global model, so any DP-on detection number would reflect noise rather than the detector.
Privacy in the deployed system is instead carried by the \emph{output} (incentive) channel of
\S6.6 (Contribution~1): differential privacy is applied to the published Shapley contribution
weights, where it protects what is actually written to the immutable ledger, rather than to the
shared model parameters where it destroys utility. The whole point of that result is that
weight-channel DP is \emph{not} required to obtain private incentives; we therefore report
detection with the weight channel clear and privacy enforced where it is both meaningful and
affordable. The full DP-on weight-channel collapse is reported transparently in \S6.2 and is the
characterised cost, not a hidden caveat.
```

> Note: these are honest numbers for a lightweight 1-D CNN on raw PCA features. Do **not**
> replace them with the base paper's 95.45\% or the old 99.31\%/0.9249 -- those are not from
> this implementation.

---

## 6.2 Federated ablation (VERIFIED baselines) — run `run_baselines.py`

The real comparison is FedAvg / FedAvg+Krum / FedAvg+DP / proposed (Krum+DP+Shapley), produced
by `run_baselines.py` (three-bank federation on the ULB test set, $n=56{,}962$; DB-BOA-tuned
ADTCN, $\epsilon=1.0$, $\delta=10^{-5}$). The verified metrics (replaces the fabricated 8-model
table):
```latex
\begin{table}[H]\centering
\caption{Federated ablation on the ULB dataset (three-bank consortium, test set $n=56{,}962$).
The two DP-enabled rows use the deliberately tight deployment budget $\epsilon=1.0$.}
\label{tab:fed_ablation}
\begin{tabular}{|p{4.6cm}|r|r|r|r|}
\hline \textbf{Configuration} & \textbf{Accuracy} & \textbf{MCC} & \textbf{FPR} & \textbf{NPV} \\ \hline
FedAvg                         & 99.72\% & 0.569 & 0.26\%  & 99.98\% \\ \hline
FedAvg + Krum                  & 99.92\% & \textbf{0.776} & 0.05\%  & 99.97\% \\ \hline
FedAvg + DP ($\epsilon=1.0$)   & 99.83\% & 0.000 & 0.00\%  & 99.83\% \\ \hline
\textbf{Proposed (Krum+DP+Shapley, $\epsilon=1.0$)} & 0.21\% & 0.001 & 99.96\% & 100.00\% \\ \hline
\end{tabular}
\end{table}
```

> Source: `results/baselines.json` (saved by `run_baselines.py`; full DB-BOA search, $\epsilon=1.0$,
> $\delta=10^{-5}$, optimal params 142/76). The two DP rows collapse to a \emph{single-class}
> (degenerate) predictor with $\text{MCC}\approx 0$; the exact failure \emph{direction} is set by the
> random Gaussian noise draw, so it varies run-to-run (here FedAvg+DP collapses to predict-all-normal,
> the proposed pipeline to predict-all-fraud). Report the collapse as MCC$\approx 0$ rather than
> quoting a precise (noise-draw-specific) accuracy.

**Interpretation paragraph (paste):**
```latex
The ablation isolates the cost of each layer. Adding \textbf{Krum} to FedAvg is purely
beneficial here: by selecting the single most consensus-aligned update rather than averaging, it
lifts MCC from $0.569$ to $0.776$ and cuts the false-positive rate to $0.05\%$, the strongest
configuration in the table. The two \emph{differentially private} configurations, however,
both collapse to a degenerate single-class predictor ($\text{MCC}\approx 0$) at the deployment
budget $\epsilon=1.0$: the Gaussian weight noise so dominates the signal that each aggregated model
predicts a single class for every transaction. The \emph{direction} of the collapse is set by the
random noise draw and so varies run-to-run --- in the saved run FedAvg+DP collapses to
predict-all-normal ($\text{FPR}=0\%$, recall $0\%$) while the proposed Krum+DP+Shapley pipeline
collapses to predict-all-fraud ($\text{FPR}\approx 100\%$) --- but in every case $\text{MCC}\approx
0$, i.e.\ no better than chance. This is not an implementation defect but the direct, expected
consequence of the privacy/utility trade-off characterised in \S6.6: at $\epsilon=1.0$ the Gaussian
weight noise ($\sigma\approx 4.84$ at $C=1$) overwhelms the signal, and Krum does not help because
selecting a single noised model forgoes the partial noise-cancellation that averaging provides. The result
therefore reinforces the thesis's central finding --- $\epsilon=1.0$ is far below the usable
budget (\S6.6 places the incentive-fidelity threshold at $\epsilon^\star=3000$) --- and motivates
the DP-SGD / relaxed-$\epsilon$ future work. For a usable end-to-end detector the privacy budget
must be relaxed or DP applied at the gradient level rather than to the shared weights.
```

> Honest note: the IID/non-IID split promised in the abstract is the stratified vs.\ volume-skewed
> shard partition used throughout; if a reviewer wants an explicit IID-vs-non-IID row, re-run
> `run_baselines.py` with the non-IID partitioner and add a second block. The collapse of the two
> $\epsilon=1.0$ rows is real and consistent with \S6.6 --- present it as evidence, not as a
> failure to hide.

---

## 6.3 Shapley contribution weights (VERIFIED — deployment run, weight-DP off, 3 federation rounds)

```latex
\begin{table}[H]\centering
\caption{Per-round exact Shapley contribution weights at the deployment operating point
(three-bank federation, weight-channel DP off). All three banks are stratified volume-splits of the
same ULB dataset, so their marginal contributions are near-identical and the split is near-uniform.}
\label{tab:shapley_weights}
\begin{tabular}{|c|r|r|r|l|l|}
\hline
\textbf{Round} & \textbf{BankA} & \textbf{BankB} & \textbf{BankC} & \textbf{Krum selects} & \textbf{Weight DP} \\ \hline
1 & 0.336 & 0.332 & 0.332 & BankA & off \\ \hline
2 & 0.333 & 0.333 & 0.333 & BankA & off \\ \hline
3 & 0.333 & 0.333 & 0.333 & BankA & off \\ \hline
\end{tabular}
\end{table}
```

**Interpretation (paste):**
```latex
At the deployment operating point (weight-channel DP off, \S6.1), exact Shapley assigns each bank
its marginal contribution to the global model's balanced accuracy on a shared validation set, and
the result is a \emph{near-uniform} split ($\approx 0.333$ each, token allocation $\approx 6.67$ of
the 20-token pool). This is the honest and \emph{correct} behaviour of the mechanism on this data:
the three ``banks'' are stratified volume-splits of a single ULB dataset
(\S\,Limitations), and -- because the pipeline performs no inter-round local training -- every
single-bank and coalition model evaluates to essentially the same balanced accuracy
($v(S)\approx 0.50$ for all coalitions $S$), so no bank has a larger marginal contribution than
another. Exact Shapley therefore neither invents spurious differentiation nor collapses any bank to
zero; it returns the equal split that near-identical contributors deserve, replacing FedAvg's
\emph{assumed} equal weighting with a \emph{measured} one. The corollary is important and stated
plainly: on genuinely identical contributors the contribution signal carries little information,
so the \emph{value} of Shapley attribution -- and the privacy$\to$fidelity$\to$reward-error coupling
that is this thesis's central contribution -- is demonstrated where contributions actually differ,
by inducing a real heterogeneity gradient (training-label noise BankA$<$BankB$<$BankC) in the
privacy--incentive characterisation of \S6.6 and the private-incentive mechanism of \S6.B (B1),
not on the single-source deployment split shown here.
```

> Honest note: this run is the weight-DP-off deployment configuration, so there is no per-round
> $\epsilon$ to report here; the privacy budget enters only on the incentive (output) channel of
> \S6.6/B1. The near-uniform split is a direct, disclosed consequence of single-source data plus
> no-inter-round-local-training (both in Limitations) -- present it as the mechanism behaving
> correctly on identical contributors, and lean on \S6.6/B1 for the case where contributions differ.

---

## 6.4 Byzantine resilience (VERIFIED — `--attack`, BankC always reports fraud, 15 rounds)

```latex
\begin{table}[H]\centering
\caption{Byzantine attack: BankC always reports fraud (15 rounds)}
\label{tab:attack}
\begin{tabular}{|l|r|}
\hline \textbf{Outcome} & \textbf{Value} \\ \hline
Rounds with verdict disputed by consensus & 15 / 15 \\ \hline
Attacker reputation (start $\to$ end)    & 1.00 $\to$ 0.50 (floor, reached round 10) \\ \hline
Attacker token balance (start $\to$ end) & 100 $\to$ 70 \\ \hline
\textbf{Attacker Shapley aggregation weight under attack} & \textbf{0.000} \\ \hline
\end{tabular}
\end{table}
```

**Interpretation (paste):**
```latex
Under the Byzantine scenario both defence layers penalise the attacker, and they do so
consistently. BankC issues an always-fraud verdict on legitimate transactions, so the honest
consensus (which correctly predicts ``normal'') \emph{disputes} it in all 15 rounds; the incentive
contract debits the attacker two tokens per disputed round, draining its balance from 100 to 70,
and its reputation decays by the per-dispute step to the $0.50$ floor by round~10. In parallel, the
exact-Shapley aggregation layer assigns the poisoned update a contribution weight of $0.000$, so the
attacker's model is excluded from the global aggregate without any human intervention. The two
mechanisms are complementary: the reputation/token loop makes the attack economically unrewarding
(monotone token drain, reputation floored), while the Shapley layer makes it technically
ineffective (zero aggregation weight). We are careful not to over-read a single-attacker
illustration: as the systematic economic sweep in \S6.7 shows, a \emph{lone} attacker is in any
case out-voted by the two honest banks and does not need to be formally isolated from the quorum --
reputation-floor \emph{quorum isolation} only becomes decisive against a colluding \emph{majority}
(\S6.7). What this run demonstrates is the end-to-end mechanism firing correctly on the headline
pipeline: dispute detection $\to$ token penalty $\to$ reputation decay $\to$ zero Shapley weight.
```

---

## 6.5 DB-BOA hyperparameter optimisation vs. default (VERIFIED — single full run, fixed objective)

The objective degeneracy was fixed (the unbounded `1/FPR` term, which exploded to ~1e8 whenever a
candidate reached FPR=0 and made the search unable to rank configurations, was replaced by a
bounded, MCC-dominated fitness `2·MCC + Spec + Pre + NPV`). The headline run and this comparison are
produced by that **current, fixed-objective code in a single full run** (`main.py --attack`, 7364 s;
`db_boa_results.json`), so there is no longer any pre-fix/post-fix mixing of runs:

| Config | Filters / steps | Accuracy | MCC |
|--------|-----------------|----------|-----|
| Hand-set default | 128 / 150 | 99.92% | **0.785** |
| DB-BOA-tuned | 142 / 76 | 99.85% | 0.677 |

**Honest finding:** with the corrected (non-degenerate) objective, the DB-BOA hyperparameter
search did **not** beat the hand-set default on this dataset -- the default is ~0.11 MCC higher
(0.785 vs 0.677). The objective fix was still necessary and correct -- it makes the search rank
configurations meaningfully rather than tie everything at a `1/FPR` ceiling -- but it does not, on
its own, make DB-BOA the source of detection accuracy here. We report this gap openly rather than
hide it.

**Why (defensible explanation for the discussion):** the test set contains only 98 fraudulent
transactions, so MCC is sensitive to a handful of TP/FN flips, but a 0.11 gap is a real difference,
not pure noise -- we do **not** claim the two are statistically indistinguishable without a
multi-seed study. The more honest explanation is that the fast validation surrogate used inside the
search (a 2{,}000-row subsample trained for 5 epochs) is a weak proxy for the fully trained model,
especially for steps-per-epoch (which sets the batch size and does not transfer across dataset
scale), and that the bounded fitness landscape is genuinely flat near a well-chosen configuration
(`db_boa_stats`: Min = Max = Mean = $-5.0$, Std $\approx 0$ -- see below). DB-BOA's demonstrated
value is therefore \emph{automation} (selecting a working detector without manual search) and
\emph{leader selection}, not a detection-accuracy gain over a careful hand-tune.

**Recommended framing (honest) — DECIDED:**
- Present the **DB-BOA-tuned model (MCC 0.677)** as the single headline detector everywhere
  (§6.1, §6.5, abstract, conclusion), with the honest note that it is a working detector selected
  **automatically (no manual search)** that **does not exceed, and slightly trails, a hand-tuned
  default** on this dataset. Match the named framework, but do **not** dress the gap up as
  "within variance," and do **not** quote 0.785 as the headline.
- Describe DB-BOA's demonstrated contributions as **(i) automated consensus leader selection**
  (its clear, deterministic win) and **(ii) automated hyperparameter tuning without manual search**
  -- explicitly **not** as a source of accuracy gains.
- Do **not** claim "DB-BOA improves detection accuracy over the default" -- the evidence says the
  opposite (0.677 vs 0.785).
- Optional rigour: a 3-seed mean$\pm$std study would let you quantify whether any of the 0.11 gap is
  noise; until then, report it as a real gap (recommended if a reviewer presses on this).
- **Do not present the optimiser's own fitness statistics as evidence of a hard search.** In
  `db_boa_results.json` the `db_boa_stats` block reports Min $=$ Max $=$ Mean $= -5.0$ with
  Std $\approx 0$ \emph{from the first iteration}: the population is pinned at the
  objective ceiling immediately, so the Min/Max/Mean/Std table demonstrates a \emph{flat} fitness
  landscape, not a meaningful search. Report the convergence curve honestly --- it converges in
  one iteration \emph{because} the landscape is saturated at the top, and say so --- and lead the
  DB-BOA value argument with the Phase-4 side-by-side (DB-BOA-optimal vs default hyperparameters)
  and the deterministic leader-selection win (\S on RL / leader selection), not with the
  saturated statistics table.

> **Paste sentence for the discussion:**
> ```latex
> The DB-BOA fitness statistics ($\mathrm{Min}=\mathrm{Max}=\mathrm{Mean}=-5.0$,
> $\mathrm{Std}\approx 0$ from the first iteration) indicate that the bounded
> objective surface is flat at its ceiling around a well-chosen configuration: the optimiser
> reaches the maximum immediately and stays there. We therefore do not present this convergence as
> evidence of a hard search problem; the demonstrated value of DB-BOA is the automation of
> hyperparameter and leader selection (without manual search, though it does not beat the hand-set
> default on detection accuracy), as quantified by the Phase-4 side-by-side comparison, rather than
> the discovery of a non-obvious optimum.
> ```

The DB-BOA objective change is documented in `REWRITE_05_methodology.md` (bounded $\mathrm{Obf}_2$).

---

## 6.6 Privacy–incentive characterisation: when do on-chain rewards stay honest? (CONTRIBUTION 1)

> Source data: `db_boa_framework/experiments/privacy_incentive_sweep.py`
> (full run, 12 epochs, **50 noise draws/ε** to stabilise the low-$\varepsilon$ region); raw
> drafts in `TASKA_privacy_incentive_results.md`; figures
> `results/privacy_incentive_tradeoff.png`, `results/privacy_incentive_reward_bars.png`.
> Framing: **characterisation novelty, not new-algorithm novelty.** The novel
> coupling is *noise → Shapley fidelity → on-chain reward error* — the reverse of
> the closest prior FedSDP/FedSVA (arXiv 2503.12958), which runs Shapley → noise.

This experiment sweeps the DP weight-sharing budget $\varepsilon$ and measures how the
Gaussian noise that buys privacy corrupts the Shapley contribution signal that drives the
on-chain token split (tokens $=20\,w_i$ in chaincode `recordFederationRound`). Three banks
are given a genuine contribution gradient via training-label noise (BankA 0\%, BankB 10\%,
BankC 25\%), so the honest ($\varepsilon=\infty$) Shapley ordering is non-degenerate:
weights $[0.619, 0.250, 0.131]$, i.e. tokens $[12.38, 5.00, 2.62]$. To make the mechanism
converge to this honest split as $\varepsilon\to\infty$, the DP layer uses adaptive per-tensor
sensitivity ($C=\lVert w\rVert_2$, Andrew et al. NeurIPS 2021); the per-element
noise-to-signal ratio $\approx\sqrt{2\ln(1.25/\delta)}\sqrt{\dim}/\varepsilon$ is independent of
$C$, so the transition budget is unchanged.

```latex
\begin{table}[H]\centering
\caption{Privacy budget vs incentive fidelity (3-bank federation, mean over 50 noise draws).
$\rho$ = Spearman rank-correlation of the DP reward ordering against the honest $\varepsilon=\infty$ ordering.}
\label{tab:privacy_incentive}
\begin{tabular}{|r|r|r|r|r|r|}
\hline
$\varepsilon$ & Global bal-acc & cosine & Spearman $\rho$ & Token error & Rank-inversion \\
              & (\%)           & ($w_{DP},w_\infty$) & (reward rank) & (of 20 pool) & rate \\ \hline
3      & 49.59 & 0.591 & $+0.007$ & 21.04 & 80\% \\ \hline
10     & 49.74 & 0.544 & $-0.060$ & 22.53 & 78\% \\ \hline
30     & 50.05 & 0.464 & $-0.232$ & 25.83 & 78\% \\ \hline
100    & 52.07 & 0.504 & $-0.144$ & 24.57 & 80\% \\ \hline
300    & 78.49 & 0.923 & $+0.739$ & 12.64 & 42\% \\ \hline
1000   & 88.83 & 0.953 & $+0.790$ & 8.60  & 42\% \\ \hline
3000   & 90.75 & 0.989 & $+0.947$ & 3.34  & 10\% \\ \hline
10000  & 90.78 & 0.999 & $+1.000$ & 1.01  & 0\%  \\ \hline
$\infty$ & 90.78 & 1.000 & $+1.000$ & 0.00 & 0\% \\ \hline
\end{tabular}
\end{table}
```

**Interpretation paragraph (paste):**
```latex
The privacy and incentive layers are in direct tension. At the privacy budgets normally called
``private'' for this task ($\varepsilon\le 100$), the Gaussian weight noise destroys the Shapley
contribution signal: averaged over 50 noise draws the reward ordering is uncorrelated to inverted
relative to the honest split (Spearman $\rho\approx 0$ to $-0.23$; the $\varepsilon=3$ point is
$+0.007$, statistically indistinguishable from zero), and $21$--$26$ of the $20$-token pool is
mis-allocated on an immutable ledger. This region is the \emph{noise floor}: with 50 draws the
low-$\varepsilon$ correlations are stable and never rise meaningfully above zero, so we make the
clean claim only for $\varepsilon\ge 300$. As $\varepsilon$ grows the global detector recovers
first --- balanced accuracy climbs from $\sim$$50\%$ back to $78\%$ by $\varepsilon\approx 300$
and $89\%$ by $\varepsilon=1000$ --- but incentive fairness lags far behind: at $\varepsilon=1000$
the model is effectively recovered (balanced accuracy $88.8\%$) yet $42\%$ of reward orderings are
still inverted ($\rho=0.79$), and only at $\varepsilon\gtrsim 3000$ do rewards approach honesty
($\rho=0.95$, $10\%$ inversions), with the fully trustworthy regime ($\rho=1.0$, zero inversions)
reached only at $\varepsilon\ge 10000$. We define $\varepsilon^\star=3000$ as the largest budget
at which the on-chain reward order still flips. The key finding is a \emph{decoupling}: accuracy
and incentive-fairness have different privacy thresholds, so a DP-FL system can publish an
accurate global model while paying provably wrong, permanent token rewards. The contribution
signal (a differential of near-equal coalition scores) is intrinsically more fragile to noise
than the aggregate model.
```

Figure~\ref{fig:privacy_tradeoff} (the 3-panel `privacy_incentive_tradeoff.png`) shows the
balanced-accuracy, Shapley-fidelity and token-error curves with $\varepsilon^\star$ marked; the
reward-bar figure shows the BankA$\to$BankC reward ordering inverting at $\varepsilon=10$ and
being restored only by $\varepsilon\ge 10000$.

---

## 6.B Private incentive mechanism: moving privacy off the weight channel (CENTRAL CONTRIBUTION, B1)

> Source data: `db_boa_framework/experiments/private_incentive_sweep.py`
> (full run, 12 epochs, **100 noise draws/$\varepsilon$**, 3 orgs, $d=111{,}874$ weights);
> raw draft `TASKB1_private_incentive_results.md`; figure `results/private_incentive_channel.png`;
> JSON `results/private_incentive_sweep.json`.
> Framing: this is the **central contribution** — it turns the §6.6 negative result into a
> built, positive mechanism. **Characterisation + composition of published parts** (Gaussian
> mechanism, Dwork 2006; output perturbation, Chaudhuri 2011; on-chain Shapley incentive,
> FedCoin 2020), **not** a new privacy primitive.

Section~6.6 established a negative result: at any practical privacy budget, adding DP to the
\emph{weight} channel destroys the Shapley contribution signal and pays provably wrong on-chain
rewards. This section shows that the failure is a property of the \emph{channel}, not of
DP-plus-Shapley itself. The incentive split needs only the $n$-dimensional contribution vector
$\varphi$, not the $\sim$$10^{5}$-dimensional weights. We therefore compute Shapley on the
\emph{clean} models and apply the privacy noise directly to the released contribution statistic
(\emph{output perturbation}: clip $\lVert\varphi\rVert_2\le C$, add Gaussian
$\sigma=C\sqrt{2\ln(1.25/\delta)}/\varepsilon$). The per-element noise-to-signal ratio scales as
$k\sqrt{\dim}/\varepsilon$, so moving from $\dim=d\approx111{,}874$ weights to $\dim=n=3$
contributions improves the budget by up to $\sqrt{d/n}\approx 193\times$ in the worst case.

```latex
\begin{table}[H]\centering
\caption{Head-to-head privacy of the on-chain reward split: differential privacy on the
weight channel ($d=111{,}874$) versus on the released contribution channel ($n=3$), mean over
100 noise draws/$\varepsilon$. Ground-truth honest split (induced by a BankA$<$BankB$<$BankC
training-label-noise gradient) is weights $[0.619,0.250,0.131]$, tokens $[12.38,5.00,2.62]$ of a
20-token pool. $\rho$ = Spearman rank-correlation of the DP reward order against honest.}
\label{tab:private_incentive}
\begin{tabular}{|r|r|r|r|r|r|r|}
\hline
& \multicolumn{3}{c|}{\textbf{Weight channel}} & \multicolumn{3}{c|}{\textbf{Output channel (ours)}} \\ \hline
$\varepsilon$ & Inversion & Token err & $\rho$ & Inversion & Token err & $\rho$ \\ \hline
1    & 75\% & 19.23 & $+0.06$ & 67\% & 20.49 & $+0.15$ \\ \hline
5    & 78\% & 20.04 & $+0.01$ & 65\% & 15.53 & $+0.32$ \\ \hline
10   & 78\% & 21.04 & $-0.05$ & 54\% & 10.90 & $+0.55$ \\ \hline
30   & 84\% & 25.42 & $-0.25$ & 21\% & 4.62  & $+0.90$ \\ \hline
50   & 81\% & 26.71 & $-0.29$ & 10\% & 2.89  & $+0.95$ \\ \hline
100  & 82\% & 24.83 & $-0.27$ & 0\%  & 1.43  & $+1.00$ \\ \hline
300  & 65\% & 12.20 & $+0.65$ & 0\%  & 0.48  & $+1.00$ \\ \hline
1000 & 57\% & 8.37  & $+0.71$ & 0\%  & 0.14  & $+1.00$ \\ \hline
3000 & 20\% & 3.20  & $+0.90$ & 0\%  & 0.05  & $+1.00$ \\ \hline
\end{tabular}
\end{table}
```

**Interpretation paragraph (paste):**
```latex
Relocating the privacy noise from the weight channel to the released contribution vector moves
the budget at which on-chain rewards stay rank-faithful from $\varepsilon^\star=3000$ (weight) to
$\varepsilon^\star=50$ (output) --- the largest budget at which any reward order still flips ---
a $60\times$ improvement that lands honest, DP-protected incentives inside the practical
differential-privacy regime ($\varepsilon\approx 10$--$50$) for the first time. On the output
channel the reward ordering is already positively correlated at $\varepsilon=10$ ($\rho=0.55$),
rank-faithful with zero inversions by $\varepsilon=100$, and its token error decays monotonically
to under one token of the 20-token pool, whereas the weight channel is still inverting four of
five orderings at the same budgets. The realised $60\times$ is below the $\sqrt{d/n}\approx 193\times$
dimensional ceiling, as expected: $\varepsilon^\star$ is read off discrete sweep points and is set
by the finite inter-organisation signal gaps, not by the worst-case bound. We keep the boundary
honest in three ways. First, at $\varepsilon=1$ \emph{both} channels still fail (output
noise-to-signal $\approx k\sqrt{n}\approx 8$ exceeds the inter-org gaps), so the mechanism buys
roughly two orders of magnitude of budget, not unconditional privacy. Second, the privacy unit is
explicitly the \emph{released contribution statistic} --- the quantity written to the immutable
ledger --- and is decoupled from model-weight privacy, which remains the separate, off-by-default
weight-DP channel of §6.6. Third, the honest ordering is the constructed label-noise gradient
(disclosed in Limitations), as in §6.6. The result is the thesis's central positive contribution:
the privacy$\leftrightarrow$incentive incompatibility of §6.6 is not intrinsic to combining
differential privacy with Shapley incentives but an artefact of privatising the wrong channel,
and output perturbation on $\varphi$ restores private, rank-faithful, chaincode-enforced rewards.
```

Figure (`private_incentive_channel.png`) overlays the weight- and output-channel inversion-rate,
token-error and Spearman-$\rho$ curves against $\varepsilon$, with $\varepsilon^\star$ marked on
each channel ($3000$ vs $50$).

---

## 6.7 Economic Byzantine tolerance: incentive-as-defense under $f=0$ Krum (CONTRIBUTION 2)

> Source data: `db_boa_framework/experiments/economic_byzantine_sweep.py`
> (full run, 12 epochs, 12 rounds); raw drafts in
> `TASKB_economic_byzantine_results.md`; figures
> `results/economic_isolation_trajectory.png`, `results/economic_accuracy_protection.png`.
> Framing: applied characterisation on the real stack — **not** a new mechanism.
> All accuracy is **balanced** accuracy (raw accuracy is uninformative at 0.17\% fraud).

In the default $n=3$ deployment Krum runs at $\texttt{byzantine\_f}=0$ (the largest $f$ admissible
under $n\ge 2f+3$), so at that size it provides outlier rejection rather than a statistical BFT
guarantee; the guarantee itself is demonstrated separately in the $n\ge 2f+3$ regime in
Section~\ref{sec:byzantine-robustness}. Here we instead test the *complementary* question: whether
the *economic* loop
(Shapley $\to$ tokens $\to$ reputation $[0.5,2.0]$ $\to$ leader-selection exclusion) supplies a
complementary, economic Byzantine resilience. Attackers are behavioural (prediction overrides,
the path the Shapley coalition evaluator was built for): always-fraud, label-flip, and a
passive free-rider.

```latex
\begin{table}[H]\centering
\caption{Economic isolation and consensus-accuracy protection ($n=3$, balanced accuracy).
``WITH'' drops reputation-floored attackers from the voting quorum; ``WITHOUT'' lets all orgs vote.}
\label{tab:economic_byzantine}
\begin{tabular}{|l|c|c|r|r|r|}
\hline
Strategy & \#Atk & Isolation round & Bal-acc WITH & WITHOUT & Gap \\ \hline
always-fraud & 1 & never  & 92.29\% & 92.29\% & $+0.00$ \\ \hline
always-fraud & 2 & r3, r3 & 90.78\% & 50.00\% & $+40.78$ \\ \hline
label-flip   & 1 & never  & 91.78\% & 91.78\% & $+0.00$ \\ \hline
label-flip   & 2 & r3, r3 & 90.78\% & 11.26\% & $+79.52$ \\ \hline
free-rider   & 1 & never  & 90.80\% & 90.80\% & $+0.00$ \\ \hline
free-rider   & 2 & never  & 50.00\% & 50.00\% & $+0.00$ \\ \hline
\end{tabular}
\end{table}
```

**Interpretation paragraph (paste):**
```latex
The economic mechanism is strongest precisely where vote-based Byzantine fault tolerance
provably fails --- against a \emph{coordinated majority}. When two of three banks attack, an
equal-weight consensus is dragged to $50\%$ (always-fraud) or $11\%$ (label-flip) balanced
accuracy, whereas the Shapley$\to$reputation loop drives both attackers to the $0.5$ reputation
floor within three rounds and, by excluding them from the quorum, restores the lone honest
bank's $\sim$$91\%$ --- a $+41$ to $+80$ percentage-point swing. This holds because Shapley scores
each bank against a trusted validation set rather than by counting votes, so a numerical majority
cannot out-vote the contribution signal; it is genuinely complementary to Krum. The honest
limitation is the single-minority and free-rider regime: a lone attacker is out-voted by the two
honest banks and barely moves consensus accuracy, an always-fraud bank is even mildly rewarded
because flagging the rare fraud class raises balanced accuracy, and a passive free-rider that
never flags not only evades isolation but --- under majority-vote coalition scoring --- can be
mis-attributed almost the entire token pool ($w=0.99$ in the single-free-rider run). The economic
layer therefore deters coordinated manipulation that $f=0$ Krum cannot, but does not by itself
solve free-riding.
```

Figure (the `economic_isolation_trajectory.png`) shows the two label-flip attackers' token share
held at zero and their reputation decaying to the floor by round 3, while the honest bank's
reputation climbs to the $2.0$ ceiling; `economic_accuracy_protection.png` shows the protection
gap peaking at the coordinated-majority cases.

---

## 6.8 Statistical Byzantine fault tolerance: Krum where the theorem holds ($f\ge 1$)

> Source data: `db_boa_framework/experiments/byzantine_robustness_sweep.py`
> (full run, 12 epochs, eval on 5000 test transactions); raw draft in
> `TASKD_byzantine_robustness_results.md`; figure
> `results/byzantine_robustness_krum_vs_fedavg.png`.
> Framing: this is the experiment that backs the title's **``Secure''** at the parameter level —
> Krum is run where its $n\ge 2f+3$ theorem actually applies, distinct from the $f=0$ default
> pipeline (\S6.7) and from the behavioural/economic attacks above.
> All accuracy is **balanced** accuracy.

Section~\ref{tab:economic_byzantine} characterised the *economic* defence at $f=0$. We now close
the *statistical* side: we run the same Krum aggregator
(\texttt{FederationManager.\_krum\_aggregate}, general in $f$) in the two smallest configurations
where its Byzantine-tolerance precondition $n\ge 2f+3$ is satisfied --- $n=5,f=1$ and $n=7,f=2$ ---
against four **weight-level** poisoning attacks injected into the shared parameter vector that Krum
scores: \emph{sign-flip} ($w\mapsto-w$), \emph{scaled} norm-boosting ($w\mapsto 50w$),
\emph{Gaussian} junk, and a genuinely retrained \emph{label-flip} model. For each we compare the
Krum-selected global model against plain FedAvg.

```latex
\begin{table}[H]\centering
\caption{Statistical Byzantine fault tolerance: Krum vs FedAvg under four weight-poisoning attacks,
in the regime where $n\ge 2f+3$ holds. ``Rejected'' = the poisoned organisation is \emph{not}
selected by Krum. Balanced accuracy of the resulting global model on 5000 held-out transactions.}
\label{sec:byzantine-robustness}
\begin{tabular}{|l|c|c|c|c|}
\hline
Attack & Attacker rejected? & Krum bal-acc & FedAvg bal-acc & Krum advantage \\ \hline
\multicolumn{5}{|l|}{\emph{Regime $n=5,\ f=1$ (one attacker)}} \\ \hline
sign-flip  & yes & 99.95\% & 99.92\% & $+0.03$ \\ \hline
scaled     & yes & 99.95\% & 87.46\% & $+12.49$ \\ \hline
gaussian   & yes & 99.95\% & 99.96\% & $-0.01$ \\ \hline
label-flip & yes & 99.95\% & 99.91\% & $+0.04$ \\ \hline
\multicolumn{5}{|l|}{\emph{Regime $n=7,\ f=2$ (two attackers)}} \\ \hline
sign-flip  & yes & 99.95\% & 99.82\% & $+0.13$ \\ \hline
scaled     & yes & 99.95\% & 87.47\% & $+12.48$ \\ \hline
gaussian   & yes & 99.95\% & 99.95\% & $+0.00$ \\ \hline
label-flip & yes & 99.95\% & 99.95\% & $+0.00$ \\ \hline
\end{tabular}
\end{table}
```

**Interpretation paragraph (paste):**
```latex
Run where its theorem applies, Krum rejects the Byzantine update in all eight (regime $\times$
attack) cases: its score (the sum of squared $L_2$ distances to the $k=n-f-2$ nearest neighbours)
identifies the poisoned organisation as the cluster outlier, so it is never the argmin and never
enters the global model, which stays at $99.95\%$ balanced accuracy throughout. The separation
margin between the attacker's score and the worst honest score is graded by how aggressive the
attack is --- of order $10^{6}$ for norm-boosting, $10^{3}$ for sign-flip and Gaussian junk, and
only $10^{0}$ for the subtle retrained label-flip, whose parameters sit just outside the honest
spread. The FedAvg comparison is reported honestly rather than uniformly: plain averaging collapses
only under the magnitude-dominant \emph{scaled} attack ($87.5\%$ versus Krum's $99.9\%$, a
$+12.5$-point gap), because there a single large-norm vector dominates the mean; for the other three
attacks one ($\le f$) poisoned vector is diluted by the honest majority, so FedAvg happens to
survive as well. The contribution is therefore not that FedAvg always fails, but that Krum provides
a \emph{uniform} guarantee --- a constant $99.9\%$ and a provably-rejected adversary for up to $f$
colluding organisations --- whereas an unprotected mean is safe only when the attack is not
norm-dominant. This is genuine statistical Byzantine fault tolerance, in contrast to the $f=0$
outlier rejection of the default $n=3$ pipeline. Two boundaries are kept honest: the guarantee
covers at most $f$ colluding organisations (a coordinated \emph{majority} is the regime handled by
the economic mechanism of Section~\ref{tab:economic_byzantine}, making the two defences
complementary), and the thin $10^{0}$ label-flip margin shows that a sufficiently subtle,
close-to-honest poisoning attack would eventually fall within the honest cluster's own spread. The
sweep is run with differential privacy disabled to isolate the robustness effect, in a single
process rather than a live multi-host testnet.
```

Figure (`byzantine_robustness_krum_vs_fedavg.png`) shows (a) Krum's flat $\sim$$99.9\%$ against
FedAvg's drop under norm-boosting and (b) the per-organisation Krum scores on a log axis, with the
attacker as the clear outlier that is never selected.

---

## 6.9 Scalable contribution attribution: making Shapley tractable as the federation grows (CONTRIBUTION 3)

> Source data: `db_boa_framework/experiments/scalability_sweep.py`
> (full run 2026-06-06, $n_{\text{orgs}}\in\{3,\dots,20\}$, 8000 samples/org, 10 epochs,
> 200 MC permutations, fraud-stratified validation set of 400 / 49 fraud); raw drafts in
> `TASKC_scalability_results.md`; figures `results/scalability_shapley_runtime.png`,
> `results/scalability_fidelity_accuracy.png`.
> Framing: **engineering-scalability characterisation, not new-algorithm novelty.** This is the
> *only* sense in which the title's ``Scalable'' is defensible — *scalable contribution
> attribution*, not blockchain throughput or distributed training (see `title_issue.md` \S3).

The system's single anti-scalable component is exact Shapley attribution: it evaluates all
$2^{n}-1$ coalitions, so the cost of the on-chain reward split explodes with the federation size
$n$ (hardcoded at $n=3$ in the baseline). We parameterise $n_{\text{orgs}}$ end-to-end and add a
Monte-Carlo permutation estimator (\texttt{shapley\_method=mc}) costing $O(\text{samples}\cdot n)$,
then \emph{measure} (real wall-clock, single process) the runtime, the fidelity of the MC reward
weights against exact, and the global-model accuracy as $n$ grows. Organisations are made
genuinely heterogeneous (graded feature noise $\sigma\in\{0,0.5,1.0,1.5\}$) from a shared weight
initialisation, so (i) the Shapley-weighted weight-average is a valid global model and (ii) the
true Shapley values are non-uniform, making the exact-vs-MC fidelity check a real test.

```latex
\begin{table}[H]\centering
\caption{Exact vs Monte-Carlo Shapley as the federation size grows (real wall-clock, single
process). $\rho$ = Spearman rank-correlation of the MC reward weights against exact; L1 = total
reward-weight error. Exact is infeasible beyond $n\approx 12$ ($2^{20}\approx 1.05$M coalitions).}
\label{tab:scalability}
\begin{tabular}{|r|r|r|r|r|r|r|r|}
\hline
$n$ & Exact (s) & MC (s) & Speed-up & Exact coal. & MC coal. & L1 & Spearman $\rho$ \\ \hline
3   & 0.14   & 0.15  & 0.9$\times$ & 7    & 7    & 0.021 & $+1.00$ \\ \hline
6   & 3.06   & 1.45  & 2.1$\times$ & 63   & 63   & 0.221 & $+0.20$ \\ \hline
8   & 6.90   & 6.42  & 1.1$\times$ & 255  & 249  & 0.180 & $+0.64$ \\ \hline
10  & 27.78  & 17.21 & 1.6$\times$ & 1023 & 697  & 0.207 & $+0.56$ \\ \hline
12  & 113.53 & 34.74 & 3.3$\times$ & 4095 & 1265 & 0.257 & $-0.01$ \\ \hline
14  & ---    & 48.08 & ---         & 16383 (infeasible) & 1820 & --- & --- \\ \hline
16  & ---    & 66.72 & ---         & 65535 (infeasible) & 2321 & --- & --- \\ \hline
20  & ---    & 95.09 & ---         & 1048575 (infeasible) & 3247 & --- & --- \\ \hline
\end{tabular}
\end{table}
```

**Interpretation paragraph (paste):**
```latex
The contribution-attribution layer is made scalable in the sense of \emph{feasibility}, not raw
speed-up. Exact Shapley grows geometrically --- $113$\,s already at $n=12$ --- and is simply
intractable beyond it, whereas the Monte-Carlo estimator evaluates every federation size up to
$n=20$ in under $100$\,s. The two costs form a \emph{crossover}: below $n\approx 10$ the faithful
(untruncated) MC samples nearly all coalitions, so its runtime matches exact ($\sim$$1\times$),
which is acceptable because exact is cheap there; above the crossover MC touches only a small
fraction of $2^{n}$, giving a measured $3.3\times$ at $n=12$ and, more importantly, remaining the
only estimator that terminates at all. The speed-up is also not monotone and is sometimes
\emph{below} $1\times$ at small $n$ ($0.92\times$ at $n=3$, $0.44\times$ at $n=5$), because when the
untruncated estimator samples nearly all coalitions it does the same work as exact with extra
permutation overhead; MC only earns its keep above the crossover, where exact is infeasible. The
most important honest caveat concerns \emph{fidelity, not speed}, and we report it as a finding
rather than bury it: at a \emph{fixed} $200$-permutation budget the reward ranking degrades as $n$
grows. The estimator stays \textbf{top-1-faithful only to $n\approx 5$} --- the single
best-contributing organisation is recovered at $n\le 5$ but is missed at $n=6,8,10,11,12$ --- and
the full rank-correlation falls from $\rho=1.0$ at $n=3$ to $\rho\approx 0.2$ at $n=6$ and to
$\rho\approx 0$ (a statistically random ordering) at $n=12$, where $200$ permutations cannot resolve
twelve near-equal contributors each with true weight $\approx 0.08$. The finding is therefore
two-edged and stated as such: MC-Shapley makes the on-chain reward split \emph{tractable} at scale,
but its $200$-draw fidelity is trustworthy only to $n\approx 5$ --- quantifying the practical limit
of Shapley-based incentives at a fixed budget. Recovering fidelity at larger $n$ requires the
sample count to scale with $n$ (approximately $O(n\log n)$), so the reported speed-ups are an upper
bound on the cost saving once fidelity is held fixed. Accuracy under scaling is reported under two regimes to separate the federation effect
from data dilution: with each organisation's data volume held fixed (equal-shard) the
Shapley-weighted global model stays roughly flat at $79$--$87\%$ balanced accuracy as $n$ grows,
whereas splitting one fixed dataset $n$ ways (fixed-pool) produces the expected dilution decline
from $92.3\%$ to $85.2\%$ as each bank's scarce fraud positives thin out --- a data-budget limit of
the ULB set, not a failure of the attribution scaling. This result narrows the title's scalability
claim to its defensible form (\emph{scalable contribution attribution}) while leaving blockchain
throughput and distributed training in the Limitations (statistical Byzantine tolerance is
demonstrated separately in Section~\ref{sec:byzantine-robustness}). We make no claim that the
end-to-end system scales: the measured Fabric layer (Section on consensus measurement) peaks at
$\approx 40$ committed tps at concurrency~10 and its goodput collapses beyond that (50\% of
transactions fail with \texttt{MVCC\_READ\_CONFLICT} at concurrency~20), so blockchain throughput
is explicitly out of scope for the ``Scalable'' claim and is reported honestly as a limitation.
```

Figure~\ref{fig:scalability_runtime} (`scalability_shapley_runtime.png`) plots the exact runtime
tracking the $2^{n}$ reference and terminating at $n=12$ while MC bends sub-linear out to $n=20$;
`scalability_fidelity_accuracy.png` shows the bounded reward-weight error (L1$\le 0.26$,
L$\infty\le 0.10$) and the equal-shard vs fixed-pool accuracy curves.

---

## 6.10 Validity and simulation scope (paste — strengthens the chapter)
```latex
\paragraph{Validity and scope.} The consensus latency and throughput reported by the
simulation layer are computed from node resource scores, not measured on a multi-host network;
they characterise relative behaviour rather than production performance. The three
institutions are stratified volume-splits of a single bank's dataset, a controlled rather than
a cross-institution setting. The differential-privacy budget $\epsilon=1.0$ is intentionally
tight. These constraints are revisited in the Limitations and Future Work.
```
