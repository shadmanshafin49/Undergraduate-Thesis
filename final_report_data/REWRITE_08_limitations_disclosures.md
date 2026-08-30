# REWRITE — Threat Model, Limitations & Disclosures (paste-ready)

This is the consolidated honesty block that closes `new_issues.md` items #3 (Secure scope),
#4 (economic-incentive scope), and #8 (disclosures). It is written so that every word of the
**fixed title** is honestly bounded in the body — the title is not changed. Place the Threat
Model subsection in Chapter 3/5 (Requirements / Methodology) and the Limitations subsection at the
end of Chapter 6 / start of the Conclusion. All claims are code- and results-traceable.

---

## Threat model (paste — Chapter 3 or 5)

```latex
\paragraph{Threat model and security scope.} The framework's ``Secure'' guarantees are stated
against an explicit, bounded adversary, not as blanket security. Two distinct defences cover two
distinct regimes. (i) \emph{Statistical Byzantine fault tolerance:} the Krum aggregator provides a
provable robustness guarantee against up to $f$ colluding organisations that submit arbitrary
\emph{weight-level} poisoning (sign-flip, norm-scaling, Gaussian junk, retrained label-flip), but
only in the regime where its precondition $n\ge 2f+3$ holds; we demonstrate it at $n=5,f=1$ and
$n=7,f=2$ (\S\ref{sec:byzantine-robustness}). (ii) \emph{Economic isolation:} the
Shapley$\to$reputation$\to$quorum-exclusion loop additionally defends against a \emph{colluding
majority} of label-corrupting participants that out-numbers any statistical defence
(\S\ref{tab:economic_byzantine}). The default three-organisation production pipeline, by contrast,
runs Krum at $f=0$: at $n=3$ the precondition $n\ge 2f+3$ forces $f=0$, so there Krum performs
\emph{consensus-aligned outlier rejection} with no adversary assumed, not a BFT guarantee. We do
not claim robustness to a sufficiently subtle, close-to-honest poisoning attack: the separation
margin for the retrained label-flip attack is only $O(10^{0})$, so an attacker whose update sits
inside the honest cluster's own spread would eventually evade Krum.
```

---

## Limitations & honest disclosures (paste — end of Chapter 6 / Conclusion)

```latex
\paragraph{Limitations and disclosures.} We state the boundaries of the evidence plainly.

\textbf{Security and incentive scope (precise claims).} The economic incentive mechanism is
\emph{conditional}, and its failure modes are part of the contribution rather than hidden:
reputation-driven isolation triggers only under a \emph{colluding majority} of label-corrupting
participants (where it restores $+41$ to $+80$ percentage points of balanced accuracy); it does
\emph{not} deter a lone attacker or a passive free-rider. A single attacker is out-voted by the
honest majority and barely moves consensus accuracy (\texttt{acc\_gap}$=0$, never isolated), and a
free-rider that never flags is never isolated and can even be mis-attributed almost the entire
token pool under majority-vote coalition scoring ($w\approx 0.99$ in the single-free-rider run).
This is an inherent property of reputation that decays only on \emph{detectable disagreement}.
Deterring free-riding requires a contribution \emph{floor} (a minimum useful-work threshold), not
reputation alone, which we leave to future work. Likewise, the ``Scalable'' claim covers
contribution \emph{attribution} (MC-Shapley, faithful to $n\approx 5$ at a fixed sample budget),
\emph{not} blockchain throughput: the measured Fabric layer peaks near $40$ committed tps and its
goodput collapses past concurrency~10 ($\sim$$50\%$ \texttt{MVCC\_READ\_CONFLICT} failures at
concurrency~20).

\textbf{Single-source data.} The three ``banks'' (BankA/B/C) are a stratified volume-split of a
single institution's dataset (the ULB Credit Card Fraud set), not genuinely distinct cross-
institution sources. The federation is therefore a controlled simulation of heterogeneity, not a
real multi-bank deployment, and cross-institution distribution shift is not exercised.

\textbf{No inter-round local training.} Organisations do not run local gradient steps between
federation rounds; the global model is re-aggregated from per-shard models rather than warm-started
and locally fine-tuned each round. The three-round convergence reported here therefore should not
be read as a general FL convergence result.

\textbf{Shapley validation circularity.} Shapley contribution values are computed on a held-out
validation slice (\texttt{X\_val}, distinct from the final-evaluation \texttt{X\_test} after the
validation-set fix) drawn from the same data pool used for training and final evaluation. The
attribution signal thus carries a small, disclosed circularity: it measures marginal contribution
to balanced accuracy on an in-distribution validation set, not on an independent held-out
institution's data.

\textbf{Empirically chosen / future-work parameters.} Several settings are chosen empirically and
flagged as such rather than derived: the input sequence length \texttt{SEQ\_LEN}$=10$; the default
Byzantine tolerance $f=0$ at $n=3$; the sequence zero-padding that affects $\approx 0.003\%$ of
samples; and the deployment privacy budget $\epsilon=1.0$ on the (disabled) weight channel. None of
these is claimed to be optimal; sensitivity to them is left to future work.

\textbf{Simulation scope.} Consensus latency and throughput, and the RL leader-selection
adaptivity result, are obtained on a single-host Fabric test-network / resource-score simulation,
not a geo-distributed multi-host banking network; the transferable findings are the \emph{shapes}
(batching amortises the orderer timeout; RL avoids a node whose reliability drifts), not the
absolute constants.
```

---

### Mapping to `new_issues.md`
- **#3 (Secure too broad):** Threat-model subsection + the ``Security and incentive scope''
  paragraph restate Krum's guarantee as $f\ge 1, n\ge 2f+3$ and label the $f=0$ default as outlier
  rejection — matching the docstrings. The subtle-poisoning margin is disclosed.
- **#4 (economic does nothing vs single attacker / free-rider):** framed conditionally with the
  lone-attacker and free-rider gaps stated, and the contribution-floor fix named as future work.
- **#8 (disclosures):** single-source data, no inter-round training, Shapley `X_val` circularity,
  and `SEQ_LEN=10` / `f=0` / padding / $\epsilon=1.0$ all moved from docstrings into the report.
- The same paragraph reinforces #1 (scalable = attribution, Fabric throughput out of scope) so the
  Limitations section reads consistently with \S6.9.
