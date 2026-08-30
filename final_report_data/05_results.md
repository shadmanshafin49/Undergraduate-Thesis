# Chapter 6 — Result Analysis (what can be honestly reported)

**Chapter 6 is the highest-risk chapter.** Most of its tables are fabricated or copied
from the base paper, and some contradict each other (see divergences D4–D14). The safe
path: **re-run the current code, then report only what the code outputs.**

## Step 0 — Regenerate results (mandatory, 🔴 D12)
The saved `results/db_boa_results.json` is from the **old DB-BOA Job-3 path** (epoch=23).
Before writing Chapter 6:
```
cd db_boa_framework
python3 main.py                 # full pipeline → new db_boa_results.json + plots
python3 main.py --attack        # Byzantine BankC simulation
python3 run_baselines.py        # FedAvg / FedAvg+Krum / FedAvg+DP / proposed
```
Then paste the printed `baseline_metrics()` dict into `utils/metrics.py` so comparison
plots populate.

## What you CAN report (real, code-backed)

### A. Centralised ADTCN fraud detection
From the VERIFIED full run of `main.py` (DB-BOA-tuned ADTCN, ULB test set n=56,962, 98 fraud).
Reportable fields: Accuracy, Precision, Sensitivity, Specificity, NPV, FPR, FNR, FDR, F1, MCC,
and the confusion matrix (TP/TN/FP/FN). Emphasise **MCC** as the honest metric under 0.17%
imbalance (the report's point that 99.83% is the trivial "all-normal" baseline is good — keep it).

> **VERIFIED headline values (use everywhere — fresh fixed-objective run, MCC 0.677):** Acc 99.85%,
> Precision 54.25%, Sens 84.69%, Spec 99.88%, NPV 99.97%, FPR 0.12%, F1 66.14%, **MCC 0.677**,
> TP 83 / TN 56,794 / FP 70 / FN 15. The DB-BOA-tuned detector is the single headline (selected
> automatically without manual search; it does not beat — and slightly trails — a hand-set default
> at MCC 0.785, see REWRITE_06 §6.5). **Do not** reuse the
> old 0.941 / 0.9931 / 0.9249 / 97.38 figures; full table in [REWRITE_06_results.md](REWRITE_06_results.md) §6.1.

### B. Federated comparison (the real baseline set — replaces the 8-model table, D4)
Report the table `run_baselines.py` produces:

| Configuration | Acc | MCC | FPR | NPV | Precision |
|---------------|-----|-----|-----|-----|-----------|
| FedAvg | 99.72% | 0.569 | 0.26% | 99.98% | 37.07% |
| FedAvg + Krum | **99.92%** | **0.776** | 0.05% | 99.97% | 72.97% |
| FedAvg + DP (ε=1.0) | 99.83% | 0.000 | 0.00% | 99.83% | — |
| **Proposed (Krum + DP + Shapley, ε=1.0)** | 0.21% | 0.001 | 99.96% | 100.00% | 0.17% |

_VERIFIED — saved `results/baselines.json` (`run_baselines.py`), ULB test set n=56,962,
DB-BOA-tuned ADTCN (142/76)._ This is an honest ablation showing the marginal effect of each
technique. **Key reading:** Krum alone is the strongest config (MCC 0.776); both DP rows
**collapse to a degenerate single-class predictor (MCC≈0)** at the deliberately tight ε=1.0 — the
failure direction (all-normal vs all-fraud) is set by the random noise draw and varies run-to-run,
because the Gaussian weight noise overwhelms the signal and Krum cannot help (selecting one noised
model forgoes the noise-cancellation averaging gives). This is *not* a defect — it is the privacy/utility trade-off
characterised in REWRITE_06 §6.6 (incentive-fidelity threshold ε\*=3000), and motivates the
DP-SGD / relaxed-ε future work. Present it as evidence, not as a failure to hide. The full
LaTeX-ready table + interpretation is in [REWRITE_06_results.md](REWRITE_06_results.md) §6.2.

### C. DB-BOA convergence
Plot the **real** convergence history (`opt_history` / `cost_function_convergence.png`).
Report the actual final Obf2 and the actual optimum `(n_filters, steps_per_epoch)`.
**Do not** state 5.82 vs 4.91 vs 5.21 unless you actually run pure-BOA / pure-DBOA
ablations (`algorithms/boa.py`, `dboa.py` exist — you *could* run them and then the claim
becomes real).

### D. Federation / Shapley weights
Report the **Shapley** contribution weights and coalition values per federation round
(the federation result dict carries `shapley_values`, `coalition_values`, `krum_scores`).
Show how the +20 pool splits by Shapley weight. **Do not** claim convergence to
[0.52,0.32,0.16] unless the fresh run shows it.

### E. Incentive / token evolution
Report the real per-org token trajectory from the run (chaincode `getNodeStatus` /
`token_balance_history.png`). Use the actual final balances, not 458/312/178.

### F. Byzantine resilience (real, from `--attack`)
Reportable: number of disputed rounds out of 15, attacker token depletion curve,
reputation hitting the 0.5 floor, and the attacker's Shapley weight collapsing when present
in coalitions. **Remove** the invented "global model Acc 0.9121→0.9612 by round window"
table (D14) unless you add code to measure per-window global accuracy.

## What you MUST remove or re-derive (fabricated)
- 🔴 8-model base-paper comparison table (DTCN…DBOA-ADTCN) — D4.
- 🔴 "97.38% / MCC 0.966" headline — D5 (and it contradicts the report's own 0.9786/0.7812).
- 🔴 Statistical-significance section (paired t-tests, 3 seeds, p<0.001) — D8.
- 🔴 Leader-selection 28/18/4 over 50 rounds — D6 (code: 5 rounds, leader Node 7, Org1/Org2 only).
- 🔴 85 TPS / 180 ms / 28.4% latency reduction — D7 (simulated; not measured).
- 🔴 Token balances 458/312/178 and weight convergence [0.52,0.32,0.16] — D9.
- 🔴 Activation comparison ReLU/LeakyReLU/ELU/SELU — D13 (only ReLU exists).
- 🔴 FedProx μ-sensitivity table — D3 (FedProx not implemented).

## Plots available in `db_boa_framework/results/` (regenerate, then reuse)
`cost_function_convergence`, `confusion_matrix`, `roc_curve`, `federation_weights`,
`token_balance_history`, `incentive_tokens`, `leader_selection`, `throughput_latency`,
`org_accuracy_progression`, `classifier_comparison`, `summary_comparison`,
`activation_accuracy` (the last one is only valid if you actually run an activation
ablation; otherwise drop it).

## Honesty framing to include
Add a short "Validity and Simulation Scope" paragraph: consensus latency/throughput are
**simulated**; the three orgs are volume-splits of one bank's data; ε=1.0 DP is an
intentionally tight budget. Examiners reward this kind of explicit scoping far more than
inflated numbers.
