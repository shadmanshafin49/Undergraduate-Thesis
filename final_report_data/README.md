# final_report_data — Ground-Truth Drafts for the Final Thesis Report

**Purpose.** This folder documents *what the project actually implements*, derived
by reading the real code (`db_boa_framework/`, `db_boa_fabric/`), the saved results
(`db_boa_framework/results/db_boa_results.json`), and `build_log.md`. It exists so
the final report (`FINAL YEAR THESIS REPORT/`) can be made **truthful** — every claim
in the report should trace back to something in these notes.

**Guiding rule (from the author):** *If something has not been used in the actual
project work, it must not appear in the final report.*

> **STATUS (2026-06-07).** Drafts are verified and internally consistent. **Framing: the approved
> title is fixed and sticks word-for-word** (Secure, Incentivized, Scalable ML, Consensus
> Mechanisms, Reinforcement Learning); each title word is earned in its *qualified* sense and the
> body says so (RL = minimal linear-Q leader rotation, secondary; Scalable = scalable contribution
> *attribution*; Secure = Krum BFT at $n\ge 2f+3$ + economic isolation of a majority). The
> **central contribution is B1** — the private-incentive characterisation ($\epsilon^\star\,3000\to
> 50$ via output-channel DP); the other tasks (privacy/utility decoupling, economic isolation,
> statistical BFT, scalable attribution) are the supporting characterisation. The headline detector
> is settled at **Acc 99.85% / MCC 0.677** (fresh fixed-objective run) and is explicitly reported at the **weight-DP-off**
> deployment operating point (privacy carried on the incentive channel). The false "first to bind
> Shapley to blockchain" claim is **struck** everywhere and FedCoin (2020) is cited instead. The
> `new_issues.md` (2026-06-07) honesty fixes are folded into REWRITE_00/01/06/08/09. The actual
> `FINAL YEAR THESIS REPORT/*.tex` files are still untouched (drafts-before-tex).
>
> _Note: an earlier "Option A" plan proposed dropping RL and renaming the title; it is **superseded** —
> the title is kept and RL was integrated (linear-Q leader selection) to honour it._

## How to use these files

1. Read [00_report_vs_code_divergences.md](00_report_vs_code_divergences.md) first —
   it is the audit of everything the current LaTeX report claims that the code does
   **not** do. These are the items to remove or rewrite.
2. Use [00_ground_truth_implementation.md](00_ground_truth_implementation.md) as the
   single source of truth for what the system really is.
3. Per-chapter drafts (01–06) describe what each chapter *should* say, using only
   verified facts.
4. [07_actions_checklist.md](07_actions_checklist.md) is the concrete to-do list to
   reach a submission-ready report (including which experiments must be re-run).

## File index

### Audit / ground truth (read first)
| File | Covers |
|------|--------|
| [00_report_vs_code_divergences.md](00_report_vs_code_divergences.md) | **Critical.** Every report claim not backed by code, with severity + fix |
| [00_ground_truth_implementation.md](00_ground_truth_implementation.md) | Authoritative description of the actual system |
| [07_actions_checklist.md](07_actions_checklist.md) | Step-by-step path to a ready report |

### Per-chapter analysis notes (what to change & why)
| File | Covers |
|------|--------|
| [01_introduction.md](01_introduction.md) … [06_conclusion.md](06_conclusion.md) | Chapter-by-chapter correction notes |

### ✅ Option A paste-ready rewrites (apply to `.tex` after review)
These contain the **actual LaTeX-ready content** for the Shapley+Krum+DP (Option A) framing,
using **verified numbers from a real full run**. Draft here first, then apply to the `.tex`.
| File | Covers |
|------|--------|
| [REWRITE_00_title_abstract.md](REWRITE_00_title_abstract.md) | **Title kept unchanged** + B1-centred abstract (DP-off operating point, RL secondary, qualified Scalable/Secure) |
| [REWRITE_01_introduction.md](REWRITE_01_introduction.md) | Chapter 1 blocks (novelty, dataset, objectives) |
| [REWRITE_02_literature_and_bib.md](REWRITE_02_literature_and_bib.md) | Ch 2 DP/Krum/Shapley subsection + bib entries + missing-key flag |
| [REWRITE_03_requirements.md](REWRITE_03_requirements.md) | Ch 3 specs, baselines, risk table |
| [REWRITE_05_methodology.md](REWRITE_05_methodology.md) | Ch 5 real architecture + DP/Krum/Shapley federation layer |
| [REWRITE_06_results.md](REWRITE_06_results.md) | Ch 6 results from the verified full run (DP-off headline note §6.1; tightened scalability §6.9; flat-landscape note §6.5) |
| [REWRITE_08_limitations_disclosures.md](REWRITE_08_limitations_disclosures.md) | **Threat model + Limitations & disclosures** — closes `new_issues.md` #3/#4/#8 (Secure scope, economic free-rider, single-source data, no inter-round training, Shapley X_val, SEQ_LEN/f=0/padding) |
| [REWRITE_09_conclusion.md](REWRITE_09_conclusion.md) | Conclusion (B1-centred, RL secondary, DP-off headline) |

> **Workflow:** content is staged here first; the real `FINAL YEAR THESIS REPORT/*.tex`
> files are edited only after your review.

## One-paragraph summary of the gap

The LaTeX report (`FINAL YEAR THESIS REPORT/`) is well written and *mostly* aligned to
the real work — far better than the older `reports/P2_REPORT_T2430460.pdf`. However it
**overstates the model and several experiments**. The two largest issues: (1) the
report's *primary novel contribution* is "DB-BOA Job 3 for federated aggregation
weights," but the code's default path uses **exact Shapley values + Krum + differential
privacy** and keeps DB-BOA Job 3 only as a disabled fallback; (2) the model is described
as a dilated-TCN with 64-dim embeddings and softmax attention + FedProx, while the code
is a **2-layer 1D-CNN with global max-pooling and no FedProx**. Many headline numbers
(97.38% / MCC 0.966, 85 TPS, 28.4% latency cut, 28/18/4 leader splits, paired t-tests,
the 8-model comparison table) are **not produced by any code path** and must be either
re-derived from a real run or removed.
