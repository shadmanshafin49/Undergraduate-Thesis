# Actions Checklist — Path to a Submission-Ready Report

Ordered, concrete steps. Tackle the 🔴 decision (A1) first — it shapes everything else.

> **STATUS (updated 2026-06-06).** A1 decided = **Option A** (Shapley+Krum+DP, characterisation
> not priority). `run_baselines.py` re-run (verified ablation, see `05_results.md` §B /
> REWRITE_06 §6.2). Novelty experiments re-run: privacy↔incentive sweep at 50 draws
> (**ε\*=3000**, TASKA), economic-Byzantine (TASKB), Krum-BFT (TASKD), scalability (TASKC).
> Headline detector settled at **Acc 99.85% / MCC 0.677** (fresh fixed-objective run). The "first to bind Shapley to
> blockchain" claim is **struck** everywhere (FedCoin cited instead). All REWRITE_* drafts and
> the per-chapter notes (01–06) reflect these. Remaining: optional 3-seed study, optional
> pure-BOA/DBOA + activation ablations, and the actual `.tex` edits (still staged here per the
> drafts-before-tex workflow).

## A. Decisions (do these first)
- [x] **A1 (🔴 the big one). DECIDED → Option A** (keep code as-is: DP + Krum + **Shapley**;
  reframe novelty as *characterisation* of Shapley-weighted, blockchain-coupled incentives +
  DB-BOA for hyperparameters/leader selection). Original options retained below for the record:
  - **A1 — original** Choose the novelty framing:
  - **Option A (recommended):** keep the code as-is (DP + Krum + **Shapley**), and reframe
    the report's novelty around Shapley-weighted, blockchain-coupled incentives + DB-BOA
    for hyperparameters/leader selection. Less rewriting of code, honest, defensible.
  - **Option B:** make DB-BOA Job 3 the real federation method — set `use_shapley=False`,
    `use_krum=False` in `config.py`, fix its degenerate/non-converging objective, re-run.
    More work, and the current Job-3 results do not converge.
- [ ] **A2.** Decide whether to run the pure-BOA / pure-DBOA ablation (enables the
  convergence-comparison claim) and the activation ablation (enables that figure). If not,
  delete those claims/figures.

## B. Re-run experiments (mandatory — current JSON is stale)
- [x] `python3 db_boa_framework/main.py` → fresh `results/db_boa_results.json` + plots.
- [x] `python3 db_boa_framework/main.py --attack` → Byzantine simulation outputs.
- [x] `python3 db_boa_framework/run_baselines.py` → FedAvg / +Krum / +DP / proposed (verified;
      pasted into `05_results.md` §B and REWRITE_06 §6.2). *Remaining (optional code step): paste
      the printed dict into `utils/metrics.py::baseline_metrics()` if you want the comparison plot
      to repopulate.*
- [x] Novelty re-runs: privacy↔incentive (50 draws, ε\*=3000), economic-Byzantine, Krum-BFT,
      scalability → TASKA–D + REWRITE_06 §6.6–6.9.
- [ ] (Optional) multi-seed loop (≥3 seeds) if you want to keep any significance claim.
- [x] Record every real number into a results table cited consistently (REWRITE_06 is the
      single source; per-chapter notes point to it).

## C. Report edits by file
- [ ] `core/titlepage.tex` — remove "Reinforcement Learning"; new title (D15).
- [ ] `core/abstract.tex` — align novelty to the chosen framing; remove unverified headline
      numbers (D1, D5).
- [ ] `chapters/chapter_1.tex` — fix dataset to ULB 284,807/0.17% (D10); novelty (D1); move
      DP to implemented (D17); objectives list per `01_introduction.md`.
- [ ] `chapters/chapter_2.tex` — add DP/Krum/Shapley related-work subsection; make FedProx
      background-only (D3); update gap table.
- [ ] `chapters/chapter_3.tex` — dataset (D10); DB-BOA roles (D1); baseline list (D4); SDK
      version (D16); add the real risk-management table.
- [ ] `chapters/chapter_5.tex` — rewrite ADTCN architecture to the real 1-D CNN (D2); 2-D
      hyperparameter search (D11); federation = DP+Krum+Shapley (D1); remove FedProx + its
      sensitivity table (D3); fix config table + dev-environment (D16).
- [ ] `chapters/chapter_6.tex` — rebuild entirely from the fresh run; delete fabricated
      tables (D4, D5, D6, D7, D8, D9, D13, D14); add "Validity and Simulation Scope" note.
- [ ] `chapters/chapter_9.tex` — rewrite per `06_conclusion.md` (remove D1/D5/D7 claims).

## D. Code hygiene (small, supports the report)
- [ ] Pin `torch` in `db_boa_framework/requirements.txt` (currently commented optional,
      but `adtcn.py` imports it).
- [ ] Fix `leader_block.py` node→org mapping if you want a real 3-bank leader table
      (currently `node_id % 2` → Org1/Org2 only) — otherwise describe it honestly.
- [ ] Confirm `datasets/creditcard.csv` path matches `config.DATASET_PATH` (the CSV sits at
      repo root as `creditcard.csv`; config points to `datasets/creditcard.csv`).

## E. Front-matter / template
- [ ] Fill approval page, co-supervisor/coordinator/HoD (template placeholders remain).
- [ ] Ensure all `\ref{}` targets exist (e.g. risk-management table referenced but missing).
- [ ] Build the LaTeX and check `\listoftables` / `\listoffigures` resolve.

## F. Final consistency pass
- [ ] One number per metric, used identically in abstract, Ch.5, Ch.6, and conclusion.
- [ ] Every method named in the report appears in the code (and vice-versa).
- [ ] Every figure in `results/` cited is from the fresh run.

---

### Priority order if time is short
1. A1 decision → 2. B re-run → 3. Chapter 6 rebuild (biggest risk) → 4. Chapter 5
architecture/federation fixes → 5. Ch.1/Ch.3 dataset + baseline fixes → 6. title/abstract/
conclusion → 7. front-matter + LaTeX build.
