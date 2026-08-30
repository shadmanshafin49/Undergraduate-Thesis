# Final Report Update — Progress Checklist

_Tracks the work to bring the thesis report (`FINAL YEAR THESIS REPORT/`) in line with the
upgraded, honest, real-data research. Last updated: 2026-06-08._

**Legend:** ✅ done · 🟡 in progress · ⬜ not started · ⛔ blocked

**Workflow rule:** report rewrites are drafted in `final_report_data/*.md` first, then merged
into the `.tex` chapters once approved. Standing mandate: **honest, real-data, zero fabrication.**

---

## 1. Foundations (research + sources)

- [x] ✅ Novelty experiments built, run, written up — Task A, B, B1, C, D (see `final_report_data/TASK*.md`)
- [x] ✅ Ground-truth implementation audit (`00_ground_truth_implementation.md`)
- [x] ✅ Report-vs-code divergences documented (`00_report_vs_code_divergences.md`)
- [x] ✅ `all papers/` folder — 30 verified PDFs for every method/related-work paper used
- [x] ✅ `references.bib` — 71 entries, no dup keys, braces balanced, compiles
  - [x] All verifiable citations have entries (verified via Crossref/arXiv, no fabrication)
  - [ ] ⛔ `ahamad2022`, `zhang2019` — unverifiable → **drop their sentences during lit-review pass** (decided)
  - [ ] ⬜ 8 paywalled PDFs not downloaded (refs are complete; PDFs optional) — see §5

---

## 2. Chapter rewrites: draft → `.tex` merge

Each row: draft is written; the remaining work is merging into the `.tex` + applying the listed fixes.

_Audited 2026-06-08 against the live `.tex`: the merge was far more complete than this table
previously recorded. All chapters are merged; the four genuinely-outstanding items (abstract,
two ch.5 sentences, `amssymb` preamble, recompile) were completed in the same audit pass._

| Ch | `.tex` file | Draft source(s) | Draft | Merged to .tex |
|----|-------------|-----------------|:-----:|:--------------:|
| Title/Abstract | `core/titlepage.tex`, `core/abstract.tex` | `REWRITE_00_title_abstract.md` | ✅ | ✅ (abstract merged; title is fixed/approved, unchanged) |
| 1 Introduction | `chapters/chapter_1.tex` | `REWRITE_01_introduction.md`, `01_introduction.md` | ✅ | ✅ |
| 2 Literature Review | `chapters/chapter_2.tex` | `REWRITE_02_literature_and_bib.md`, `02_literature_review.md` | ✅ | ✅ |
| 3 Requirements | `chapters/chapter_3.tex` | `REWRITE_03_requirements.md`, `03_requirements_impacts_constraints.md` | ✅ | ✅ |
| 5 Methodology | `chapters/chapter_5.tex` | `REWRITE_05_methodology.md`, `04_methodology.md` | ✅ | ✅ ("all three jobs" overstatement fixed → two live roles) |
| 6 Result Analysis | `chapters/chapter_6.tex` | `REWRITE_06_results.md`, `05_results.md`, `TASK*` | ✅ | ✅ |
| (new) Limitations | section in ch.6/ch.9 | `REWRITE_08_limitations_disclosures.md` | ✅ | ✅ |
| 9 Conclusion | `chapters/chapter_9.tex` | `REWRITE_09_conclusion.md`, `06_conclusion.md` | ✅ | ✅ |

> Note: `chapters/chapter_7.tex` exists but is **not** included in `main.tex` — confirm whether
> it's needed before final submission.

---

## 3. Specific content fixes to apply during the merge

### Report-wide formatting
- [x] ✅ All display equations numbered (4.1)–(4.13) — converted 10 `\[...\]` blocks in ch.5 to `\begin{equation}` with labels (2026-06-10; other chapters had none)
- [x] ✅ All table captions moved above tables (2026-06-10; 17 table floats + 2 `\captionof` blocks in ch.5 — ch.2/3/6 were already top-captioned)

### Literature Review (ch.2) — 🟡 partially done
- [x] ✅ Remapped 8 named cites to existing numeric keys (truong/ying/zhao/machhale/saveetha/yang/hussain/nourmohammadi)
- [x] ✅ Added 7 verified related-work entries (zhuang2019, li2022fdia, abdallah2020, tsoulias2020, li2023, wang2022, arora2019)
- [ ] ⬜ **Drop** the `zhang2019` sentence (ch.2 opening) and `ahamad2022` sentence (incentive para)
- [ ] ⬜ **Cite the ~22 new method/related-work entries** currently in bib but uncited in .tex
      (dwork2006, blanchard2017, fedcoin2020, bai2018, wang2020fedsv, watkins1992, chaudhuri2011,
      ghorbani2019, hsieh2020, andrew2021, mcmahan2017/2018, li2020fedprox, givi2023, tubishat2020,
      liu2021pickchoose, lopezrojas2016, suttonbarto2018, commey2025, fraboni2020, li2025shapley,
      jaramillo2026, zhao2026sichainfl) — so they appear in the compiled bibliography
- [ ] ⬜ Fix `li2022fdia` description: it's "Secure Federated **Deep Learning**," NOT "transformer" (verify against paper)
- [ ] ⬜ FedProx → background-only ("natural extension"), not "employed"
- [ ] ⬜ Add "Robust and Fair Federated Aggregation" subsection (DP / Krum / Shapley) per `REWRITE_02` §A2

### Methodology (ch.5)
- [x] ✅ §4.1 methodology overview diagram designed & approved (2026-06-10; `images/methodology_overview.svg` + `.png` — banks/DB-BOA/federation/Fabric/phase strip, novelty path in crimson, honesty notes kept)
- [x] ✅ §4.1 figure block inserted into `chapter_5.tex` (2026-06-10; 300-dpi PNG render, Figure 4.1 + lead-in paragraph; latexmk compiles clean, verified visually on printed p.23)
- [ ] ⬜ Fix numberless `\section*{FL-ADTCN: Design, Evaluation, and Results}` block after §4.2.10 (merge leftover, ~line 450): demote to `\subsection*{4.2.11 ...}`, demote its child headings to `\subsubsection*`, refit the stray "4.2.11 Performance Metric Definitions"; do NOT shift rubric sections 4.3/4.4. Same block also holds the 80/10/10-split and epoch-wording audit leftovers — fix together.
- [ ] ⬜ DB-BOA: "three optimisation roles" → **two roles** (hyperparam + leader selection) + RL leader policy
- [ ] ⬜ Remove FedProx (`μ=0.05`) claims + μ-sensitivity table (not implemented)
- [ ] ⬜ ADTCN architecture: remove dilated/64-dim/softmax-attention claims → 2-layer 1D-CNN + global max-pool
- [ ] ⬜ Replace "DB-BOA Job 3 (novel)" box → DP + Krum + exact Shapley aggregation (per `REWRITE_05` §5.C)
- [ ] ⬜ Fix experimental-config table (remove FedProx/Job-3 rows)

### Results (ch.6)
- [ ] ⬜ Replace overstated/hardcoded numbers with measured results (Tasks A–D, B1, B2 consensus)
- [ ] ⬜ Add limitations/disclosures (ε=1.0 budget, Krum f=0 in 3-org, etc.) per `REWRITE_08`

### Intro (ch.1) + Title/Abstract + Conclusion (ch.9)
- [ ] ⬜ Reframe contribution as **characterisation** (privacy↔incentive trade-off), not new algorithm
- [ ] ⬜ Position relative to FedCoin (extend, not claim priority)
- [ ] ⬜ Title wording per `REWRITE_00`

---

## 4. Final verification (do last)
- [x] ✅ `latexmk` (pdflatex+biber) compiles cleanly — **0 undefined control sequences, 0 undefined references/citations, no `[?]`** (verified 2026-06-08; added missing `\usepackage{amssymb}` to fix `\mathbb`/`\gtrsim` in ch.5/ch.6 math)
- [ ] ⬜ All figures referenced exist and match measured results
- [ ] ⬜ Every claim traces to a PDF read or a verified abstract (honesty pass)
- [ ] ⬜ Remove leftover template entries from bib if desired (cricket/NBA papers, `prabanand2025_orig`)

---

## 5. Outstanding / blocked
- ⛔ `ahamad2022`, `zhang2019` — no verifiable source found → sentences to be dropped (see §3)
- ⬜ Paywalled PDFs not in folder (refs complete; download via library if wanted):
  `ying2025` (10.1109/TMC.2024.3477616), `zhao2024` (10.1109/TSC.2024.3399653),
  `machhale2024` (10.1109/MITADTSoCiCon60330.2024.10575309), `hussain2024` (10.1109/M2VIP62491.2024.10746050),
  `li2022fdia` (10.1109/TSG.2022.3204796), `arora2019` (10.1007/s00500-018-3102-4),
  `zhuang2019` (10.1145/3343147.3343169), `wang2022` (10.1016/j.aej.2021.04.079 — gold OA, free in browser)

---

## 6. Structural restructure (2026-06-10)
- [x] ✅ Moved from ch.4 (Methodology) → ch.5 (Result Analysis): *Implementation of Selected Design* (now **5.1 Implementation**, incl. Experimental Configuration as closing subsection), *Performance Metric Definitions* (now **5.2**); ch.5 continues with 5.3 Centralised ADTCN Detection Performance etc.
- [x] ✅ Ch.4 *Quantitative Results* block folded into existing **5.4 Federated Ablation** (was a verbatim duplicate of `tab:fed_ablation`); `run_baselines.py` provenance preserved in 5.4 intro
- [x] ✅ Numberless FL-ADTCN `\section*` block folded into §4.2 as **4.2.11** (inner headings demoted to `\subsubsection*`) — closes the pending numbering issue
- [x] ✅ Fixed wrong split claim in moved Implementation text: "time-aware 70:15:15" → stratified 70:10:20 (matches `data_loader.py`/`config.py`, train 199,364 / val 28,481 / test 56,962) — closes audit item
- [x] ✅ `latexmk` recompiled clean: 89 pages, 0 undefined references; TOC shows 5.1/5.2/5.3 as requested
- [x] ✅ Added FL-ADTCN (federated) confusion matrices to §5.4: Table 5.7 (TP/FP/FN/TN per ablation config, straight from `results/baselines.json`) + Figure 5.3 (`images/confusion_matrix_federated.png`, generated by new `experiments/plot_federated_confusion.py` — pure visualisation of saved counts, no re-run)
- [x] ✅ Margin-violation sweep (user spotted Table 5.12 off-margin): fixed Table 5.12 (fixed-width cols), resized 15 over-wide hand-set p{} tables across ch4/ch5 to fit the 150mm text block, \small+tighter \tabcolsep on 3 wide numeric tables, widened the Raft-orderer TikZ node, \allowbreak in DP/Krum/Shapley + fabric-contract-api literals, added \emergencystretch=2em to preamble. Overfulls: 19 over 20pt → worst remaining 7.5pt (≈2.6mm, invisible). Verified by page renders.
- [x] ✅ Conclusion (ch.6 / chapter_9.tex) restructured into 6.1 Summary, 6.2 Research Limitations, 6.3 Future Work — limitations/future-work prose converted to itemised lists verbatim (no claims added/removed), cross-refs to Tables 4.7/4.8 added; now in TOC; compiles clean
- [x] ✅ Merged 6.2+6.3 → single "6.2 Limitations and Future Work" (user preference): five bold-headed items, each limitation paired with its remedy (data realism→multi-bank streams + n≥2f+3; infra→multi-host Fabric; ε=1.0→DP-SGD; subtle poisoning→hardened aggregator; +quantum-resistant crypto); claims unchanged

---

## 7. Gaps vs approved example thesis (`example_thesis/T2430427_P3`, audited 2026-06-10)

Benchmark: approved BRAC CSE thesis "Avantgarde" (Oct 2025, 87 pp). Ours: 91 pp, 74 bib entries
(vs ~50), same template, same 6-chapter skeleton. Ours is **stronger** on lit-review organisation,
results depth, honesty, and writing; gaps below are rubric-conformance items the example has and we lack.

_All seven fixed 2026-06-10; report recompiles clean at **98 pages**, 0 errors, 0 undefined refs, no >20pt overfulls._

- [x] ✅ **Removed template Appendices A/B** from `main.tex` (the approved example has no appendices)
- [x] ✅ **Nomenclature populated**: `core/nomenclature.tex` (28 abbreviations + 5 symbol groups, all
      verified in-use in the body); `.latexmkrc` added with the `nlo→nls` makeindex cus_dep; renders
      as a proper two-page Nomenclature in front matter
- [x] ✅ **Ch.1 §1.6 "Summary of Contributions"** added between Methodology-in-Brief and Scopes —
      bulleted primary (characterisation + channel fix) / secondary (4-experiment suite) / substrate
      (Fabric, DP+Krum+Shapley, DB-BOA+RL) — wording lifted from existing prose, no new claims
- [x] ✅ **Ch.6 "Comparison with Prior Work"** (new section before Discussion): Table 5.16 capability
      matrix (FedCoin, Yang, SI-ChainFL, Jaramillo-Velez, Commey, base paper, ours) + Table 5.17
      quantitative anchors with explicit not-directly-comparable caveat. Every cell verified against
      the PDFs in `all papers/` (incl. Yang 2024 DP = future-work only; base paper dataset private,
      accuracy relative-only +1.7–7.3pp; SI-ChainFL 90%-malicious/+14.12%; Commey 96.7%@50% flip)
- [x] ✅ **Ch.6 "Discussion"** close-out: Interpretations / Implications / Limitations-and-Validity
      (existing validity text kept verbatim under its `sec:validity` label; synthesis restates only
      established results)
- [x] ✅ **Methodology evidence figures** (all real artefacts): `images/ulb_class_distribution.png`
      (computed from raw creditcard.csv; matches Table consortium_split 142,403/85,442/56,962 +
      246/148/98), TikZ ADTCN forward-path diagram with tensor shapes (matches `models/adtcn.py`:
      Conv1d 30→F→2F k3 + ReLU, global max-pool, Linear 2F→2, F=142), and
      `images/fabric_live_capture.png` — live capture of the **rebooted** Fabric network (docker ps,
      `peer channel getinfo` height 7, real `getNodeStatus` chaincode query), captioned honestly as
      the single-host two-org measurement deployment per §validity
- [x] ✅ **Ch.1 Figure 1.1 research workflow** (six-phase TikZ, phase 6 highlighted) + **Ch.9
      "Concluding Remarks"** section added between Summary and Limitations
- [x] ✅ (bonus honesty fix found during capture) `tab:chaincode_functions` renamed to the **real**
      contract functions: `recordNodeMetrics→updateNodeMetrics`, `getTokenBalance/getReputation→
      getNodeStatus`, `updateReputation→updateIncentive` (chaincode bounds ρ∈[0.5,2.0], floors
      tokens at 0; the ±0.02/−0.05 policy lives in the off-chain layer — verified in
      `db_boa_fabric/chaincode/lib/db_boa_chaincode.js` + `blockchain/leader_block.py`)

---

## 8. Final audit + fix batch (2026-06-11)

Full report audit (every results table vs JSON ✅, figures byte-identical ✅, compile clean ✅)
found 10 substantive errors; **all fixed same day** (changelog: `final_report_data/REWRITE_FIX_2026-06-11.md`).
Recompiles clean at **100 pages**, 0 undefined refs, worst overfull 7.5pt (pre-existing).

- [x] ✅ `tab:dbboa_vs_default` — disclosure route: 3 rows (default 0.785 paired / tuned 0.677 headline /
      tuned **0.313** paired), caption + Honest-finding rewritten to report the 0.47 paired gap and the
      run-to-run variance; ch9 + ch5 echoes updated
- [x] ✅ Feature count 29/30 → **33** (30 base + 3 recurrence, `use_graph_features: True`) everywhere incl.
      equations, TikZ forward-path diagram, (N,10,33); recurrence features now described
- [x] ✅ Unimplemented claims removed: Threshold-Optimisation subsection + workflow step (argmax in code),
      AdamW/weight-decay/grad-clip/LR-scheduler → plain Adam lr 1e-3 (ch5 §Training Config + dev-env table),
      "weighted sampling" → class-weighted loss only, "re-runs on degradation" → manual
- [x] ✅ "50 consensus / 10 federation rounds" → 5 consensus + 3 federation + 15-round attack (2 places)
- [x] ✅ 3-org Fabric "deployment" rewritten honestly: 2-org single-host CCaaS network (matches
      fig:fabric_live; new Docker table from the live capture), 3-org topology framed as design target
- [x] ✅ DB-BOA math matched to implementation: range-normalised switch threshold τ_t; LSAM = Gaussian
      mutation μ=0.1×range, 5 iters (Lévy noted as base-paper description)
- [x] ✅ `tab:stat_summary` + z-score table corrected against raw creditcard.csv (Time mean 94,814 etc.)
- [x] ✅ ε* definition unified ("largest budget at which reward ordering still inverts") in §private-incentive
      + nomenclature z02
- [x] ✅ Approval: "B.Sc. in Computer Science and Engineering **on** June 13, 2026" (⚠ user to confirm
      "Spring, 2026" vs June date); ch9 repo-URL sentence deleted (both URLs 404); ch2 receptive-field
      wording; empty `chapter_7.tex` deleted
