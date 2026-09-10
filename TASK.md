# ▰▰▰ OPERATION: HONEST FEDERATION ▰▰▰

**FL-ADTCN** · Blockchain-Integrated Federated Fraud Detection · BRAC University CSE

```
CLASSIFICATION ...... INTERNAL / TEAM T2430460
OPERATOR ............ Shadman Sakib  (+ Amiya, Mahima, Nafi, Muntasir)
AREA OF OPS ......... DB-BOA-FEL-ADTCN-Hyperledger-Fabric-main
PHASE ............... THESIS ACCEPTED (2026-06-13) -> PAPER EXTENSION ACTIVE
DEADLINE ............ 2026-09-19  ·  T-11 days at this sitrep  ·  OBJ-16 go/no-go 2026-09-12 (T-4)
LAST SITREP ......... 2026-09-08  (OBJ-16 LOADER LANDED — Handbook wired + 42 checks pass, zero results yet · file-order trap does NOT fire (1.02x) · NEW: TERMINAL_ID linkage 71.65x, the strongest in the project · cost measured at ~11 h, the ~6 h budget was low)
                      2026-09-05  (OBJ-13 SIDE-BY-SIDE LANDED — 0 of 9 searched configs beat the hand-set default · CRN averaging is what converges the search, determinism is NOT)
                      2026-09-04  (OBJ-13 — leak FIXED + verified 0/11 · side-by-side launched at production budget, operator-approved)
                      2026-09-04  (OBJ-13 — ULB surrogate VALIDATES ON MEMORISED ROWS (9/9) · knee scored, prereg 6 WRONG both halves)
                      2026-09-04  (OBJ-13 patch APPLIED — legacy verified bit-for-bit · 3 silent-switch defects caught · runs in flight)
                      2026-09-04  (OBJ-17 CLOSED — factor RETIRED as a headline: 20x–>=100x on definition · ordering + rho-gap carry the claim)
                      2026-09-04  (OBJ-17 grid landed — all 3 de-censored · ULB fails reproduction, cause unexplained)
                      2026-09-04  (CONFOUND CONTROL SCORED — 2 of 6 pre-registrations WRONG · plan LOCKED: consolidate first)
                      2026-09-04  (OBJ-18 CLOSED — withdrawal propagated · paradox cells MAPPED)
                      2026-09-04  (board re-planned · ≥30x premise CORRECTED — 1 draw of 100 · OBJ-13 patch found in temp)
```

---

## ▮ RULES OF ENGAGEMENT

> Read before every mission. These are not negotiable.

1. **NOVEL AND COMPLETELY HONEST.** Both. When they conflict, honesty wins.
2. **No number ships that a script did not produce.** If it does not exist, run it or cut it.
3. **Negative results stay negative.** DB-BOA buys no measured accuracy gain over the hand-tuned default (+0.047 MCC, p=0.34 — and no measured loss either; the old "loses by 0.11/0.47" claim was seed and float noise, see CONFIRMED KILLS). MC-Shapley is only top-1 faithful to n≈5. The economic layer does not stop a lone attacker or a free-rider. Krum has no guarantee at n=3. These are the contribution — do not "fix" them.
4. **No result-shopping.** Never pick a dataset, seed, or split because it makes ADTCN look better.
5. **The repo is the truth.** Handbooks, decks, and chat exports are claims to check against `results/*.json` — never status.
6. **Scope every claim.** "Scalable" = contribution attribution, not throughput. "Secure" = ≤f colluders at n≥2f+3.
7. **No invention claims.** DP, Krum, Shapley, Fabric are prior art. FedCoin (2020) owns Shapley-on-blockchain.
8. **A new dataset must be able to change a conclusion, and breadth across *claims* beats breadth
   across *datasets*.** Before adding a dataset, write down what it could *overturn*. BankSim earned
   its place by flipping the DP collapse direction and killing "temporal complexity does not pay".
   But a claim tested on three datasets while four sibling claims sit on one is not rigour — it is
   a lopsided table. **Every title claim goes off ULB once before any claim goes to a third dataset.**
9. **Judge the system, not a component.** The unit is FL-ADTCN: federated ADTCN + DP + Krum +
   Shapley + DB-BOA + RL/Fabric. The title makes six claims and a direction review must score all
   six. ADTCN's rank among seven classifiers is a component ablation and backs none of them —
   never let a component result stand in for the system verdict. (This rule exists because a
   review did exactly that on 2026-09-01; see DIRECTION REVIEW rev. B.)
10. **Krum is security, Shapley is fairness.** Krum picks *which* weights become global; Shapley
   decides *how much* each org earns. They are independent by design — a Krum-rejected org can
   still be paid. Never describe them as two stages of one pipeline.
11. **State which metric you are quoting, and why that one.** If a script computes two ratios and the
   prose quotes the flattering one without naming the switch, that is rule 4 with extra steps.
   (Caught live in OBJ-13 — see OBJ-14.)
12. **A control that can overturn a finding you already have outranks a dataset that can add a
   new one.** When two factors moved together, the next run separates them — before anything
   builds on top. Added 2026-09-04, when a ~3 h control killed OBJ-15's most quotable claim; a
   third dataset run first would have inherited the dead claim *and* added a third confounded
   column. **Corollary:** when a control answers the question a planned experiment existed to
   answer, that experiment loses its priority — re-gate it, do not just re-schedule it (OBJ-16).

---

## ▮ SITREP

The thesis is submitted and accepted. The fabricated-numbers purge is complete and the report now matches the code. Current phase is **extending toward a publishable paper**.

The three loose ends a reviewer would have found are now closed: the MCC contradiction (OBJ-2), the missing Fabric measurements (OBJ-3), and the single-seed detector (OBJ-4). The detector headline is now a distribution over seeds, not a point estimate, and **every reported detector number requires a pinned torch version and thread count** — that is the lesson OBJ-2 cost 13 runs to learn.

**Status 2026-09-04 — the confound control is scored, and it cost us a headline.** The
BankSim/`stratified` run OBJ-15 demanded landed **2026-09-01 23:41** (four sweeps, ~3 h 20 m,
clean). All three conditions are now on disk and collated by
`experiments/sweeps_cross_condition.py` → `final_report_data/OBJ15_two_factor_decomposition.md`.

**Two of the six pre-registered expectations were WRONG, both in the same direction: the
divergences track the DATASET, not the partition.** OBJ-15's flagship reading — *"Krum pays
1.5–12.6 pp of utility under entity-disjointness"* — is **withdrawn as a federation-structure
claim**. Krum is negative under BankSim/**stratified** too (7 of 8 cells, dataset effect −1.06 to
−17.27 pp), so the mechanism OBJ-15 argued (orgs hold different customers ⇒ one org's weights
generalise worse) is not what drives it. Scoring in RESUME HERE, consequences in CONFIRMED KILLS.

**Status 2026-09-04 (later) — OBJ-18 closed, and the paradox mapping came with it.** The
withdrawal is out of the generators and out of the supervisor brief, and the OBJ-17 zero-CPU
sibling was pulled forward. It produced a result nobody predicted: **the entity-disjoint headline
range was carried by contaminated cells** (clean-only it is −4.73 to −1.48 pp, not −12.57 to
−1.48), **and the withdrawal survives the filter anyway** — on the 3 cells with a clean baseline
in all three conditions the dataset effect is negative 3/3 and the partition effect positive 2/3.
So the kill is now established on the cleanest evidence in the objective, not just the fullest.
Board top is now **OBJ-17's ε-grid half**.

**All six title claims remain off ULB** (five measured on BankSim, RL/Consensus/Blockchain
dataset-independent by design). Rule 8's gate stays satisfied — but see OBJ-16: the control run
**voided AMLSim's specific justification**, so the third dataset had to be re-gated, not just
re-scheduled.

**Status 2026-09-04 (later still) — OBJ-17's grid landed, and it is a mixed result that has to
be reported as one.** All three conditions were re-run on one shared 11-point grid out to
ε=30000 (~78 min). **Every one of them de-censored**, so the objective's goal is met: the budget
factors are **measurements, not lower bounds**, for the first time. Four of five
pre-registrations scored; item (2) was **wrong** and had pre-named that as the more valuable
outcome. Scorecard in [OBJ-17].

Live contribution: *weight-channel DP destroys the Shapley signal that pays participants;
moving the noise to the published contribution channel restores rank-faithful rewards at a
lower ε.* The **ordering still holds** — but ⚠ **on the pre-registered metric it is 11/11 ·
11/11 · 10/11, not "9/9 in all three"**: that phrase came from the ρ table, and on *inversion
rates* BankSim/entity-disjoint has a counter-example at ε=1 (weight 85/100, output **88/100**).
Both channels fail near-totally there, so it falsifies the wording rather than the mechanism —
but it was true when "9/9" was written and nobody had checked the metric that was actually
pre-registered.

⚠ **The factor is now measured, and the trustworthiness ordering has INVERTED.**
**60× (ULB) · 30× (BankSim/strat) · 33× (BankSim/entity-disj)** — all measured. But ULB and
BankSim/stratified are each a ratio of **two single-draw thresholds** (1/100 at both ends;
exact 95 % CI 0.0003–0.0545), while **BankSim/entity-disjoint is the only one on double-digit
draws at both ends** (13/100 and 11/100). ~~Prefer ≥60× (ULB) when a single figure is needed~~
is **withdrawn — it now points at the weakest column.** **Prefer 33× (BankSim/entity-disjoint).**
Note also that ≥10× → 33× because the extension *raised* the numerator: ε\* is monotone
non-decreasing in both draws and grid extent, so no factor here is a fixed point.

**Status 2026-09-05 — OBJ-13's side-by-side landed, and the negative result is now decisive
rather than merely unrefuted.** Nine searches at the deployed budget (BankSim, pop 20×30, filters
5–255, three surrogate protocols × three seeds, 16.3 h, exit 0). Scored on a held-out yardstick
drawn from a **row-disjoint** pool, compared **paired** on the seven shared draws because the
yardstick uses common random numbers:

**Not one of the nine beats a hand-set 128 filters / 150 steps-per-epoch.** Mean paired
Δ **−0.1982**; the best single search reaches −0.0720 and wins 4 of 7 draws. The shipped 142/76
also loses, at −0.1900. So rule 3's *"DB-BOA buys no measured accuracy gain over the hand-tuned
default"* survives the repair — and now has a mechanism *and* a direct measurement behind it,
rather than a p=0.34 tie.

**Two pre-registrations failed, and one failure is the most useful thing in the objective.**
*(i)* **Determinism does not make the search reproducible — shared-draw averaging does.** Filter
spreads across seeds: `legacy` 66, `deterministic` **81 (wider than legacy)**, `averaged` **12**.
The common-random-numbers design — added as an implementation detail the original decision did
not contain — is empirically the half of the repair that does the work. *(ii)* **`averaged` is
the *worst* mode on held-out draws** (3.4237 vs 3.5857 / 3.5552), inverting the predicted
ordering: it converges onto high-filter configurations that generalise worse, because with only
k=3 shared draws a search that can finally rank reliably will exploit *those three draws*.
**The noise the repair removed was acting as accidental regularisation.**

⛔ **And a new defect: `ADTCN.fit:681` still uses the pre-repair floored formula.** The patch only
touched `_score_once`, so at ~400 k training rows the floor never binds and **the final model's
`steps_per_epoch` axis was always live — only the surrogate's was dead.** The search was blind to
a dimension the deployed model responds to, which is worse than "one axis was wasted".

**Status 2026-09-04 (later still) — OBJ-13's repair is applied, and applying it turned up three
ways the patch would have quietly corrupted its own evidence.** The staged files were rescued out
of the session temp dir into `db_boa_framework/scratchpad/` (tracked), `--check` resolved all four
anchors, and the patch went in clean. **`legacy` is now verified bit-for-bit** against
`models/adtcn.py.pre_obj13` — three configurations agreeing to ten decimals — so the side-by-side
runs against a live baseline rather than a remembered one, exactly as the design claimed but had
never checked.

The three defects, all of which would have shipped: the patch **silently re-pointed
`objective_noise_audit.py`**, the script that produced OBJ-13's whole evidence base, at the new
`deterministic` default — it would have reported a within-seed std of **0.0000** and read as *"the
objective was always fine"*; the same silent switch in **`basepaper_comparison.py`**, whose
optimiser JSONs are the "all five hit 5.0000" evidence; and a **pre-existing filename collision**
in which the classifier and optimiser drafts overwrote each other, which is the *third* instance
of the unsuffixed-artefact failure after OBJ-15's figures and the shared log directory. All three
fixed. Six expectations pre-registered before any repaired run. Two corrections to OBJ-13's own
downstream-cost list — the BankSim baselines are **not** invalidated (their search was skipped),
and `detector_multiseed.py` **is** affected and was not on the list. See [OBJ-13].

⛔ **And ULB failed a reproduction check, so no ULB number is quotable yet.** Re-running the 9
historical budgets at identical seeds moved **13 of 18 cells** and moved ULB's ground-truth
Shapley split with them. **Both BankSim conditions reproduced 9/9 bitwise**; ULB reproduces
*itself* same-day (11/11), so it is deterministic under fixed conditions and the Aug-30
divergence is a change in conditions. Every diff on the ULB path was audited and **none of them
should affect it** — the cause is not yet identified. See [OBJ-17].

**Status 2026-09-04 (latest) — the shipped-path check fired, and it did not just confirm the
suspicion; it moved the defect one layer below where the repair operates.** The
deployed ULB surrogate does not hold 30 fraud rows. It holds **5 unique fraud
transactions repeated 6.01×**, because `get_eval_subset` hands it a 3,000-row pool that
carries 5 positives at the real 0.173 % rate, and `__init__` then samples 30 **with
replacement**. Measured on the objective's own draw: **9 of 9 validation positives are
copies of a training row** — and `deterministic` leaks **identically**. Stratifying the
split guarantees positives on both sides, which with 5 uniques is precisely how copies
get onto both sides. **`eval_mode` is the wrong layer for this defect.** So `Obf2 =
5.0000` on the path that ships is not luck at n≈9; it is recall of memorised rows, and
that is the mechanism behind `db_boa_results.json`'s ceiling and the five-optimiser tie.
BankSim is clean on both pools (38 and 75 uniques). ⚠ Consequence that reaches backwards:
**neither `objective_noise_audit.py` nor the knee sweep describes deployment** — both
build pools through `basepaper._eval_subset`, whose own `max(30, …)` floor guarantees the
≥30 unique positives the shipped path lacks. ULB's `noise_to_signal = 0.74` answers a
different question than the one it is quoted for.

**And the knee sweep finished — during the session that was told nothing was running.**
All 8 cells landed 20:38. **Pre-registration 6 is wrong on both halves**, and it had
pre-named that as the valuable outcome: BankSim *worsens* with size (0.51 → 0.64) while
gaining 8× the fraud rows, and ULB makes the **largest improvement in the table**
(0.79 → 0.39 by 10k) with its fraud count pinned at 30 throughout. Its named alternative —
*"if ULB improves anyway, the fraud count is not the driver and the 'n≈9 positives'
mechanism is wrong"* — is therefore taken. **Two independent pre-registered falsification
routes both landed on the same day, on the same claim.** The curve is non-monotone on both
datasets (best at 10k, worse at 20k, ending 0.643 vs 0.644), BankSim's std is flat across
the whole grid so its ratio is carried by a single-draw denominator, and the run is
one seed per size — the script has since been rewritten to refuse to name a best size
under exactly that condition. **Do not quote 10,000 rows as a chosen surrogate size.**

✅ **Both operator calls were ANSWERED 2026-09-04 and the board is no longer blocked.**

| Call | Decision | State |
|---|---|---|
| **Pool fix** (ULB surrogate validating 9/9 on memorised rows) | **Fix it, then regenerate** — option (a) | ✅ **applied**: `eval_subset` 3,000 → **36,000** in both configs, plus a `RuntimeWarning` guard on the replacement branch. Re-verified **0 of 11** leaked. ULB regenerations are queued behind the CPU |
| **Side-by-side scope** | **Production budget**, not the ~1.1 h basepaper-matched option | ⏳ **running detached since 20:54**, BankSim, ~16.4 h. Chosen because only this budget yields configurations comparable to `db_boa_results.json`'s 142/76 |

⚠ The recommendation at the time was the cheap option; the operator took the expensive one
deliberately, for comparability. **Do not "optimise" it back down** — if this run has to be
repeated, repeat it at pop=20×30.

---

## ▮ RESUME HERE — handoff 2026-09-04 (rev. 2, after OBJ-18)

> First thing a new session reads. ⚠ **Check before trusting this line.** It said "nothing is running" on 2026-09-04 while the knee sweep was still going; the sweep finished at **20:38** and a scout timed against it read **12 % slow** (2.68 vs 2.36 s/eval). `Get-Process python` costs a second.

> # ✅ NOTHING IS RUNNING — the side-by-side LANDED 2026-09-05 13:13, exit 0, 9/9 runs, 16.3 h
>
> **Verify anyway before trusting this line — `Get-Process python` costs a second.** It has been
> wrong once already in this objective.
>
> **Result: not one of the nine searched configurations beats the hand-set default.** Mean paired
> Δ **−0.1982**, best single −0.0720. Scorecard for items 1–6 is in [OBJ-13]; the draft is
> `final_report_data/OBJ13_surrogate_repair_banksim.md` (regenerate any time with `--redraft`,
> seconds, no training).
>
> ### ▸ Three costing rules the run produced. Keep them; each would misprice the next run.
>
> **(i) Do not scale eval count from a small scout.** The scout extrapolated linearly from
> pop=4×3 and predicted ~1700 evals per search; real searches take **~800** at pop=20×30. DB-BOA's
> calls-per-iteration is not linear in `population × iterations`. Scale a scout's *seconds per
> eval*; take the eval count from a real search.
>
> **(ii) ⚡ The repair itself made the surrogate 1.6–5.4× more expensive, and that is a genuine
> measured cost of it, not overhead.** Legacy's `batch_size = max(32, n_train // spe)` floor
> pinned **every** candidate to 43 gradient steps per epoch regardless of `spe` — that was the
> dead axis. The repaired `ceil(n_train / spe)` makes the surrogate actually perform `spe` steps:
>
> | spe | legacy steps/epoch | repaired steps/epoch |
> |---|---|---|
> | 70 | 43 | 70 |
> | 111 | 43 | 107 |
> | 150 | 43 | 140 |
> | 250 | 43 | 233 |
>
> So `deterministic` costs ~**4,580 s** per search against `legacy`'s ~**2,470 s**, and `averaged`
> (k=3) ~**11,800 s**. **Reviving a dead search axis is not free**, and the repaired budget is not
> comparable to the pre-repair one at equal wall-clock — a point that belongs in the write-up,
> since the base paper's Table 5 compares optimisers on wall-clock.
>
> **(iii) ⚠ The repaired `steps_per_epoch` axis is ALIVE BUT COARSE, and that qualifies the fix.**
> Noticed because `deterministic`/42 (spe=**111**) and `averaged`/42 (spe=**109**) returned
> *identical* yardstick scores to four decimals — 3.5543 ± 0.4462 — which looked like a bug and is
> not. `batch_size = ceil(1400 / spe)` sends both to **batch 13**, so they train identically.
> Across `spe ∈ [50, 250]` — 201 values — the repaired formula yields only **23 distinct batch
> sizes**, and the bins widen sharply toward the top of the range:
>
> | batch | spe values mapping to it |
> |---|---|
> | 13 | 108–116 (9) |
> | 10 | 140–155 (16) |
> | 9 | 156–174 (19) |
> | 8 | 175–199 (25) |
> | 7 | 200–233 (34) |
> | 6 | 234–250 (17) |
>
> **A third of the axis (spe 200–250) is two bins.** So the repair moved the axis from *one* value
> to *23*, which is the difference between dead and alive — but a reported `steps_per_epoch`
> optimum is still only determined **to within a bin**, and barely at all above 200. Do not quote
> a repaired `spe` optimum as a precise value; quote the batch size it implies. This does not
> affect the filter axis, which is continuous in the relevant range.
>
> **(iv) The detached-runner pattern worked and should be reused.**
> `experiments/run_obj13_sidebyside.ps1` launched via `Start-Process -WindowStyle Hidden` ran
> 16.3 h across a session boundary without incident. Its preflight (assert the patch is applied
> *and* the pool fix is in, before spending the CPU) cost seconds and is the reason there was no
> 16-hour surprise. **Do not launch multi-hour work as a session-bound background task.**

### ▸ NEXT UP — in order. Do not reorder without a result forcing it.

| # | Task | Blocked? | Cost | First command |
|---|---|---|---|---|
| **1** | **[OBJ-13]** finish the re-runs the repair forces: regenerate `db_boa_results.json` + `baselines.json` on the fixed pool, then `detector_multiseed` with the repaired arms | no — **CPU is free** | ~3–5 h | `python main.py --dataset ulb` (see the ⚠ below first) |
| **2** | **[OBJ-16]** third dataset (Fraud Detection Handbook), ⛔ go/no-go **2026-09-12**. ✅ **loader LANDED + verified 2026-09-08** (42 checks); what remains is the rule-4 pre-registration and the runs | needs item 1 first, or its DB-BOA arm is paid for twice | ~~loader +~~ **~11 h CPU, measured — the ~6 h was low** | `python experiments/check_handbook_loader.py` |
| **3** | **[OBJ-7]** + **[OBJ-9]** — writing, ⛔ protected block | free | writing | — |

> ⚠ **Before regenerating anything ULB, read the ULB reproduction freeze in the SITREP.** Those
> two files are now invalid for **three** separate reasons — noise-selected config, the memorised-
> rows leak, and the unexplained Aug-30 ULB divergence — and only the first two are fixed. A
> regenerated ULB number is quotable only if the third is addressed or explicitly scoped.

**OBJ-13 progress, 2026-09-05 — the search work is DONE; only the downstream re-runs remain.**
Full write-up in [OBJ-13].

| | Step | State |
|---|---|---|
| ✅ | Rescue the staged patch out of the session temp dir | now at `db_boa_framework/scratchpad/`, **tracked** |
| ✅ | `--check` (all four anchors) → apply → both files parse | backups `models/adtcn.py.pre_obj13`, `config.py.pre_obj13` |
| ✅ | **Verify `legacy` is bit-for-bit** against the pre-patch module | `scratchpad/verify_obj13_legacy.py` — agrees to 10 dp on 3 configs; the other five claims of the patch checked too |
| ✅ | Stop the patch silently re-pointing `objective_noise_audit.py` and `basepaper_comparison.py` | both now pin/expose `eval_mode`; see the three defects in [OBJ-13] |
| ✅ | Pre-register six expectations (rule 4) | in [OBJ-13], written before any repaired run |
| ✅ | Surrogate-size knee sweep | **COMPLETE — landed 2026-09-04 20:38**, all 8 cells. ⚠ the board's "nothing is running" was wrong; it was still going. **Pre-registration 6 is WRONG on both halves** and the curve is non-monotone on both datasets — scored in [OBJ-13] |
| ✅ | ⛔ **`obj13_shipped_path_check.py`** — **RUN 2026-09-04, CONFIRMED and worse than written.** The shipped ULB pool holds **5 unique fraud transactions**, so the surrogate's 30 fraud rows are those 5 repeated **6.01×** — and **9 of 9 validation positives are copies of a training row**. The repair does **not** close it: `deterministic` leaks 9/9 too. BankSim is clean on both. See [OBJ-13] | ran in seconds |
| ✅ | `deterministic` vs `averaged` vs `legacy`, side by side | **COMPLETE 2026-09-05 13:13, exit 0, 9/9, 16.3 h** at the operator-chosen production budget (BankSim, pop 20×30, filters 5–255, 3 modes × 3 seeds). **0 of 9 beat the hand-set default**; items 1–5 scored in [OBJ-13]. → `results/obj13_surrogate_repair_banksim.json`, draft `OBJ13_surrogate_repair_banksim.md` |
| ☐ | Regenerate `db_boa_results.json` + `baselines.json` (ULB) **on the fixed pool** | **UNBLOCKED — the CPU is free.** ⚠ but see the ULB reproduction freeze; three defects invalidate these files and only two are fixed |
| ☐ | ⛔ **NEW: `ADTCN.fit:681` still uses the pre-repair floored formula** `max(32, len(X_train) // spe)` — the repair only touched `_score_once`. At ~400 k rows the floor never binds, so the **final model's `spe` axis was always live and only the surrogate's was dead**. Decide whether `fit` should adopt `ceil` too | zero CPU to decide; changing it invalidates every trained model |
| ☐ | `detector_multiseed.py` with the repaired config as a new arm | plumbing is done: the side-by-side writes `results/_multiseed_runs/extra_configs.json` and `detector_multiseed.py` picks it up with no code edit. ⚠ the arm is fixed to the **production seed 42**, never the best of the nine — selecting on the yardstick and then quoting the yardstick would be this objective's own defect one level up |

⚠ **Two corrections to OBJ-13's own downstream-cost list, both found by checking rather than
trusting it.** (i) `baselines_banksim_stratified.json` and `baselines_banksim_customer.json` are
**not** invalidated — both were produced with the search *skipped* (`detector_width_source:
"pinned …"`, params 32/150 = the hand-set default). Only `baselines.json` and
`db_boa_results.json` (both ULB, both 142/76) carry a searched configuration. (ii) The list
**missed** `experiments/detector_multiseed.py:72`, which hardcodes that same 142/76 as
`dbboa_tuned` — so OBJ-4's **+0.047 MCC, p=0.34** headline compares a *noise-selected* config
against the default. Net: fewer files than written, but one script that was not on the list.

**⚠ RAISED BY OBJ-17, NOT SCHEDULED — needs an operator call before it goes on the board.**
Both are consequences of OBJ-17 that fall outside it, and inserting either into the locked order
without a decision would be re-litigating the plan. Written down here so they are not lost:

| | Follow-up | Why it matters | Cost |
|---|---|---|---|
| **A** | **Does the ULB reproduction failure reach the detector track?** Only `private_incentive_sweep` was re-verified. Every ULB number generated **before 2026-09-01** sits on a code+environment state that no longer exists and cannot be reconstructed — including `detector_multiseed`, `baselines`, `architecture_ablation` | If a detector number moved the way the Shapley split did, it is a Chapter-5 problem, not a paper-extension one. If nothing moved, that is a clean reproducibility statement worth having in writing | re-run to check; unknown until one is tried |
| **B** | **Task A carries the same ε\* defect.** `privacy_incentive_sweep.py:249` defines `epsilon_star` with the identical `max{ε : any inversion}` rule, so everything OBJ-17 proved about monotonicity and definition-sensitivity applies to it verbatim — and its grid stops at 1e4 | Task A's ε\* is quoted in `config.py:341` and in the report. It is the *same* unstable statistic under a different name | small: the fix is the OBJ-17 one, already written |

**Closed 2026-09-04, do not re-open:** **[OBJ-17]** (ε-grid extension + `--repeats 1000` pass,
both scored; the budget factor **retired as a headline**; ULB reproduction failure closed as
*unexplained* with the environment now stamped into every sweep JSON; brief propagated to
`.md`/`.tex`/PDF) · **[OBJ-18]** (withdrawal propagated into both generators, the supervisor brief
and its PDF) · OBJ-17's **zero-CPU sibling task** (paradox cells mapped — it produced a correction
to a quoted range *and* showed the OBJ-15 kill survives the filter; see CONFIRMED KILLS).

> ⚠ **Carried forward from OBJ-17 — two corrections that outlive it, both made before the
> runs.** They apply to any future ε\* work, including follow-up **B** above. (i) The "one hair from de-censoring at 0.01" reading was backwards: 0.01 at
> `n_repeats=100` is **one inverted draw**, so that threshold was the *least* firmly established,
> not the most nearly confirmed. (ii) Bigger: **ε\* = max{ε : rate > 0} is monotone
> non-decreasing in both the repeat count and the grid extent** — seeds are `3000+rep`, so more
> draws strictly *contain* fewer, and a new budget can only add inversions. **Neither knob can
> ever lower ε\***, which is why item (4)'s "drops to 1000" branch was impossible as written and
> the `--repeats` pass is now scored on whether ε\* **rises**. Both are written up in [OBJ-17].
>
> ⛔ **And do not quote a ULB number until the reproduction failure is explained.** BankSim is
> unaffected — both partitions reproduced bitwise.

### ▸ THE RECORD BELOW IS HISTORY — the confound control, its scorecard, and its four sweeps

*Kept because the pre-registrations and their scoring are the evidence for everything above.
Nothing in this section is an outstanding action.*

### ✅ RUN COMPLETE — BankSim/`stratified` confound control, landed 2026-09-01 23:41

All four sweeps finished clean (economic 20:43 · byzantine 21:32 · private-incentive 22:00 ·
scalability 23:41). Collated 2026-09-04:

```powershell
cd "d:\THESIS\FINAL PROJECT\DB-BOA-FEL-ADTCN-Hyperledger-Fabric-main\db_boa_framework"
python experiments\sweeps_cross_condition.py     # seconds, reads JSON only, safe any time
```

→ `final_report_data/OBJ15_two_factor_decomposition.md`, all three conditions, dataset effect and
partition effect as separate columns.

**Why this run existed.** Every BankSim↔ULB comparison in OBJ-15 changed **dataset and partition
together**, so not one of its divergences could be attributed to either alone. This run separates
them:

| Comparison | What moves | What it isolates |
|---|---|---|
| ULB vs **BankSim/stratified** | dataset only | **the dataset** |
| **BankSim/stratified** vs BankSim/customer | partition only | **entity-disjointness** |

**✅ The single-factor claim was verified before launching, not assumed.** None of the four sweeps
passes `groups=` to `ADTCN.fit()` (grepped: `economic_byzantine_sweep:194`,
`byzantine_robustness_sweep:119`, `private_incentive_sweep:107`, `scalability_sweep:234,309` all
call `split_for_orgs` and then `fit()` with no groups), and `fit(groups=None)` builds **global**
windows (`adtcn.py:593`). So **both partitions window identically** and the partition really is the
only factor moving. The windowing confound recorded for the federated ablation in CONFIRMED KILLS
(`baselines_banksim_*`) does **not** apply to these four sweeps — those two runs differ in
partition *and* windowing; these do not.

> ⚠ **The other side of that same fact, and it must be reported:** the OBJ-15 entity-disjoint
> sweeps therefore measured a federation whose orgs hold **disjoint customers but global windows**.
> ~~"Krum pays 1.5–12.6 pp of utility under entity-disjointness"~~ was measured *without*
> entity-linked windows inside each org. That is honest scoping, not a bug — but do not describe
> those sweeps as using customer-linked sequences, because they do not.
>
> **↳ 2026-09-04: that sentence is now withdrawn on stronger grounds than scoping** — the control
> run shows the cost is not a property of entity-disjointness at all (scorecard row 1 below). The
> windowing caveat still stands for any *other* claim built on those sweeps.

**What each sweep is expected to say (pre-registered 2026-09-01 20:20, before any output — rule 4).**
Do not edit these after results land; score them underneath.

- **Krum's utility cost tracks the *partition*, not the dataset.** Predicted: BankSim/stratified
  puts Krum back at or near parity with unprotected FedAvg (ULB stratified was +12.49 pp,
  BankSim/customer was −12.57 to −1.48 pp). The mechanism argued in OBJ-15 — Krum selects one
  org's weights, which generalise worse only when orgs hold *different* customers — predicts the
  cost is a property of entity-disjointness. **If Krum is still negative under stratified, that
  mechanism is wrong and the finding is about BankSim, not about federation structure.** That is
  the most valuable outcome available here.
- **Krum still rejects 8/8.** It is geometric; nothing in the partition should touch it.
- **The lone-attacker isolation result follows the partition too.** Predicted: back toward ULB's
  0/3 firing. If it still fires 3/3 under stratified, the cause is the dataset (or the even-voter
  quorum arithmetic already flagged), not org heterogeneity.
- **ε\* ordering holds a third time** (output channel ≫ weight channel at every ε). The constants
  may move again; the ordering is the claim. **ε\* will still be right-censored** — the grid still
  stops at 3000 and the weight channel was still inverting there on both previous runs.
- **Exact-Shapley cost is unchanged.** O(2ⁿ) over the same 4,095 coalitions; genuinely nothing to
  learn, stated so it is not later reported as a finding.
- **MC top-1 stays intermittent.** Withdrawn as a threshold in OBJ-15; predicted to stay
  unreliable at every n, with no clean partition story.

### ▸ SCORECARD — all six, scored 2026-09-04. The six bullets above are unedited.

| # | Pre-registered (verbatim above) | Outcome |
|---|---|---|
| 1 | Krum's utility cost tracks the **partition**, not the dataset | ❌ **WRONG — and the pre-registration named this exact alternative in advance.** Krum is still negative under stratified in **7 of 8 cells**. Dataset effect **−1.06 to −17.27 pp**; partition effect changes sign (−11.50 to +5.28). By its own words: *"that mechanism is wrong and the finding is about BankSim, not about federation structure."* |
| 2 | Krum still rejects 8/8 | ✅ **held.** 8/8 in all three conditions. |
| 3 | Lone-attacker isolation follows the partition | ❌ **WRONG — dataset.** Fires 3/3 on **both** BankSim partitions, 0/3 on ULB — same orgs, same rounds. |
| 4 | ε\* ordering holds a third time, **and stays right-censored** | ✅ **both halves held.** Output > weight at **9/9 budgets in all three**; weight still inverting at ε=3000 everywhere (rate 0.20 / 0.01 / 0.38). |
| 5 | Exact-Shapley cost unchanged | ✅ **held, structural.** 113.5 / 140.3 / 120.6 s at n=12 over the same 4,095 coalitions. |
| 6 | MC top-1 stays intermittent, no clean partition story | ✅ **held.** 5/10 · 5/10 · 3/10, and the per-`n` pattern is inconsistent across conditions. |

> **Verdict: 4 of 6 held, 2 were wrong, and both were wrong the same way — we attributed to
> *federation structure* what belongs to *BankSim*.** That is the second time in this objective
> that pre-registration converted a wrong guess into a finding instead of an embarrassment
> (OBJ-15 scored 2 of 4 wrong for the same reason). Keep doing it.
>
> **What it protects:** the withdrawn claim was the most quotable thing OBJ-15 produced. Without
> this ~3 h control it would have gone into the paper as a federation-structure result, and a
> reviewer with one extra partition would have taken it apart.

**Sweep 1 of 4 — `economic_byzantine_sweep`, BankSim/stratified, 22.7 min, landed 20:43:49.**

> **❌ The pre-registration was WRONG, and the way it was wrong is the finding.** It predicted
> lone-attacker isolation would go “back toward ULB's 0/3 firing” under stratified, and named the
> alternative in advance: *“if it still fires 3/3 under stratified, the cause is the dataset … not
> org heterogeneity.”* **It fires 3/3.** The isolation behaviour is a property of **BankSim**, not
> of entity-disjointness.

| Lone attacker | ULB / strat | BankSim / strat | BankSim / entity-disj |
|---|---|---|---|
| always-fraud | not isolated | **BankC @ r8** | BankC @ r8 |
| label-flip | not isolated | **BankC @ r3** | BankC @ r3 |
| free-rider | not isolated | **BankC @ r4** | BankC @ r3 |

The two BankSim columns isolate **the same org at the same rounds**. The partition moves almost
nothing; the dataset moves everything.

| Accuracy gap (pp) | dataset effect | partition effect | reading |
|---|---|---|---|
| `always-fraud` ×1 | **−4.74** | −0.06 | dataset |
| `label-flip` ×1 | **−1.52** | −0.10 | dataset |
| `free-rider` ×2 | **+44.39** | +3.02 | dataset |
| `always-fraud` ×2 | +3.60 | +3.02 | **both — not decomposable** |
| `label-flip` ×2 | +7.43 | +2.78 | **both — not decomposable** |

**Three OBJ-15 statements this forces a change to:**
1. ~~“Isolation now fires on all 3 single-attacker scenarios” as a *BankSim/entity-disjoint*
   result~~ — it is a **BankSim** result, at both partitions. Do not attribute it to the split.
2. ~~“The free-rider at n=2 is now caught (+47.40)” as an entity-disjointness effect~~ — the
   dataset carries +44.39 of it and the partition +3.02.
3. The **2-of-3 collusion headline replicates in all three conditions** (+40.78 / +44.39 / +47.40
   always-fraud; +79.52 / +86.95 / +89.73 label-flip). Its *increase* over ULB splits roughly
   evenly between the two factors, so say “replicates, slightly larger on BankSim” and **do not
   attribute the increase**.

**What did NOT move, and is now three-condition solid:** isolation improves accuracy on **0 of 3**
lone attackers in *every* condition. That negative result (rule 3) is the strongest it has ever
been — the layer fires and does not help, on two datasets and two partitions.

⚠ The negative ×1 gaps are still partly **quorum arithmetic**, not defence quality: consensus is
`preds.sum()*2 > n_voters`, so isolating one of three voters leaves two, where the same expression
demands unanimity. That applies identically in all three conditions, so it does not explain the
ULB↔BankSim difference — but it does mean −4.74 / −1.52 are not purely a defence-quality
measurement.

✅ **Comparability check done, not assumed:** the stored ULB JSONs predate OBJ-15's plumbing, so
they are only comparable if that plumbing is a no-op on ULB. It is — `FinancialDataLoader.raw_feature_count`
is **33**, exactly the historical `min(X.shape[1], N_RAW_FEATURES + 3)` cap, so
`apply_to_model_cfg()` changes nothing for ULB.

**Sweep 2 of 4 — `byzantine_robustness_sweep`, BankSim/stratified, 49 min, landed 21:32.**

> **❌ This is the one that killed the headline.** Predicted: stratified puts Krum "back at or near
> parity with unprotected FedAvg". It does not — Krum is negative in **7 of 8 cells**.

| Krum − FedAvg (pp, balanced acc) | ULB / strat | BankSim / strat | BankSim / entity-disj | dataset | partition |
|---|---|---|---|---|---|
| n=5 `sign-flip` | +0.03 | **−4.62** | −11.20 | −4.65 | −6.58 |
| n=5 `scaled` | +12.49 | **+0.54** | −1.48 | −11.95 | −2.03 |
| n=5 `gaussian` | −0.01 | **−1.07** | −12.57 | −1.06 | **−11.50** |
| n=5 `label-flip` | +0.04 | **−5.50** | −11.13 | −5.54 | −5.63 |
| n=7 `sign-flip` | +0.13 | **−9.15** | −4.73 | −9.28 | **+4.43** |
| n=7 `scaled` | +12.48 | **−4.79** | −1.65 | **−17.27** | **+3.14** |
| n=7 `gaussian` | +0.00 | **−11.96** | −10.60 | −11.96 | **+1.36** |
| n=7 `label-flip` | +0.00 | **−11.61** | −6.32 | −11.61 | **+5.28** |

**Two things to take from this table, and the second is worse for OBJ-15 than the first.**
1. The **dataset column is negative in all 8 rows**; the partition column changes sign. The cost
   is carried by BankSim.
2. **At n=7 the partition effect is *positive* in all four rows** — entity-disjointness makes Krum
   look **better**, which is the exact opposite of OBJ-15's stated mechanism. That is not a weaker
   version of the old story; it is the reverse of it. Do not salvage the mechanism by softening it.

⚠ **The artefact is now condition-graded and it is worst exactly where we headlined.** Attacked
FedAvg runs beating their own no-attack reference by >0.5 pp: **0/8 (ULB) → 2/8 (BankSim/strat)
→ 5/8 (BankSim/entity-disj)**. So on the condition OBJ-15 quoted, the *majority* of Krum−FedAvg
cells have a contaminated baseline.

> ✅ ~~**Open item: the collator reports counts only — nobody has yet mapped which flagged cells
> correspond to which utility rows.**~~ — **CLOSED 2026-09-04 (OBJ-18 window, zero CPU).** Both
> the sweep's own draft generator and `sweeps_cross_condition.py` now mark the affected cells
> individually, and the collator adds a **clean-subset table**. The mapping and what it changes
> are in CONFIRMED KILLS; the short version is that **the two most-quoted entity-disjoint figures
> (−12.57 and −11.20) are both contaminated**, and **the withdrawal survives the filter**.

⚠ **Unpredicted and still unexplained:** Krum's accuracy is attack-invariant on ULB and on
BankSim/entity-disjoint, but **not** on BankSim/stratified. Nothing in the pre-registration
covers it and no mechanism is on record. Small, but it is a loose thread in the *security* half,
which is the half we claim replicates cleanly.

**Sweep 3 of 4 — `private_incentive_sweep`, BankSim/stratified, 28 min, landed 22:00.**

✅ **The ordering held a third time — output > weight at 9/9 budgets in all three conditions.**
That is the claim, and it is now the most robust thing in the objective.

| | ULB / strat | BankSim / strat | BankSim / entity-disj | dataset | partition |
|---|---|---|---|---|---|
| ε\* weight | ≥3000 | ≥3000 | ≥3000 | 0 | 0 |
| ε\* output | 50 | **100** | 300 | +50 | +200 |
| factor | ≥60× | **≥30×** | ≥10× | −30 | −20 |

The factor degrades under **both** effects, so it is **not decomposable** — say "the factor is
condition-dependent and ranges ≥10× to ≥60×", and do not attribute the shrinkage to either.

🔑 **The lead that OBJ-17 is built on:** weight-channel inversion rate at ε=3000 is
**0.20 (ULB) · 0.01 (BankSim/strat) · 0.38 (BankSim/entity-disj)**. BankSim/stratified is *one
hair* from de-censoring. ~~Push the grid past 3000 and at least one condition likely converts
from a lower bound into a measurement.~~

> ⚠ **2026-09-04 — that last sentence is withdrawn as an expectation.** "One hair" is literally
> **one inverted draw out of `n_repeats=100`**, and ε\* is a hard `> 0` threshold on that rate.
> So BankSim/stratified is the condition whose ε\* is *least* firmly established, not the one
> closest to confirmation — and the other two sit at 0.20 and 0.38, nowhere near zero. The
> extension is still worth running; it is just not the near-certain recovery this line implied.
> [OBJ-17] carries the corrected reasoning and a pre-registration.

**Sweep 4 of 4 — `scalability_sweep`, BankSim/stratified, 1 h 41 m, landed 23:41 · run complete.**

✅ Cost structural as predicted (113.5 / 140.3 / 120.6 s at n=12). ✅ Top-1 intermittent as
predicted (5/10 · 5/10 · 3/10, no consistent per-`n` pattern — BankSim/stratified fails at n=5
yet recovers at n=11 **and** n=12).

⚠ **Unpredicted, and large:** MC rank fidelity ρ at n=12 is **−0.014 (ULB) → +0.769
(BankSim/strat) → +0.448 (BankSim/entity-disj)** — a **dataset effect of +0.783** against a
partition effect of −0.322. MC-Shapley's rank fidelity is mostly a property of the dataset. L1
(split fidelity) barely moves at all (0.2572 / 0.2553 / 0.2262). Rule 11 applies: **name which
fidelity metric you are quoting**, because they behave completely differently here.

**Operational notes worth keeping:**
- `.\experiments\check_obj15.ps1 [-Partition customer|stratified]` reports liveness, elapsed,
  per-sweep state and landed JSONs. Both runner and checker now take `-Partition`.
- **Logs are now per-run**: `results/_obj15_logs/banksim_<partition>/`. The 2026-09-01
  entity-disjoint logs were moved into `banksim_customer/` — a second run would otherwise have
  overwritten the first one's evidence, which is the same class of mistake as the unsuffixed
  figures caught during OBJ-15.
- Judge a running sweep by **CPU time climbing**, not log freshness — prints are flushed but
  sparse (the first line after the banner waits for a whole org to train).
- The per-sweep logs are **UTF-16LE** (PowerShell 5.1 `*>`), so `grep` from Git Bash reads
  nothing from them; `iconv -f UTF-16LE` first. `_runner.out.log` is ASCII.
- ULB figures for the four sweeps are backed up in `results/_ulb_figures_backup/`.
- ⚠ ~~Do not edit `private_incentive_sweep.py` (the ε grid) while this run is queued~~ — **lifted
  2026-09-04, the run is done.** OBJ-17 now deliberately edits that grid. The constraint it
  encoded still holds in a new form: **all conditions must share one grid**, so when OBJ-17
  extends it, every condition is re-run on the extended grid or none is.

**Always run the collator; never diff the JSONs by hand.**

```powershell
python experiments\sweeps_cross_condition.py     # seconds, reads JSON only
```

`experiments/sweeps_cross_condition.py` (new, 2026-09-01) prints every headline across all
three conditions with the **dataset effect** and **partition effect** as separate columns, and
names the metric on every row (rule 11). It reproduces every ULB and BankSim number in the INTEL
table from the JSONs, so the extractors were verified against known values before the third
condition arrived; it prints `--` for anything not on disk. It also carries `_PARADOX_PP = 0.5`
**copied deliberately** from
`byzantine_robustness_sweep.py` — with a tighter threshold it flags ULB's `gaussian` +0.01 pp
wobble and reports a paradox the sweep's own draft denies, which is exactly the
two-parts-of-the-repo-disagree failure OBJ-14 was about.

**Regenerating any draft is free** — no re-run needed:

```powershell
python experiments\<sweep>.py --dataset banksim --partition stratified --redraft
python experiments\objective_noise_audit.py --render-only
```

### ▮ THE LOCKED PLAN — T-15 to 2026-09-19

> **Locked with the operator 2026-09-04.** Question asked: consolidate or expand? Answer: *both,
> consolidate first.* Do not re-litigate the ordering; if it has to change, change it because a
> result forced it, and write down which one.

**Deadline 2026-09-19. Progress checkpoint 2026-09-05.**

| Window | Work | Cost |
|---|---|---|
| ~~**now**~~ | ✅ ~~**OBJ-18** — propagate the withdrawal into the report/brief prose~~ **DONE 2026-09-04**, plus the OBJ-17 paradox-cell mapping pulled forward | ~2 h, no CPU |
| ~~**now (Sep 4–7)**~~ | ✅ ~~**OBJ-17** — ε-grid **+ repeat-count** extension~~ **DONE 2026-09-04, three days early** — both passes, both scorecards, brief propagated. Ran to completion in one day because the grid pass was 78 min rather than the budgeted 85, and the `--repeats` pass was approved at full scope | ~4 h CPU, spent |
| **Sep 5** | ✅ checkpoint — the material is the two OBJ-17 scorecards plus the confound-control one. **Brief rebuilt 2026-09-04; no further preparation needed** | — |
| **now (Sep 4–10)** | **OBJ-13** — surrogate repair + the re-runs it forces. ⚠ **Pulled forward from Sep 7** because OBJ-17 closed early; the *ordering* is unchanged, only the start date. The run-blocker cleared 2026-09-01 23:41, but ⚠ **the staged patch is not in the repo** — it is in a session temp dir and must be rescued first (path in OBJ-13) | code + CPU |
| **Sep 10–16** | **OBJ-16** — third dataset, **go/no-go 2026-09-12** | loader + ~6 h CPU |
| **Sep 16–19** | **OBJ-7** (DP threat model) + **OBJ-9** (title scoping) + buffer | writing |

**Why consolidate first — three reasons, one of which is a hard dependency:**

1. **OBJ-13 is an ordering constraint, not a preference.** It invalidates `db_boa_results.json`
   and the DB-BOA arm of every `baselines*.json`. Run a third dataset before it and that dataset's
   DB-BOA arm is generated on a surrogate we already know is broken, then paid for twice.
2. **AMLSim's gate was void as of 2026-09-01 23:41** and rule 8 says the gate is written before
   the CPU is spent. Its justification was *"Krum's cost appeared only under our synthetic
   entity-disjoint split"* — the control answered that question. See OBJ-16, re-gated.
3. **Bounded vs unbounded cost.** OBJ-17 is a hardcoded-list edit plus ~26-minute runs; the
   paradox-cell mapping is zero CPU. Those land. AMLSim is a Java generator with a config
   pipeline before a single transaction exists — that is the item that can silently eat a week.

**And the value density is inverted from how it looks: OBJ-17 is the only item that can
un-withdraw a claim.** Every other item on the board adds a column or fixes prose.

> ⚠ **Correction, 2026-09-04 — this paragraph used to overstate the odds.** It said
> BankSim/stratified "sits at inversion rate **0.01** — one hair away", implying the extension
> very likely converts "≥30×" into a measurement. **0.01 at `n_repeats=100` is one inverted draw**,
> so that condition is the *least* firmly established of the three, not the most nearly
> confirmed, and the grid extension alone cannot settle it. OBJ-17 is still the right next item —
> it is cheap, it is decisive either way, and continuing to publish a factor that rests on a
> coin-flip is the thing to avoid — but **it should not be sold as a likely recovery.** Full
> reasoning and the pre-registration are in [OBJ-17].

**⛔ Protect the Sep 16–19 block.** OBJ-7 and OBJ-9 are both flagged must-fix-before-publication
and both are writing, not code. If the third dataset overruns, **the dataset gets cut, not the
writing.** A half-run third column is worse than two clean ones — it is unusable *and* it invites
the lopsided-table criticism rule 8 exists to prevent.

**⛔ Go/no-go 2026-09-12 on OBJ-16.** If the third dataset's loader is not producing sane sweeps
by then, drop it and spend the remainder on OBJ-7/OBJ-9. Decide on the date, not on sunk cost.

### Standing instruction from the operator

**Ask questions before reaching any conclusion.** Verifying numbers is not sufficient — on
2026-09-01 a direction review with entirely correct numbers reached an entirely wrong conclusion,
because it judged the system by one component. Check scope and yardstick *first*.

---

## ▮ DIRECTION REVIEW — 2026-09-01 (rev. B)

> Re-read before opening any new dataset objective. Answers "are we on the right path?"
> **Rev. B corrects rev. A, which judged the thesis by ADTCN's classifier ranking. That was the
> wrong yardstick and it dropped four of the title's six claims.** Kept as a warning, below.
>
> **↳ AMENDED 2026-09-04.** Rev. B's *direction* stands and is confirmed — the expansion was
> under-done, and pushing on it is what produced the control run. Two things in it are now stale:
> the **Krum row** in the claims table below (the utility flip is not an entity-disjointness
> result — withdrawn, see CONFIRMED KILLS) and its **order of operations**, superseded by
> [THE LOCKED PLAN](#-the-locked-plan--t-15-to-2026-09-19) in RESUME HERE. Read the plan, not
> the list at the bottom of this section.

### What the system actually is

The unit of work is **FL-ADTCN**, not ADTCN. The detector is one component of six:

| Component | Job in the system | Title claim it backs |
|---|---|---|
| **Federated ADTCN** | per-org local detector; `extract_weights_with_dp()` is the sharing channel | (substrate — backs none on its own) |
| **DP** (Gaussian, Dwork 2006) | noises shared weights so raw gradients never leave an org | **Secure** (privacy half) |
| **Krum** (Blanchard 2017) | picks *which* org's weights become the global model; rejects outliers/poisoners | **Secure** (Byzantine half) |
| **Shapley** (FedSV 2020) | decides *how much* each org earns — contribution attribution, not security | **Incentivized**, **Scalable** |
| **DB-BOA** | hyperparameter + leader search (Job 3 aggregation-weight role is dead code) | (automation; backs none directly) |
| **RL leader / Fabric** | Q-learning leader election; on-chain token + reputation ledger | **RL**, **Blockchain-Integrated**, **Consensus** |

Krum and Shapley are deliberately independent: **Krum = security, Shapley = fairness.** A
Krum-rejected org can still earn tokens. Do not describe them as two steps of one pipeline.

### ⚠ The correction to rev. A

Rev. A concluded "ADTCN keeps losing, so reframe away from it." Two errors:

1. **Wrong unit.** ADTCN's rank among seven classifiers is a component ablation. It bears on none
   of the six title claims. Whether ResNet-1D out-scores ADTCN by 0.27 MCC on ULB does not touch
   whether DP breaks the incentive layer, whether Krum rejects poisoners, or whether Shapley
   attribution scales.
2. **Wrong contribution set.** Rev. A's "C1 + C2" silently dropped Secure-Byzantine, Blockchain,
   Consensus and RL. The title makes six claims; a direction review must score all six.

Rev. A's *findings* stand (the numbers were verified). Its *conclusion* — deprioritise datasets,
reframe the title — does not. **Reverse it: the dataset expansion is under-done, not over-done.**

### The real gap: five of six title claims are still single-dataset

This is the finding that matters, and rev. A missed it entirely.

| Title claim | Backing evidence | Datasets tested | Replicated? |
|---|---|---|---|
| **Secure** (privacy) | DP collapse → MCC ≈ 0 | ULB **+ BankSim ×2 splits** | ✅ **yes — and direction flips** |
| **Secure** (Byzantine) | Krum rejects 8/8 | ULB **+ BankSim ×2 splits** | ✅ **rejection replicates 8/8 in all three.** Utility does go negative off ULB — but ~~*because of entity-disjointness*~~ is **withdrawn 2026-09-04**; it tracks the dataset |
| **Incentivized** | 2-of-3 collusion caught | ULB **+ BankSim/entity-disjoint** | ✅ **collusion replicates (+47.4/+89.7); lone-attacker story CHANGED — isolation fires 3/3 but helps 0/3** |
| **Incentivized** (privacy↔incentive) | output channel ≫ weight channel | ULB **+ BankSim/entity-disjoint** | ✅ **ordering replicates at every ε; factor ≥60× → ≥10×, and both are censored lower bounds** |
| **Scalable** (attribution) | exact-Shapley cost | ULB **+ BankSim/entity-disjoint** | ✅ **cost replicates (structural); “MC top-1 fails from n=6” does NOT — it is intermittent on both** |
| **RL / Consensus / Blockchain** | Gini 0.90→0.782; Fabric 2115.9 ms, 40.3 TPS | dataset-independent | ✅ n/a by design |

**Mechanical cause, verified:** only **2 of 18** scripts in `experiments/` accept `--dataset`
(`basepaper_comparison.py`, `objective_noise_audit.py`). `byzantine_robustness_sweep.py`,
`economic_byzantine_sweep.py`, `scalability_sweep.py` and `private_incentive_sweep.py` all
hardcode `FinancialDataLoader()` — the ULB loader — and **none of their JSONs record which
dataset produced them.** The expansion reached the detector and the federated ablation and
stopped at the door of the incentive and Byzantine layers. That is OBJ-15.

### Why this is the right path, restated correctly

Your instinct — "real life has multiple parameters and criteria, so bring in more datasets" — is
right, and BankSim already proved it pays at the system level:

- **The DP collapse replicated.** The privacy↔incentive contribution now rests on two datasets.
- **Its direction is dataset-dependent** — ULB all-negative, accuracy moves −0.11 pp (looks
  healthy, is useless); BankSim all-positive, 119,784 FP, accuracy −97.55 pp. A reviewer cannot
  dismiss this as a ULB artefact any more, *and* it produced a new claim neither dataset gives alone.
- **Krum's premium turned out constant** (−0.037 stratified, −0.033 entity-disjoint), killing our
  own prior that heterogeneity would give Krum something to reject. ULB's +0.207 is the outlier.

That is exactly what a second dataset is for. **Now finish the job on the other four claims.**

### ~~Order of operations~~ → **SUPERSEDED 2026-09-04 by [THE LOCKED PLAN](#-the-locked-plan--t-15-to-2026-09-19)**

Kept for the record of what was believed on 2026-09-01. Three of its items have since moved:

1. ~~**OBJ-14**~~ — **DONE 2026-09-01.** Purged, plus three breaches outside its own file list.
2. ~~**OBJ-15**~~ — **DONE 2026-09-01.** All four sweeps on BankSim/entity-disjoint.
3. ~~**BankSim/stratified control**~~ — ✅ **DONE 2026-09-01 23:41, scored 2026-09-04.** It did
   what it was for and more: it overturned one of OBJ-15's own conclusions.
4. **OBJ-13** — fix or retire the DB-BOA surrogate. **Still on the plan (Sep 7–10)**, still early,
   still for the same reason: it invalidates `db_boa_results.json` and every `baselines*.json`
   DB-BOA arm.
5. ~~**OBJ-16** … AMLSim leads~~ — **RE-GATED 2026-09-04. AMLSim is demoted; the Fraud Detection
   Handbook leads.** The control run already ran the experiment AMLSim was uniquely for.
6. **OBJ-6** — Shapley under the entity-disjoint split; values still near-uniform
   `[0.167, 0.165, 0.165]`. **Not on the locked plan** — no window before 2026-09-19.
7. **OBJ-7** — formalise the DP guarantee. **On the plan, Sep 16–19, protected.**
8. **OBJ-9** — title wording. **On the plan, Sep 16–19, protected.** Write it against the
   *post-control* qualifications: the withdrawn Krum mechanism, the condition-dependent censored
   ε\*, the intermittent top-1.

**New since this list was written:** [OBJ-17] (de-censor ε\*, the cheapest win on the board) and
~~[OBJ-18]~~ (propagate the Krum withdrawal out of TASK.md — **closed 2026-09-04**).

### Where ADTCN and DB-BOA honestly sit

Report them as **measured components, not claimed contributions.** ADTCN never ranks better than
5 of 7; DB-BOA ties four optimisers and a hand-set default, with OBJ-13 giving the mechanism.
Neither is a title claim, so neither needs to win — but neither may be *described* as winning.
"We built it, measured it honestly, and it did not beat the simple baseline" is a legitimate
component result in a systems paper. Rule 3 already covers this; it does not require a reframe.

---

## ▮ PRIMARY OBJECTIVES

### ◈ ~~OBJ-1 — RECON: BankSim entity-linked temporal test~~  → CLOSED, see CONFIRMED KILLS

**Status:** `DONE 2026-08-31` · verdict recorded in `final_report_data/OBJ1_banksim_temporal_grid.md`, numbers in `results/banksim_temporal_grid.json`.

---

### ◈ ~~OBJ-2 — CLEANUP: kill the MCC contradiction~~  → CLOSED, see CONFIRMED KILLS

**Status:** `DONE 2026-08-31` · verdict in `final_report_data/OBJ2_detector_multiseed.md`, numbers in `results/detector_multiseed.json` (13 runs). Closed together with OBJ-4.

---

### ◈ ~~OBJ-3 — EXFIL: get the measured Fabric data into the report~~  → CLOSED, see CONFIRMED KILLS

**Status:** `DONE 2026-08-31` · Chapter 6 §`sec:fabric-measured`. Report compiles clean, all refs resolve.

---

## ▮ SECONDARY OBJECTIVES

### ◈ ~~OBJ-4 — Multi-seed the detector~~  → CLOSED, see CONFIRMED KILLS

`DONE 2026-08-31` · 5 seeds × 2 configs × 30 epochs. Tuned **0.753 ± 0.055**, default **0.706 ± 0.076**, difference not significant (p=0.34).

### ◈ ~~OBJ-1b — Base-paper model comparison~~  *(added 2026-08-30)*

`DONE 2026-08-31` · The base paper compares against four classifiers (EfficientNet, ResNet, DenseNet, DTCN) and four optimisers (MBO, WSA, DBOA, BOA). D4 killed the copied table; the honest replacement is to re-implement all eight and run them ourselves.

- [x] Classifiers re-implemented for 1-D transaction windows — `models/basepaper_models.py`. DTCN is deliberately ADTCN-minus-attention, so that head-to-head is a single-factor ablation rather than two unrelated networks.
- [x] Optimisers completed — `algorithms/mbo.py` (Mine Blast, Sadollah 2013) and `algorithms/wsa.py` (Water Strider, Kaveh 2020); BOA/DBOA/DB-BOA already existed. All five verified to converge on a sphere function.
- [x] **ULB classifier track done** (2026-08-31, 9.2 h, 3 seeds, time-ordered split, 10 ep / 32 filters) — `results/basepaper_comparison_ulb.json`, draft in `final_report_data/BASEPAPER_comparison_ulb.md`. ResNet-1D 0.5995 > CNN 0.4974 > DenseNet 0.4516 > LSTM 0.3607 > **ADTCN 0.3296 (5th of 7)** > EfficientNet 0.3248 > DTCN 0.2794.
- [x] **BankSim classifier track done** — it *is* the OBJ-1 grid's customer-window column (all 7 architectures, both orderings, 3 seeds). No separate run needed: `results/banksim_temporal_grid.json`.
- [x] **BankSim optimiser track done** (2026-08-31, 6.0 h) — `results/basepaper_optimisers_banksim.json`. DB-BOA best mean Obf2 (4.2123) and fastest search, but **the five optimisers are statistically indistinguishable on test MCC**: spread 0.035 vs a largest per-optimiser seed std of 0.116.
- [x] **ULB optimiser track done** (2026-08-31, 2.0 h) — `results/basepaper_optimisers_ulb.json`. **All five hit Obf2 = exactly 5.0000**, the theoretical ceiling. That is not a result about optimisers; see **OBJ-13**.
- [x] ⚠ ~~`basepaper_comparison.py` writes its JSON only at the end~~ — **fixed**. Both `basepaper_comparison.py` and `banksim_temporal_grid.py` now checkpoint after every cell via a temp-file-and-rename (never a half-written JSON), carrying a `checkpoint.complete` flag. Verified progressive with a stubbed trainer.
- [x] ⚠ **Operational trap found the hard way:** `run_baselines.py` (and `main.py`) print `─` separators; with stdout **redirected to a file** on Windows, Python falls back to cp1252 and dies with `UnicodeEncodeError`. It never shows up interactively. Two federated jobs died *after* their 35-min DB-BOA search. **Always export `PYTHONIOENCODING=utf-8` when redirecting these scripts.**

**Correction to an earlier claim in this file.** The optimiser budget is matched on *population and iteration count*, **not** on objective evaluations — those range 90–170 across the five (a 1.9× spread) because each algorithm calls the fitness a different number of times per iteration. The drafts now print per-optimiser eval counts. Search wall-clock is likewise **confounded** — the surrogate costs more when a candidate proposes more filters, so MBO's 2727 s vs DB-BOA's 331 s mostly reflects where each wandered, not efficiency. Do not quote it as a speed comparison (which is exactly what the base paper's Table 5 does).

> **No number from Prabanand & Thanabal (2025) is reproduced.** Their results are MATLAB runs on their own data. We re-run their methods on ours; the filter ceiling is cut to 64 to fit a CPU budget, identically for every optimiser, and that cut is reported next to the numbers.

---

### ◈ OBJ-13 — The DB-BOA *search* selects on noise  *(added 2026-08-31)*

`SEARCH WORK COMPLETE 2026-09-05 · downstream re-runs outstanding` · **Priority: HIGH** · This is a mechanism for rule 3, not a reopening of OBJ-2.

> **One-line result:** the repair works, changes what the search picks, and **still does not beat
> a hand-set 128/150** — 0 of 9 configurations, mean paired Δ −0.1982. Rule 3 governs: this was
> diagnosis of a negative result, and the negative result held.

> ## ▸ APPLIED 2026-09-04 — what landed, what it cost, and what was wrong in this objective
>
> **Rescue first, as instructed.** Both staged files were still in the session temp dir and are
> now durable at `db_boa_framework/scratchpad/` (**tracked**, not ignored — they *are* the record
> of a decision, and `scratchpad/` is deliberately absent from `.gitignore`). `--check` resolved
> all four anchors on the current files; applied clean; both files parse. Backups at
> `models/adtcn.py.pre_obj13` and `config.py.pre_obj13`.
>
> ⛔ **Do not delete those two `.pre_obj13` files, and do not add them to `.gitignore`.**
> `models/adtcn.py` carried **uncommitted** modifications when the patch was applied, so the exact
> pre-patch state is **not recoverable from git history** — `.pre_obj13` is its only copy, and it
> is what the bit-for-bit verification loads. Delete it and the central claim of this objective
> stops being checkable. (Verified they are not matched by any `.gitignore` rule.)
>
> **✅ `legacy` is bit-for-bit, and that is now *measured*, not asserted.**
> `scratchpad/verify_obj13_legacy.py` loads `adtcn.py.pre_obj13` and the patched module side by
> side and compares them on identical data at identical seeds. Same subsample, and three
> configurations agree to all ten decimals (−4.1746699152 · −5.0000000000 · −4.9288277568). The
> RNG consumption order matches too (permutation, then `randint` for the torch seed), which is
> what makes it exact rather than merely close. So the side-by-side runs against a **live**
> baseline, as designed. The same script confirms the rest of the patch does what it says:
> legacy gives **4 distinct scores for one config in 4 calls**; deterministic gives **1 in 4**
> and reproduces across separately-constructed objects; `averaged` shares its draws with
> `deterministic` (common random numbers); stratified draws carry fraud on both sides; and the
> dead axis is alive — batch size **28 · 14 · 10 · 7 · 6** across spe ∈ [50,250] where it was
> **32 · 32 · 32 · 32 · 32**.
>
> **⚠ THREE DEFECTS FOUND IN THIS OBJECTIVE'S OWN PLAN — all three would have shipped.**
>
> 1. **⛔ The patch silently re-pointed `objective_noise_audit.py` — the script that produced
>    this objective's entire evidence base.** It constructs `_ADTCNObjective` with no
>    `eval_mode`, so after the repair it would have inherited the new `deterministic` default
>    and reported a within-seed std of **exactly 0.0000** — not because the noise was ever
>    absent, but because `deterministic` freezes one draw and so hides it. A 0.0000 there reads
>    as *"the objective was always fine"*, which is the precise opposite of the finding. **Fixed:
>    the audit now pins `eval_mode="legacy"` explicitly at every construction site**, with the
>    reason written at each one so it is not "simplified" away later. The audit measures the
>    pre-repair objective **on purpose**.
> 2. **⛔ Same silent switch in `basepaper_comparison.py`,** whose two optimiser JSONs on disk
>    are the "all five hit 5.0000" evidence. A re-run would have written post-repair numbers over
>    them under the same filename. **Fixed:** `--eval-mode` (default `legacy`, so a re-run still
>    reproduces what is on disk), the mode is **stamped into the JSON and into the draft's
>    header**, and the script now **refuses to overwrite** a file recorded under a different
>    mode — checked before the loader runs, so it costs seconds. A warning would not do; a
>    warning scrolls off the top of a six-hour run.
> 3. **⛔ Pre-existing, found by triggering it: `write_draft` derived the draft filename from the
>    *dataset*, not from the results file**, so the classifier track and the optimiser track both
>    wrote `BASEPAPER_comparison_<ds>.md` and silently overwrote each other — whichever ran last
>    was the only one on disk. **Fixed:** the name now derives from the JSON
>    (`basepaper_optimisers_ulb.json` → `BASEPAPER_optimisers_ulb.md`), `basepaper_comparison_ulb.json`
>    still maps to `BASEPAPER_comparison_ulb.md` so no reference breaks, and all three drafts are
>    regenerated and verified (the ULB classifier table matches the numbers recorded in OBJ-1b
>    exactly). **This is the third instance of the same class** — OBJ-15's unsuffixed figures, the
>    shared per-run log directory, and now this.
>
> **⚠ CORRECTION — the downstream cost written below is overstated, and checking cost one grep.**
> It says the repair invalidates "the `DB-BOA-ADTCN` arm of `baselines.json` /
> `baselines_banksim_stratified.json` / `baselines_banksim_customer.json`". **The two BankSim
> files were produced with the search SKIPPED.** Both record
> `"detector_width_source": "pinned (DB-BOA search skipped; see experiments/objective_noise_audit.py)"`
> and `optimal_params` 32/150 — the hand-set default, not a search output. They are **not**
> invalidated. Only `baselines.json` (ULB, 142/76) and `db_boa_results.json` (ULB, 142/76, whose
> `db_boa_stats` max is **−4.999999999991326** — the ceiling artefact itself) carry a searched
> configuration.
>
> **➕ And one consumer the list missed:** `experiments/detector_multiseed.py:72` **hardcodes**
> `"dbboa_tuned": {"hidden_neurons": 142, "steps_per_epoch": 76}`. That is the config behind
> OBJ-4's headline **+0.047 MCC, p=0.34** — so that number compares a *noise-selected* config
> against the default. It is not wrong (rule 3: the tie stands as measured), but the re-assessment
> checkbox below is precisely a re-run of this script with the repaired search's config as an
> additional arm, under identical seeds and thread count.
>
> **Net: the re-run bill is smaller than written (2 files, not 4) but reaches one script that was
> not on the list.**

> ## ⛔⛔ NEW DEFECT, FOUND 2026-09-04 — the shipped ULB surrogate validates on rows it trained on, and the OBJ-13 repair does NOT fix it
>
> Evidence: `results/obj13_shipped_path_check.json`, produced by
> `experiments/obj13_shipped_path_check.py` (**zero training, runs in ~2 min**).
>
> **The audit has never measured the pipeline that ships.** `objective_noise_audit.py` builds its
> surrogate from `basepaper_comparison._eval_subset(d, n=6000)`. `main.py:144` and
> `run_baselines.py` instead call `loader.get_eval_subset(...)`, which returns a stratified
> **`eval_subset` = 3,000-row** pool (`config.py:31`). Those two pools do not hold the same number
> of *unique* fraud transactions, and the gap lands on this branch of `_ADTCNObjective.__init__`:
>
> ```python
> n_f       = max(self._MIN_FRAUD_ROWS, int(n_rows * fraud_rate))   # = 30 on ULB
> replace_f = len(fraud_idx) < n_f            # 5 < 30  ->  True
> idx       = rng.choice(fraud_idx, n_f, replace=replace_f)
> ```
>
> | pool | unique fraud | draws n_f | with replacement? | duplication |
> |---|---|---|---|---|
> | **ULB shipped** (`get_eval_subset`, 3,000) | **5** | 30 | ⛔ **yes** | **6.01×** |
> | ULB audit (`_eval_subset`, 6,000) | 30 | 30 | no | 1.0× |
> | BankSim shipped (3,000) | 38 | 30 | no | 1.0× |
> | BankSim audit (6,000) | 75 | 30 | no | 1.0× |
>
> **And the leak is total, measured not inferred.** The check replays the index construction
> (verifying it bitwise against the live object first) and traces every validation fraud row back
> to its pool identity:
>
> | path | unique fraud | train/val fraud rows | val positives that are copies of a TRAIN row |
> |---|---|---|---|
> | ULB shipped, `legacy` | 5 | 21 / 9 | ⛔ **9 of 9** |
> | ULB shipped, `deterministic` | 5 | 21 / 9 | ⛔ **9 of 9** |
> | BankSim shipped, `legacy` | 30 | 20 / 10 | 0 of 10 |
> | BankSim shipped, `deterministic` | 30 | 21 / 9 | 0 of 9 |
>
> **What this changes.**
> 1. **It replaces OBJ-13's stated mechanism with a sharper one.** "`Obf2 = 5.0000` means
>    perfectly classifying about nine rows — reachable by luck" is not what happens on ULB. All
>    nine of those validation positives are **copies of transactions in the training half**, so
>    5.0000 means *recognising five memorised rows*. Luck is not required.
> 2. **It explains the ULB↔BankSim asymmetry that OBJ-13 could not.** The objective file says
>    "the fraud rate is not what separates ULB from BankSim; task separability at n≈9 positives
>    is." That is wrong. The fraud rate **is** what separates them — through the
>    `len(fraud_idx) < _MIN_FRAUD_ROWS` threshold, which ULB crosses (5 < 30) and BankSim does
>    not (38 ≥ 30).
> 3. ⛔ **The staged repair does not fix it.** `deterministic` leaks 9/9 exactly as `legacy` does.
>    Stratifying the split cannot help when the 30 rows being stratified are 5 transactions
>    repeated ~6× — the copies land on both sides either way. **The surrogate is still broken on
>    ULB after OBJ-13.**
> 4. **Both invalidated ULB files were produced through this path.** `db_boa_results.json` and
>    `baselines.json` (142/76) come from `main.py` / `run_baselines.py`, so their configuration was
>    chosen by a search validating on memorised rows. Regenerating them **without** fixing this
>    would faithfully reproduce the defect.
>
> **The fix is cheap and that is the surprising part.** The surrogate draws 2,000 rows from the
> pool regardless, so **enlarging the pool costs no training time at all** — only the stratified
> draw changes.
>
> ### ✅ FIXED 2026-09-04, approved by the operator. Both halves applied and verified.
>
> **(a) `eval_subset` 3,000 → 36,000** in `DATA_CONFIG` *and* `BANKSIM_CONFIG` (`config.py`).
> 36,000 gives ULB ~60 unique fraud — **2× `_MIN_FRAUD_ROWS`**, so the replacement branch has
> headroom rather than sitting one row from firing. BankSim never needed it (3,000 already held
> 38) but is raised with it so both datasets share one code path and neither carries an 8-row
> margin.
>
> **(b) A guard in `_ADTCNObjective.__init__`, because the cost of this defect was its silence.**
> The `replace_f` condition is now recorded on the object (`fraud_sampled_with_replacement`,
> `unique_fraud_available`) and raises a `RuntimeWarning` naming the duplication factor and the
> config key to change. The config value alone would fix today's datasets and let the next one
> reintroduce it without a sound.
>
> **Verified by re-running the check:**
>
> | path | unique fraud in pool | drawn | leaked val positives |
> |---|---|---|---|
> | ULB shipped, `legacy` | **62** | 30 distinct | **0 of 11** ✅ |
> | ULB shipped, `deterministic` | **62** | 30 distinct | **0 of 9** ✅ |
> | BankSim shipped, both | 455 | 30 distinct | 0 of 7 · 0 of 9 ✅ |
>
> ⚠ **This does not retract anything measured on the audit's pool.** `objective_noise_audit.py`
> and the knee sweep both build from `_eval_subset(n=6000)`, which held 30 unique fraud and never
> duplicated — their numbers stand exactly as recorded. What changes is the **shipped** path, and
> therefore `db_boa_results.json` and `baselines.json`, which are now invalid for a second and
> much stronger reason than the one OBJ-13 originally gave.

#### ▸ PRE-REGISTERED 2026-09-04, before any repaired run — rule 4. Score underneath; do not edit above the line.

**1. The ceiling survives `deterministic` and dies under `averaged`.** The 5.0000 ceiling is
reachable because the surrogate holds 30 fraud rows and the 70/30 split leaves ≈9 in validation —
perfectly classifying nine rows is luck, not search. Freezing the draw does not add fraud rows, so
a best-of-N *over candidates* can still land on an easy frozen draw. Predicted: **ULB still
reaches or nearly reaches 5.0000 under `deterministic`**, and **best Obf2 falls clearly below the
ceiling under `averaged` k=3**, which needs three draws perfect at once. ⚠ **If `deterministic`
also collapses well below the ceiling, the redraw was the whole story and the "n≈9 positives"
mechanism written above is wrong** — that is the more valuable outcome and it is named in advance.

**2. The chosen configuration still scatters across objective seeds under every mode.**
Determinism removes *within-seed* variance, not *between-seed*: the subsample still sets the
difficulty (ULB between-seed mean spread 1.0471 against a config-to-config range of 1.0794).
Predicted: filter counts chosen at seeds 42/7/123 stay widely scattered in all three modes. **If
they converge under `deterministic`, the between-seed spread was itself an artefact of the redraw
and the audit's ratio-≈1 finding has to be revisited.**

**3. On a held-out yardstick the three modes are separated by less than the seed noise within any
one of them.** The modes' own "best Obf2" values are **not comparable** — legacy's is a max over
noisy draws, averaged's is a mean — so each mode's returned config is re-scored on draws **no
search saw** (a fresh draw seed), which is the only comparison that means anything. Predicted
ordering `averaged` ≥ `deterministic` ≥ `legacy`, with **all gaps smaller than the seed-to-seed
spread inside a mode**. That is the honest expected outcome: the repair fixes the *mechanism*
without producing a better model, because `noise_to_signal` (0.74 ULB / 0.94 BankSim) says the
configs barely span a rankable band in the first place.

**4. `steps_per_epoch` now varies but buys little.** With the floor gone the axis moves batch size
28→6 across the range. Predicted: chosen `spe` still scatters, and held-out score depends on it
less than on filter count. **If `spe` turns out to matter substantially, the dead axis cost a real
dimension** — that is a finding, not merely a bug, and it must be reported as one.

**5. DB-BOA still ties the hand-set default on test MCC** (restated from the decision block below
so it gets scored). Rule 3: the repair is diagnosis of a negative result, not a rescue attempt.

<!-- SCORING BELOW THIS LINE. Nothing above it has been edited since it was written. -->

#### ▸ SCORECARD — item 6 scored 2026-09-04 from `results/objective_size_knee.json`. Items 1–5 await the side-by-side.

| # | Outcome |
|---|---|
| **1** | ⚠ **UNSCOREABLE from this run** — the claim is ULB-specific and the run is BankSim. |
| **2** | ❌ **WRONG for `averaged`, and the failure is the best result in the objective.** |
| **3** | ❌ ordering **WRONG** (`averaged` is *worst*) · ✅ structure **held**. |
| **4** | ✅ held, plus a defect found: the dead axis was **surrogate-only**. |
| **5** | ⏳ needs `detector_multiseed`; the surrogate-side evidence is **0 of 9**. |
| **6** | ❌ **WRONG ON BOTH HALVES, and the pre-registration named this exact alternative in advance.** See below. |

#### ▸ THE HEADLINE — `results/obj13_surrogate_repair_banksim.json`, 9/9 runs, 16.3 h, exit 0

**Not one of the nine searched configurations beats a hand-set 128 filters / 150 steps-per-epoch.**
Mean paired difference across all nine: **−0.1982**. The shipped 142/76 is also below, at −0.1900.
Paired per-draw on the 7 shared yardstick draws (common random numbers), so the ±0.4 spreads on
the individual means are draw-to-draw variation that **cancels** — judging these as
indistinguishable from the unpaired spreads would be wrong:

| Mode | Seed | Chose | paired Δ vs default | SE | wins/7 |
|---|---|---|---|---|---|
| `legacy` | 42 · 7 · 123 | 81f · 69f · 135f | −0.2027 · −0.1130 · −0.1780 | .036 · .096 · .087 | 0 · 1 · 1 |
| `deterministic` | 42 · 7 · 123 | 137f · 56f · **62f** | −0.1654 · −0.1647 · **−0.0720** | .031 · .191 · .168 | 0 · 2 · **4** |
| `averaged` | 42 · 7 · 123 | 137f · 146f · 149f | −0.1654 · −0.3512 · −0.3716 | .031 · .169 · .157 | 0 · 1 · 1 |
| _shipped_ | — | 142f/76spe | −0.1900 | .163 | 2 |

**Item 2 — ❌ WRONG for `averaged`, and this is the objective's most valuable finding.**
Predicted: the chosen configuration keeps scattering across objective seeds in **every** mode,
because determinism removes within-seed variance while the subsample still sets the difficulty.

| Mode | filters chosen | spread |
|---|---|---|
| `legacy` | 81, 69, 135 | 66 |
| `deterministic` | 137, 56, 62 | **81 — wider than legacy** |
| `averaged` | 137, 146, 149 | **12** |

**Determinism alone does not make the search reproducible — it is slightly *worse* than legacy.
Only shared-draw averaging converges it** (12 out of a 5–255 range). That separates the two
halves of the repair cleanly, and it was not predicted: the common-random-numbers design, added
almost as an implementation detail, is the part that does the work. `spe` still scatters
everywhere (47 · 49 · 58).

**Item 3 — ❌ ordering WRONG, ✅ structure held.** Predicted `averaged` ≥ `deterministic` ≥
`legacy`. Measured means: `deterministic` **3.5857** > `legacy` **3.5552** > `averaged`
**3.4237**. **`averaged` is the worst of the three on held-out draws.** The structural half held:
between-mode spread **0.162** is smaller than the largest within-mode seed spread **0.206**.

> **Mechanism, and it is worth stating because it is counter-intuitive.** `averaged` *converges*
> (item 2) — onto high-filter configurations (137, 146, 149) that generalise **worse**. With only
> k=3 shared draws, a search that can now rank reliably will exploit *those particular three
> draws*, and extra capacity helps it do so. **The noise the repair removed was acting as
> accidental regularisation.** Making the objective rankable made the search better at
> overfitting it. Both halves of that sentence are measured, and neither is a reason to prefer
> the broken objective.

**Item 4 — ✅ held, and it turned up a defect nobody had noticed.** `spe` scatters in every mode.
The best searched configuration is `deterministic`/123 at **62f/160spe** (Δ −0.0720, 4/7 wins) —
the *lowest*-filter pick — while the three worst are all ≥137 filters, so held-out score tracks
filter count more than `spe`, as predicted.
⛔ **But `ADTCN.fit` line 681 still uses the OLD floored formula** `batch_size = max(32,
len(X_train) // spe)`. With ~400 k training rows the floor never binds there, so **the final
model's `spe` axis was always live — only the surrogate's was dead** (n_train = 1400 put every
value under the floor). The search was therefore blind to a dimension the deployed model
genuinely responds to. That is worse than "one axis was wasted", and it is *not* fixed by the
OBJ-13 patch, which only touched `_score_once`.

**Item 6, in full.** Predicted: BankSim's `noise_to_signal` improves markedly with size (its fraud
count escapes the floor at ≈2.5k) while ULB's barely moves (pinned at 30 fraud until ≈17.6k).
Measured, at 25 draws per cell, seed 42:

| rows | ULB fraud | ULB std | ULB n2s | BankSim fraud | BankSim std | BankSim n2s |
|---|---|---|---|---|---|---|
| 2,000 | 30 | 0.5225 | 0.79 | 30 | 0.2563 | 0.51 |
| 5,000 | 30 | 0.4820 | 0.56 | 63 | 0.2947 | 0.67 |
| 10,000 | 30 | 0.3512 | **0.39** | 126 | 0.2465 | **0.36** |
| 20,000 | 36 | 0.4377 | 0.64 | 252 | 0.2553 | 0.64 |

- **ULB improved sharply *while pinned at the floor*** — 0.79 → 0.39 across 2k→10k with the fraud
  count fixed at exactly 30. The prediction was that it would barely move.
- **BankSim's noise did not fall at all despite 8× the fraud rows** — std 0.2563 → 0.2553 while
  fraud went 30 → 252. The prediction was a marked improvement.
- **By the pre-registration's own words: "if ULB improves anyway, the fraud count is not the
  driver".** It is not. On either dataset.

**Two things the sweep found that were not asked for, and both limit what it can conclude:**

1. ⚠ **The curve is non-monotone and the confound is structural.** Both datasets bottom out at
   10,000 and get *worse* at 20,000. `batch_size = max(32, n_train // spe)` means surrogate size
   and training budget move together: gradient updates run **215 → 545 → 760 → 750** while batch
   doubles to 93 at 20k. So "more rows" is never varied alone, and the 20k regression is
   plausibly undertraining rather than anything about size. **The repaired formula
   `ceil(n_train / spe)` has the same property** — fixing steps means batch grows with data — so
   this is a property of the `steps_per_epoch` parameterisation, not of the bug that was fixed.
2. ⛔ **`noise_to_signal` moves almost entirely through its DENOMINATOR, which is the noisiest
   thing in it.** Within a dataset `std` is stable (BankSim 0.2465–0.2947 across a 10× size
   change) while `range` swings 0.79–1.81. That range is the max−min of **8 single evaluations**,
   each carrying the same noise the numerator measures — so part of the denominator is noise
   divided by itself. Two consequences: **(a) the threshold for this ratio is not 1.0**, and a
   bootstrap null (resampling the fixed-config draws, now computed by the audit as
   `null_range_test`) is the only honest reference; **(b) with one objective seed per size, the
   size effect cannot be separated from the subsample effect** — the same nominal setup measured
   std **0.4439** in the main audit and **0.2563** here, a 1.7× swing from the draw alone.
   `--knee-seeds` was added to make that measurable rather than assumed.

> **So the knee checkbox is answered, but not with a number.** The honest output is: *the curve,
> plus the finding that a knee is not locatable at this precision, plus the falsified mechanism.*
> Picking 10,000 because it is the argmin of eight single-seed cells would be reading a subsample
> draw as a property of the surrogate — the same error, one level up, that this whole objective
> is about.

**6. The knee splits by dataset, and the mechanism says which way.** `n_f = max(30, rows × rate)`,
so the `_MIN_FRAUD_ROWS = 30` floor stops binding at **≈2.5k rows on BankSim** (1.211 %) but only
at **≈17.6k on ULB** (0.173 %). Across the {2k, 5k, 10k, 20k} grid BankSim therefore gains real
fraud rows from 5k upward while ULB stays pinned at 30 fraud rows until the very last point.
Predicted: **BankSim's `noise_to_signal` improves markedly with size; ULB's barely moves until
20k.** **If ULB improves anyway, the fraud count is not the driver** and the "n≈9 positives"
mechanism is wrong — the same alternative as (1), reached by a second road.

---
#### ▸ SCORED 2026-09-04 — the shipped-path check. ⛔ The surrogate's ULB validation positives were **memorised**, and the repair does not fix it.

`experiments/obj13_shipped_path_check.py` → `results/obj13_shipped_path_check.json`.
Zero training; ran in seconds. Two stages, the second added when the first confirmed:
stage 1 replays `_ADTCNObjective.__init__`'s fraud arithmetic on the pool each caller
actually supplies; stage 2 takes the real objective's own draw and asks whether the
duplicated rows land on **both** sides of the 70/30 split. **All four stage-2 replays
verified bitwise against the live object** (`replay_verified_bitwise: true`) before any
number below was read.

| Pool | rows | unique fraud | surrogate draws | distinct fraud in surrogate |
|---|---|---|---|---|
| **ULB — SHIPPED** (`get_eval_subset`, `eval_subset=3000`) | 3,000 | **5** (0.167 %) | n_f=30 **⛔ with replacement** | **4.99 — duplication 6.01×** |
| ULB — audit (`basepaper._eval_subset(n=6000)`) | 6,000 | 30 (0.500 %) | n_f=30, no replacement | 30.0 — 1.0× |
| BankSim — SHIPPED | 3,000 | 38 (1.267 %) | n_f=30, no replacement | 30.0 — 1.0× |
| BankSim — audit | 6,000 | 75 (1.250 %) | n_f=30, no replacement | 30.0 — 1.0× |

**Stage 2 — the leak, measured rather than argued:**

| Dataset / mode | unique fraud | train / val fraud rows | val positives that are copies of a train row |
|---|---|---|---|
| **ULB / `legacy`** | 5 | 21 / 9 | **9 of 9 — 100 %** ⛔ |
| **ULB / `deterministic`** | 5 | 21 / 9 | **9 of 9 — 100 %** ⛔ |
| BankSim / `legacy` | 30 | 20 / 10 | 0 of 10 — disjoint |
| BankSim / `deterministic` | 30 | 21 / 9 | 0 of 9 — disjoint |

**Three consequences, in order of how much they cost.**

1. **⛔ `Obf2 = 5.0000` on the deployed ULB path is not a classification result.** This
   objective's stated mechanism — *"perfectly classifying about nine rows — reachable by
   luck"* — is **wrong for the path that ships**. It is not luck. The nine validation
   positives are copies of transactions the surrogate trained on, every time. The
   ceiling in `db_boa_results.json` (`db_boa_stats` max −4.999999999991326) and the
   five-optimiser 5.0000 tie now have a sharper cause than "n≈9 positives is easy":
   the val set was in the train set.

2. **⛔ The repair as designed does not close this hole — and that was not predicted.**
   Stratifying the 70/30 split guarantees positives on *both* sides; with only 5 unique
   transactions repeated ~6×, that is precisely what puts copies on both sides.
   `deterministic` leaks 9/9, identically to `legacy`. **`eval_mode` is the wrong layer
   for this defect**: the duplication happens in `__init__`, before any draw. Closing it
   needs the *pool* changed, not the split — see the operator call below.

3. **⚠ Neither the audit nor the knee sweep describes the shipped path.**
   `objective_noise_audit.py` and `knee()` both build their pool with
   `basepaper._eval_subset`, whose own `n_f = min(len(f), max(30, int(n·rate)))` floor
   **guarantees ≥30 unique fraud in the pool** — the exact condition that fails at
   deployment. So ULB's headline `noise_to_signal = 0.74`, and every ULB row of the knee
   table, are measured on a surrogate holding 30 *distinct* fraud rows, where the
   deployed one holds 5. They are not wrong — they answer a different question, and one
   worth answering for the repaired design — but **no ULB audit number may be quoted as
   a description of what the search actually did.** BankSim is unaffected: its shipped
   and audit pools agree at 30 distinct.

> **Correction to this objective's own text, above the pre-registration line.** It states
> *"`_MIN_FRAUD_ROWS = 30` fires on both datasets, so both surrogates hold exactly 30
> fraud rows … the fraud rate is not what separates ULB from BankSim; separability at
> n≈9 positives is."* Both surrogates hold 30 fraud **rows** — but on the shipped path
> ULB's are 5 transactions and BankSim's are 30. **On the deployed pipeline the fraud
> rate is exactly what separates them**: it is what drops the 3,000-row pool below 30
> unique positives (ULB 5, BankSim 38). The original sentence is true of the audit pool
> only, and is left above the line unedited per rule 4.

**Consumers of the leaking path, checked not assumed.** `get_eval_subset` has exactly two
callers outside this check: `main.py:144` and `run_baselines.py:148`. So the ULB
configuration **142/76** carried by `db_boa_results.json` and `baselines.json` was chosen
by a surrogate whose validation positives were 100 % memorised — and
`detector_multiseed.py:72` hardcodes that same 142/76 as `dbboa_tuned`, so OBJ-4's
**+0.047 MCC, p=0.34** headline inherits it. The BankSim baselines skipped the search and
are untouched, as already corrected above.

⛔ **OPERATOR CALL NEEDED — do not pick one of these unilaterally; it changes what the
repaired search means.** The floor is in the *pool*, not the split, so `eval_mode` cannot
reach it. Three ways to close it, with what each costs:

| | Fix | Effect | Cost / risk |
|---|---|---|---|
| **(a)** | Raise `eval_subset` for ULB to ≳17,400 rows (30 ÷ 0.00173) | 30 genuinely distinct fraud rows at the real rate | every surrogate evaluation gets slower; the knee log already shows ULB at 12.5 s/eval by 20k vs 3.1 s at 2k — a ~4× search bill |
| **(b)** | Give the pool the same fraud floor `_eval_subset` has, i.e. oversample fraud **into the pool** | matches what the audit and knee sweep already measure, so those become descriptions of deployment | the surrogate then sees a fraud rate that is not the dataset's — the exact defect `get_eval_subset`'s docstring says it was written to fix |
| **(c)** | Clamp `n_f = min(n_f, len(fraud_idx))` — never sample with replacement | honest: no leak, no duplication | ULB's surrogate then trains on **5** fraud rows; MCC at n≈2 validation positives is near-meaningless. Honest and useless |

**Recommendation: (a).** It is the only one that leaves the fraud rate real *and* the
positives distinct, and rule 2 prefers a slower honest measurement to a fast ambiguous
one. But it re-prices every remaining OBJ-13 run, which is why it is an operator call and
not a checkbox.

---

#### ▸ SCORED 2026-09-04 — the surrogate-size knee. Pre-registration 6 is **WRONG on both halves**, and it named this outcome as the valuable one.

`results/objective_size_knee.json` → `final_report_data/OBJECTIVE_size_knee.md`.
Completed 2026-09-04 20:38 (25 draws/cell, `legacy` mode, seed 42, fixed 40,000-row
pool so size is the only thing moving). `noise_to_signal` = 2·std ÷ config range;
**lower is better**, ≥1 means the search is ranking draws rather than models.

| rows | ULB fraud | ULB std | ULB range | **ULB n2s** | BankSim fraud | BankSim std | BankSim range | **BankSim n2s** |
|---|---|---|---|---|---|---|---|---|
| 2,000 (shipped) | 30 | 0.5225 | 1.3209 | **0.79** | 30 | 0.2563 | 1.0132 | **0.51** |
| 5,000 | 30 | 0.4820 | 1.7304 | **0.56** | 63 | 0.2947 | 0.8860 | **0.67** |
| 10,000 | 30 | 0.3512 | 1.8075 | **0.39** | 126 | 0.2465 | 1.3718 | **0.36** |
| 20,000 | 36 | 0.4377 | 1.3618 | **0.64** | 252 | 0.2553 | 0.7931 | **0.64** |

**The premise was right and the prediction drawn from it was wrong.** The fraud counts
land exactly where the mechanism said: the `_MIN_FRAUD_ROWS = 30` floor stops binding on
BankSim above 2,000 rows (30→63→126→252) and holds ULB pinned to the top of the grid
(30→30→30→36). So the arithmetic in pre-registration 6 is confirmed. What it predicted
from that arithmetic is not:

* **BankSim was predicted to "improve markedly with size".** It **worsens**, 0.51 → 0.64,
  while gaining 8× the fraud rows. ✗
* **ULB was predicted to "barely move until 20k".** It makes the **largest move in the
  table** — 0.79 → 0.39 by 10,000, a 51 % improvement — with its fraud count pinned at 30
  the whole way. ✗

> ⛔ **Pre-registration 6 named this exact result in advance: *"If ULB improves anyway,
> the fraud count is not the driver and the 'n≈9 positives' mechanism is wrong."*** ULB
> improved anyway. So the mechanism is falsified **by a second, independent road** — and
> the shipped-path check above reached the same verdict by the first road that same day
> (pre-registration 1's named alternative). **Two pre-registered falsification routes,
> both taken.** The "n≈9 positives is easy by luck" story is retired.

**And the curve does not support a knee at all.** Both datasets are **non-monotone**: best
at 10,000, worse again at 20,000, ending within 0.001 of each other (0.643 vs 0.644). The
decomposition says why the two halves fail differently:

* **ULB** moves on *both* terms — std falls 0.52→0.35 and range rises 1.32→1.81 to 10k,
  then both reverse. Real movement, but reversing.
* **BankSim's std is flat** across the whole grid (0.2465–0.2947, ±10 %). Its `n2s` is
  driven almost entirely by `config_range`, which swings 0.79–1.37 **with no trend** — and
  `config_range` is itself a single-draw spread over the five `SWEEP_CFGS`.

⚠ **The run is one objective seed per size, and the script has already been rewritten to
forbid reading it.** `knee()` in the working tree now takes `seeds` and refuses to name a
best size unless the size effect exceeds the between-seed effect — the JSON on disk
predates that rewrite (no `seeds` key; the log format has no `sd=` field). Its draft
therefore takes the single-seed branch and prints *"Lowest `noise_to_signal` at 10,000
rows"* for both datasets. **That line is an argmin over a noisy curve, not a knee**; the
draft's own one-seed warning immediately below it is the operative sentence, and the
non-monotonicity is exactly the symptom the multi-seed guard was written to catch.
**Do not quote 10,000 rows as a chosen surrogate size.**

#### ▸ SCOUTED 2026-09-04 — the side-by-side costs ~16.7 h on BankSim alone

`python experiments/obj13_surrogate_repair.py --dataset banksim --scout`, on a confirmed
idle CPU (4 physical cores, `torch` threads=2): **34 evals in 80 s = 2.36 s/eval**, search
pool 19,999 rows / 252 fraud, yardstick pool row-disjoint. Projected at the **deployed**
budget (`DB_BOA_CONFIG` pop=20 × 30 iters, so the projection is of the real search, not a
toy):

| mode | evals × draws × seeds | hours |
|---|---|---|
| `legacy` | ~1700 × 1 × 3 | 3.3 |
| `deterministic` | ~1700 × 1 × 3 | 3.3 |
| `averaged` (k=3) | ~1700 × 3 × 3 | 10.0 |
| **TOTAL, BankSim only** | | **~16.7 h** |

⚠ **Measured twice on purpose.** The first scout read 2.68 s/eval → 19.0 h, because it
overlapped the tail of the knee run the board said was not running. Idle re-run: 2.36 s/eval
→ 16.7 h, **12 % lower**, with the search itself bit-identical (both chose 76f/214spe,
own-best 4.3086). The correct number is 16.7 h.

⛔ **This does not fit the calendar without a decision.** T-15 to 2026-09-19, OBJ-16's
go/no-go is 2026-09-12, and ULB would roughly double the bill *on top of* the pool fix it
now needs. The levers, with what each one costs:

| Lever | Saves | What it costs |
|---|---|---|
| Drop `averaged` to k=1 | 6.7 h of 16.7 | **Kills pre-registration 1.** `averaged` is the arm that tests whether the ceiling dies under a mean; without it the side-by-side has no result |
| Drop to 1 objective seed | ~11 h | **Kills pre-registrations 2 and 3**, both of which are explicitly about seed scatter |
| Cut pop×iters below 20×30 | scales linearly | The comparison stops being about the search that actually ships |
| `--filters-max` | scales with the cap | Changes the search space, so `legacy` no longer reproduces what is on disk |
| Raise `torch` threads 2→4 | unreliable | Breaks the thread pin every detector number depends on (OBJ-2), on a 4-core mobile part that will throttle over a run this long |

**Recommendation: BankSim only, all three modes, all three seeds — pay the 16.7 h once,
overnight, and leave ULB out of this experiment entirely.** BankSim is the arm the
shipped-path check leaves *valid*; ULB's would be measured on a pool that does not describe
its own deployment, so it would cost ~17 h to produce a number that then needs a caveat
longer than the result. That is a lever the operator should confirm, not one to take
silently — hence the block.

---

Evidence: `results/objective_noise_audit.json`, `final_report_data/OBJECTIVE_noise_audit.md`,
produced by `experiments/objective_noise_audit.py`.

OBJ-2 established that the *final 30-epoch training* is bitwise deterministic given
(seed, torch version, threads). This is about the layer above it: **the surrogate fitness
that chooses the configuration in the first place.**

Found while running the base-paper optimiser track: on ULB **all five optimisers**
(MBO, WSA, DBOA, BOA, DB-BOA) reported best Obf2 = **exactly 5.0000** — the theoretical
maximum of `2·MCC + Spec + Pre + NPV`. Five different search algorithms agreeing to four
decimals on the ceiling is a property of the objective, not the algorithms.

- `_ADTCNObjective.__call__` redraws its 70/30 split **and** its torch seed on every call, so fitness is a *random function of its input*.
- A **fixed** config re-evaluated 25× on ULB spans 3.4497–5.0000 and **touches the ceiling once, with no search involved**.
- Between-seed mean spread **1.0471** vs config-to-config range **1.0794** (ratio 0.97): changing which rows are sampled moves the score as much as changing the model does.
- **One search axis is arithmetically dead:** `batch_size = max(32, 1400 // spe)` = **32 for every `spe` in [50,250]**. The advertised 2-D search is 1-D and every reported `steps_per_epoch` optimum is arbitrary.

**⚠ CORRECTION (2026-09-01) — the "BankSim is 6× better" line was wrong, and it was wrong in
our own favour.** The audit script computes *two* noise ratios and the generated prose quoted
only the flattering one. Both are in `objective_noise_audit.json`:

| Ratio | What it asks | ULB | BankSim |
|---|---|---|---|
| between-seed spread ÷ config range | does the *subsample* set the difficulty? | 0.97 | 0.16 |
| `noise_to_signal` = 2·within-seed std ÷ config range | can *one draw* rank two configs? | **0.74** | **0.94** |

The first says BankSim is better. The second — which is the script's own headline field, and
the one that matters for a best-of-N search — says BankSim is **worse**. The honest statement is
narrower: on BankSim the objective's *mean* is stable across subsamples, but a single evaluation
still carries ±0.383 against a config-to-config range of 0.812, so it cannot rank either.
**The objective is uninformative on both datasets, for two different reasons.** Do not repeat
"the objective is informative on BankSim."

**The real mechanism is sharper than "0.17% fraud".** `_MIN_FRAUD_ROWS = 30` fires on *both*
datasets (ULB 0.17%×2000 = 3.4 → 30; BankSim 1.211%×2000 = 24 → 30), so both surrogates hold
exactly **30 fraud rows**, confirmed by `surrogate_fraud_rows: 30` in both JSON blocks. The 70/30
split leaves roughly **9 fraud rows in validation**. ~~`Obf2 = 5.0000` therefore means *perfectly
classifying about nine rows* — reachable by luck, on any dataset, with no search involved. The
fraud rate is not what separates ULB from BankSim; task separability at n≈9 positives is.~~

> ⛔ **STRUCK 2026-09-04 — the two struck sentences are wrong, and both refutations are
> measured.** They are true of the **audit's** 6,000-row pool and false of the pipeline that
> ships. (i) On the deployed 3,000-row pool ULB held **5 unique fraud transactions**, so its 30
> "fraud rows" were those 5 repeated ~6×, and **9 of 9 validation positives were copies of
> training rows** — not luck, memorisation. (ii) The fraud rate **is** what separates the two
> datasets, via the `len(fraud_idx) < _MIN_FRAUD_ROWS` threshold that ULB crosses (5 < 30) and
> BankSim does not (38 ≥ 30). (iii) The knee sweep separately falsified the fraud-count story
> altogether: BankSim's noise std held flat (0.2563 → 0.2553) across **8× more fraud rows**.
> Full detail in the NEW DEFECT block near the top of this objective.

> **Why this matters for rule 3.** "DB-BOA buys no measured accuracy gain over the
> hand-tuned default (+0.047, p=0.34)" now has a mechanism: a search that cannot rank
> candidates returns an essentially arbitrary one, so tying a sensible hand-set choice is
> the *expected* outcome. That strengthens the negative result rather than softening it.

**⚡ STAGED 2026-09-01 20:30, NOT YET APPLIED.** The patch is written and dry-run clean but is
deliberately not in the repo yet: the BankSim/stratified sweeps were still running, each sweep
launches as a fresh process that imports `models/adtcn.py`, and a syntax slip there would have
silently killed every sweep still queued. **Apply it the moment the run above lands.**

> ✅ **The blocker is gone as of 2026-09-01 23:41** — the run it was waiting on landed that
> evening and was scored 2026-09-04. **Nothing is running and nothing is queued.** This is
> item 2 on the board; the only reason it is not item 1 is that [OBJ-17] is bounded and this is
> not. Start with `python scratchpad\apply_obj13.py --check`, which reports and writes nothing.
>
> ⚠ **Sequencing that has not changed:** applying this invalidates `db_boa_results.json` and the
> DB-BOA arm of every `baselines*.json`, which is exactly why [OBJ-16]'s third dataset must come
> *after* it — see the LOCKED PLAN's reason 1.

⚠ **The two files are NOT in the repo — `scratchpad/` does not exist here and is not in
`.gitignore`.** "scratchpad" in the note below meant the *session's own* temp directory, which is
per-session and garbage-collected. **Located 2026-09-04, intact and parsing clean, at:**

```
C:\Users\Shadman\AppData\Local\Temp\claude\
  d--THESIS-FINAL-PROJECT-DB-BOA-FEL-ADTCN-Hyperledger-Fabric-main\
  2c93901b-4553-4f98-aace-eba6fd0fddff\scratchpad\
      obj13_new_objective.py   13,278 B   2026-09-01 20:30   the repaired _ADTCNObjective
      apply_obj13.py            6,582 B   2026-09-01 20:31   idempotent applier; --check writes nothing
```

> ⛔ **FIRST ACTION ON THIS OBJECTIVE: copy those two files somewhere durable**, before running
> anything. They are three days of design work sitting in a temp directory that can be cleaned
> at any time, and nothing else in the repo can reconstruct them. Suggested home:
> `db_boa_framework/scratchpad/` plus a `scratchpad/` line in `.gitignore` — or track them, since
> they *are* the record of a decision. **Then** run `--check` from wherever they land.
>
> This is the same class of mistake as the unsuffixed figures (OBJ-15) and the shared per-run log
> directory: **an artefact whose only copy lives somewhere the next run can destroy.**

`python apply_obj13.py --check` resolved all four anchors (class body, `import math`, the
`optimise_hyperparams` constructor call, the `ADTCN_CONFIG` keys) **as of 2026-09-01**. It backs
both touched files up to `*.pre_obj13` and re-parses them before declaring success.

⚠ **Re-run `--check` before trusting that.** `models/adtcn.py` and `config.py` have both been
edited since the anchors were resolved, so an anchor may have moved. `--check` writes nothing,
so this costs seconds and is the correct first command either way.

What the staged patch does:
- **`eval_mode` ∈ {`legacy`, `deterministic`, `averaged`}** on `_ADTCNObjective`, defaulting to
  `deterministic`, plumbed through `ADTCN_CONFIG` as `surrogate_eval_mode` / `surrogate_k` /
  `surrogate_rows`. `legacy` reproduces the pre-repair behaviour bit-for-bit, so the side-by-side
  comparison is against a **live** baseline rather than a remembered one.
- **Draws are pre-drawn once in `__init__` and shared by every candidate** (common random
  numbers). This is the part that matters and it was not in the original decision: re-drawing per
  candidate would leave the search comparing a good config on an easy draw against a good config
  on a hard one — the original defect wearing a mean.
- **The 70/30 split is stratified**, so fraud is present in both halves at every surrogate size.
  Without it, “more rows” and “fraud actually present in validation” move together and the size
  knee below would not be attributable.

**➕ New mechanism found while writing the patch: the dead axis is caused by the *floor*, and the
floor is removable.** `batch_size = max(32, n_train // spe)` — with `n_train = 1400` and
`spe ∈ [50, 250]` the right-hand term is 28 down to 5, **every value below the 32 floor**. So the
floor, not the arithmetic, is the bug. The repaired modes use
`batch_size = ceil(n_train / spe)`, which is what `steps_per_epoch` has always claimed to mean and
makes the axis live across the whole range (spe=50 → batch 28, spe=250 → batch 6).
**Decision, and the reason:** repair the axis rather than drop it. If `spe` turns out not to
matter once it is genuinely alive, that is a *measured* finding and dropping it to a 1-D search is
then justified by data — dropping it now would assert the same conclusion without evidence.

**DECIDED 2026-09-01 — repair is mandatory, not optional.** Approach chosen by the operator:

- [x] **Both fixes, reported side by side.** ✅ **DONE 2026-09-05** —
      `experiments/obj13_surrogate_repair.py`, 9 searches, 16.3 h.
      ⚠ **The gap between them did not measure what this bullet expected it to.** It predicted the
      deterministic↔averaged gap would quantify "how much of the original behaviour was
      noise-chasing". What it actually shows is that **`deterministic` scatters as widely as
      `legacy` (81 vs 66) and only `averaged` converges (12)** — so the informative contrast is
      *legacy+deterministic vs averaged*, not *deterministic vs averaged*. Common random numbers,
      not determinism, is the active ingredient.
- [x] **Surrogate size: measure the knee, do not guess it.** ✅ **DONE 2026-09-04**,
      `--knee` → `OBJECTIVE_size_knee.md`. ⚠ **Answered, but not with a number.** The curve is
      non-monotone on both datasets (min at 10 k, worse at 20 k), it is confounded (batch size and
      surrogate size move together), and at one seed per cell the size effect is smaller than the
      subsample effect. **The honest output is the curve plus "a knee is not locatable at this
      precision" — do not quote 10,000 as a chosen size.**
- [x] **Stratify the validation split** so fraud appears in both halves at every size — applied.
- [x] **Make `steps_per_epoch` affect training** — applied in the *surrogate*. ⚠ **Two
      qualifications found by running it.** (a) The axis is alive but **coarse**: 23 distinct batch
      sizes across 201 `spe` values, and spe 200–250 is two bins — quote the implied batch size,
      never the `spe`. (b) ⛔ **`ADTCN.fit:681` was never changed**, so the *final model* still uses
      `max(32, len(X_train) // spe)`. At ~400 k rows that floor never binds, meaning the final
      model's axis was always live and **only the surrogate's was dead**.
- [x] **Only then re-assess whether DB-BOA beats the default.** ✅ **Re-assessed on the surrogate
      side: it does not. 0 of 9, mean paired Δ −0.1982.** ☐ Still open on the **test set** —
      `detector_multiseed.py` with the repaired arms, which is what pre-registration 5 needs.

> **Pre-registered expectation (rule 4 — write it before running).** A repaired surrogate is a
> *better-behaved* DB-BOA, not necessarily a winning one: the five-optimiser tie already survives
> on BankSim, where the objective's mean is at least unbiased. If DB-BOA still ties the hand-set
> default after the repair, that is the honest outcome and rule 3 applies — **the repair is
> diagnosis of a negative result, not a rescue attempt.** If it *does* win post-repair, the claim
> is "the search works once the objective is fixed", and the old tie must be reported alongside it.

**⚠ Downstream cost — this is why OBJ-13 goes early.** `optimise_hyperparams` has exactly two
callers: `main.py:164` and `run_baselines.py:176`. So every result those two produce is invalidated
by the repair and must be regenerated — `db_boa_results.json` (the deployment figures, MCC 0.677),
and the `DB-BOA-ADTCN` arm of `baselines.json` / `baselines_banksim_stratified.json` /
`baselines_banksim_customer.json`. **The longer the repair waits, the more results accrue that
have to be re-run.** OBJ-15's four sweeps are *not* affected — verified, see below.

---

### ◈ ~~OBJ-14 — Retracted numbers are back in a live draft~~  → CLOSED, see CONFIRMED KILLS

**Status:** `DONE 2026-09-01` · generator fixed + `--render-only` added; draft regenerated; 8 historical
files stamped; **3 breaches found outside the stated scope** (`SUPERVISOR_BRIEF.tex`/`.md`,
`_seqtrain.py`). Repo-wide sweep is clean. Detail in CONFIRMED KILLS.

<details><summary>original objective (kept for the record)</summary>

`NOT STARTED` · **Priority: HIGHEST — this is a rule 2 and rule 3 breach, not a tidy-up.**

`final_report_data/OBJECTIVE_noise_audit.md`, written 2026-08-31, quotes two numbers that
OBJ-2 withdrew *the same day*:

- §"What this means" ¶2 — "`db_boa_results.json` and `dbboa_vs_default.json` report the same
  configuration at MCC 0.677 and **0.313**". 0.313 is withdrawn; lowest observed in 7 runs is 0.6598.
- §"What this means" ¶3 — "**consistent with the standing negative result that DB-BOA loses to
  the hand-set default (MCC 0.785)**". That negative result was replaced by +0.047, p=0.34 —
  *no gain and no loss*. Quoting a loss is now a fabricated claim.

`chapter_6.tex` is **correct** and already states both withdrawals. The regression is confined to
the auto-generated draft — which means the bug is in the generator, not the file.

- [ ] Fix the prose block in `experiments/objective_noise_audit.py` (the `L.append(...)` section,
      ~lines 225–245 and the "What this means" writer): drop 0.313 and 0.785, quote **both** noise
      ratios with their names, and state the ~9-validation-fraud-row mechanism.
- [ ] Add a `--render-only` flag that rebuilds the `.md` from the existing
      `results/objective_noise_audit.json` — currently the only way to regenerate the prose is to
      re-run 150 training draws, which is why a stale draft is cheaper to leave broken than to fix.
      That is the actual defect.
- [ ] Regenerate `OBJECTIVE_noise_audit.md` and diff it against this file's OBJ-13 section.

**Second, wider half — superseded drafts are unmarked.** Seven pre-OBJ-2 files still assert
"DB-BOA does not beat the default (0.677 vs 0.785)" with nothing saying they are dead:
`01_introduction.md`, `04_methodology.md`, `05_results.md`, `06_conclusion.md`,
`REWRITE_06_results.md`, `REWRITE_09_conclusion.md`, `REWRITE_FIX_2026-06-11.md`.

- [ ] Stamp each with a `> **SUPERSEDED 2026-08-31 by OBJ-2**` header naming the live source
      (`OBJ2_detector_multiseed.md` / `chapter_6.tex §sec:dbboa-hpo`). Do **not** silently edit the
      numbers — these are the historical record of what was believed and when; that provenance is
      worth keeping. Mark them, do not rewrite them.

> **The lesson to carry.** The `.tex` was purged and the drafts were not, so a *new* file written
> after the purge quoted a dead number in good faith. Rule 5 says the repo is the truth — this is
> the case where two parts of the repo disagreed. `results/*.json` outranks any `.md`.

</details>

---

### ◈ ~~OBJ-15 — The system experiments are locked to ULB~~  → **CLOSED**, see CONFIRMED KILLS

**Status:** ✅ **CLOSED 2026-09-04.** Both runs done — `BankSim/entity-disjoint 2026-09-01 16:20`
(3 h 15 m) and the `BankSim/stratified` confound control `2026-09-01 23:41` (~3 h 20 m). All four
sweeps × three conditions, clean exits, provenance in every JSON. **Five of the six title claims
are off ULB**; the sixth group is dataset-independent by design.

> ⚠ **Read the RESUME HERE scorecard before quoting anything in this objective.** The control run
> **withdrew this objective's flagship reading.** Krum's utility cost is **not** a property of
> entity-disjointness — it tracks the dataset, and at n=7 the partition effect runs the *other
> way*. Every per-sweep block below was written before that was known; the corrections are inline
> and marked `↳ 2026-09-04`.

The expansion reached the detector (OBJ-1/1b) and the federated ablation (CONFIRMED KILLS) and
stopped. **Four of the six title claims are still backed by ULB alone**, because their scripts
cannot take another dataset:

| Script | Hardcodes | Title claim left single-dataset |
|---|---|---|
| `byzantine_robustness_sweep.py:115` | `FinancialDataLoader()` | Secure (Byzantine) — Krum 8/8 |
| `economic_byzantine_sweep.py:190` | `FinancialDataLoader()` | Incentivized — collusion caught, lone attacker not |
| `private_incentive_sweep.py:104` | `FinancialDataLoader()` | Incentivized — ε\* 3000→50, ρ +0.950 |
| `scalability_sweep.py:217` | `FinancialDataLoader()` | Scalable — Shapley cost, MC top-1 fidelity |

`rl_leader_sweep.py` needs no change — it is a consensus simulation with no dataset. Fabric
measurements are likewise dataset-independent.

- [x] **Plumbing done 2026-09-01.** New `experiments/_dataset.py` holds the resolution once
      (`add_dataset_args` / `resolve` / `apply_to_model_cfg` / `provenance` / `suffix`); all four
      scripts now take `--dataset {ulb,banksim} [--partition {stratified,customer}]`. `--partition`
      on ULB fails loudly rather than silently falling back — ULB has no entity IDs.
- [x] **The 33-feature trap is now handled in one place.** `ADTCN.fit` caps consumed features at
      `N_RAW_FEATURES + 3` = 33, correct for ULB and silently wrong for BankSim (79 → 33, dropping
      46). `apply_to_model_cfg()` forwards `loader.raw_feature_count`, so a future sweep cannot
      forget it. `scalability_sweep` needed it threaded through `_train_org` as well.
- [x] **Every sweep JSON now carries `dataset` / `dataset_label` / `partition` / `raw_features`.**
- [x] Output filenames suffixed via `suffix()`; ULB/stratified keeps its bare historical name so
      existing report and TASK.md references stay valid.
- [x] **Verified independent of OBJ-13:** all four call `m.fit()` directly, which falls back to
      hand-set config defaults (`adtcn.py:478`). `optimise_hyperparams` is called only by `main.py`
      and `run_baselines.py`. **These sweeps do not need re-running after the surrogate repair.**
- [x] **Two latent bugs found by the smoke test, both fixed** — they would have spoiled the
      overnight run:
      1. `write_report()` opened its markdown with a bare `open(md, "w")`, so it died with
         `UnicodeEncodeError` on `→` under cp1252 **even with `PYTHONIOENCODING=utf-8` set** — that
         env var governs stdout, not file handles. The JSON was already saved, so the failure looked
         survivable while silently costing every draft. `encoding="utf-8"` added to all 15 scripts
         in `experiments/`, not just the four.
      2. The drafts had **fixed filenames** (`TASKB_economic_byzantine_results.md` …), so a BankSim
         run would have overwritten the ULB drafts in place. Now suffixed via `suffix_of(summary)`,
         read back off the run's own provenance so JSON and draft names can never disagree.
#### ⚑ Sweep 1 of 4 landed — `economic_byzantine_sweep`, BankSim/customer, 23.5 min

`results/economic_byzantine_sweep_banksim_customer.json` · draft
`TASKB_economic_byzantine_results_banksim_customer.md`. Provenance recorded correctly
(`dataset=banksim`, `partition=customer`, `raw_features=79`).

| Strategy | atk | with | without | gap | isolated |
|---|---|---|---|---|---|
| always-fraud | 1 | 92.95 | 97.75 | **−4.80** | BankC@r8 |
| always-fraud | 2 | 97.40 | 50.00 | **+47.40** | BankB@r4, BankC@r4 |
| label-flip | 1 | 92.95 | 94.58 | **−1.62** | BankC@r3 |
| label-flip | 2 | 97.40 | 7.68 | **+89.73** | BankB@r3, BankC@r3 |
| free-rider | 1 | 92.95 | 92.95 | +0.00 | BankC@r3 |
| free-rider | 2 | 97.40 | 50.00 | **+47.40** | BankB@r8, BankC@r8 |

**Replicates:** the headline claim holds on a second dataset — 2-of-3 collusion is caught,
+47.40 (ULB +40.78) always-fraud and +89.73 (ULB +79.52) label-flip. Direction and rough
magnitude both survive.

**Diverges from ULB in two ways, neither yet attributable:**
- Isolation now **fires on all 3 single-attacker scenarios** (ULB: none of 3).
- The **free-rider at n=2 is now caught** (+47.40); on ULB it was isolated by nothing.

⚠ **Two confounds block any "BankSim behaves differently" conclusion. Do not skip these.**
1. **Dataset AND partition both changed.** ULB ran stratified; this ran entity-disjoint.
   Same trap already flagged for the federated ablation in CONFIRMED KILLS. **A
   BankSim/stratified run is what separates them** — until then "BankSim" and
   "entity-disjoint" are the same column.
2. **The negative single-attacker gaps are partly quorum arithmetic, not defence quality.**
   Consensus is `preds.sum()*2 > n_voters`. At 3 voters that is majority (2 of 3); isolating
   exactly one leaves 2 voters, where the identical expression demands **unanimity**. That
   alone suppresses positives and depresses balanced accuracy on a rare-positive task. The
   even-voter case is a property of the voting rule. Now stated in the draft.

**Against the pre-registration:** "lone attacker / free-rider not caught — expected to
replicate" did **not** replicate cleanly. Isolation fires but does not help (3 of 3 fire,
0 of 3 improve accuracy). That is a *different* negative result, not the same one — rule 3
still applies, but the wording must change.

#### ⚑ Sweep 2 of 4 — `byzantine_robustness_sweep`, BankSim/customer, 48 min

**The pre-registered expectation HELD: Krum rejected the attacker 8/8, exactly as on ULB.**
Krum's accuracy is also perfectly attack-invariant — 83.74 % at n=5 and 88.50 % at n=7, in
*every* row, identical to the no-attack reference. The security property replicates cleanly.

**But the utility flipped sign, and that is the finding.**

| | ULB | BankSim / entity-disjoint |
|---|---|---|
| Attacker rejected | 8/8 | **8/8** ✅ replicates |
| Krum advantage over unprotected FedAvg | **+12.49 pp** (scaled) | ~~−12.57 to −1.48 pp~~ → **−4.73 to −1.48 pp on the clean cells** (still negative in 8/8) |
| Krum global model | 99.95 % | 83.74–88.50 % |

> ⚠ **2026-09-04 — the range in that row was carried by contaminated cells.** 5 of the 8
> entity-disjoint cells have an attacked-FedAvg baseline above its own no-attack reference, and
> **every cell worse than −5 pp is one of them** — including the −12.57 endpoint that got quoted.
> On the 3 cells with a clean baseline the cost is **−4.73 to −1.48 pp**. Quote that, and say it
> is the clean subset. Full mapping in CONFIRMED KILLS.

~~**Mechanism.** Krum *selects one org's weight vector*; FedAvg averages all of them. Where orgs
hold disjoint customers, a single org's model generalises worse than the average, so Krum pays
a real utility cost for its guarantee. On ULB's stratified split the orgs are near-interchangeable
and the cost is ~0.~~

> **↳ 2026-09-04 — THIS MECHANISM IS WITHDRAWN.** The BankSim/stratified control falsifies it
> directly: orgs there are *not* customer-disjoint and Krum is still negative in 7 of 8 cells.
> Worse for the mechanism, **at n=7 entity-disjointness makes Krum look *better* in all four
> rows** (partition effect +1.36 to +5.28 pp) — the opposite sign to what it predicts. The
> utility cost is carried by the **dataset**. Do not soften this into a weaker version of the
> same story; there is no measured mechanism on record for it right now, and saying so is the
> honest position. See the RESUME HERE scorecard, row 1.

> **What survives, and it is the part worth keeping:** Krum buys *robustness* (attacker rejected
> 8/8 in **all three** conditions; accuracy attack-invariant on ULB and BankSim/entity-disjoint)
> and **can pay utility for it**. **Security and utility are separate claims and must be reported
> separately** — that lesson is unaffected by the withdrawal, and it is what refines the standing
> CONFIRMED KILLS reading that Krum's value "shows up under attack".

⚠ **A metric artefact that blocks reading this as "FedAvg wins".** In **5 of 8** cases the
*attacked* FedAvg scores **above its own no-attack reference** (sign-flip 94.94 % vs 90.10 %).
An attack cannot genuinely improve a model — this is balanced accuracy rewarding noise that
pushes the boundary toward the positive class, the same direction-of-collapse effect the DP
sweep found on BankSim. So "FedAvg survived" is not evidence of robustness, and the
Krum−FedAvg gap is not a clean utility comparison. Now printed in the draft (fires on BankSim,
correctly silent on ULB). ⚠ Dataset **and** partition changed together here too.

#### ⚑ Sweep 3 of 4 — `private_incentive_sweep`, BankSim/customer, 26 min

**The ordering replicates; the constant moved — exactly the pre-registered contingency.**
The output channel beats the weight channel at *every* ε on both datasets. At ε=50 the weight
channel is at ρ=−0.070 (ULB −0.288) while the output channel is at ρ=+0.805 (ULB +0.950).

| | ULB | BankSim / entity-disjoint |
|---|---|---|
| ε\* weight | ≥3000 | ≥3000 |
| ε\* output | 50 | **300** |
| Improvement | ≥60× | **≥10×** |

Per the pre-registration, **stop quoting 60× as universal** — report ≥60× (ULB) and ≥10×
(BankSim).

> ⚠ **New, and it applies to the ULB number too: ε\* is right-censored on both datasets.**
> ε\* is defined as `max{ε : inversion_rate > 0}` — the largest budget at which rewards are
> *still* mis-ranked. The weight channel is **still inverting at ε=3000, the largest budget
> swept** (ULB 20 % of 100 draws, BankSim 38 %), so its true ε\* lies beyond the grid.
> **60× and 10× are therefore lower bounds, not measurements**, and the INTEL table's
> "ε\* 3000 (weight)" was quoting a censored value as if measured. Both drafts now say so.
> Extending the sweep past ε=3000 is what would turn this into a measurement.

#### ⚑ Sweep 4 of 4 — `scalability_sweep`, BankSim/customer, 1 h 38 m · **RUN COMPLETE 16:20:32, 3 h 15 m total**

**The cost claim replicates, and it is structural.** Exact Shapley at n=12: **113.53 s (ULB) vs
120.64 s (BankSim)** over an identical 4,095 coalitions. O(2ⁿ) is a property of the coalition
lattice, not the data, so this was never going to move — worth stating as *confirmed and
expected*, not as a new finding.

**But "MC top-1 fails from n≈6" is not supported on either dataset.** Top-1 agreement is
**intermittent, not a threshold**:

| | n=3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ULB** | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | 5/10 |
| **BankSim** | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | 3/10 |

ULB *first* fails at n=6 but **recovers at n=7 and n=9**; **BankSim fails already at n=3 and
n=4**. So the honest statement is "top-1 first fails at n=6 on ULB and at n=3 on BankSim, and
is unreliable at every n on both" — **not** "fails for all n≥6". The mechanism is in the
script's own prose and is correct: top-1 flips on sub-noise differences whenever two orgs are
near-tied, which makes it the *hardest* fidelity bar rather than a size threshold.

⚠ **Rule 11 — the two fidelity metrics disagree about which dataset is better.** At n=12
BankSim's ρ is **+0.448** against ULB's **−0.014** (BankSim looks far better), yet BankSim
recovers top-1 in only 3/10 cases against ULB's 5/10 (BankSim looks worse). **State which
metric you are quoting.** ρ and L1 are the fairer measures of whether the *token split* is
preserved; top-1 answers a different question (who gets the largest single payout).

⚠ **The MC estimator is barely a speed-up in the tested range, on both datasets.** Speed-up is
≈1× below n≈10 and reaches only ~3.2× at n=12 — and on ULB at n=5 MC is *slower* than exact
(0.44×). The "cheap approximation" framing only starts to earn its name past the largest n we
can check against exact, which is precisely where fidelity can no longer be verified.

- [x] **Re-run on BankSim — DONE 2026-09-01, all four sweeps, entity-disjoint.**
- [ ] Re-run on BankSim, both partitions where meaningful. Entity-disjoint is the interesting one
      for the economic and Shapley sweeps — that is where orgs genuinely differ.
      **Runner: `.\experiments\run_obj15_banksim.ps1 [-Partition ...]`** (cheapest sweep first,
      per-run log dir,
      one failure cannot take the others down). ⚠ **Use the `.ps1`, not the `.sh`** — `python`
      is absent from Git Bash's PATH here and `python3` is the Store stub with no torch, which
      is what killed the two earlier launch attempts. Progress: `.\experiments\check_obj15.ps1`.
- [x] **Pre-registered and LOCKED 2026-09-01 13:05, before the run produced any output**
      (rule 4). The run started 13:05:29; the first sweep's first result line had not been
      written when these were fixed, so none of them is retro-fitted. **Do not edit these
      four bullets after results land** — if an expectation turns out wrong, that is the
      finding, and it gets recorded next to the original wording, not in place of it.
      - **Krum 8/8** — expected to hold; it is a geometric property of the weight space, and the
        constant-premium result (−0.037 vs −0.033) suggests dataset-independence. A *failure* to
        replicate would be the most valuable outcome in this objective.
      - **Lone attacker / free-rider not caught** — expected to replicate. This is a standing
        negative result (rule 3); do not treat replication as a disappointment.
      - **ε\* 3000 → 50** — the ε values themselves are noise-scale-dependent and **may well move**.
        The claim under test is the *ordering* (output channel ≫ weight channel), not the constants.
        If the 60× shrinks on BankSim, report the new factor; do not keep quoting 60×.
      - **MC-Shapley top-1 fails from n≈6** — untested at BankSim's org sizes; genuinely open.

      **↳ SCORED 2026-09-01 16:20, after the run. The four bullets above are unedited.**

      | Pre-registered | Outcome |
      |---|---|
      | Krum 8/8 holds | ✅ **held.** 8/8 on BankSim too, and Krum's accuracy is attack-invariant. *But the expectation was silent on utility, and utility is where it broke:* −12.57 to −1.48 pp vs unprotected FedAvg, negative 8/8. We predicted the right answer to the wrong question. |
      | Lone attacker / free-rider not caught | ⚠️ **did not replicate as worded.** Isolation now *fires* on 3/3 lone attackers (ULB: 0/3) but improves accuracy on 0/3. Still a negative result — a **different** one. "Not caught" must become "caught and not helped". |
      | ε\* ordering holds, constants may move | ✅ **exactly as pre-registered.** Ordering held at every ε; the factor moved 60× → 10×. The instruction "report the new factor, do not keep quoting 60×" is now executed. **Unforeseen:** ε\* is right-censored on *both* datasets, so both factors are lower bounds. |
      | MC top-1 from n≈6 — genuinely open | ❌ **the premise was wrong.** Top-1 is intermittent, not a threshold: ULB recovers at n=7 and n=9; BankSim fails from n=3. "Fails from n≈6" is withdrawn. |

      > **What the pre-registration bought.** Two of four expectations were wrong, and both
      > were wrong in ways that are now findings rather than embarrassments — which is the
      > entire point of writing them down first. The Krum row is the sharpest lesson: a
      > correctly predicted *security* result concealed an unpredicted *utility* reversal,
      > because the expectation never named which property it was about. **Pre-register the
      > property, not just the number.**
- [x] ⚠ Export `PYTHONIOENCODING=utf-8` when redirecting these to a file (see OBJ-1b's trap).
      Done — set in the runner. Note the per-sweep logs still land as **UTF-16LE** because
      PowerShell 5.1's `*>` redirection writes UTF-16 regardless; that env var governs
      Python's stdout, not the shell's file handle. `grep` from Git Bash cannot read them.

> **Why this outranked a third dataset.** A third dataset would have given claim #1 a third data
> point while four other claims still had one. Breadth across *claims* beats breadth across
> *datasets* until every claim has been off ULB at least once.
> **✅ That condition was met on 2026-09-01.** The gate is now open — see OBJ-16.

---

### ◈ OBJ-16 — Proceed to the third dataset  *(added 2026-09-01)*

`NOT STARTED` · **Priority: MEDIUM · scheduled Sep 10–16 · ⛔ GO/NO-GO 2026-09-12.**

**Rule 8's gate is satisfied.** Every title claim has been off ULB at least once: five measured
on BankSim (OBJ-15 + the federated ablation), and RL/Consensus/Blockchain are dataset-independent
by design. Breadth across *datasets* is a legitimate next move rather than a lopsided table.

- [x] ~~Do the BankSim/`stratified` control first~~ — **DONE 2026-09-01 23:41, scored 2026-09-04.**

#### ⚠ RE-GATED 2026-09-04 — the control run voided the old ranking

**The previous gate ranked AMLSim first for one specific reason, and that reason is now gone.**
It read: *"Krum's utility cost appeared only under our **synthetic** entity-disjoint split, so a
dataset with native bank IDs is the deciding experiment."* The control answered that question
without AMLSim — the cost appears under **stratified** BankSim too, so it is not an artefact of
our customer-dealing script, and at n=7 entity-disjointness pushes Krum the *other* way. AMLSim
is no longer *deciding*; it is merely *nicer*, and rule 8 does not fund nicer.

**The live question the third dataset must answer is now simpler and cheaper:** *is Krum's
utility cost a **BankSim artefact** or a general property?* **Any** third dataset answers that,
so take the cheapest one that can run all four sweeps.

| Candidate | What it could overturn *now* | Verdict (rev. 2026-09-04) |
|---|---|---|
| **Fraud Detection Handbook** (OBJ-5) | (a) **Is Krum's utility cost BankSim-specific?** — the live question, and this answers it. (b) Finer-than-daily resolution + `TERMINAL_ID` + built-in time-dependent fraud scenarios: still the only remaining shot at overturning "no architecture genuinely exploits time", which BankSim only half-answered. Has `CUSTOMER_ID` **and** `TERMINAL_ID`, so all four sweeps and both partitions run on it. | **⭐ PROMOTED TO FIRST.** Two live questions, lightest lift, fits the window. |
| **IBM AMLSim** (OBJ-12) | Native bank IDs — a federation we do not synthesise. Still genuinely valuable, and still the only candidate exercising **Secure + Incentivized + Scalable** under real institutional boundaries. But its *deciding* role is spent, and it is a Java generator with a config pipeline before a single transaction exists. | **DEMOTED TO SECOND.** Right dataset, wrong fortnight. Revisit post-deadline. |
| **PaySim** (OBJ-11) | Little. No bank IDs, and origin accounts may appear once or twice, so it tests neither the federated/incentive story nor entity linkage. | **Weak — recon only.** Unchanged. |

> **Housekeeping, 2026-09-08 — `datasets/bsNET140513_032310.csv` is not a fourth dataset and
> never was.** It sits unused next to the BankSim file and looks like spare data; it is not.
> Verified row-for-row: its `Source,Target,Weight,typeTrans,fraud` is **byte-identical** to
> `customer,merchant,amount,category,fraud` from `bs140513_032310.csv` across all 594,643 rows.
> It is a strict column *projection* of the file we already load, dropping `step`, `age`,
> `gender` and the two zip columns. **It carries zero information the loaded file lacks, so
> there is nothing to wire up and no rule-8 gate it could pass.** Left on disk (it is
> gitignored and costs nothing); do not spend a second on it again.

> **This reordering is a result following data, not convenience** — record it that way. If the
> deadline had been the only reason, that would be rule-4 reasoning and it would not be allowed.
> The reason AMLSim moved is that the experiment it was uniquely for **has already been run**.

#### What to run when a third dataset lands

- [ ] Wire it into `config.DATASETS` + a loader, using `data/banksim_loader.py` as the template.
      **Check the file-order trap first** (see the BankSim note in CONFIRMED KILLS).
- [ ] `--dataset <new>` already works for all four system sweeps and both pipelines — OBJ-15's
      plumbing is generic, so this should be loader work only, not sweep work.
- [ ] Re-run the four sweeps **and** the federated ablation, both partitions where meaningful.
- [ ] **Pre-register expectations before running (rule 4), and per OBJ-15's lesson, name the
      *property*, not just the number.** Two of the four questions the old version listed are
      now answered — Krum's cost tracks the **dataset**, and lone-attacker isolation fires
      without helping in all three conditions. The genuinely open ones for a third dataset:
      **(a)** is Krum's utility cost BankSim-specific, or does it appear on any non-ULB data?
      **(b)** does the isolation-fires-but-does-not-help negative hold a third time?
      **(c)** does the ε\* ordering hold a fourth time?
      **(d)** does *any* architecture exploit time at finer-than-daily resolution?
- [x] ~~Extend the ε grid beyond 3000~~ — **promoted out of this objective to [OBJ-17]**, because
      it needs no new dataset and must not be held hostage to one. Run it on the extended grid
      when it lands here, so all four conditions share one grid.

> **Rule 8 still applies within this objective.** Adding AMLSim must be justified by what it can
> overturn — written above, before any CPU is spent — not by the fact that three datasets sound
> more thorough than two.

---

### ◈ OBJ-5 — Fraud Detection Handbook dataset

`LOADER LANDED 2026-09-08 — no CPU spent on results yet` · **⭐ THIS IS THE THIRD DATASET —
promoted to FIRST 2026-09-04**, sequenced under [OBJ-16], scheduled Sep 10–16 with a **go/no-go
on 2026-09-12**.

> **▸ 2026-09-08 — the loader is in and verified. The sweeps are NOT run.**
> Data: `git clone --depth 1 github.com/Fraud-Detection-Handbook/simulated-data-raw` →
> `datasets/handbook_raw/` (183 daily pickles, gitignored), consolidated on first use into
> `datasets/handbook_transactions.csv` (105 MB, also gitignored).
> Code: `data/handbook_loader.py`, `config.HANDBOOK_CONFIG` + `DATASETS["handbook"]` +
> `ENTITY_PARTITIONS`, and `experiments/check_handbook_loader.py` — **42 acceptance checks,
> all passing**, no training involved. `--dataset handbook` now resolves everywhere.
>
> **Measured shape** (not quoted from the paper): 1,754,155 tx · 4,990 customers · 10,000
> terminals · 183 days · 14,681 fraud (**0.837 %**) · timestamp resolution **1 second**.
> Temporal cuts at the 70/80 % row-mass quantiles: train day 0–127, val 128–145, test 146–182.
>
> **① The file-order trap does NOT fire here — checked, not assumed.** BankSim's raw CSV had
> 3,635 adjacent fraud pairs collapsing to 84 after a within-step shuffle. This dataset's raw
> file order has **125 adjacent pairs, P(fraud | prev fraud) = 0.0085 against a 0.0084 base rate
> — lift 1.02×, i.e. nothing**, and the seeded tie-break moves it to 124. The shuffle is kept
> anyway (6.788 % of rows do share a timestamp to the second) so this stays a measured property.
>
> **② NEW, and the gate did not predict it: `TERMINAL_ID` is a far stronger entity link than
> `CUSTOMER_ID`.** Fraud-adjacency lift over base rate, same rows, same seed —
> global **1.01×** · customer **13.35×** · terminal **71.65×**. Cause is in the generator:
> scenario 2 compromises a *terminal* for 28 days and is 9,077 of the 14,681 fraud rows (62 %),
> against scenario 3's customer-card compromise (4,631) and scenario 1's pure amount rule (973,
> no entity at all). BankSim's customer-linked lift was 30×, so **`ordering="terminal"` is the
> strongest entity linkage anywhere in this project**. It is exposed as a *third* ordering arm
> and a *third* partition, not folded into "customer" — which is why `_dataset.resolve`'s
> `dataset != "banksim"` guard had to become the `ENTITY_PARTITIONS` table (it would otherwise
> have rejected a perfectly valid `--dataset handbook --partition customer`).
>
> **③ Two traps this project has already paid for do not reproduce here.** The OBJ-13
> memorised-rows leak: the 36,000-row eval pool holds **293 unique fraud** against the floor of
> 30, so the oversample branch never fires (ULB held 5 and repeated them 6.01×). ⚠ but the
> inherited 36,000 is **load-bearing, not merely copied** — at the old 3,000 this pool would
> hold ~24, *below* the floor. And the 33-feature cap: width is **35**, so `apply_to_model_cfg`
> is doing real work; without it `ADTCN.fit` would silently drop 2 columns.
>
> **④ `TX_FRAUD_SCENARIO` is the label under another name** (non-zero exactly when
> `TX_FRAUD == 1`, verified) and is never read. Nor is `TRANSACTION_ID`, a row counter encoding
> arrival order. Strongest single-feature correlation with the label is |r| = 0.168.
>
> **⑤ ⚠ The feature matrix is deliberately thinner than the Handbook's own baseline, and our
> absolute MCC will be lower than their published numbers.** Their reference features are
> `CUSTOMER_ID_NB_TX_*_WINDOW` / `TERMINAL_ID_RISK_*_WINDOW` aggregates — entity-derived, and
> therefore forbidden by the BankSim design decision that keeps the global-vs-linked comparison
> honest. 35 columns: log1p(amount), amount, hour-of-day one-hot, day-of-week one-hot,
> is_weekend, is_night. **This is the design working. Do not "fix" it by adding their features
> (rule 3), and do not compare our MCC to theirs.**
>
> **⑥ ⚠ COST: the "~6 h CPU" budget in the calendar is low.** Measured, one ADTCN epoch at the
> hand-set default: BankSim 417,848 train rows / 79 feat / batch 2,785 → **42.4 s**; Handbook
> 1,226,990 rows / 35 feat / batch 8,179 → **78.7 s**. That is **1.86×**, not the 2.94× the row
> ratio suggests — the thinner matrix pays back part of the size. A BankSim-equivalent sweep
> suite therefore costs **~11 h, not ~6 h**. That is a go/no-go input, not a blocker.
>
> **Still open before any number is quotable:** the rule-4 pre-registration for questions
> (a)–(d) is **not written** — it is a scientific call for the operator, and rule 4 says it goes
> down *before* the runs. And per the NEXT UP table this objective still sits behind board item
> 1, or its DB-BOA arm is paid for twice.
**Rule-8 gate — what could this overturn?** Two live things, both written before any CPU:
**(1)** whether Krum's utility cost is a **BankSim artefact** — the question the confound control
left open, and the reason this dataset now outranks AMLSim; **(2)** "no architecture genuinely
exploits time", which BankSim only half-answered — this is the only candidate with
*finer-than-daily* resolution and built-in time-dependent fraud scenarios. It has `CUSTOMER_ID`
**and** `TERMINAL_ID`, so all four system sweeps and both partitions run on it, not just the
detector track. · **Now unblocked and more valuable than before.** The better temporal dataset — `TERMINAL_ID`, exact datetime, ~1.75M tx over 183 days, fraud scenarios built to be time-dependent. OBJ-1 established that entity linkage is worth +0.10 to +0.17 MCC, so a dataset with *finer* temporal resolution than BankSim's daily `step` is the natural next test — and it would show whether LSTM/DTCN keep their lead over ADTCN on a second entity-linked dataset. The loader work is now mostly a template: `data/banksim_loader.py` plus `config.DATASETS`. **Check the file-order trap first** (see the BankSim note in CONFIRMED KILLS).

### ◈ OBJ-6 — Entity-level federated partitioning  *(partially done 2026-08-31)*

`IN PROGRESS` · Kills our biggest limitation: the three banks are currently a stratified volume split of *one* institution, so cross-institution shift is never exercised. Partition by customer, no history in two banks. Makes the Shapley numbers mean something — they are currently near-uniform (`[0.167, 0.165, 0.165]`).

- [x] **Implemented and run on BankSim.** `BankSimDataLoader.split_for_orgs(partition="customer")` deals whole customers to banks; verified **0 customers shared** between BankA/B/C (2055 / 1233 / 823 customers). `run_baselines.py --dataset banksim --partition customer` → `results/baselines_banksim_customer.json`.
- [ ] **Shapley under the entity-disjoint split has NOT been re-measured** — that is the part that would make the near-uniform `[0.167, 0.165, 0.165]` meaningful, and it is still open. The ablation run only covers FedAvg / Krum / DP.
- [ ] ULB cannot support this at all (no account IDs), so this stays BankSim-only until OBJ-12.

### ◈ OBJ-7 — Formalise the DP guarantee

`NOT STARTED` · Define adjacency, clipping norm C, σ, δ, and composition across rounds. Add a privacy accountant. Flagged as **must-fix before publication** — right now ε is stated without a threat model.

### ◈ OBJ-8 — Freeze the on-chain / off-chain execution map

`NOT STARTED` · List every function — local training, Krum, Shapley, DP, DB-BOA, RL, reward update, reputation update, ledger write — and mark exactly where each runs. Fabric chaincode must be deterministic; stochastic search cannot live inside it.

### ◈ OBJ-9 — Scope the two overstated words in the title

`NOT STARTED` · Per `title_issue.md`, four of the title's six claims are supported and **two are
overstated**: "Scalable Machine Learning" (we have scalable *contribution attribution* only) and
"Consensus Mechanisms" (stock Fabric Raft; our contribution is RL leader selection + incentives).
"Secure" and "Reinforcement Learning" are backed in code — `byzantine_robustness_sweep.py` at
n≥2f+3 and `blockchain/rl_leader.py` respectively — and need scoping sentences, not softening.

- [ ] *Scalable* → scope to **scalable contribution attribution**.
- [ ] *Consensus* → state that Raft is stock and the consensus round is simulated.
- [ ] *Secure* → keep, scoped to ≤f colluders at n≥2f+3, single-process simulation.
- [ ] Report ADTCN and DB-BOA as **measured components**, not claimed contributions. Neither is a
      title claim, so neither has to win — but neither may be described as winning (rule 3).

**Do this after OBJ-15**, not before: the wording should be written once, against six claims that
have each been tested on more than one dataset. Writing task, not code.

> ⚠ **Rev. A of the DIRECTION REVIEW proposed rewriting the title around a two-claim "C1/C2"
> story. That was wrong and is withdrawn** — it dropped Secure-Byzantine, Blockchain, Consensus
> and RL, all of which are backed by code and results. Do not resurrect it.

### ◈ ~~OBJ-10 — Update the stale objective in metrics.py~~  → CLOSED

`DONE 2026-08-31` · `obf2_value` now holds the bounded `2·MCC + Spec + Pre + NPV` and `adtcn.py` calls it, so there is one definition instead of two. Verified bit-identical over 2,000 random metric dicts (max diff 0.0).

### ◈ OBJ-11 — Recon PaySim before committing

`NOT STARTED` · **Ranked LAST under [OBJ-16] — recon only, and not before the 2026-09-19
deadline.** Unaffected by the 2026-09-04 re-gate: it was weak on the old gate and it is weak on
the new one, for the same reason (no bank IDs, probably no reusable per-customer histories).
**Rule-8 gate: weak.** PaySim has no bank IDs and probably no reusable per-customer
histories, so it can test neither the federated/incentive story nor the entity-linkage story.
Recon only — do not commit CPU-hours until OBJ-12 is ruled out. · 6.36M mobile-money transactions with sender/receiver/balances. **Check first:** origin accounts may appear only once or twice, which would make per-customer sequences useless. It is the multi-bank/network dataset, not the temporal one.

### ◈ OBJ-12 — IBM AMLSim for the real multi-bank story

`NOT STARTED` · **DEFERRED past the 2026-09-19 deadline — demoted from first to second
2026-09-04.** Sequenced under [OBJ-16] behind the Handbook.
⚠ **Its promotion rationale has been spent.** OBJ-15 raised AMLSim's value on the grounds that
*"Krum's utility cost showed up only under our **synthetic** entity-disjoint split, so a
federation with native institutional boundaries is the deciding experiment."* **The
BankSim/stratified control ran that experiment for free**: the cost is not an artefact of our
customer-dealing script, it tracks the dataset. AMLSim is therefore no longer deciding.
**Rule-8 gate, restated honestly:** it remains the only candidate exercising **Secure +
Incentivized + Scalable** under *native* institutional boundaries, and it would make Shapley
non-uniform for a real reason rather than a constructed noise gradient. That is still worth
doing — after the deadline, when a Java generator and a config pipeline are affordable. · Generates accounts, transactions and laundering patterns **with actual bank IDs** — so Bank A / B / C stop being a pretend split. Highest-value dataset for the FL + incentive + Byzantine architecture, and the heaviest lift.

---

### ◈ ~~OBJ-17 — De-censor ε\*: extend the privacy grid past 3000~~ → **CLOSED 2026-09-04**  *(added 2026-09-04)*

`DONE` — both passes landed and are scored (grid 14:25, `--repeats 1000` 17:53). **It succeeded at
its stated goal and the success is what killed the claim:** the factors did become measurements,
and the measurement then proved to be an unstable statistic (20× to ≥100× depending on where the
threshold is drawn). **The budget factor is retired as a headline**; the *ordering* and the *ρ-gap
at a stated ε* carry the contribution instead, and both are threshold-free. Two scorecards below,
ten pre-registrations, three wrong — each wrong one informative.

⚠ **It also cost a claim nobody was auditing:** ULB results from before 2026-09-01 do not
reproduce and the cause could not be determined. See the verdict block below, and follow-ups
**A** and **B** in RESUME HERE.

Every other open objective adds a column or fixes prose. This one aims to convert
**"≥60× / ≥30× / ≥10×, all censored lower bounds"** back into a *measurement*.

#### ⚠ The premise needs a correction before any CPU is spent — read this first

ε\* is `max{ε : inversion_rate > 0}` (`private_incentive_sweep.py:198-201`), the largest budget
at which rewards are still mis-ranked. The weight channel is still inverting at ε=3000, the top
of the grid, in all three conditions, so ε\* lies beyond it and every factor is a lower bound.
That much is unchanged. **What was missed is what the rate at the top actually is, in draws:**

| Condition | weight inv. @ ε=3000 | in draws (`n_repeats=100`) | reading |
|---|---|---|---|
| ULB / stratified | 0.20 | 20 of 100 | solidly non-zero; far from de-censoring |
| **BankSim / stratified** | **0.01** | **1 of 100** | ⚠ **a single draw decides ε\*** |
| BankSim / entity-disjoint | 0.38 | 38 of 100 | solidly non-zero; furthest from de-censoring |

**Three consequences, and the first one corrects a line that has been repeated in the SITREP.**

1. **"BankSim/stratified is one hair from de-censoring" is true but was read backwards.** It is
   not the condition closest to being *confirmed* — it is the one whose ε\* is **least firmly
   established**, because a `> 0` threshold sitting on 1 of 100 draws is not distinguishable
   from a much smaller true rate (binomial 95 % CI on 1/100 is roughly 0.0003–0.054). Had that
   one draw not inverted, ε\*(weight) would drop to **1000**, and BankSim/stratified would
   already be **un-censored at a measured 10×** — a different number *and* a different kind of
   number.
2. **It is worse than that: on BankSim/stratified *both* ends of the ratio are single-draw
   thresholds.** ε\*(output)=100 also sits at inversion rate **0.01**. So **≥30×** is a ratio of
   two 1-of-100 thresholds. ULB (ε\*out=50 at rate 0.10) and BankSim/entity-disjoint
   (ε\*out=300 at rate 0.11) are ~10× better supported on the output end.
3. **Extending the grid does not address any of this.** More ε points test *further out*; the
   fragility is in the *estimator* at a fixed ε. The two are independent fixes and the objective
   needs both.

✅ **What is solid, and stays solid:** the draws are seeded (`np.random.seed(3000 + rep)`,
`private_incentive_sweep.py:166,174`) and **the same seed drives both channels in the same
iteration** — a paired design across channels, ε, and all three conditions. So every re-run
reproduces bitwise, and the **ordering** claim (output > weight at 9/9 budgets in all three) is
paired-comparison evidence, not three independent noisy runs. **The ordering was never the
fragile part and is not affected by anything above.**

#### ▸ PRE-REGISTERED, 2026-09-04, before any run (rule 4). Do not edit after results land.

- **(1) The ordering holds at every new ε too.** Output ≥ weight at every added budget, all
  three conditions. Falsifier: any budget where the weight channel inverts less than the output
  channel. This is the claim, and it is the one I expect to hold.
- **(2) ULB and BankSim/entity-disjoint stay right-censored at ε=30000.** Their rates at the top
  (0.20, 0.38) are far from zero and the ULB curve only fell 0.57→0.20 across a 3× budget
  increase. Falsifier: either reaches 0.00. **If they de-censor, the extrapolation was wrong and
  that is worth more than the extension** — say so rather than quietly banking the number.
- **(3) BankSim/stratified de-censors, and the honest reading of that is ambiguous.** Predicted:
  its weight rate hits 0.00 at ε=10000. **But a de-censoring driven by 1→0 draws is not evidence
  the true rate crossed zero**, so per (4) below this must not be reported as a measured ε\*
  unless the repeat count is also raised. Falsifier: it stays > 0.
- **(4) Raising `n_repeats` at the top of the grid moves BankSim/stratified's ε\*(weight) off
  3000.** At 1000 repeats a true rate near 0.01 yields ~10 inversions and ε\* stays 3000; a true
  rate near 0.001 likely yields 0 and ε\* drops to 1000. **Either outcome is publishable and
  they are opposite**, which is what makes this worth the CPU. Falsifier for the whole framing:
  the rate is stable near 0.01, in which case ε\*=3000 was right all along and only the
  confidence interval improves.
- **(5) The three conditions stay ordered ULB > BankSim/strat > BankSim/entity-disj on ε\*
  (output).** 50 < 100 < 300 now. Predicted unchanged — the output channel de-censors well
  inside the existing grid in every condition, so extending it cannot move these.

#### ⚠ CORRECTION TO ITEM (4), found 2026-09-04 **before any of the run landed**

Recorded here rather than by editing the pre-registration above, which stays verbatim.
This came out of reading `private_incentive_sweep.py:165-180`, not out of a result.

**Item (4)'s "ε\* drops to 1000" branch cannot happen.** The draws are seeded
`np.random.seed(3000 + rep)` and the seed is re-set immediately before *each* channel, so
the outcome of repeat *r* depends only on *r*. A run with `--repeats 1000` therefore
reproduces repeats 0–99 **bitwise** and strictly *contains* the 100-repeat run. The single
inverted draw that puts BankSim/stratified's weight channel at rate 0.01 for ε=3000 is one
of those 100 — so it is still there at 1000 repeats, `rate > 0` still holds, and
ε\*(weight) stays at **≥ 3000**. It cannot fall to 1000 no matter how many draws are added.

**The general form, which is the more useful statement:** ε\* = max{ε : rate > 0} is
**monotone non-decreasing in both the repeat count and the grid extent**. More draws can
only turn a 0/100 cell into a non-zero one; more budgets can only add candidates to the
max. So neither knob can ever *lower* ε\*, and "de-censoring" means only that the added
budgets happened to show zero inversions — not that a population quantity was located.
**ε\* is properly read as "the largest budget at which an inversion was *caught*", and it
gets worse the harder anyone looks.** That is a property of the estimator, not of the
mechanism.

**What this changes, and what it does not.**

- **It does not cancel the `--repeats` pass.** The pass still buys the thing the premise
  correction actually asked for: a 10× tighter interval on the *rate* at the deciding
  budget, which is what separates "true rate ≈ 0.01" from "true rate ≈ 0.0003". The rate is
  estimable; the threshold built on it is not stable. Run it — and report the **rate with
  its interval**, not only ε\*.
- **It redirects the pass.** The budgets worth 1000 draws are no longer only the top ones.
  They are (a) the budget that *decides* each ε\*, and (b) the budgets **above** it that
  currently read 0/100 — because 0/100 is consistent with a true rate up to ≈0.03, and
  those are the only cells that can move ε\* at all. For BankSim/stratified that is
  ε∈{100, 300} on the output end and ε∈{3000, 10000…} on the weight end.
- **It gives item (4) a falsifier that can actually fire.** As written its two outcomes were
  "ε\* stays 3000" and "ε\* drops to 1000"; the second is impossible, so the real question
  is whether ε\* **rises** — i.e. whether a currently-zero cell starts inverting under 10×
  the draws. Scored that way when the pass lands.
- **It is checkable, and it will be checked.** Nesting predicts the 9 historical budgets
  reproduce their inversion counts *exactly* in the 11-point run (same seeds, same models,
  same 100 repeats). If any of them moves, the determinism claim is false and both the
  "reproduces bitwise" line in the draft and this correction are wrong — which would be the
  OBJ-2 lesson (pinned torch version and thread count) recurring. Verified after the run.

#### ▸ The work

- [x] **Add `--eps-grid`** rather than editing the literal at `private_incentive_sweep.py:93-94`
      (`[1, 5, 10, 30, 50, 100, 300, 1000, 3000]`), so the grid is recorded in provenance and not
      only in git. `argparse` there (`def main()`) currently takes only `--quick` and `--no-plots`
      plus the `_dataset.py` trio.
- [x] **Add `--repeats`** in the same edit — it is the same two lines of argparse and item (4)
      above is un-runnable without it. `n_repeats` is hardcoded at `:92`.
- [x] **Write the grid, the repeat count, and an explicit `censored` boolean into the JSON.**
      Today `epsilon_star_weight` is just a float; a reader cannot tell a censored ε\* from a
      measured one, and *that is precisely the bug this objective exists to fix.* Recompute
      `censored = (inversion_rate at max(grid)) > 0`. The grid is recoverable from
      `sweep[].epsilon`, but recoverable is not stated.
- [x] ⚠ **All conditions must share one grid.** Re-run ULB **and** both BankSim partitions, or
      the three columns stop being comparable. **Measured cost from the stored JSONs**
      (`elapsed_sec`): ULB **954.7 s**, BankSim/strat **1675.2 s**, BankSim/cust **1542.4 s** —
      **~70 min for 9 points**, so ~**85 min** for 11. A `--repeats 1000` pass at the top budgets
      only is the cheaper way to buy item (4); a full 10× re-run is ~12 h and is **not** approved.
- [x] Re-run `sweeps_cross_condition.py` and update SITREP / INTEL / CONFIRMED KILLS **only for
      whichever conditions actually de-censor.** A condition still inverting at the new ceiling
      stays a lower bound and must keep saying so (rule 3 — a partial success does not upgrade
      the rest).
- [x] ~~**Zero-CPU sibling task:** map the paradox-flagged cells to the Krum−FedAvg rows.~~ —
      **DONE 2026-09-04, pulled forward into the OBJ-18 window.** Both
      `byzantine_robustness_sweep.py` and `sweeps_cross_condition.py` now name and mark the
      affected cells, and the collator adds a **clean-subset table**. Two results, the second
      unanticipated: **(a)** the entity-disjoint headline range was carried by contaminated
      cells — clean-only it is **−4.73 to −1.48 pp**, not −12.57 to −1.48; **(b)** on the 3 cells
      clean in *all three* conditions the dataset effect is negative 3/3 and the partition effect
      positive 2/3, so **the OBJ-15 withdrawal survives the filter** rather than being produced
      by it. Full table in CONFIRMED KILLS.

#### ▸ SCORECARD — the grid run, landed 2026-09-04 14:25. The five items above are unedited.

Three conditions, one shared 11-point grid `[1, 5, 10, 30, 50, 100, 300, 1000, 3000, 10000,
30000]`, 100 draws per ε, ~78 min total (ULB 1138.2 s · BankSim/cust 1767.8 s · BankSim/strat
1718.0 s). Collated by `experiments/sweeps_cross_condition.py`.

| # | Pre-registered (verbatim above) | Outcome |
|---|---|---|
| 1 | The ordering holds at every new ε too | ✅ **HELD at both added budgets, all three conditions** — and see the correction below, because the *existing* budgets were never scored on this metric. |
| 2 | ULB and BankSim/entity-disjoint **stay right-censored** at ε=30000 | ❌ **WRONG, and it pre-named this as the more valuable outcome.** Both de-censored, and so did the third. **All three weight channels reach 0/100** — ULB at ε=10000, BankSim/strat at ε=10000, BankSim/entity-disj at ε=30000. |
| 3 | BankSim/stratified de-censors, at ε=10000, for a reason too weak to quote | ✅ **CORRECT on both halves.** Weight rate hits 0.00 at exactly ε=10000, and the de-censoring is a 1→0 draw transition, so ε\*=3000 still rests on one draw. |
| 4 | Raising `n_repeats` moves BankSim/stratified's ε\*(weight) off 3000 | ⏸ **NOT RUN YET**, and its "drops to 1000" branch was already shown impossible — see the CORRECTION above. Re-scoped to "does ε\* **rise**". |
| 5 | Conditions stay ordered ULB > BankSim/strat > BankSim/entity-disj on ε\*(output) | ✅ **CORRECT, unchanged.** 50 < 100 < 300. |

**The objective's stated goal was met: every factor is now a measurement, not a lower bound.**

| Condition | ε\*(weight) | ε\*(output) | factor | draws deciding it (weight / output) |
|---|---|---|---|---|
| ULB / stratified          | 3000  | 50  | **60×** | ⚠ **1/100 and 1/100** |
| BankSim / stratified      | 3000  | 100 | **30×** | ⚠ **1/100 and 1/100** |
| BankSim / entity-disjoint | 10000 | 300 | **33×** | 13/100 and 11/100 |

**⚠ Three findings the pre-registration did not anticipate, and two of them invert standing
advice in the SITREP.**

1. **The trustworthiness ordering of the three conditions has flipped.** The SITREP says
   *"Prefer ≥60× (ULB) when a single figure is needed"* and calls BankSim/entity-disjoint
   *"furthest from de-censoring"*. Both are now false. ULB's 60× is a ratio of **two
   single-draw thresholds** (exact 95 % CI on 1/100 is 0.0003–0.0545 at each end), whereas
   BankSim/entity-disjoint is the **only** condition whose ε\* rests on double-digit draws at
   both ends. **The best-supported single figure is now 33× (BankSim/entity-disjoint), not
   60× (ULB).**
2. **≥10× became 33× — the lower bound was not merely loose, the extension *raised* the
   numerator.** BankSim/entity-disjoint's ε\*(weight) moved 3000 → 10000 because ε=10000
   inverts 13/100. That is the monotonicity in the CORRECTION above doing exactly what it
   says: more budgets can only push ε\* up. A "de-censored" factor is therefore not a fixed
   point either — a 12-point grid could raise it again.
3. **The pre-registration's own metric was never the one being reported.** Item (1) is about
   *inversion rates*; the SITREP's *"output ≫ weight at 9/9 budgets in all three conditions"*
   came from the ρ table. On ***inversion rates*** BankSim/entity-disjoint is **10/11, not
   11/11** — at ε=1 the weight channel inverts 85/100 and the output channel **88/100**, so
   the falsifier as literally written fires there. The same cell in the archived 9-point run
   was 85 vs 88 too, so **this was true when "9/9" was written and nobody had checked it on
   the pre-registered metric.** Two metrics ranking the same cell oppositely is the rule-11
   hazard the repo already documents. The collator now scores **both**, side by side.
   *Scope it honestly:* at ε=1 both channels are near-total failure (85 % and 88 %), so this
   is a counter-example to the claim as written, not a case where the weight channel is
   usefully better.

#### ⚠ THE CORRECTION'S OWN PREDICTION WAS CHECKED, AND IT PARTLY FAILED — ULB does not reproduce

The addendum above staked itself on a check: *"the 9 historical budgets reproduce their
inversion counts exactly."* `experiments/check_obj17_nesting.py` ran it against the archived
9-point results at identical seeds and repeat count.

| Condition | shared budgets | result |
|---|---|---|
| BankSim / stratified      | 9/9 | ✅ every count identical |
| BankSim / entity-disjoint | 9/9 | ✅ every count identical |
| **ULB / stratified**      | 9   | ❌ **13 of 18 cells moved** |

ULB's ground-truth Shapley split moved with them: `[0.619, 0.250, 0.131]` →
`[0.554, 0.336, 0.110]`. Its weight inversions at ε=3000 went **20/100 → 1/100**, at ε=300
**65/100 → 28/100**, and its output inversions at ε=50 went **10/100 → 1/100** — which is
*the entire reason* ULB's 60× now reads as fragile when this morning it read as the
best-supported factor on the board.

**So the noise draws are deterministic — BankSim proves that twice — and something in the ULB
path is not.** Two candidates, and git cannot separate them because `models/adtcn.py`,
`models/federated_adtcn.py` and `data/data_loader.py` all carry **uncommitted** modifications,
so the code state that produced the 2026-08-30 ULB result no longer exists anywhere:

- **(a) genuine nondeterminism in ULB org-model training** — the OBJ-2 lesson (pinned torch
  version *and* thread count) recurring on a different metric; or
- **(b) a code change since 2026-08-30** that altered ULB training only.

Ruled out already: the `n_raw_features` forwarding added by OBJ-15 resolves to 33 for ULB
either way (`adtcn.py:306` — `min(cols, N_RAW_FEATURES + 3)` = 33), and `get_loader("ulb")` is
exactly the old bare `FinancialDataLoader()`. Neither changes the ULB training path.

**⛔ Until this is settled, no ULB number in this objective is quotable** — not 60×, not
ε\*=3000, not ε\*=50. The BankSim columns are unaffected: both reproduce exactly.

**The decisive test ran, and it came back (a)-negative: ULB reproduces ITSELF exactly.**
`check_obj17_nesting.py --pair` on `private_incentive_sweep.json` vs
`private_incentive_sweep_rep2.json` — **11/11 budgets identical, ground truth identical to
six decimals**. So ULB training is deterministic under today's conditions, and the Aug-30
divergence is a change in *conditions*, not run-to-run noise. **Today's three columns are
mutually consistent** (all three produced by the same current code; BankSim reproduces from
2026-09-01, ULB from today), which is what the OBJ-17 numbers need in order to stand.

**But the cause of the Aug-30 divergence is still NOT identified, and that is the honest
statement.** Every file on the ULB B1 path was diffed against the committed Aug-30 state:

| File | Change | Can it affect ULB's B1 run? |
|---|---|---|
| `models/adtcn.py` | `build_sequences` rewrite + `groups=` plumbing + architecture factory | **No.** For `groups=None` the new index-clamp form is *provably equivalent* to the old pad-and-stack: window *i* = rows `max(0, i-L+1)…i` in both. `n_raw` resolves to 33 either way (`min(cols, N_RAW_FEATURES+3)` = 33 = ULB's `raw_feature_count`). The factory returns the identical `_Conv1dClassifier` for `"cnn"`. |
| `utils/metrics.py` | `obf2_value` re-derived (Eq.11 → bounded form) | **No.** `compute_all_metrics` is untouched, and `coalition_score` uses **balanced accuracy**, not Obf2. This sweep never calls `obf2_value`. |
| `data/data_loader.py` | `+raw_feature_count` property, `+last_org_groups` bookkeeping | **No.** Consumes no RNG; `split_for_orgs` splits identically. |
| `models/federated_adtcn.py` | docstring only | **No.** |
| `config.py` | BankSim entries + `get_loader` added | **No.** `ADTCN_CONFIG` / `DATA_CONFIG` unchanged; `get_loader("ulb")` *is* the old bare `FinancialDataLoader()`. |
| `blockchain/federation_manager.py` | **not modified at all** | — |

**So no code change on the path explains it.** The next candidate was the one the repo
documents in its own `requirements.txt` — *"bitwise reproducible only for a fixed (torch
version, CPU thread count) pair … changing either changes the float reduction order inside the
convolutions"* — and thread count is pinned nowhere in the code.
`experiments/check_thread_sensitivity.py` trained one ULB org model (BankA, 99,682 samples, 12
epochs) at **1 vs 4 threads** and compared the weights exactly.

**❌ Ruled out too: all six weight tensors bit-identical, max |A−B| = 0.000e+00.** The config
dicts were then compared literal-by-literal against the Aug-30 commit — `ADTCN_CONFIG`,
`DATA_CONFIG`, `FEDERATION_CONFIG`, `INCENTIVE_CONFIG`, `ORG_DATA_SPLITS`, `DB_BOA_CONFIG` —
**all identical**.

#### ⛔ VERDICT: the ULB divergence is UNEXPLAINED, and it is recorded as unexplained

Everything reachable has been eliminated: the code on the path (each change provably inert for
ULB), every config dict, and the thread count. What remains is the **torch/numpy version at the
time** — and that is **unrecoverable**, because `torch==2.12.0` was pinned in `requirements.txt`
*after* the 2026-08-30 result was generated, and **no stored result records the stack it ran
under**. Guessing which version it was would be inventing a cause; rule 3 says a negative result
stays negative, and "we do not know" is the negative result here.

**What this costs, stated plainly:** the 2026-08-30 ULB private-incentive result is **not
reproducible under the current environment and its cause cannot be determined**. Today's ULB
numbers are internally sound — they reproduce exactly on re-run and were produced by the same
stack as both BankSim columns — so **OBJ-17's ULB column stands on its own evidence**; what is
lost is the ability to reconcile it with the number the supervisor brief currently quotes.

**✅ The one durable fix, shipped 2026-09-04:** `experiments/_dataset.py` gained
`environment()`, and `provenance()` now writes **torch version, torch thread count, numpy
version, Python version and platform into every sweep JSON**. All four system sweeps route
through `provenance()`, so this closes the gap for every one of them at once. The three
2026-09-04 results were stamped by hand, marked `_backfilled`, so a reader can tell a hand
stamp from a run-written one. *This diagnosis took a day and produced no number; the guard is
what stops the next one costing the same.*

⛔ **Quote the BankSim columns, not ULB.** BankSim/entity-disjoint's **33×** is the only factor
in this objective that is simultaneously *measured*, *reproduced across two dates and two code
states*, and *supported by double-digit draws at both ends*.

#### ▸ PRE-REGISTERED, 2026-09-04 — the `--repeats 1000` pass, written BEFORE launch (rule 4)

Approved by the operator at full scope: **all three conditions**, ~135 min. Grids are
per-condition because the informative budgets are per-condition — four each, not the whole
sweep, because **only a budget currently reading 0/100 can move ε\***, plus the budget that
currently decides it (to tighten its rate interval).

| Condition | grid | why these four | ε\* now (w / o) |
|---|---|---|---|
| ULB / stratified          | `50,100,3000,10000`    | 50 and 3000 *decide* ε\*, each on 1/100; 100 and 10000 are the 0/100 cells directly above them | 3000 / 50 |
| BankSim / stratified      | `100,300,3000,10000`   | same shape: 100 and 3000 decide, each on 1/100; 300 and 10000 sit above | 3000 / 100 |
| BankSim / entity-disjoint | `300,1000,10000,30000` | 300 and 10000 decide, on 11/100 and 13/100 — already solid, so this condition is testing the *other* thing: whether 1000 and 30000 (both 0/100) start inverting | 10000 / 300 |

**Predictions. Do not edit after results land; score underneath.**

- **(A) No ε\* falls.** Structural, not empirical — seeds are `3000+rep`, so 1000 draws contain
  the 100. Recorded as a **check on the harness, not a finding**: if any ε\* falls, the nesting
  argument is broken and `check_obj17_nesting.py` should have caught it. Falsifier: any ε\*
  lower than the value in the table above.
- **(B) The two single-draw thresholds survive as non-zero, at a rate near 0.001–0.01.** ULB and
  BankSim/stratified each showed 1/100 at their deciding budget. At 1000 draws I expect
  **1–15 inversions**, i.e. the rate is real but small, and the interval narrows by ~3×.
  Falsifier: exactly 1 inversion in 1000 at either (⇒ the true rate is ~10× smaller than the
  point estimate and those ε\* are effectively artefacts of catching one outlier).
- **(C) At least one 0/100 cell starts inverting, and I expect it on BankSim/entity-disjoint at
  ε=30000.** Its weight channel was still at 13/100 one decade below, so 0/100 at 30000 is the
  cell least likely to be a true zero. **If it inverts, ε\*(weight) rises 10000 → 30000 and the
  factor rises 33× → 100×** — which would be the second time in one day that looking harder
  made the headline bigger, and the clearest possible demonstration that ε\* is not a
  population quantity. Falsifier: every 0/100 cell stays 0/1000 in all three conditions.
- **(D) ULB's ε\*(output) rises from 50 to 100.** ε=100 read 0/100 while ε=50 read 1/100, and
  the output curve is steep there (15/100 at ε=30). A single decade from 1/100 to a true zero
  is a big drop. **If it rises, ULB's factor falls 60× → 30×** and the three conditions
  collapse to ~30× / 30× / 33× — i.e. the *condition-dependence* of the factor would itself
  be an artefact of thin sampling. Falsifier: ε=100 stays 0/1000.
- **(E) The ordering claim is untouched.** Output ≤ weight inversions at every probed budget in
  every condition. This is paired within each draw, so 10× the draws cannot change its
  direction, only its precision. Falsifier: any probed budget where weight < output.

> ⚠ **(C) and (D) point in opposite directions and both are live.** One raises a factor, the
> other flattens the spread between conditions. That is why this pass is worth 135 minutes even
> though the grid already de-censored everything: **it tests whether the three "measured"
> factors are stable, and the honest outcome may be that they are not.**

#### ▸ SCORECARD — the `--repeats 1000` pass, landed 2026-09-04 17:53. The five items above are unedited.

Three conditions × 4 budgets × 1000 draws, ~2 h 50 m (ULB 3044.5 s · BankSim/strat 3946.5 s ·
BankSim/cust 3360.4 s). Nesting verified for all three: **4/4 shared budgets consistent**, the
1000-draw runs strictly contain the 100-draw ones.

| # | Pre-registered (verbatim above) | Outcome |
|---|---|---|
| A | No ε\* falls (harness check, not a finding) | ✅ **HELD.** No ε\* fell anywhere; one rose. The nesting argument survives its own test. |
| B | The two single-draw thresholds survive as non-zero, 1–15 inversions in 1000 | ✅ **HELD in substance** — and this is the reassuring result. 1/100 became **7/1000, 11/1000, 16/1000, 10/1000**, all ≈ 0.01. ⚠ One (BankSim/strat weight, 16) is *one draw above* the predicted band; recorded, not rounded away. **The falsifier (exactly 1 in 1000) fired nowhere.** |
| C | A 0/100 cell starts inverting; **I expect BankSim/entity-disjoint at ε=30000**; ε\*(weight) rises 10000 → 30000 and the factor 33× → 100× | ✅ **CORRECT, and precisely** — the named cell, the named direction, the named number. ε=30000 inverted **1/1000**. Exactly one 0/100 cell moved, out of six probed. |
| D | ULB's ε\*(output) rises 50 → 100, dropping its factor 60× → 30× | ❌ **WRONG.** ε=100 stayed **0/1000**. The falsifier as written — *"ε=100 stays 0/1000"* — fired verbatim. ULB stays at 60× and is now *better* supported. |
| E | The ordering is untouched | ✅ **HELD, 12/12 probed budgets** (4 per condition). |

#### ⚠ THE PASS OVERTURNED THIS AFTERNOON'S HEADLINE — twice over

**1. BankSim/entity-disjoint is RIGHT-CENSORED AGAIN, at ≥100×, on one draw in a thousand.**
Six hours ago the grid pass de-censored it at a measured 33× and it was the **best**-supported
column on the board (13/100 and 11/100). Looking 10× harder found an inversion at ε=30000, the
top of the probe grid — so its ε\*(weight) rose 10000 → 30000, its factor rose 33× → **≥100×**,
and it is **a lower bound again**. Nothing above 30000 has been tested.

> This is the monotonicity note from this morning demonstrated in the sharpest possible form:
> **more effort made the headline bigger AND re-censored it, in the same run.** A "de-censored"
> ε\* is not a fact about the mechanism; it is a statement about how hard anyone has looked.

**2. The trustworthiness ordering flipped BACK.** After the grid pass I wrote *"prefer 33×
(BankSim/entity-disjoint), not 60× (ULB)"* and propagated it into the supervisor brief. At 1000
draws that is **wrong**:

| Condition | ε\*(w) | ε\*(o) | factor | draws deciding it | status |
|---|---|---|---|---|---|
| ULB / stratified          | 3000 | 50  | **60×**   | **7/1000** and **11/1000** | measured (0/1000 at ε=10000) |
| BankSim / stratified      | 3000 | 100 | **30×**   | **16/1000** and **10/1000** | measured (0/1000 at ε=10000) |
| BankSim / entity-disjoint | ≥30000 | 300 | **≥100×** | ⚠ **1/1000** and 115/1000 | ⛔ **RIGHT-CENSORED** |

⚠ Caveat on rows 1–2: ε=30000 was swept at **100 draws only** in those two conditions. Their
weight rates fall monotonically (0.21 → 0.007 → 0 across ε=1000/3000/10000), so an inversion at
30000 would be anomalous — but *anomalous is what BankSim/entity-disjoint just did*. Say
"measured on the evidence swept", not "measured".

**3. The single-draw thresholds were REAL.** This is the one unambiguously good result. 1/100 at
four different (condition, channel) thresholds became 7, 11, 16 and 10 per 1000 — every one
consistent with a true rate near 0.01, none an outlier. **So the fragility was about *precision*,
never about *existence*:** ULB's 60× and BankSim/stratified's 30× were correctly located all
along, and their intervals are now ~4× tighter (e.g. ULB weight 0.0003–0.0545 → 0.0028–0.0144).

#### ⛔ THE FINDING THAT MATTERS MORE THAN ANY OF THE ABOVE: ε\* IS NOT A ROBUST STATISTIC

ε\* is `max{ε : inversion rate > 0}`. That definition snaps to a grid point, is monotone in
effort, and sits on a hard `> 0` threshold. Re-deriving the factor under two *other* rate
thresholds, on the best evidence available per cell (1000 draws where probed, else 100):

| Condition | `rate > 0` (as published) | `rate > 0.01` | `rate > 0.05` |
|---|---|---|---|
| ULB / stratified          | **60×** | 20× | 33× |
| BankSim / stratified      | **30×** | 60× | 20× |
| BankSim / entity-disjoint | **≥100×** *(censored)* | 33× | 33× |

**The factor swings between 20× and ≥100× on an arbitrary threshold choice, and the *ranking of
the three conditions* changes with it too.** It is a ratio of two grid-snapped thresholds, so it
moves with the threshold, the grid resolution *and* the sample size. **No single number from this
family is defensible as a headline.**

**What IS robust, and should carry the claim instead:**

1. **The ordering.** Output channel inverts no more often than the weight channel — 12/12 probed
   budgets, and 11/11 · 11/11 · 10/11 on the full grid. Paired within each draw, so sample size
   changes its precision and not its direction. This has now survived three conditions, two
   sample sizes and two metrics.
2. **The rank-fidelity gap at a stated ε**, which needs no threshold at all: at ε=50, weight
   ρ = **−0.199** vs output ρ = **+0.995** (ULB); **+0.017 vs +0.945** (BankSim/strat);
   **−0.070 vs +0.805** (BankSim/entity-disj). Same direction in all three, no grid-snapping, no
   arbitrary cut-off.
3. **The rates with intervals**, quoted at named budgets, rather than the threshold built on them.

⚠ **This is a rule-3 result: the negative part is the valuable part.** The objective set out to
convert a censored bound into a measurement. It did that — and then showed the measurement was
never the right thing to quote. **Do not "fix" this by picking whichever threshold flatters the
mechanism** (rule 4); the mechanism is well supported by the ordering and the ρ-gap without it.

> **Value, restated honestly after the correction above.** This objective may not recover a
> measurement — items (2) and (3) predict that two conditions stay censored and the third
> de-censors for a reason too weak to quote. **That is still the right thing to run**, because
> the alternative is continuing to publish a factor whose most-quoted variant rests on one
> coin-flip, and because item (4) is decisive either way. If it ends with all three still
> censored and a firmer ≥ bound, **that is a successful outcome, not a failed one** — write it
> up as one.

---

### ◈ ~~OBJ-18 — Propagate the Krum withdrawal out of TASK.md~~  → **CLOSED 2026-09-04**, see CONFIRMED KILLS

---

## ▮ CONFIRMED KILLS

**OBJ-13 — "the DB-BOA search selects on noise" is now DIAGNOSED, REPAIRED, and the repair
CHANGES NOTHING about the verdict** — *2026-09-04/05, ~20 h CPU.* Four things are settled:

1. **The negative result survives the repair, decisively.** 9 searches at the deployed budget
   (3 protocols × 3 seeds, BankSim, pop 20×30). **0 of 9 beat a hand-set 128f/150spe**; mean
   paired Δ **−0.1982** on 7 shared held-out draws. Rule 3's "DB-BOA buys no measured accuracy
   gain" no longer rests on a p=0.34 tie — it rests on a direct paired measurement, under three
   surrogate protocols, and the shipped 142/76 loses too (−0.1900).
2. ⛔ **The shipped ULB surrogate had been validating on rows it trained on** — a 3,000-row pool
   held **5 unique fraud transactions**, `_MIN_FRAUD_ROWS = 30` duplicated them ~6×, and **9 of 9
   validation positives were copies of training rows**. `Obf2 = 5.0000` was memorisation, not
   luck on nine rare rows. **The staged repair did not close it** (`deterministic` leaked 9/9
   identically); only enlarging the pool did. Fixed: `eval_subset` 3,000 → 36,000 plus a
   `RuntimeWarning` guard. Re-verified **0 of 11** leaked.
3. **Common random numbers, not determinism, is what makes a noisy search reproducible.** Filter
   spread across objective seeds: `legacy` 66 · `deterministic` **81 — wider than legacy** ·
   `averaged` **12**. This was pre-registered the other way and is the objective's most
   transferable finding.
4. ⚠ **And the converged mode generalises worst.** `averaged` 3.4237 vs `deterministic` 3.5857 vs
   `legacy` 3.5552 on held-out draws: it converges onto high-filter configurations that overfit
   its k=3 shared draws. **The noise the repair removed was doing accidental regularisation.**

**Three mechanisms this objective previously asserted are withdrawn**, all struck at source in
TASK.md, in `OBJECTIVE_noise_audit.md`'s generator, and in the `adtcn.py` comment: *"the fraud
rate is not what separates ULB from BankSim"* (it is — via the `len(fraud_idx) < 30` threshold);
*"Obf2 = 5.0000 is reachable by luck"* (no luck required on the shipped path); and the fraud
count as the driver of surrogate noise (BankSim's std held flat at 0.2563 → 0.2553 across **8×**
more fraud rows).

**Still open, and small:** pre-registration 5 needs `detector_multiseed` on the test set; the two
ULB files need regenerating on the fixed pool; and `ADTCN.fit:681` still carries the pre-repair
floored formula, so the dead axis was **surrogate-only** — a defect found by this work and not
yet decided on.

---

~~**OBJ-18 — the withdrawn Krum claim propagated out of TASK.md**~~ — **CLOSED 2026-09-04, ~2 h,
no CPU.** The sweep found the withdrawn mechanism alive in **two generators** and the exposed
brief carrying a separate over-claim. All fixed at the generator, then regenerated.

**What the sweep actually found** (grep of every `.md`/`.tex` for `entity-disjoint`, `12.57`,
`12.6 pp`, `pays utility`, `generalise worse`, `Krum pays` — 4 files hit, 3 of them generated):

| Where | What it said | Fix |
|---|---|---|
| `byzantine_robustness_sweep.py:374` (generator) | *"Krum selects a single org's weights; where orgs hold disjoint entities that model generalises worse…"* — emitted into **both** BankSim drafts, including the **stratified** one, where orgs are **not** customer-disjoint | Sentence removed; the draft now states the measured advantage, says **no mechanism is established**, names the withdrawal date, and points at the collator. A comment above the branch forbids re-introducing an explanation |
| `federated_cross_dataset.py:193` (generator) | *"Its value shows up under attack (8/8 attacks rejected)"* — kill-list wording #1, conflating rejection with utility | Split into two claims: rejection 8/8 ×3 conditions, utility negative 7/8 and 8/8 on BankSim. Also now states that this ablation **corroborates** the withdrawal from a second experiment and a different metric |
| `SUPERVISOR_BRIEF.md` / `.tex` | (a) *"a ~60× improvement"* / *"ε\* = 3000"* — kill-list wording #2, stated as a measurement; (b) Krum's +12.49 pp presented as the general result; (c) *"single-source data"* — two datasets since | (a) → **≥**60×, ε\* **≥**3000, plus a caveat box on censoring and condition-dependence, and the ordering (9/9 ×3) promoted to *the* claim; (b) → rejection replicates ×3, accuracy does not, no mechanism offered; (c) → rewritten as *simulated, not genuinely multi-institution* |
| `TASKB_economic_*`, `OBJ15_two_factor_*`, `TASKB1_*` | checked — already correctly scoped, no edit needed | — |

**Nothing needed stamping.** Every affected draft is machine-generated, so the honest fix was the
generator plus `--redraft`; the brief is a live document under git, and git is its record. The
OBJ-14 stamping rule applies to hand-written drafts that are *superseded*, and none were.

**A second bug the sweep exposed, which the checklist did not predict.** The mechanism sentence
was emitted by a **single-run** generator — a script reading one JSON structurally *cannot* see a
dataset-vs-partition effect, so any attribution it prints is unfalsifiable by its own evidence.
That is a class of bug, not an instance: **a generator may only assert what its own input can
falsify.** Attribution belongs in `sweeps_cross_condition.py`, which reads all three.

⚠ **Not fixed, deliberately:** `FINAL YEAR THESIS REPORT/chapters/chapter_6.tex` and the stamped
`REWRITE_06_results.md` still say "60×" unqualified. The thesis was **submitted and accepted
2026-06-13** on ULB alone and the REWRITE files are stamped historical record — neither is a live
claim. The paper extension is where this gets restated; see [OBJ-9].

✅ **Toolchain, incidental:** `SUPERVISOR_BRIEF.pdf` had been unbuildable since before this session
(MiKTeX had no scalable Type1 CM fonts, so `microtype` auto-expansion aborted — verified by
building the **HEAD** version, which fails identically). Installing `cm-super` fixed it with no
change to the document's typography, and the PDF is rebuilt (9 pp, 2026-09-04). That is why the
PDF was dated 8/30 while the `.tex` had been edited on 9/1 — **the brief the supervisor reads had
been two revisions behind and nobody would have seen it.**

~~**"Krum pays utility under entity-disjointness"**~~ — **KILLED 2026-09-04 by our own confound
control.** Four sweeps on BankSim/`stratified`, landed 2026-09-01 23:41, scored three days later.
This was OBJ-15's most quotable finding and it did not survive its first control.

**What was claimed:** Krum rejects attackers perfectly but costs 1.5–12.6 pp of utility *because*
orgs holding disjoint customers are not interchangeable, so selecting one org's weights
generalises worse than averaging.

**What killed it:** Krum is negative in **7 of 8 cells under the stratified partition too**,
where orgs are *not* customer-disjoint. The dataset effect is negative in all 8 rows (−1.06 to
−17.27 pp); the partition effect changes sign. **And at n=7 the partition effect is positive in
all four rows** (+1.36 to +5.28) — entity-disjointness makes Krum look *better*, the opposite of
the claimed mechanism.

**Say instead:** Krum rejects the attacker **8/8 in all three conditions** (security replicates,
three-condition solid) and **can cost utility, by an amount that tracks the dataset**. No
mechanism for the utility cost is currently on record. **Say that too** — an unexplained measured
effect is an honest result; a mechanism invented to cover it is not (rule 3).

⚠ **Which pp figures are quotable — mapped 2026-09-04, no longer an open item.** In **5 of 8**
entity-disjoint cells the attacked FedAvg baseline beats its own no-attack reference. The
artefact is condition-graded (0/8 · 2/8 · 5/8), i.e. worst precisely where OBJ-15 headlined.
The cells are now named in every generated draft and in
`final_report_data/OBJ15_two_factor_decomposition.md`:

| Condition | contaminated cells (do NOT quote) | clean cells (quotable) |
|---|---|---|
| ULB / stratified | *none* | all 8 |
| BankSim / stratified | n=7 `gaussian` (−11.96) · n=7 `label-flip` (−11.61) | the other 6 |
| BankSim / entity-disjoint | n=5 `sign-flip` (−11.20) · n=5 `gaussian` (−12.57) · n=5 `label-flip` (−11.13) · n=7 `gaussian` (−10.60) · n=7 `label-flip` (−6.32) | n=5 `scaled` (−1.48) · n=7 `sign-flip` (−4.73) · n=7 `scaled` (−1.65) |

**Two consequences, and the first one is a correction to how this objective has been quoted.**
1. **The headline range was carried by contaminated cells.** ~~"−12.57 to −1.48 pp"~~ — both its
   endpoints are unusable as a *range* because the −12.57 end is contaminated. On the clean rows
   only, the entity-disjoint cost is **−4.73 to −1.48 pp**, and BankSim/stratified's clean range
   is **−9.15 to +0.54 pp**. Every cell worse than −5 pp in the entity-disjoint column is
   contaminated. **Quote the clean ranges, and say they are the clean subset.**
2. ✅ **The withdrawal does not rest on the artefact.** Exactly **3 of 8** cells have a clean
   baseline in *all three* conditions (n=5 `scaled`, n=7 `sign-flip`, n=7 `scaled`). On those
   three the **dataset effect is negative 3/3** (−17.27 to −9.28 pp) and the **partition effect
   is positive in 2 of 3** (−2.03 to +4.43). That is the same reading as the full table, so
   filtering the artefact out does **not** rescue the entity-disjointness mechanism — it
   reproduces the kill on the cleanest evidence available.

**Two other pre-registrations died the same way** — lone-attacker isolation tracks the **dataset**
(fires 3/3 on both BankSim partitions, 0/3 on ULB), and MC rank fidelity ρ@n=12 moves +0.783 on
the dataset axis against −0.322 on the partition axis. **Every divergence OBJ-15 attributed to
federation structure was actually BankSim.** Four of six expectations held; the two that failed
failed identically.

> **The lesson, and it is worth more than the claim it cost:** OBJ-15 named the falsifying
> outcome in advance — *"if Krum is still negative under stratified, that mechanism is wrong."*
> Because that sentence existed before the run, a dead headline became a finding in one reading
> instead of a reviewer's objection six months later. **The control cost ~3 h. Skipping it would
> have cost the claim in public.** Full scorecard in RESUME HERE; three-condition tables in
> `final_report_data/OBJ15_two_factor_decomposition.md`.

~~**OBJ-15 — the four system sweeps taken off ULB**~~ — 2026-09-01, 3 h 15 m, four sweeps on
BankSim/entity-disjoint, clean exit. **Five of six title claims are now two-dataset.**
**Every headline replicated; every one came with a qualification the second dataset revealed.**
⚠ **Read the entry above first — one of the qualifications below has since been withdrawn.**

| Claim | Replicates? | What the second dataset added |
|---|---|---|
| Krum rejects 8/8 | ✅ **yes** (and 8/8 in all three conditions) | ~~Krum's utility flips *because of entity-disjointness*~~ **← WITHDRAWN 2026-09-04**, see entry above. The −12.57 to −1.48 pp is real but tracks the **dataset**, and 5/8 of those cells have a contaminated baseline |
| 2-of-3 collusion caught | ✅ **yes**, +47.4 / +89.7 | **Lone-attacker story changed**: isolation fires 3/3 but helps 0/3 (ULB fired 0/3) |
| Output channel ≫ weight channel | ✅ **yes**, at every ε | Factor **≥60× → ≥10×**, and *both are censored lower bounds* |
| Exact-Shapley cost | ✅ yes, 113.5 → 120.6 s | Structural (O(2ⁿ)); confirmed, expected, not news |
| "MC top-1 fails from n≈6" | ❌ **not supported** | **Intermittent on both** — ULB 5/10 (recovers at 7, 9), BankSim 3/10 (fails from **n=3**) |

**Three claims to stop making, in these words:**
1. ~~"Krum's value shows up under attack"~~ — on entity-disjoint BankSim it does not; Krum buys
   robustness and **pays utility**. Security ≠ utility; report them separately.
2. ~~"a 60× budget improvement"~~ — ε\* is right-censored (weight channel still inverting at
   ε=3000, the top of the grid, in **all three** conditions). **≥60× (ULB) · ≥30×
   (BankSim/strat) · ≥10× (BankSim/entity-disj)** — condition-dependent, and all lower bounds.
   The **ordering** is the claim and it holds 9/9 budgets in all three. **[OBJ-17] de-censors this.**
3. ~~"MC top-1 fails from n=6"~~ — withdrawn; the failure is intermittent, not a threshold.

✅ ~~**The confound that limits all of it: dataset and partition changed together.**~~ —
**RESOLVED 2026-09-04.** ULB ran stratified and BankSim ran entity-disjoint, so no comparison in
this entry separated "BankSim" from "entity-disjoint". The BankSim/stratified control was run and
**the answer was: the dataset, essentially every time.** See the entry above — it did not confirm
these readings, it overturned one of them.
⚠ **A metric artefact, now known to be condition-graded:** attacked FedAvg beats its own no-attack
reference in **0/8 (ULB) · 2/8 (BankSim/strat) · 5/8 (BankSim/entity-disj)** cases. An attack
cannot improve a model — balanced accuracy is rewarding noise-induced positive bias, the same
direction-of-collapse effect the DP sweep found here. So "FedAvg survived" is not robustness
evidence, Krum−FedAvg is not a clean utility gap, **and the contamination is worst in exactly the
condition this entry headlined.** ✅ Which cells are affected was **mapped 2026-09-04** — see
the table in the entry above; the clean-subset reading reproduces the kill.

~~**Two OBJ-15 plumbing holes the first BankSim sweep exposed**~~ — 2026-09-01, found by
reading sweep 1's output instead of trusting its exit code. **OBJ-15 made the JSON and the
draft filenames dataset-aware and stopped there.** Two things were still ULB-locked, and
both would have silently corrupted every remaining result.

**(1) Figures were never suffixed — the run was overwriting ULB data in place.**
All six `savefig` paths across the four sweeps used bare names, so the BankSim run wrote
straight over the ULB figures at 13:28. Caught after sweep 1; `economic_isolation_trajectory.png`
and `economic_accuracy_protection.png` were **recovered byte-identical from HEAD** (`ccdd19f`
postdates the 8/30 ULB run), the BankSim versions preserved as `*_banksim_customer.png`, and
the **six not-yet-overwritten ULB figures backed up to `results/_ulb_figures_backup/`** before
the remaining sweeps could reach them. All six paths now carry `suffix_of(summary)`, as do the
figure references inside the draft prose.

**(2) The drafts asserted ULB conclusions regardless of what was measured.** This is the
worse one. `economic_byzantine_sweep`'s "Honest limitations" paragraph was **static text**:
it claimed "on this 0.17%-fraud data", "so it is not isolated", and a free-rider "evades
isolation entirely" — while the table **printed directly above it** showed BankSim at 1.211%
fraud with all three lone attackers isolated at r8/r3/r3. A generated draft contradicting its
own table is the OBJ-14 failure mode with the numbers left in.
- `economic_byzantine_sweep` — limitations *and* the coordinated-majority figures
  (hardcoded "~6%", "~92%", "+43% to +86%") now computed from the run.
- `byzantine_robustness_sweep` — asserted the attacker is flagged "*every time*" plus a
  hardcoded ≈99.9%/≈87.5%. **Pre-registration calls a Krum non-replication the most valuable
  outcome in OBJ-15, and that sentence would have paraphrased it away.** Now conditional:
  a miss is named in the headline. Verified against the ULB JSON — reproduces 8/8, 99.95 %,
  87.46 % under `scaled`, margins 6.9e0–1.4e6, all from data.
- `scalability_sweep` — "on this 0.17%-fraud set", "a limitation of the ULB set". Fixed
  **before it runs**, so it picks the fix up on its own.
- `private_incentive_sweep` — clean, no hardcoded dataset claims.

**`--redraft` added to all four** (`_dataset.py:redraft`), plus `--render-only` on
`objective_noise_audit.py` under OBJ-14. **This is the same root cause both times:** while a
draft can only be rebuilt by re-running the experiment, a stale draft is cheaper to leave
broken than to fix. Rebuilding is now seconds and exact — every sweep already writes its full
summary to JSON, and both `make_plots` and `write_report` take nothing else.

> **The lesson.** "Dataset-aware" was scoped to *filenames* and stopped at the prose. Any
> generated narrative that states a conclusion must read it from the data, or it is a
> hardcoded claim wearing a script's authority. Grep new drafts for `0.17`, `ULB`, and any
> bare percentage before trusting them.

~~**OBJ-14 — retracted numbers in live drafts**~~ — 2026-09-01, ~40 min, no compute.
**The generator was the bug, and the fix is that regenerating prose now costs seconds
instead of 150 training draws.** `objective_noise_audit.py` gained **`--render-only`**,
which rebuilds `OBJECTIVE_noise_audit.md` from the existing JSON with no training. That
was the actual defect: while the only way to refresh the prose was a multi-hour re-run, a
stale draft was *cheaper to leave broken than to fix*, which is precisely how MCC 0.313
survived in a file written after the purge.
Three prose defects fixed in the generator: the withdrawn **0.313**, the fabricated
**"loses to the default (0.785)"**, and the rule-11 breach where the script quoted the
flattering noise ratio without naming it. The draft now prints **both** ratios side by
side with the question each answers (ULB 0.97 / **0.74**, BankSim 0.16 / **0.94**) and
states the mechanism: `_MIN_FRAUD_ROWS = 30` fires on *both* datasets, so the 70/30 split
leaves **~9 validation fraud rows** and `Obf2 = 5.0000` means classifying about nine rows
by luck. **The objective is uninformative on both datasets, for two different reasons.**
⚠ **Three rule-2 breaches were found that the objective did not list** — the file list in
OBJ-14 was incomplete, and a repo-wide sweep is now part of closing any purge:
  1. `SUPERVISOR_BRIEF.tex` §"honest caveat" and `SUPERVISOR_BRIEF.md` both asserted the
     tuned detector "trails a hand-set default (0.785)" **as a live claim**. This is the
     document the FIELD MAP calls the one-read walkthrough — the most exposed of all of
     them. Both now state +0.047, p=0.34, with the withdrawal named.
  2. `experiments/_seqtrain.py` docstring cited "OBJ-2 showed the same config swinging
     0.677 ↔ 0.313" — which **misstates OBJ-2's actual finding**, since OBJ-2 showed 0.313
     is *not* reproducible. Corrected to the measured spans (0.660–0.792 across seeds;
     0.708 → 0.800 on thread count alone).
  3. `finalreport_checklist.md` recorded the June disclosure route (0.47 paired gap built
     on 0.313) with no marker.
8 historical files stamped `SUPERSEDED` — the 7 named drafts plus the checklist — **unedited
on purpose**, per the objective: they are the record of what was believed and when.
Stamping is idempotent. Repo-wide `.md`/`.tex` sweep now shows every surviving mention of
0.313/0.785 is either a stamped historical file or an explicit in-context retraction.

~~**Federated ablation on BankSim, both partition schemes**~~ — 2026-08-31, 3.0 h
(`baselines_banksim_stratified.json`, `baselines_banksim_customer.json`, collated by
`experiments/federated_cross_dataset.py` → `FEDERATED_cross_dataset.md`).
**The DP result replicates on a second dataset — this is the important one**, because
the whole privacy↔incentive contribution rests on it. FedAvg 0.757 → +Krum 0.720 →
+DP(ε=1) **0.018** → full 0.026 (stratified); 0.827 → 0.793 → **−0.000** → 0.010
(entity-disjoint).
**New nuance the second dataset exposed: the collapse direction is not universal.**
On ULB the DP model goes **all-negative** (FP=0, predicts no fraud, accuracy moves
*+0.11 pp* — it looks almost unharmed while being useless); on BankSim it goes
all-positive (119,784 FP, accuracy −97.55 pp). Both MCC≈0. An accuracy-reported DP
pipeline on a 0.17 %-fraud dataset hides the failure completely — an argument for the
metric choice, not just the privacy claim.
**Krum's cost does not depend on federation heterogeneity** (−0.037 stratified vs
−0.033 entity-disjoint; the prior that a heterogeneous federation would give Krum
something worth rejecting is **not supported**). ULB's +0.207 remains the outlier.
Reading: Krum is insurance with a roughly constant premium; its value shows up under
attack (`byzantine_robustness_sweep.json`, 8/8 rejected), not in clean-federation accuracy.
⚠ **Do not report entity-disjoint partitioning as an accuracy win.** Its two BankSim
columns differ in *two* ways — partition scheme AND windowing, since only the
entity-disjoint split leaves org rows customer-contiguous and can therefore use
entity-linked windows. Given OBJ-1 priced linkage alone at +0.10–0.17, linkage plausibly
accounts for most of the +0.070 FedAvg gap. The experiment cannot separate them.

~~**OBJ-2 / OBJ-4 — the MCC contradiction, and multi-seeding the detector**~~ —
13 full 30-epoch runs. **Neither historical number is reproducible.** Training is
bitwise deterministic for a fixed *(seed, torch version, thread count)* triple —
two 4-thread runs gave identical weight hashes — but `main.py` uses torch's
4-thread default while `dbboa_vs_default.py` forces 8, so the two files were never
running the same arithmetic. Only the *tuned* config is sensitive to it: at seed 42
it scores 0.708 / 0.746 / 0.800 at 4 / 2 / 8 threads, while the hand-set default is
bitwise **invariant** (identical hash at 2 and 8). The report's "large batch →
instability" hypothesis was right about the cause (batch 2,997 vs 1,518) and wrong
about the nature: it is float-reduction-order sensitivity, not run-to-run randomness.
Over 5 seeds: tuned **0.753 ± 0.055**, default **0.706 ± 0.076**, paired diff
**+0.047, p=0.34** (Welch 0.29, Wilcoxon 0.44) — statistically indistinguishable, and
the *default* is the noisier of the two across seeds. **0.313 is withdrawn**: the
lowest tuned MCC in 7 runs is 0.6598, and re-running its own script's conditions gives
0.8004. 0.677 sits inside the observed range and stays as the deployment operating
point. Root cause of the irreproducibility: torch was listed *optional and unpinned*
in `requirements.txt`; now pinned at 2.12.0. ULB code path cleared — `build_sequences`
is byte-identical to the pre-BankSim windowing.
Report: Table 6.4, §sec:centralised, §sec:dbboa-hpo, abstract, ch5, ch9 all re-based.

~~**OBJ-3 — measured Fabric data into the report**~~ — new Chapter 6 section
§`sec:fabric-measured`: 2115.9 ms mean (n=50, p95 2172, spread only 163 ms → the 2 s
Raft `BatchTimeout` floor), the 6-point concurrency/goodput table peaking at **40.3 TPS
@ c=10**, and the MVCC knee past it (50 % fail @ c=20, 70 % @ c=40 — goodput peaks when
concurrency ≈ the 10 conflict-free keys). States plainly that this replaces the retired
85 TPS / 180 ms. Also corrected six stale "throughput is simulated" claims in ch5/6/9.
New finding the measurement exposed: the **+15-token bonus for latency < 300 ms is
unreachable** on a 2.1 s floor — an incentive-design mismatch the simulated layer hid.

~~**OBJ-10 — stale objective in metrics.py**~~ — `obf2_value` now holds the bounded
`2·MCC + Spec + Pre + NPV`; `adtcn.py` calls it instead of duplicating it. Bit-identical.

~~**OBJ-1 — BankSim entity-linked temporal test**~~ — 2×7 grid, 3 seeds, 5.3 h.
**Neither pre-registered branch was right.** Entity-linked windows help *every*
architecture (+0.0999 to +0.1717 MCC) and the ranking inverts between arms, so the
ULB "temporal complexity does not pay" result does **not** generalise — it was a
statement about entity-less windows. But ADTCN ranks **7 of 7** on linked windows
(0.7089), losing to its own no-attention ablation DTCN (0.7733, **−0.0645**) and to
the plain CNN (**−0.0466**). Verdict: *the window was the problem, not the sequence
model — but ADTCN is still the wrong sequence model.* Best linked: LSTM 0.7805,
DTCN 0.7733. Best global: DenseNet-1D 0.6508. No-window reference 0.6257.
Trap disarmed: BankSim's file order clusters fraud within a step (3,635 adjacent
pairs → 84 after a within-step shuffle); ties are broken by a seeded permutation, so
the global arm sits at P(fraud|prev fraud)=0.0117 vs a 0.0121 base rate.

~~**Purge the fabricated numbers**~~ — 97.38% accuracy, MCC 0.966, 85 TPS, 180 ms, the 8-model comparison table, paired t-tests, the 28/18/4 leader split, the 458/312/178 token balances, the activation ablation. All gone from the `.tex`.

~~**A1 — decide the novelty story**~~ — Option A. DB-BOA Job 3 is dead code; the contribution is the privacy↔incentive characterisation.

~~**Strike the priority claim**~~ — FedCoin (2020) cited instead of "first to bind Shapley to blockchain."

~~**Re-run the federated ablation**~~ — FedAvg 0.569 → +Krum 0.776 → +DP(ε=1) ≈0 → full pipeline ≈0 (collapse, opposite class direction).

~~**Privacy-channel sweep**~~ — ε\* 3000 → 50, ρ −0.288 → **+0.950** at ε=50, over **100 noise draws** per ε with inversion rates.

~~**Krum where the theorem holds**~~ — n=5/f=1 and n=7/f=2, attacker rejected **8/8** across 4 attack types; unprotected FedAvg collapses to 87.46% under scaling.

~~**Economic Byzantine sweep**~~ — 2-of-3 collusion caught (+40.78 always-fraud, +79.52 label-flip); lone attacker and free-rider **not** caught, reported honestly.

~~**RL leader policy**~~ — 5 seeds, Gini 0.90 → 0.782, degraded-node election 100% → 15.2%.

~~**Measure Fabric for real**~~ — live test-network, 2115.9 ms / 40.3 TPS. (Report insertion is OBJ-3.)

~~**Rewrite the abstract**~~ — states its own scope limits.

~~**Sync the local checkout**~~ — commit `ccdd19f`, all Chapter 5 scripts now present locally.

---

## ▮ INTEL — VERIFIED NUMBERS

> Every figure below traces to a JSON in `db_boa_framework/results/`. Nothing here is estimated.

| Metric | Value | Source |
|---|---|---|
| **Detector (headline, 5 seeds)** | **MCC 0.753 ± 0.055** [0.660, 0.792] | `detector_multiseed.json` |
| Hand-set default (5 seeds) | MCC 0.706 ± 0.076 [0.611, 0.796] | `detector_multiseed.json` |
| Tuned − default | +0.047, **p=0.34** (not significant) | `detector_multiseed.json` |
| Deployment run (figures come from it) | Acc 99.85 / MCC 0.677 / TP 83, FP 70, FN 15 | `db_boa_results.json` |
| ⛔⛔ **`db_boa_results.json` and `baselines.json` are INVALID for THREE reasons** *(2026-09-05)* | (1) their 142/76 was chosen by a surrogate that **could not rank candidates**; (2) that surrogate was **validating on memorised rows** — 5 unique ULB fraud transactions duplicated ~6×, **9 of 9 validation positives were copies of training rows**; (3) the **unexplained Aug-30 ULB divergence**. (1) and (2) are fixed; **(3) is not.** Regenerate before quoting, and scope (3) explicitly | `obj13_shipped_path_check.json`, [OBJ-13] |
| **DB-BOA vs hand-set default — the repaired search, 9 runs** *(2026-09-05)* | **0 of 9 searched configurations beat a hand-set 128f/150spe.** Mean paired Δ **−0.1982** on 7 shared held-out draws; best single −0.0720 (4/7 wins). Shipped 142/76 also loses at −0.1900. Holds under **all three** surrogate protocols at **all three** seeds | `obj13_surrogate_repair_banksim.json` |
| **What actually makes the search reproducible** | **Common-random-numbers averaging, NOT determinism.** Filter spread across seeds: `legacy` 66 · `deterministic` **81 (wider)** · `averaged` **12**. Freezing the draw does not converge the search; sharing draws across candidates does | `obj13_surrogate_repair_banksim.json` |
| ⚠ **`averaged` generalises WORST despite converging** | Held-out means: `deterministic` 3.5857 > `legacy` 3.5552 > `averaged` **3.4237**. It converges onto high-filter configs (137/146/149) that overfit the k=3 shared draws. **The noise the repair removed was accidental regularisation** | `obj13_surrogate_repair_banksim.json` |
| ⚠ Repaired `spe` axis is **alive but coarse** | 23 distinct batch sizes across 201 `spe` values; **spe 200–250 is two bins**. Quote the implied **batch size**, never the `spe` optimum. (`deterministic`/42 spe=111 and `averaged`/42 spe=109 both → batch 13 and scored *identically*) | `obj13_surrogate_repair_banksim.json` |
| ⛔ The dead axis was **surrogate-only** | `ADTCN.fit:681` still uses `max(32, len(X_train) // spe)`; at ~400 k rows the floor never binds, so the **final model's `spe` axis was always live**. The search was blind to a dimension the deployed model responds to | `models/adtcn.py:681` |
| Cost of reviving the axis | `legacy` ~2,470 s/search · `deterministic` ~4,580 s · `averaged` (k=3) ~11,800 s. **1.6–5.4× more gradient steps** — a repaired search is not comparable to a pre-repair one at equal wall-clock | `_obj13_logs/sidebyside_*.log` |
| Thread-count sensitivity (tuned) | 0.708 / 0.746 / 0.800 @ 4/2/8 threads, seed 42 | `detector_multiseed.json` |
| Thread-count sensitivity (default) | bitwise identical @ 2 and 8 threads | `detector_multiseed.json` |
| ~~0.313~~ | **withdrawn — not reproducible under any condition tested** | `detector_multiseed.json` |
| Federated ablation | 0.569 → 0.776 → ≈0 → ≈0 | `baselines.json` |
| **ε\* ordering (the claim)** | ⚠ **metric-dependent, corrected 2026-09-04.** On **ρ**: output > weight at **11/11 in all three**. On **inversion rate** (the pre-registered metric): **11/11 · 11/11 · 10/11** — BankSim/entity-disj has a counter-example at **ε=1** (weight 85/100, output **88/100**). The old "9/9 in all 3" was the ρ table; nobody had scored the pre-registered metric. Scope it: at ε=1 both channels fail near-totally | `sweeps_cross_condition.py` |
| **Privacy threshold — ALL THREE NOW MEASURED, not censored** *(11-point grid to ε=30000, 2026-09-04)* | every weight channel reaches **0/100** inside the grid ⇒ no lower bounds left | `private_incentive_sweep*.json` |
| Privacy threshold (ULB) | ε\* **3000** (weight) → **50** (output) = **60×** · *measured* · **7/1000 and 11/1000 draws** | `private_incentive_sweep{,_r1000}.json` |
| Privacy threshold (BankSim/**stratified**) | ε\* **3000** → **100** = **30×** · *measured* · **16/1000 and 10/1000 draws** | `private_incentive_sweep_banksim_stratified{,_r1000}.json` |
| Privacy threshold (BankSim/entity-disjoint) | ε\* **≥30000** (weight) → **300** = **≥100×** · ⛔ **RIGHT-CENSORED AGAIN** · ⚠ **1/1000 and 115/1000 draws** | `private_incentive_sweep_banksim_customer{,_r1000}.json` |
| ⛔⛔ **DO NOT HEADLINE ANY BUDGET FACTOR** | ε\* = max{ε : rate > 0} is a ratio of two **grid-snapped** thresholds. Re-derived at other rate cut-offs it gives ULB **60/20/33×**, BankSim-strat **30/60/20×**, BankSim-cust **≥100/33/33×** for `rate > 0 / 0.01 / 0.05`. **Swings 20× to ≥100×, and the ranking of conditions changes too.** Report "one to two orders of magnitude, definition-sensitive" | [OBJ-17] scorecard |
| ✅ **Quote these instead — both threshold-free** | (1) the **ordering**, 12/12 probed budgets and 11/11 · 11/11 · 10/11 on the grid, *paired within each draw*; (2) the **ρ gap at a stated ε** — at ε=50, **−0.199 → +0.995** (ULB) · **+0.017 → +0.945** (BS/strat) · **−0.070 → +0.805** (BS/cust) | `private_incentive_sweep*.json` |
| ⚠ Both trustworthiness reversals, in one day | 100 draws said *"prefer 33× (entity-disj), not 60× (ULB)"*; 1000 draws reversed it — entity-disj re-censored to ≥100× on **1/1000**, while ULB/strat firmed up. **Neither ordering is a fact about the mechanism**; both were facts about sample size | [OBJ-17] scorecard |
| ✅ The single-draw thresholds were **REAL** | 1/100 → **7, 11, 16, 10 per 1000** at the four fragile (condition, channel) pairs — all ≈ 0.01, none an outlier. Fragility was **precision, not existence**; intervals tightened ~4× | `private_incentive_sweep*_r1000.json` |
| ⚠ ε\* is **monotone non-decreasing** in draws *and* in grid extent | Seeds are `3000+rep`, so more repeats strictly *contain* fewer, and a new budget can only add inversions. Neither knob can lower ε\*. A "de-censored" factor is **not a fixed point** — a 12-point grid could raise it again. Read ε\* as *the largest budget at which an inversion was CAUGHT* | `private_incentive_sweep.py:165-180` → [OBJ-17] |
| Factor is still **condition-dependent** | 60× / 30× / 33× · **not decomposable**; never quote one as universal | `sweeps_cross_condition.py` |
| Weight-channel inversion @ ε=3000 (new run) | ULB **1/100** · BankSim/strat **1/100** · BankSim/entity-disj **38/100** | `private_incentive_sweep*.json` |
| Weight-channel inversion @ ε=10000 / 30000 | ULB 0/100 · 0/100 · BankSim/strat 0/100 · 0/100 · BankSim/entity-disj **13/100** · 0/100 | `private_incentive_sweep*.json` |
| ✅ Draws are **seeded and paired** | `np.random.seed(3000 + rep)` drives both channels in the same iteration ⇒ the **ordering** claim is paired evidence, and it was never the fragile part | `private_incentive_sweep.py:166,174` |
| ⛔ **ULB does not reproduce its 2026-08-30 result** | Re-run at identical seeds/grid/repeats: **13 of 18 shared cells moved**; ground truth `[0.619, 0.250, 0.131]` → `[0.554, 0.336, 0.110]`. **Both BankSim conditions reproduced 9/9 bitwise.** ULB *does* reproduce **itself** same-day (11/11), so it is deterministic under fixed conditions. Cause **not yet identified** — every diff on the ULB path was audited and none of them should affect it | `check_obj17_nesting.py` → [OBJ-17] |
| Rank fidelity @ ε=50 (ULB) | weight −0.20 vs output **+0.99** | `private_incentive_sweep.json` |
| Rank fidelity @ ε=50 (BankSim strat / entity-disj) | weight +0.02 / −0.07 vs output **+0.94 / +0.81** — same ordering | `private_incentive_sweep_banksim_*.json` |
| Krum BFT (rejection) | **8/8 in all three conditions** — replicates, three-condition solid | `byzantine_robustness_sweep*.json` |
| Krum utility vs unprotected FedAvg | ULB **+12.49 pp** (scaled) · BankSim/strat **−11.96 to +0.54** · BankSim/entity-disj **−12.57 to −1.48** — ⚠ **raw ranges; both BankSim lower endpoints are contaminated cells.** Use the clean-cells row below. ⚠ Tracks the **dataset** (negative in all 8 dataset-effect rows); at n=7 the partition effect is **positive** in all 4 rows. ~~Entity-disjointness mechanism~~ **withdrawn** | `sweeps_cross_condition.py` |
| ⚠ FedAvg-under-attack artefact (condition-graded) | **0/8 (ULB) · 2/8 (BankSim/strat) · 5/8 (BankSim/entity-disj)** attacked runs beat their own no-attack ref — metric, not robustness. **Cells mapped 2026-09-04**; marked `(!)` in every draft | `sweeps_cross_condition.py` |
| Krum utility, **clean cells only** | BankSim/strat **−9.15 to +0.54 pp** (6/8 clean) · BankSim/entity-disj **−4.73 to −1.48 pp** (3/8 clean). Quote these, not the raw ranges — every entity-disjoint cell worse than −5 pp is contaminated | `sweeps_cross_condition.py` |
| Clean in **all three** conditions | 3 of 8 cells (n=5 `scaled`, n=7 `sign-flip`, n=7 `scaled`): dataset effect **negative 3/3** (−17.27 to −9.28), partition effect **positive 2/3**. The withdrawal survives the filter | `sweeps_cross_condition.py` |
| ⚠ Krum accuracy attack-invariant? | ULB **yes** · BankSim/strat **no** · BankSim/entity-disj **yes** — unpredicted, unexplained, open | `sweeps_cross_condition.py` |
| Temporal ablation | CNN/rand 0.770 · CNN/time 0.474 · dil+attn/rand 0.714 · dil+attn/time 0.459 | `temporal_pipeline_ablation.json` |
| Fabric latency | mean 2115.9 ms, n=50 | `fabric_consensus_measured.json` |
| Fabric throughput | 40.3 TPS @ c=10, MVCC knee after (50 % fail @ c=20) | `fabric_consensus_measured.json` |
| Base-paper classifiers (ULB) | ResNet 0.5995 > CNN 0.4974 > … > **ADTCN 0.3296 (5/7)** > DTCN 0.2794 | `basepaper_comparison_ulb.json` |
| Exact Shapley cost | **113.5 / 140.3 / 120.6 s** at n=12, same 4,095 coalitions — structural, replicates ×3 | `scalability_sweep*.json` |
| MC speed-up | ≈1× below n≈10 (ULB n=5 is **0.44×**, i.e. slower); 3.27 / 4.13 / 3.24× at n=12 | `scalability_sweep*.json` |
| MC-Shapley top-1 | **intermittent, not a threshold**: **5/10 · 5/10 · 3/10**, no consistent per-`n` pattern (BankSim/strat fails n=5, recovers n=11 **and** n=12). “fails from n=6” is **withdrawn** | `scalability_sweep*.json` |
| MC-Shapley ρ @ n=12 (rank fidelity) | **−0.014 / +0.769 / +0.448** — dataset effect **+0.783** vs partition −0.322; mostly a property of the dataset | `sweeps_cross_condition.py` |
| MC-Shapley L1 @ n=12 (split fidelity) | 0.2572 / 0.2553 / 0.2262 — barely moves. **ρ and L1 behave completely differently; name the metric (rule 11)** | `scalability_sweep*.json` |
| Economic isolation (lone attackers) | fires **0/3 (ULB) · 3/3 · 3/3**, helps **0/3 in all three** — tracks the **dataset**; the negative is three-condition solid | `sweeps_cross_condition.py` |
| 2-of-3 collusion caught | **+40.78 / +44.39 / +47.40** (always-fraud) · **+79.52 / +86.95 / +89.73** (label-flip) — replicates ×3, slightly larger on BankSim; **do not attribute the increase** | `economic_byzantine_sweep*.json` |
| Dataset (1) | ULB 284,807 tx / 492 fraud / 0.17% / no customer IDs | `creditcard.csv` |
| Dataset (2) | BankSim 594,643 tx / 7,200 fraud / 1.211% / 4,112 customers / 180 days | `datasets/bs140513_032310.csv` |
| Entity-linkage gain (BankSim) | +0.0999 to +0.1717 MCC, **every** architecture | `banksim_temporal_grid.json` |
| Best on customer-linked windows | LSTM 0.7805 · DTCN 0.7733 · ADTCN **0.7089 (last of 7)** | `banksim_temporal_grid.json` |
| Best on global windows | DenseNet-1D 0.6508 · CNN 0.6435 · ADTCN 0.5892 (last) | `banksim_temporal_grid.json` |
| ADTCN − DTCN, BankSim linked windows | **−0.0645** (attention costs here) | `banksim_temporal_grid.json` |
| ADTCN − DTCN, ULB global windows | **+0.0502** (attention pays here — sign flips) | `basepaper_comparison_ulb.json` |
| No-window reference (BankSim) | MCC 0.6257 | `banksim_temporal_grid.json` |
| BankSim conditional fraud | global 0.0117 vs customer-linked 0.3747, base 0.0121 | `banksim_temporal_grid.json` |
| Base-paper optimisers (ULB) | **all 5 at Obf2 = 5.0000**; test MCC spread 0.057 vs std 0.080 | `basepaper_optimisers_ulb.json` |
| Base-paper optimisers (BankSim) | best DB-BOA mean Obf2 4.2123; MCC spread 0.035 vs std 0.116 | `basepaper_optimisers_banksim.json` |
| Surrogate fraud rows (**both** datasets) | 30 (the `_MIN_FRAUD_ROWS` floor fires on each) → ~9 in validation | `objective_noise_audit.json` |
| Objective noise:signal (`noise_to_signal`) | ULB **0.74** · BankSim **0.94** — uninformative on both | `objective_noise_audit.json` |
| Objective between-seed ÷ config range | ULB 0.97 · BankSim 0.16 — *different question, do not conflate* | `objective_noise_audit.json` |
| DP collapse direction | ULB all-negative (acc −0.11 pp) · BankSim all-positive (acc −97.55 pp) | `federated_cross_dataset.json` |
| DP replication (MCC after DP) | ULB 0.000 · BankSim strat 0.018 · BankSim entity-disjoint −0.000 | `federated_cross_dataset.json` |

---

## ▮ FIELD MAP

```
TASK.md                                    <- you are here
SUPERVISOR_BRIEF.md                        one-read walkthrough
final_report_data/
  00_ground_truth_implementation.md        what the system ACTUALLY is
  00_report_vs_code_divergences.md         the D1-D17 audit
  07_actions_checklist.md                  older checklist (superseded by this file)
  B2_fabric_consensus_measured.md          OBJ-3 payload (CONSUMED -> ch6)
  OBJ2_detector_multiseed.md               OBJ-2/OBJ-4 verdict
  BASEPAPER_comparison_ulb.md              OBJ-1b ULB classifier track
  OBJECTIVE_noise_audit.md                 OBJ-13 draft. ⚠ its "n~9 positives" MECHANISM was
                                           WITHDRAWN 2026-09-05 (the numbers stand; the
                                           interpretation does not). Regenerate in seconds:
                                             python experiments/objective_noise_audit.py --render-only
  OBJECTIVE_size_knee.md                   OBJ-13 surrogate-size sweep. Answers "measure the
                                           knee" with "a knee is NOT locatable at one seed".
                                             ... objective_noise_audit.py --knee-redraft
  OBJ13_surrogate_repair_banksim.md        OBJ-13 HEADLINE — legacy vs deterministic vs averaged,
                                           9 searches, 16.3 h. 0 of 9 beat the hand-set default.
                                           Paired on 7 shared draws; read the paired column, not
                                           the means +/- spreads.
                                             python experiments/obj13_surrogate_repair.py \
                                                    --dataset banksim --redraft
  FEDERATED_cross_dataset.md               DP replication across both datasets (Secure/privacy)
  OBJ15_two_factor_decomposition.md        dataset effect vs partition effect, all four sweeps.
                                           Generated; re-run the script, never hand-edit.
  01/04/05/06_*.md, REWRITE_*.md           PRE-OBJ-2, now STAMPED `> **SUPERSEDED**` (OBJ-14).
                                           Left unedited on purpose — historical record.
SUPERVISOR_BRIEF.md / .tex                 ⚠ was asserting the retracted 0.785 as a LIVE claim
                                           until 2026-09-01; fixed. Re-check it after any purge —
                                           it is the doc a supervisor actually reads.  OBJ-18
                                           (2026-09-04) also fixed "~60x" -> ">=60x" and split
                                           Krum's rejection from Krum's utility in both files.
                                           ✅ The PDF builds again: MiKTeX was missing scalable
                                           Type1 CM fonts, so microtype's auto-expansion aborted
                                           and SUPERVISOR_BRIEF.pdf had been stuck at 8/30 while
                                           the .tex moved on.  `mpm --install=cm-super` fixed it
                                           with no typography change.  REBUILD AFTER EVERY EDIT:
                                             pdflatex -interaction=nonstopmode SUPERVISOR_BRIEF.tex
                                           (twice, for the tcolorbox/hyperref pass)
finalreport_checklist.md                   PRE-OBJ-2 tracker, stamped (OBJ-14).
title_issue.md                             OBJ-9 evidence
results/_ulb_figures_backup/               ULB figures saved before the BankSim run
results/*_banksim_customer.{json,png,md}   OBJ-15 BankSim/entity-disjoint outputs
db_boa_framework/
  config.py                                every real parameter; DATASETS + get_loader()
  data/data_loader.py                      ULB loader (unchanged behaviour)
  data/banksim_loader.py                   BankSim loader — customer IDs, entity-disjoint split
  models/adtcn.py                          build_sequences(groups=...) + model factory
  models/basepaper_models.py               EfficientNet/ResNet/DenseNet/DTCN/LSTM, 1-D
  models/federation_manager.py             DP -> Krum -> Shapley
  algorithms/{mbo,wsa,boa,dboa,db_boa}.py  the five optimisers compared
  models/adtcn.py.pre_obj13                ⛔ DO NOT DELETE, DO NOT GITIGNORE. adtcn.py had
  config.py.pre_obj13                      UNCOMMITTED edits when OBJ-13 was applied, so these
                                           are the ONLY copy of the pre-patch state -- git
                                           history does not have it. verify_obj13_legacy.py
                                           loads them to prove `legacy` is bit-for-bit.
  scratchpad/                              TRACKED, deliberately not in .gitignore.
    obj13_new_objective.py                 the staged OBJ-13 objective (the record of a decision)
    apply_obj13.py                         idempotent applier; --check writes nothing
    verify_obj13_legacy.py                 proves legacy == pre-patch, and that the other five
                                           claims of the patch hold. ~1 min, run after any
                                           change to _ADTCNObjective.
  experiments/                             one script per Chapter 5 result
    objective_noise_audit.py               OBJ-13 diagnosis. ⚠ PINS eval_mode="legacy" ON
                                           PURPOSE -- on the repo default it would report a
                                           within-seed std of 0.0000 and read as "the objective
                                           was always fine". Do not "simplify" that away.
                                           Modes: (default) | --knee | --components
                                                | --render-only | --knee-redraft
    obj13_shipped_path_check.py            ⛔ the check that found the memorised-rows leak.
                                           Zero training, ~2 min. Replays the fraud-selection
                                           arithmetic on the pool EACH CALLER ACTUALLY SUPPLIES
                                           -- the audit's 6,000 vs the shipped 3,000 were never
                                           the same experiment.
    obj13_surrogate_repair.py              the 3-mode side-by-side + held-out yardstick.
                                           Writes _multiseed_runs/extra_configs.json, which
                                           detector_multiseed.py picks up with no code edit.
                                           --scout first: it prices the run in ~10 min.
    run_obj13_sidebyside.ps1               detached runner (Start-Process -WindowStyle Hidden).
                                           Survives the session; preflights the patch AND the
                                           pool fix before spending 16 h.
  experiments/run_obj15_banksim.ps1        OBJ-15 runner (USE THIS, not the .sh)
                                             -Partition customer|stratified
  experiments/check_obj15.ps1              OBJ-15 progress checker (read-only, run any time)
                                             -Partition customer|stratified [-Tail N]
  results/_obj15_logs/banksim_<part>/      per-RUN logs + runner PID/start time.  Split per run
                                             on 2026-09-01: a second partition would otherwise
                                             overwrite the first run's evidence.  UTF-16LE -
                                             iconv before grepping.
  experiments/sweeps_cross_condition.py    dataset-vs-partition decomposition of the four
                                             sweeps across ULB / BankSim-strat / BankSim-cust.
                                             Since 2026-09-04 it also marks contaminated cells
                                             `(!)` and prints the CLEAN SUBSET table — the only
                                             cells any Krum-FedAvg effect may be read from.
                                             Reads JSON only, seconds, safe mid-run:
                                               python experiments/sweeps_cross_condition.py
                                             -> final_report_data/OBJ15_two_factor_decomposition.md
  experiments/banksim_temporal_grid.py     OBJ-1 grid
  experiments/basepaper_comparison.py      OBJ-1b base-paper classifiers + optimisers
  experiments/detector_multiseed.py        OBJ-2/OBJ-4 seed sweep + determinism probe
  results/_multiseed_runs/                 one JSON per run (restartable sweep)
  results/*.json                           the only source of truth for numbers

Both pipelines take --dataset:
  python main.py --dataset banksim [--partition customer]
  python run_baselines.py --dataset banksim [--partition customer]
db_boa_fabric/chaincode/lib/               DBBOAContract
FINAL YEAR THESIS REPORT/                  the LaTeX
```

---

```
▰ END OF BRIEFING ▰
To close an objective: wrap its heading in ~~strikethrough~~
and move it down to CONFIRMED KILLS.
```
