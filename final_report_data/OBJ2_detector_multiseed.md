# OBJ-2 / OBJ-4 — The MCC contradiction, and what actually reproduces

_Produced by `experiments/detector_multiseed.py`; every number below traces to
`results/detector_multiseed.json` (13 full 30-epoch runs) and its per-run files
in `results/_multiseed_runs/`. Dataset: ULB, fixed stratified split
(`random_state=42`, test n=56,962 / 98 fraud) — the same split both contested
files used._

---

## The contradiction

Two files reported the same configuration on the same split with irreconcilable
numbers, and one of them claimed to cross-check the other:

| File | Config | MCC | Accuracy | FP |
|---|---|---|---|---|
| `db_boa_results.json` | 142 / 76 / 30 | 0.677 | 99.85 % | 70 |
| `dbboa_vs_default.json` | 142 / 76 / 30 | **0.313** | 98.77 % | 688 |
| `dbboa_vs_default.json` (default) | 128 / 150 / 30 | 0.785 | 99.92 % | 31 |

`dbboa_vs_default.json`'s note read *"Tuned row cross-checks db_boa_results.json
(MCC 0.677)"*. It did not.

## Finding 1 — training is deterministic; the thread count is not held fixed

Two runs of the tuned config at 4 threads produced **bitwise identical weights**
(`6272587cd34429db` twice). A run at 8 threads produced different weights
(`2e7d2759f8b463a1`). So nothing here is stochastic: the pipeline is exactly
reproducible for a fixed *(seed, torch version, thread count)* triple.

The two scripts never shared that triple. `main.py` leaves torch at its default
thread count (4 on this machine); `experiments/dbboa_vs_default.py` calls
`torch.set_num_threads(os.cpu_count())` (8). Different thread counts partition
the convolution reductions differently, so the two runs were doing genuinely
different floating-point arithmetic from the first batch onward.

## Finding 2 — only the tuned config is sensitive to it

Same seed (42), same data, thread count varied:

| Config | batch | 2 threads | 4 threads | 8 threads | weights |
|---|---|---|---|---|---|
| DB-BOA-tuned (142/76) | 2997 | 0.7465 | 0.7080 | 0.8004 | three different hashes |
| Hand-set default (128/150) | 1518 | 0.646116 | — | 0.646116 | **identical hash** |

The hand-set default is bitwise thread-count-invariant. The tuned configuration
moves by **0.092 MCC** across thread counts with everything else held fixed.

This confirms the mechanism the report had hypothesised — the tuned config's low
steps-per-epoch implies a very large batch, and large batches are where the
reduction order changes — but corrects its description. The instability is *not*
"run-to-run variance"; each run is perfectly repeatable. It is sensitivity to
the floating-point reduction order, which is an environment property, not a
random one.

## Finding 3 — the seed sweep: the two configs are indistinguishable

Five seeds per configuration (42–46), split held fixed, threads pinned to 2, so
the only thing varying is `torch.manual_seed` (weight init and batch order):

| Seed | DB-BOA-tuned | Hand-set default |
|---|---|---|
| 42 | 0.7465 | 0.6461 |
| 43 | 0.6598 | 0.7320 |
| 44 | 0.7780 | 0.7435 |
| 45 | 0.7921 | 0.6106 |
| 46 | 0.7894 | 0.7959 |
| **mean ± std** | **0.7531 ± 0.0553** | **0.7056 ± 0.0756** |
| range | [0.6598, 0.7921] | [0.6106, 0.7959] |

Paired difference (tuned − default) = **+0.0475 ± 0.0977**.

| Test | Statistic | p |
|---|---|---|
| Paired t | t = +1.088 | 0.338 |
| Welch t | t = +1.135 | 0.292 |
| Wilcoxon signed-rank | W = 4.0 | 0.438 |

**The two configurations are statistically indistinguishable at n=5.** Note also
that the hand-set default is the *noisier* of the two across seeds (std 0.076 vs
0.055) — it is stable against thread count, not against seed.

## Finding 4 — none of the three historical point estimates regenerates

Re-running seed 42 at each historical script's own thread count:

| Target | Claimed | Observed | Verdict |
|---|---|---|---|
| `db_boa_results.json` tuned @ 4t | 0.6772 | 0.7080 | did not reproduce |
| `dbboa_vs_default.json` tuned @ 8t | 0.3133 | **0.8004** | did not reproduce |
| `dbboa_vs_default.json` default @ 8t | 0.7849 | 0.6461 | did not reproduce |

The ULB code path was checked and cleared as the cause: the BankSim refactor's
`build_sequences(groups=None)` is byte-identical to the previous windowing code
across every length tested, and the `data_loader` changes are additive. The
remaining free variable is the environment — `requirements.txt` listed torch as
*optional* and unpinned, so the June runs' torch version is unrecoverable. It is
now pinned at `torch==2.12.0`.

**0.313 is the one number that must be withdrawn.** Across 7 runs of the tuned
configuration (5 seeds × 2 threads, plus 4- and 8-thread reproductions) the
lowest MCC observed anywhere is **0.6598**. 0.313 is not a tail of this
distribution; it is not reproducible under any condition tested. By contrast
0.677 sits comfortably inside the observed range [0.6598, 0.8004] and is a
plausible single draw.

---

## What ships

- **Headline detector performance: MCC 0.753 ± 0.055** (DB-BOA-tuned, 5 seeds),
  against **0.706 ± 0.076** for the hand-set default.
- The single full-pipeline run at MCC 0.677 stays as the *deployment operating
  point* — it is the run the confusion matrix and ROC figures were generated
  from — and is now labelled as one draw from the distribution above, not as a
  point estimate of the configuration's quality.
- **Retire the 0.47 gap.** Both the "default beats tuned by ≈0.11" and the
  "≈0.47" claims were single-seed artefacts, and the larger one rests on an
  unreproducible run.
- **The negative result survives in its defensible form.** DB-BOA still buys no
  demonstrated detection-accuracy gain: +0.047 MCC at p=0.34 is not a gain. Its
  contribution remains automation of hyperparameter and leader selection. What
  changes is that it no longer *loses* either — the earlier claim that it did
  was measuring seed and arithmetic noise, not the optimiser.
- Any future detector claim needs ≥3 seeds and a pinned thread count. A single
  run of the tuned config carries roughly ±0.06 MCC of seed noise and a further
  ±0.05 of thread-count noise.
