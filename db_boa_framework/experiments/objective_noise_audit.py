"""
experiments/objective_noise_audit.py
====================================
Is the DB-BOA fitness function actually measuring anything?

Why this exists
---------------
Running the base-paper optimiser comparison (`basepaper_comparison.py`) on ULB
produced a result that cannot be taken at face value: **all five optimisers —
MBO, WSA, DBOA, BOA and DB-BOA — reported a best Obf2 of exactly 5.0000**, which
is the theoretical maximum of

    Obf2 = 2*MCC + Specificity + Precision + NPV      (bounded form)

attainable only by perfectly classifying the surrogate's validation split.  Five
structurally different search algorithms landing on the identical ceiling value
is not a result about the algorithms.  This script finds out what it is instead.

What it measures
----------------
1.  **Objective noise.**  `_ADTCNObjective.__call__` redraws its 70/30
    train/validation split and its torch seed on *every* call, so the same
    configuration does not score the same twice.  Evaluate one fixed
    configuration N times and report the spread.

2.  **Signal vs noise.**  Evaluate several *different* configurations once each,
    and compare that config-to-config range against the single-config noise from
    (1).  If the noise is as large as the range, the search cannot rank.

3.  **Seed sensitivity.**  Rebuild the objective under different
    `random_state` values — which changes only *which normal rows* are
    subsampled — and compare the resulting distributions.  Large shifts here
    mean the objective's difficulty is set by the sample, not by the candidate.

4.  **The steps_per_epoch axis.**  Pure arithmetic, no training: the surrogate
    set `batch_size = max(32, n_train // spe)` with `n_train = 1400`, so every
    `spe` in the searched range [50, 250] yielded batch 32.  If so, the
    advertised 2-D search is 1-D and one whole axis is dead.  Reported next to
    the repaired formula `ceil(n_train / spe)`, so the fix is visible as a
    measurement rather than asserted in prose.

5.  **Surrogate size (`--knee`).**  How many rows does the surrogate need
    before one draw can rank two candidates?  OBJ-13 decided to measure this
    rather than pick a number; the curve, not just the chosen value, is what
    makes the choice defensible.  Falls out of the same `noise_to_signal`
    ratio, plus a free k-averaging column (std of a k-mean is std/sqrt(k)).

Why it matters beyond the optimiser table
------------------------------------------
OBJ-4 measured DB-BOA's tuned configuration against the hand-set default over
five seeds and found them statistically indistinguishable: **+0.047 MCC,
p=0.34**.  This audit supplies the *mechanism* for that tie — a search whose
fitness cannot rank its candidates returns an essentially arbitrary one, so
tying a sensible hand-set choice is the expected outcome, not a surprise.

That strengthens the negative result rather than softening it.

⚠ Two numbers that earlier versions of this script quoted are **withdrawn** by
OBJ-2 and must not come back (rule 2 — no number ships that a script did not
produce):

  * **MCC 0.313** — not reproducible under any condition tested; the lowest
    tuned MCC observed in 7 runs is 0.6598.
  * **"DB-BOA loses to the default (0.785)"** — replaced by +0.047, p=0.34.
    There is no measured loss *and* no measured gain.  Quoting a loss is a
    fabricated claim.

⚠ **This script measures the PRE-repair objective on purpose.**  OBJ-13's repair
landed in `models/adtcn.py` on 2026-09-04 and made `deterministic` the default
everywhere.  Every objective built here pins `eval_mode="legacy"` explicitly, so
the audit keeps measuring the behaviour that is the finding.  Left on the repo
default it would report a within-seed std of exactly 0.0 — not because the noise
was ever absent, but because `deterministic` scores every call on one frozen
draw.  Do not "simplify" those explicit `eval_mode=` arguments away.

Usage
-----
    python experiments/objective_noise_audit.py
    python experiments/objective_noise_audit.py --draws 40 --dataset ulb
    python experiments/objective_noise_audit.py --render-only    # prose only
    python experiments/objective_noise_audit.py --knee           # OBJ-13 size sweep
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import RESULTS_DIR
from models.adtcn import _ADTCNObjective
from experiments.basepaper_comparison import (prepare_ulb, prepare_banksim,
                                              _eval_subset)

#: one fixed configuration, re-evaluated to expose pure noise
FIXED_CFG = np.array([32.0, 150.0])

#: distinct configurations spanning the search box, one evaluation each
SWEEP_CFGS = [(8, 50), (16, 100), (24, 150), (32, 80),
              (40, 200), (48, 120), (56, 60), (64, 250)]

CEILING = 5.0   # 2*1 + 1 + 1 + 1, perfect classification


def _objective(d, seed, arch="dilated_attn", eval_mode="legacy", rows=None,
               pool=6000):
    """
    Build the objective for the audit.

    ⚠ `eval_mode` is passed **explicitly and defaults to "legacy"**, which is
    not the repo default.  OBJ-13's repair landed in `models/adtcn.py` and made
    `deterministic` the default everywhere; this script must keep measuring the
    *pre-repair* behaviour, because that behaviour is the finding.  Left on the
    repo default it would report a within-seed std of exactly 0.0 — not because
    the noise was ever absent, but because `deterministic` hides it by scoring
    every call on one frozen draw.  A 0.0 here would read as "the objective was
    always fine", which is the opposite of what happened.
    """
    # ⚠ `pool` is the size of the candidate pool the surrogate subsamples FROM,
    # and it has to exceed the surrogate size or the sweep lies.  `_eval_subset`
    # defaults to 6,000 rows, so a `--knee` point asking for 10,000 or 20,000
    # would silently receive ~6,000 (`n_n = min(len(normal_idx), ...)` caps it),
    # the curve would go flat above 6k, and that flat stretch would read as
    # "the knee is at 5k".  The pool is recorded in the JSON so the reader can
    # check it was never the binding constraint.
    X_opt, y_opt = _eval_subset(d, n=max(int(pool), int(rows or 0) * 2))
    return _ADTCNObjective(X_opt, y_opt, random_state=seed,
                           architecture=arch, n_raw=int(d["X"].shape[1]),
                           eval_mode=eval_mode, surrogate_rows=rows)


def _repaired_bs(n_train, spe):
    """
    The repaired batch size — pure arithmetic, no state.

    Computed at render time rather than read out of the JSON so that
    `--render-only` produces the right table against *any* results file,
    including the ones written before OBJ-13 landed. A draft that silently
    degrades to em-dashes against an older JSON is the same stale-draft failure
    `render_only()` was written to prevent.
    """
    return int(max(1, math.ceil(int(n_train) / max(1, int(spe)))))


def spe_axis_is_dead(n_rows=2000, lo=50, hi=250):
    """
    Pure arithmetic — no training.  Returns the batch size the surrogate would
    use for each `steps_per_epoch` in the searched range, under both the
    pre-repair formula and the repaired one.

    The pre-repair formula was `max(32, n_train // spe)`.  With n_train = 1400
    the right-hand term runs 28 down to 5 across spe ∈ [50, 250] — every value
    below the floor — so the *floor* was the bug, not the arithmetic.  OBJ-13
    replaced it with `ceil(n_train / spe)`, which is what `steps_per_epoch` has
    always claimed to mean.  Both are reported so the repair is visible as a
    measurement rather than asserted in prose.
    """
    n_train = int(0.7 * n_rows)
    sizes = {spe: max(32, n_train // spe) for spe in range(lo, hi + 1, 10)}
    fixed = {spe: int(max(1, math.ceil(n_train / spe))) for spe in range(lo, hi + 1, 10)}
    return {"n_train": n_train, "distinct_batch_sizes": sorted(set(sizes.values())),
            "is_dead_axis": len(set(sizes.values())) == 1,
            "sample": {k: sizes[k] for k in (lo, 100, 150, 200, hi)},
            "repaired_distinct_batch_sizes": sorted(set(fixed.values())),
            "repaired_is_dead_axis": len(set(fixed.values())) == 1,
            "repaired_sample": {k: fixed[k] for k in (lo, 100, 150, 200, hi)}}


def audit(dataset, draws, seeds, arch="dilated_attn", verbose=True,
          eval_mode="legacy", rows=None, d=None, pool=6000):
    prep = prepare_ulb if dataset == "ulb" else prepare_banksim
    kw = {"ordering": "time"} if dataset == "ulb" else {"ordering": "customer"}
    if d is None:
        d = prep(verbose=False, **kw)

    out = {"dataset": dataset, "architecture": arch, "draws": draws,
           "seeds": seeds, "ceiling": CEILING,
           # Stamped so a JSON can never be mistaken for one produced under a
           # different surrogate protocol — the mistake OBJ-15 made with
           # unsuffixed figure filenames, in a different costume.
           "eval_mode": eval_mode,
           "surrogate_rows": rows or _ADTCNObjective._SURROGATE_ROWS}

    # 1 + 3: fixed config, repeated, under each objective seed
    per_seed = {}
    for sd in seeds:
        obj = _objective(d, sd, arch, eval_mode=eval_mode, rows=rows, pool=pool)
        v = np.array([-obj(FIXED_CFG) for _ in range(draws)])
        per_seed[str(sd)] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "ceiling_hits": int((np.abs(v - CEILING) < 1e-9).sum()),
            "surrogate_fraud_rows": int(obj.y.sum()),
            # ACHIEVED size, not requested.  The two diverge silently when the
            # pool is too small, and the whole knee sweep depends on them being
            # equal — so record the achieved number and let the draft compare.
            "surrogate_rows_achieved": int(len(obj.y)),
            # Raw per-draw values, kept so the null test below can be run (and
            # re-run) at render time without retraining anything.  25 floats.
            "draws_raw": [float(x) for x in v],
        }
        if verbose:
            s = per_seed[str(sd)]
            print(f"  [{dataset}] fixed config x{draws}, objective seed {sd:>3}: "
                  f"mean={s['mean']:.4f} std={s['std']:.4f} "
                  f"range=[{s['min']:.4f}, {s['max']:.4f}]", flush=True)
    out["fixed_config_by_seed"] = per_seed

    means = [s["mean"] for s in per_seed.values()]
    out["between_seed_mean_spread"] = float(max(means) - min(means))
    out["within_seed_noise_std"] = float(np.mean([s["std"] for s in per_seed.values()]))

    # 2: different configs, one draw each, first seed
    obj = _objective(d, seeds[0], arch, eval_mode=eval_mode, rows=rows, pool=pool)
    sweep = [float(-obj(np.array(c, dtype=float))) for c in SWEEP_CFGS]
    out["config_sweep"] = {"configs": [list(c) for c in SWEEP_CFGS],
                           "obf2": sweep,
                           "range": float(np.ptp(sweep))}
    if verbose:
        print(f"  [{dataset}] {len(SWEEP_CFGS)} distinct configs, 1 draw each: "
              f"range={out['config_sweep']['range']:.4f}", flush=True)

    # the decisive ratio
    out["noise_to_signal"] = (out["within_seed_noise_std"] * 2
                              / max(out["config_sweep"]["range"], 1e-9))

    # ── the null test the ratio alone cannot give you ────────────────────────
    #
    # `noise_to_signal` has a denominator — the config-to-config RANGE — that is
    # itself measured from 8 *single* noisy evaluations.  The range of 8 draws of
    # one fixed configuration is already large, so a chunk of that denominator is
    # noise being compared against itself, and the ratio flatters the objective.
    #
    # The fix costs nothing: the fixed-config draws collected above ARE samples
    # under the null "configuration makes no difference".  Resample 8 of them and
    # take the range, many times, and that is the distribution of the range one
    # would see from 8 configurations that are genuinely identical.  If the
    # observed config range sits inside that distribution, the eight
    # configurations are statistically indistinguishable from eight redraws of
    # ONE configuration — which is a far stronger and far more honest statement
    # than "the ratio is below 1".
    pooled = [x for s in per_seed.values() for x in s["draws_raw"]]
    out["null_range_test"] = _null_range_test(pooled, len(SWEEP_CFGS),
                                              out["config_sweep"]["range"])
    return out


def _null_range_test(pooled, n_cfgs, observed_range, n_boot=20000, seed=0):
    """
    Bootstrap the range of `n_cfgs` draws of a SINGLE configuration.

    Returns the null distribution's summary plus the percentile of the observed
    config-to-config range within it.  A high percentile means the configurations
    really do spread more than noise alone; a middling one means they do not.

    Non-parametric on purpose: Obf2 is bounded above by 5 and its draws pile up
    against that ceiling, so it is visibly not normal and a normal
    approximation would misstate the tail that this test lives in.
    """
    a = np.asarray(pooled, dtype=float)
    if a.size < 2:
        return None
    rs = np.random.RandomState(seed)
    draws = rs.choice(a, size=(int(n_boot), int(n_cfgs)), replace=True)
    null = draws.max(axis=1) - draws.min(axis=1)
    pct = float((null < observed_range).mean() * 100.0)
    return {
        "n_pooled_draws": int(a.size),
        "n_configs": int(n_cfgs),
        "observed_config_range": float(observed_range),
        "null_range_mean": float(null.mean()),
        "null_range_p50": float(np.percentile(null, 50)),
        "null_range_p95": float(np.percentile(null, 95)),
        "observed_percentile_in_null": pct,
        # The ratio's value when configurations genuinely make NO difference.
        # This is the number `noise_to_signal` should be compared against —
        # NOT 1.0.
        "null_noise_to_signal": float(2 * a.std() / max(null.mean(), 1e-9)),
        "configs_distinguishable_from_noise": bool(pct >= 95.0),
    }


#: surrogate sizes for the knee sweep (OBJ-13: "measure the knee, do not guess it")
KNEE_ROWS = (2_000, 5_000, 10_000, 20_000)


def knee(datasets=("ulb", "banksim"), draws=12, seed=42, rows=KNEE_ROWS,
         arch="dilated_attn", seeds=None):
    """
    How large does the surrogate have to be before one draw can rank two
    candidates?  OBJ-13 chose to *measure* this rather than pick a number.

    Measured in `legacy` mode, and that is a deliberate choice, not an
    oversight.  `deterministic` reports a within-seed std of 0 by construction —
    it freezes one draw — but freezing an arbitrary draw does not make it a good
    draw.  The draw-to-draw spread is exactly what says how arbitrary the frozen
    one is, so the diagnostic has to be taken on the unfrozen objective.  It
    then answers both open questions at once:

      * which `surrogate_rows` to set, and
      * how many draws `averaged` needs — the mean of k draws has std/sqrt(k),
        so the k column below is free once the std is measured.

    Wall-clock per evaluation is recorded alongside, because a knee is a
    cost/benefit curve and quoting only the benefit half is how OBJ-13's first
    draft ended up quoting only the flattering noise ratio.
    """
    # The pool has to strictly exceed the largest surrogate size, or the top of
    # the grid silently caps and the curve goes flat for the wrong reason.
    pool = max(rows) * 2
    # ⚠ More than one objective seed per size is not a luxury here.  The noise
    # level is itself a property of the draw: the same nominal configuration at
    # the same size measured a within-seed std of 0.4439 in the main audit (pool
    # 6,000) and 0.2563 in the first knee run (pool 40,000) on BankSim — a 1.7x
    # swing from the subsample alone.  With one seed per size, a difference
    # between two sizes cannot be told apart from that swing, and the "knee"
    # would be an artefact of which rows happened to be drawn.
    seeds = list(seeds) if seeds else [seed]
    out = {"eval_mode": "legacy", "draws": draws, "seed": seeds[0],
           "seeds": seeds,
           "rows_grid": list(rows), "architecture": arch, "pool_rows": pool,
           "datasets": {}}
    for ds in datasets:
        prep = prepare_ulb if ds == "ulb" else prepare_banksim
        kw = {"ordering": "time"} if ds == "ulb" else {"ordering": "customer"}
        d = prep(verbose=False, **kw)      # load once, reuse across sizes
        cells = []
        for n_rows, sd in ((n, s) for n in rows for s in seeds):
            t0 = time.time()
            r = audit(ds, draws, [sd], arch=arch, verbose=False,
                      eval_mode="legacy", rows=n_rows, d=d, pool=pool)
            dt = time.time() - t0
            n_evals = draws + len(SWEEP_CFGS)
            s = r["fixed_config_by_seed"][str(sd)]
            rng = r["config_sweep"]["range"]
            cell = {
                "surrogate_rows": n_rows,
                "objective_seed": sd,
                "surrogate_rows_achieved": s["surrogate_rows_achieved"],
                "pool_capped": s["surrogate_rows_achieved"] < n_rows,
                "surrogate_fraud_rows": s["surrogate_fraud_rows"],
                "null_range_test": r.get("null_range_test"),
                "mean_obf2": s["mean"], "std_obf2": s["std"],
                "min_obf2": s["min"], "max_obf2": s["max"],
                "ceiling_hits": s["ceiling_hits"],
                "config_range": rng,
                "noise_to_signal": r["noise_to_signal"],
                # mean of k draws has std/sqrt(k); free, no extra training
                "noise_to_signal_if_averaged": {
                    str(k): float(2 * s["std"] / math.sqrt(k) / max(rng, 1e-9))
                    for k in (1, 3, 5, 10)},
                "seconds_total": dt,
                "seconds_per_eval": dt / n_evals,
            }
            cells.append(cell)
            print(f"  [{ds}] rows={n_rows:>6} sd={sd:>3}"
                  f"{'  ⚠CAPPED@' + str(cell['surrogate_rows_achieved']) if cell['pool_capped'] else ''}"
                  f"  fraud={cell['surrogate_fraud_rows']:>4}  "
                  f"std={s['std']:.4f}  range={rng:.4f}  "
                  f"n2s={cell['noise_to_signal']:.2f}  "
                  f"ceil={s['ceiling_hits']}/{draws}  "
                  f"{cell['seconds_per_eval']:.1f} s/eval", flush=True)
        out["datasets"][ds] = cells

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "objective_size_knee.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {path}", flush=True)
    write_knee_draft(out)
    return out


#: Obf2 = 2*MCC + Spec + Pre + NPV — the weight each component carries into it.
OBF2_TERMS = {"MCC": 2.0, "Specificity": 0.01, "Precision": 0.01, "NPV": 0.01}


def components(datasets=("ulb",), rows=(2_000, 20_000), draws=25, seed=42,
               arch="dilated_attn"):
    """
    Where in Obf2 does the noise actually live?

    Only reached for if the knee sweep falsifies pre-registration 6 — i.e. if
    `noise_to_signal` improves with surrogate size *while the fraud count stays
    pinned at the `_MIN_FRAUD_ROWS` floor*.  That would mean the "Obf2 = 5.0000
    is perfectly classifying about nine validation fraud rows" mechanism written
    into OBJ-13 and into `OBJECTIVE_noise_audit.md` cannot be the whole story,
    and leaving that prose standing while the evidence contradicts it is not an
    option.

    `Obf2 = 2*MCC + Spec + Pre + NPV` splits cleanly along the class imbalance:

      * **MCC and Precision** are governed by the ~9 validation fraud rows — the
        scarce quantity, and the one the fraud floor pins.
      * **Specificity and NPV** are governed by the validation *normals*, which
        grow with the surrogate whether or not the fraud count does.

    So if the noise falls with size while fraud stays fixed, the fall has to be
    concentrated in Spec and NPV, and the prediction is checkable term by term.
    Each component's contribution is reported as `weight x std`, in Obf2 units,
    so the four numbers are directly comparable and sum to something meaningful.
    """
    out = {"eval_mode": "legacy", "draws": draws, "seed": seed,
           "rows_grid": list(rows), "architecture": arch,
           "terms": {k: v for k, v in OBF2_TERMS.items()}, "datasets": {}}
    pool = max(rows) * 2
    for ds in datasets:
        prep = prepare_ulb if ds == "ulb" else prepare_banksim
        kw = {"ordering": "time"} if ds == "ulb" else {"ordering": "customer"}
        d = prep(verbose=False, **kw)
        cells = []
        for n_rows in rows:
            obj = _objective(d, seed, arch, eval_mode="legacy", rows=n_rows,
                             pool=pool)
            per_term = {k: [] for k in OBF2_TERMS}
            obf2s = []
            for _ in range(draws):
                obf2s.append(-obj(FIXED_CFG))
                m = getattr(obj, "last_metrics", None)
                if m:
                    for k in OBF2_TERMS:
                        per_term[k].append(float(m[k]))
            cell = {"surrogate_rows": n_rows,
                    "surrogate_rows_achieved": int(len(obj.y)),
                    "surrogate_fraud_rows": int(obj.y.sum()),
                    "val_fraud_approx": int(round(int(obj.y.sum()) * 0.3)),
                    "obf2_std": float(np.std(obf2s)),
                    "obf2_mean": float(np.mean(obf2s)),
                    "terms": {}}
            for k, w in OBF2_TERMS.items():
                v = np.array(per_term[k], dtype=float)
                cell["terms"][k] = {
                    "mean": float(v.mean()) if v.size else None,
                    "std": float(v.std()) if v.size else None,
                    # in Obf2 units, so the four are comparable to each other
                    # and to obf2_std
                    "contribution_std": float(w * v.std()) if v.size else None,
                }
            cells.append(cell)
            terms = "  ".join(
                f"{k[:4]}={cell['terms'][k]['contribution_std']:.4f}"
                if cell["terms"][k]["contribution_std"] is not None else f"{k[:4]}=?"
                for k in OBF2_TERMS)
            print(f"  [{ds}] rows={n_rows:>6}  fraud={cell['surrogate_fraud_rows']:>4}"
                  f"  obf2_std={cell['obf2_std']:.4f}  |  {terms}", flush=True)
        out["datasets"][ds] = cells

    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = os.path.join(RESULTS_DIR, "objective_noise_components.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {p}", flush=True)
    write_components_draft(out)
    return out


def write_components_draft(r):
    L = ["# Where in Obf2 does the surrogate's noise live?\n",
         "_Auto-generated by `experiments/objective_noise_audit.py --components`._\n",
         "`Obf2 = 2·MCC + Spec + Pre + NPV`. The four terms split along the class "
         "imbalance: **MCC and Precision** are governed by the handful of "
         "validation *fraud* rows that `_MIN_FRAUD_ROWS = 30` pins, while "
         "**Specificity and NPV** are governed by the validation *normals*, which "
         "grow with the surrogate regardless. So if noise falls as the surrogate "
         "grows while the fraud count does not, the fall must sit in Spec and "
         "NPV — and that is checkable term by term rather than arguable.\n",
         "Each contribution is `weight × std`, in Obf2 units, so the columns are "
         "comparable with each other and with `Obf2 std`.\n"]
    for ds, cells in r["datasets"].items():
        L += [f"## {ds.upper()}\n",
              "| rows | fraud rows | ≈val fraud | Obf2 std | 2·std(MCC) | "
              "std(Spec)/100 | std(Pre)/100 | std(NPV)/100 |",
              "|---|---|---|---|---|---|---|---|"]
        for c in cells:
            t = c["terms"]
            def _f(k):
                v = t[k]["contribution_std"]
                return f"{v:.4f}" if v is not None else "—"
            L.append(f"| {c['surrogate_rows']:,} | {c['surrogate_fraud_rows']} | "
                     f"{c['val_fraud_approx']} | {c['obf2_std']:.4f} | "
                     f"{_f('MCC')} | {_f('Specificity')} | {_f('Precision')} | "
                     f"{_f('NPV')} |")
        L.append("")
        if len(cells) >= 2:
            a, b = cells[0], cells[-1]
            drops = {k: (a["terms"][k]["contribution_std"] or 0)
                        - (b["terms"][k]["contribution_std"] or 0)
                     for k in OBF2_TERMS}
            top = max(drops, key=drops.get)
            L += [f"From {a['surrogate_rows']:,} to {b['surrogate_rows']:,} rows "
                  f"the fraud count went **{a['surrogate_fraud_rows']} → "
                  f"{b['surrogate_fraud_rows']}** and Obf2's std went "
                  f"**{a['obf2_std']:.4f} → {b['obf2_std']:.4f}**. The largest "
                  f"single reduction is in **{top}** "
                  f"({drops[top]:+.4f} in Obf2 units).\n"]
    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                       "OBJECTIVE_noise_components.md"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {out}", flush=True)


def write_knee_draft(r):
    L = ["# How big does the DB-BOA surrogate have to be?\n",
         "_Auto-generated by `experiments/objective_noise_audit.py --knee`._\n",
         f"OBJ-13 decided to **measure** the surrogate size rather than pick one: "
         f"sweep `surrogate_rows` and report the curve, so the choice is "
         f"defensible rather than arbitrary. Measured in `legacy` mode "
         f"(`draws={r['draws']}`, seed {r['seed']}) — `deterministic` reports a "
         f"std of 0 by construction, which says nothing about whether the one "
         f"draw it freezes is a good one.\n",
         "`noise_to_signal` = 2·std ÷ config-to-config range. **Below 1 means a "
         "single evaluation can separate two candidates**; at or above 1 the "
         "search is ranking draws, not models.\n",
         f"> ⚠ **The 2,000-row cell here is not the same measurement as the "
         f"2,000-row headline in `OBJECTIVE_noise_audit.md`, and should not be "
         f"quoted as a reproduction of it.** Every size below draws from one "
         f"fixed {r.get('pool_rows', 0):,}-row pool — held constant precisely so "
         f"that surrogate size is the only thing moving across the rows of this "
         f"table — whereas the main audit draws from `_eval_subset`'s default "
         f"6,000. Different pool, different subsample, different number. The "
         f"table is internally comparable; it is not cross-comparable with the "
         f"audit.\n",
         f"> The pool is also **{r.get('pool_rows', 0):,} rows against a largest "
         f"surrogate of {max(r['rows_grid']):,}**, so it is never the binding "
         f"constraint. That matters: `_eval_subset` defaults to 6,000, and at "
         f"that default every point above 6,000 would have silently received "
         f"~6,000 rows and the curve would have gone flat for a reason that has "
         f"nothing to do with a knee. `surrogate_rows_achieved` is recorded per "
         f"cell so the claim is checkable rather than trusted.\n"]
    for ds, cells in r["datasets"].items():
        multi = len({c.get("objective_seed") for c in cells}) > 1
        L += [f"## {ds.upper()}\n",
              "| rows | seed | achieved | fraud rows | mean Obf2 | std | "
              "config range | **n2s (k=1)** | n2s k=3 | n2s k=5 | "
              "ceiling hits | s/eval |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            a = c["noise_to_signal_if_averaged"]
            L.append(f"| {c['surrogate_rows']:,} | "
                     f"{c.get('objective_seed', r.get('seed', '—'))} | "
                     f"{c['surrogate_rows_achieved']:,}"
                     f"{' ⚠**CAPPED**' if c['pool_capped'] else ''} | "
                     f"{c['surrogate_fraud_rows']} | "
                     f"{c['mean_obf2']:.4f} | "
                     f"{c['std_obf2']:.4f} | "
                     f"{c['config_range']:.4f} | **{c['noise_to_signal']:.2f}** | "
                     f"{a['3']:.2f} | {a['5']:.2f} | "
                     f"{c['ceiling_hits']}/{r['draws']} | "
                     f"{c['seconds_per_eval']:.1f} |")
        L.append("")

        # With several seeds per size, the honest question is not "which size
        # wins" but "is the size effect bigger than the seed effect at all".
        if multi:
            by_size = {}
            for c in cells:
                by_size.setdefault(c["surrogate_rows"], []).append(
                    c["noise_to_signal"])
            size_means = {k: float(np.mean(v)) for k, v in by_size.items()}
            within = max(max(v) - min(v) for v in by_size.values())
            between = max(size_means.values()) - min(size_means.values())
            L += ["| size | n2s per seed | mean |", "|---|---|---|"]
            for k in sorted(by_size):
                L.append(f"| {k:,} | "
                         f"{', '.join(f'{x:.2f}' for x in by_size[k])} | "
                         f"{size_means[k]:.2f} |")
            L += ["",
                  f"Spread across sizes (of the per-size means): **{between:.2f}**. "
                  f"Largest spread *between seeds at one size*: **{within:.2f}**.\n"]
            if between <= within:
                L += ["> ⛔ **The size effect does not exceed the seed effect, so "
                      "the knee is not locatable from this sweep.** Picking the "
                      "argmin here would be reading a subsample draw as a "
                      "property of the surrogate size. The honest output of this "
                      "experiment is the curve plus this warning — not a chosen "
                      "value.\n"]
            else:
                best_size = min(size_means, key=size_means.get)
                L += [f"> The size effect exceeds the seed effect, so the "
                      f"ordering is readable: lowest mean `noise_to_signal` at "
                      f"**{best_size:,} rows**.\n"]
        else:
            best = min(cells, key=lambda c: c["noise_to_signal"])
            base = cells[0]
            L.append(f"Lowest `noise_to_signal` at **{best['surrogate_rows']:,} "
                     f"rows** ({best['noise_to_signal']:.2f}), against "
                     f"{base['noise_to_signal']:.2f} at the shipped "
                     f"{base['surrogate_rows']:,} — at "
                     f"{best['seconds_per_eval'] / max(base['seconds_per_eval'], 1e-9):.1f}× "
                     f"the cost per evaluation.\n")
            L.append("> ⚠ **One objective seed per size.** The noise level is "
                     "itself a property of the draw — the same nominal "
                     "configuration and size measured std 0.4439 in the main "
                     "audit and 0.2563 in this sweep's first BankSim cell, a "
                     "1.7× swing from the subsample alone. A difference between "
                     "two sizes of that order cannot be attributed to size. "
                     "Re-run with `--knee-seeds` before quoting a chosen "
                     "value.\n")
    # Pre-registration 6 (TASK.md, OBJ-13): the `_MIN_FRAUD_ROWS = 30` floor
    # stops binding at different sizes on the two datasets, so if fraud count is
    # what drives the noise, the two curves must separate — and in a direction
    # named in advance.  Reported as a scoring table, not as prose, so the
    # prediction cannot be quietly reinterpreted after the fact.
    floor = _ADTCNObjective._MIN_FRAUD_ROWS
    if len(r["datasets"]) > 1:
        L += ["## Scoring pre-registration 6 — does the fraud floor drive the noise?\n",
              f"`n_f = max({floor}, rows x fraud_rate)`, so the "
              f"`_MIN_FRAUD_ROWS = {floor}` floor stops binding at very different "
              f"sizes on the two datasets. **Pre-registered:** BankSim's "
              f"`noise_to_signal` improves markedly with size while ULB's barely "
              f"moves until the top of the grid — and *if ULB improves anyway, "
              f"the fraud count is not the driver* and the \"n≈9 positives\" "
              f"mechanism is wrong.\n",
              "| Dataset | fraud rows across the grid | floor still binding? | "
              "n2s first → last | change |",
              "|---|---|---|---|---|"]
        for ds, cells in r["datasets"].items():
            fr = [c["surrogate_fraud_rows"] for c in cells]
            binding = [c for c in cells if c["surrogate_fraud_rows"] <= floor]
            escapes = ("never — pinned at the floor throughout"
                       if len(binding) == len(cells)
                       else f"escapes above {binding[-1]['surrogate_rows']:,} rows"
                       if binding else "never binds on this grid")
            a, b = cells[0]["noise_to_signal"], cells[-1]["noise_to_signal"]
            L.append(f"| {ds.upper()} | {' → '.join(map(str, fr))} | {escapes} | "
                     f"{a:.2f} → {b:.2f} | {b - a:+.2f} |")
        L.append("")

    L += ["## Reading the k columns\n",
          "The `k=3` and `k=5` columns are **not** separate runs: the mean of k "
          "independent draws has std ÷ √k, so they follow from the measured std "
          "at zero extra CPU. They say how far `averaged` mode gets on the same "
          "surrogate size, and are the honest way to compare 'more rows' against "
          "'more draws' — the two knobs cost the same currency.\n"]
    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                       "OBJECTIVE_size_knee.md"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {out}", flush=True)


def main(datasets=("ulb", "banksim"), draws=25, seeds=(42, 7, 123)):
    print("=" * 78, flush=True)
    print("  OBJECTIVE NOISE AUDIT — can the DB-BOA fitness rank a candidate?", flush=True)
    print("=" * 78, flush=True)

    spe = spe_axis_is_dead()
    print(f"\n  steps_per_epoch axis: n_train={spe['n_train']}, "
          f"batch sizes across [50,250] = {spe['distinct_batch_sizes']}  "
          f"-> DEAD AXIS: {spe['is_dead_axis']}", flush=True)

    results = {"steps_per_epoch_axis": spe, "datasets": {}}
    for ds in datasets:
        print(f"\n  --- {ds.upper()} ---", flush=True)
        results["datasets"][ds] = audit(ds, draws, list(seeds))

    print("\n" + "-" * 78, flush=True)
    for ds, r in results["datasets"].items():
        print(f"  {ds:>8}: single-config noise std={r['within_seed_noise_std']:.4f}  "
              f"| between-seed mean spread={r['between_seed_mean_spread']:.4f}  "
              f"| config-to-config range={r['config_sweep']['range']:.4f}", flush=True)
    print("-" * 78, flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "objective_noise_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {path}", flush=True)
    write_draft(results)
    return results


def write_draft(r):
    spe = r["steps_per_epoch_axis"]
    L = ["# Is the DB-BOA fitness function measuring anything?\n",
         "_Auto-generated by `experiments/objective_noise_audit.py`._\n",
         "**Trigger.** On ULB, all five optimisers (MBO, WSA, DBOA, BOA, DB-BOA) "
         "reported a best Obf2 of exactly **5.0000** — the theoretical maximum of "
         "`2*MCC + Spec + Pre + NPV`, reachable only by perfectly classifying the "
         "surrogate's validation split. Five structurally different search "
         "algorithms agreeing to four decimals on the ceiling is a property of "
         "the objective, not of the algorithms.\n",
         "## 1. The objective is stochastic\n",
         "`_ADTCNObjective.__call__` redraws its 70/30 train/validation split and "
         "its torch seed on every call, so one configuration does not score the "
         "same twice.\n",
         "| Dataset | Objective seed | Mean Obf2 | Std | Range | Ceiling hits |",
         "|---|---|---|---|---|---|"]
    total_hits = 0
    for ds, d in r["datasets"].items():
        for sd, s in d["fixed_config_by_seed"].items():
            total_hits += s["ceiling_hits"]
            n_draws = d["draws"]
            flag = (f"**{s['ceiling_hits']}/{n_draws}**" if s["ceiling_hits"]
                    else f"{s['ceiling_hits']}/{n_draws}")
            L.append(f"| {ds.upper()} | {sd} | {s['mean']:.4f} | {s['std']:.4f} | "
                     f"{s['min']:.4f} – {s['max']:.4f} | {flag} |")
    L.append("")
    L.append("A *single fixed configuration*, re-evaluated, spans most of the "
             "usable range of the objective.\n")
    if total_hits:
        L.append("**The decisive observation is in the last column.** One fixed "
                 "configuration reached the 5.0000 ceiling by redrawing alone, "
                 "with no search involved at all. Whatever the optimisers were "
                 "rewarding when they each reported 5.0000, it was reachable "
                 "without optimising anything.\n")

    L += ["## 2. The noise is as large as the signal\n",
          "This script computes **two different ratios**, and they answer two "
          "different questions. Quoting either one without naming it is how the "
          "first version of this draft ended up claiming the objective was "
          "healthy on BankSim. Both are reported here, always together.\n",
          "| Dataset | Single-config noise (std) | Between-seed mean spread | "
          "Config-to-config range (8 configs) | `between_seed ÷ range` | "
          "**`noise_to_signal`** |",
          "|---|---|---|---|---|---|"]
    for ds, d in r["datasets"].items():
        rng = d["config_sweep"]["range"]
        L.append(f"| {ds.upper()} | {d['within_seed_noise_std']:.4f} | "
                 f"{d['between_seed_mean_spread']:.4f} | "
                 f"{rng:.4f} | "
                 f"{d['between_seed_mean_spread']/rng:.2f} | "
                 f"**{d['noise_to_signal']:.2f}** |")
    L += ["",
          "| Ratio | What it asks | Verdict when large |",
          "|---|---|---|",
          "| `between_seed ÷ range` | Does the *subsample* set the difficulty, "
          "rather than the candidate? | The objective's **mean** is a property "
          "of the draw, not the model. |",
          "| **`noise_to_signal`** = 2·within-seed std ÷ range | Can **one "
          "evaluation** rank two different configurations? | A single draw "
          "cannot separate candidates — fatal for a best-of-N search. |",
          "",
          ""]

    # ── the null test ────────────────────────────────────────────────────────
    nulls = {ds: d.get("null_range_test") for ds, d in r["datasets"].items()}
    if any(nulls.values()):
        L += ["### ⚠ The threshold for `noise_to_signal` is NOT 1.0\n",
              "`noise_to_signal` divides by the config-to-config **range**, and "
              "that range is measured from 8 *single* evaluations — each carrying "
              "the same noise the numerator is measuring. So part of the "
              "denominator is noise being compared against itself, and **the "
              "ratio flatters the objective**. Reading \"below 1 means the search "
              "can rank\" therefore sets the bar in the wrong place.\n",
              "The correct reference costs nothing to compute. The repeated draws "
              "of one *fixed* configuration are already samples under the null "
              "*\"the configuration makes no difference\"*; resampling 8 of them "
              "and taking the range, 20,000 times, gives the range one would see "
              "from 8 configurations that are genuinely identical.\n",
              "| Dataset | observed config range | null range (mean) | null p95 | "
              "**observed percentile in null** | `noise_to_signal` | "
              "**null value of the ratio** | configs separable? |",
              "|---|---|---|---|---|---|---|---|"]
        for ds, n in nulls.items():
            if not n:
                continue
            d = r["datasets"][ds]
            L.append(f"| {ds.upper()} | {n['observed_config_range']:.4f} | "
                     f"{n['null_range_mean']:.4f} | {n['null_range_p95']:.4f} | "
                     f"**{n['observed_percentile_in_null']:.1f}%** | "
                     f"{d['noise_to_signal']:.2f} | "
                     f"**{n['null_noise_to_signal']:.2f}** | "
                     f"{'yes' if n['configs_distinguishable_from_noise'] else '**no**'} |")
        L += ["",
              "**Read the last two columns together.** The *null value of the "
              "ratio* is what `noise_to_signal` comes out at when the "
              "configurations are interchangeable — it is nowhere near 1.0. A "
              "measured ratio at or above its own null column means the eight "
              "configurations spread no more than eight redraws of a single one: "
              "**the objective carries no usable information about the "
              "configuration at all.**\n",
              "> This does not overturn the audit's conclusion — it sharpens it "
              "in the unfavourable direction. Every previous reading of these "
              "ratios against a threshold of 1.0 **understated** how bad the "
              "objective is.\n"]

    L += ["The between-seed spread is caused purely by *which normal rows get "
          "subsampled* — nothing about the candidate being scored. When that "
          "shift is comparable to the spread across genuinely different "
          "configurations, the search is ranking samples, not models.\n",
          "**So the reported \"best Obf2\" is a maximum over 90–170 noisy draws — "
          "a best-of-N noise statistic, not a measure of configuration quality.** "
          "On ULB that maximum pins at the 5.0000 ceiling for every optimiser.\n"]

    # The dataset contrast is real, but it is NOT "BankSim is better".  The two
    # ratios disagree about which dataset is worse, and `noise_to_signal` — this
    # script's own headline field, and the one that governs a best-of-N search —
    # says BankSim.  Reporting only the flattering ratio is rule 4 with extra
    # steps; that is exactly what the retracted version of this draft did.
    u = r["datasets"].get("ulb")
    b = r["datasets"].get("banksim")
    if u and b:
        u_bs = u["between_seed_mean_spread"] / u["config_sweep"]["range"]
        b_bs = b["between_seed_mean_spread"] / b["config_sweep"]["range"]
        L += ["### The two datasets fail differently — neither one passes\n",
              f"**The two ratios disagree about which dataset is worse, and that "
              f"disagreement is the finding.** On `between_seed ÷ range` ULB "
              f"scores {u_bs:.2f} and BankSim {b_bs:.2f}, so BankSim looks "
              f"~{u_bs/b_bs:.0f}× better. On `noise_to_signal` — this script's "
              f"own headline field, and the one that decides whether a "
              f"best-of-N search can rank anything — ULB scores "
              f"{u['noise_to_signal']:.2f} and BankSim "
              f"**{b['noise_to_signal']:.2f}**, so BankSim is *worse*.\n",
              f"The honest statement is the narrow one: on BankSim the "
              f"objective's **mean** is stable across subsamples "
              f"({b['between_seed_mean_spread']:.4f} between seeds), but a "
              f"single evaluation still carries ±{b['within_seed_noise_std']:.3f} "
              f"against a config-to-config range of "
              f"{b['config_sweep']['range']:.3f}, so it cannot rank two "
              f"candidates either.\n",
              "> **The objective is uninformative on both datasets, for two "
              "different reasons.** Do not write \"the objective is informative "
              "on BankSim\" — an earlier version of this draft did, by quoting "
              "one ratio and omitting the other.\n"]

        # The mechanism is sharper than "0.17 % fraud", and the JSON proves it:
        # the _MIN_FRAUD_ROWS floor fires on both datasets, so fraud rate is
        # not what separates them.
        fraud_rows = sorted({s["surrogate_fraud_rows"]
                             for d in r["datasets"].values()
                             for s in d["fixed_config_by_seed"].values()})
        if len(fraud_rows) == 1:
            n_fr = fraud_rows[0]
            n_val = int(round(n_fr * 0.3))
            L += ["### The mechanism — ⚠ CORRECTED 2026-09-04, twice\n",
                  f"Within *this* measurement, `surrogate_fraud_rows` is "
                  f"**{n_fr} on both datasets**, because the "
                  f"`_MIN_FRAUD_ROWS = {n_fr}` floor fires on each (ULB 0.17 % "
                  f"× 2,000 ≈ 3 → {n_fr}; BankSim 1.211 % × 2,000 ≈ 24 → "
                  f"{n_fr}), and the 70/30 split leaves roughly **{n_val} fraud "
                  f"rows in validation**.\n",
                  "> ⛔ **This draft used to conclude from that: _\"the fraud "
                  "rate is not what separates ULB from BankSim; task "
                  "separability at n≈9 positives is.\"_ That conclusion is "
                  "WITHDRAWN. It is wrong twice over, and both refutations are "
                  "measured.**\n",
                  "**1. The fraud rate _is_ what separates them — through a "
                  "threshold, not a rate.** This script builds its surrogate "
                  "from a 6,000-row pool (`basepaper_comparison._eval_subset`). "
                  "The deployed pipeline does not: `main.py` and "
                  "`run_baselines.py` call `loader.get_eval_subset`, which "
                  "returned a **3,000-row** pool. At ULB's 0.17 % that pool held "
                  "**5 unique fraud transactions** against a floor of 30, so "
                  "`_ADTCNObjective` took its `replace=True` branch and built "
                  "the 30 \"fraud rows\" by repeating those 5 about six times "
                  "each. BankSim's 3,000 rows held 38 and never crossed the "
                  "threshold. **That threshold is the difference between the "
                  "datasets, and it is a direct consequence of the fraud "
                  "rate.**\n",
                  "**2. \"Reachable by luck\" understates it — on the shipped "
                  "path no luck was required.** With 5 unique positives "
                  "duplicated across an unstratified split, **9 of 9 validation "
                  "fraud rows were copies of rows in the training half** "
                  "(measured in `obj13_shipped_path_check.json`, which replays "
                  "the index construction and verifies it bitwise against the "
                  "live object). `Obf2 = 5.0000` did not mean *classified nine "
                  "rare rows correctly*; it meant **recognised five rows it had "
                  "already memorised**. Fixed 2026-09-04 by raising "
                  "`eval_subset` to 36,000 (ULB now holds ~60 unique fraud) "
                  "plus a `RuntimeWarning` guard on the replacement branch; "
                  "re-verified at 0 of 11 leaked.\n",
                  "**3. The fraud count does not drive the noise on either "
                  "dataset.** The size sweep (`--knee`, "
                  "`OBJECTIVE_size_knee.md`) pre-registered that BankSim would "
                  "improve markedly as its fraud count escaped the floor and "
                  "ULB would not. Neither happened: **BankSim's noise std held "
                  "flat at 0.2563 → 0.2553 while its fraud rows went 30 → 252**, "
                  "and **ULB improved 0.79 → 0.39 while pinned at exactly 30**. "
                  "Whatever sets the noise level, it is not the number of "
                  "positives.\n",
                  "> **The numbers in the tables above are unaffected by all of "
                  "this** — they were measured on the 6,000-row pool, which held "
                  "30 unique fraud rows and never duplicated. What is withdrawn "
                  "is the *interpretation*, not the measurement.\n"]

    L += [f"## 3. One of the two search axes is arithmetically dead\n",
          f"The surrogate sets `batch_size = max(32, n_train // spe)` with "
          f"`n_train = {spe['n_train']}`. Across the searched range "
          f"`spe` ∈ [50, 250] this yields batch sizes "
          f"{spe['distinct_batch_sizes']} — i.e. **{'a single value' if spe['is_dead_axis'] else 'several values'}**.\n",
          "| steps_per_epoch | 50 | 100 | 150 | 200 | 250 |", "|---|---|---|---|---|---|",
          # JSON round-trips dict keys to strings, so accept either form
          "| batch_size | " + " | ".join(
              str(spe["sample"].get(k, spe["sample"].get(str(k))))
              for k in (50, 100, 150, 200, 250)) + " |",
          "",
          "So the advertised 2-D search (filters x steps/epoch) is effectively "
          "1-D: the second coordinate cannot change the surrogate's behaviour at "
          "all. Every reported `steps_per_epoch` optimum is arbitrary.\n",
          "**The floor was the bug, not the arithmetic.** `n_train // spe` runs "
          "28 down to 5 across the searched range — every one of those is *below* "
          "the 32 floor, which is why the floor won every time. OBJ-13's repair "
          "drops the floor for `ceil(n_train / spe)`, which is what "
          "`steps_per_epoch` has always claimed to mean:\n",
          "| steps_per_epoch | 50 | 100 | 150 | 200 | 250 |", "|---|---|---|---|---|---|",
          "| batch_size (repaired) | " + " | ".join(
              str(_repaired_bs(spe["n_train"], k)) for k in (50, 100, 150, 200, 250)) + " |",
          "",
          f"That is {len({_repaired_bs(spe['n_train'], s) for s in range(50, 251, 10)})} distinct "
          f"batch sizes where there was 1, so the axis is alive across the whole "
          f"range. **The axis was repaired rather than dropped on purpose:** if "
          f"`spe` turns out not to matter once it is genuinely live, that is a "
          f"*measured* finding and dropping to a 1-D search is then justified by "
          f"data. Dropping it now would assert the same conclusion without "
          f"evidence.\n"]

    L += ["## What this means\n",
          "1. **The optimiser comparison cannot be read as a ranking.** This is "
          "why `basepaper_comparison` reports the five optimisers as "
          "indistinguishable rather than crowning DB-BOA.\n",
          "2. **It is *not* the explanation for the old MCC contradiction — "
          "OBJ-2 settled that separately, and by a different mechanism.** "
          "Training is bitwise deterministic given (seed, torch version, thread "
          "count); the two scripts simply ran at different thread counts (4 vs "
          "8) against an unpinned torch. **MCC 0.313 is withdrawn** — it is not "
          "reproducible under any condition tested, and the lowest tuned MCC in "
          "7 runs is 0.6598. 0.677 sits inside the observed range and stands as "
          "the deployment operating point. This audit concerns the layer *above* "
          "that training run: which configuration the search hands it in the "
          "first place.\n",
          "3. **It supplies the mechanism for the measured tie.** OBJ-4 scored "
          "the tuned configuration at MCC 0.753 ± 0.055 against the hand-set "
          "default at 0.706 ± 0.076 over five seeds — **+0.047, p=0.34**, "
          "statistically indistinguishable. There is no measured gain *and no "
          "measured loss*; the earlier claim that DB-BOA loses to the default "
          "(0.785) is withdrawn and must not be requoted. A search whose fitness "
          "cannot rank its candidates returns an essentially arbitrary one, so "
          "**tying a sensible hand-set choice is the expected outcome** — which "
          "strengthens the negative result rather than softening it.\n",
          "## The repair — APPLIED 2026-09-04 (OBJ-13)\n",
          "⚠ **Everything above describes the objective as it was *before* the "
          "repair, and that is deliberate.** This script pins "
          "`eval_mode=\"legacy\"` explicitly rather than taking the repo default, "
          "which is now `deterministic`. On the default it would report a "
          "within-seed std of exactly 0.0000 — not because the noise was ever "
          "absent, but because `deterministic` freezes one draw and so hides it. "
          "A 0.0000 here would read as *the objective was always fine*, which is "
          "the opposite of what happened.\n",
          "`models/adtcn.py` now takes `eval_mode` ∈ {`legacy`, `deterministic`, "
          "`averaged`}, plumbed through `ADTCN_CONFIG` as `surrogate_eval_mode` / "
          "`surrogate_k` / `surrogate_rows`:\n",
          "| Mode | Fitness | Cost | Why it exists |",
          "|---|---|---|---|",
          "| `legacy` | fresh unstratified draw + fresh torch seed per call | 1× | "
          "reproduces the pre-repair behaviour **bit-for-bit**, verified against "
          "`models/adtcn.py.pre_obj13`, so the comparison runs against a live "
          "baseline rather than a remembered one |",
          "| `deterministic` | split + seed frozen at construction | 1× | fitness "
          "is a *pure function of the candidate*, so ranking is reproducible — "
          "but it still optimises one arbitrary draw, which is why `averaged` "
          "exists |",
          "| `averaged` | mean over `surrogate_k` pre-drawn draws **shared by "
          "every candidate** | k× | optimises an expectation, so a lucky draw "
          "can no longer win a best-of-N |",
          "",
          "Two details carry the honesty here. The draws are pre-drawn **once** "
          "and shared across candidates (common random numbers) — re-drawing per "
          "candidate would leave the search comparing a good config on an easy "
          "draw against a good config on a hard one, which is the original defect "
          "wearing a mean. And both repaired modes **stratify** the 70/30 split, "
          "so fraud is present in both halves at every surrogate size; without "
          "that, 'more rows' and 'fraud actually present in validation' would "
          "move together and the size knee would not be attributable.\n",
          "**Running `deterministic` and `averaged` side by side is the result**, "
          "because the gap between them measures how much of the original DB-BOA "
          "behaviour was noise-chasing rather than optimisation. Surrogate size "
          "is swept separately — `--knee`, written to "
          "`OBJECTIVE_size_knee.md`.\n",
          "> **Pre-registered (rule 4).** A repaired surrogate is a "
          "*better-behaved* DB-BOA, not necessarily a winning one. If DB-BOA "
          "still ties the hand-set default after the repair, that is the honest "
          "outcome and rule 3 applies — **this is diagnosis of a negative "
          "result, not a rescue attempt.**\n"]

    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                       "OBJECTIVE_noise_audit.md"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {out}", flush=True)


def render_only():
    """
    Rebuild the markdown draft from the JSON already on disk.

    This exists because of a real failure.  The prose used to be reachable only
    by re-running ~150 training draws, which made a stale draft *cheaper to
    leave broken than to fix* — so a retracted number (MCC 0.313) survived in a
    live draft after the report itself had been purged of it.  That is the
    actual defect; the wrong sentence was only the symptom.

    Regenerating the prose must cost seconds, not hours.
    """
    path = os.path.join(RESULTS_DIR, "objective_noise_audit.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"--render-only needs {path}, which does not exist.\n"
            f"Run the full audit first:  python experiments/objective_noise_audit.py")
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    print(f"  rendering from {path} (no training)", flush=True)
    write_draft(results)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--dataset", default=None, choices=["ulb", "banksim"])
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--render-only", action="store_true",
                    help="rebuild the markdown draft from the existing "
                         "results/objective_noise_audit.json without training")
    ap.add_argument("--knee", action="store_true",
                    help="sweep surrogate_rows against noise_to_signal and "
                         "write results/objective_size_knee.json (OBJ-13)")
    ap.add_argument("--components", action="store_true",
                    help="decompose the Obf2 noise into its four terms at the "
                         "ends of the size grid (OBJ-13, follow-up to the knee)")
    ap.add_argument("--knee-redraft", action="store_true",
                    help="rebuild OBJECTIVE_size_knee.md from the existing "
                         "results/objective_size_knee.json without training. "
                         "Same reason as --render-only: regenerating prose must "
                         "cost seconds, or a stale draft becomes cheaper to "
                         "leave broken than to fix.")
    ap.add_argument("--knee-draws", type=int, default=12)
    ap.add_argument("--knee-seeds", action="store_true",
                    help="run every --seeds value at every size, instead of "
                         "just the first. Costs len(seeds)x, and is what makes "
                         "a difference between two sizes distinguishable from "
                         "the subsample-to-subsample swing in the noise level.")
    ap.add_argument("--rows", default=None,
                    help="comma-separated surrogate sizes for --knee "
                         f"(default {','.join(str(r) for r in KNEE_ROWS)})")
    a = ap.parse_args()
    if a.render_only:
        render_only()
    elif a.knee_redraft:
        p = os.path.join(RESULTS_DIR, "objective_size_knee.json")
        if not os.path.exists(p):
            raise SystemExit(f"--knee-redraft needs {p}; run --knee first")
        with open(p, encoding="utf-8") as fh:
            write_knee_draft(json.load(fh))
    elif a.components:
        torch.set_num_threads(a.threads)
        components(datasets=(a.dataset,) if a.dataset else ("ulb", "banksim"),
                   rows=tuple(int(x) for x in a.rows.split(",")) if a.rows
                        else (2_000, 20_000),
                   draws=a.knee_draws, seed=int(a.seeds.split(",")[0]))
    elif a.knee:
        torch.set_num_threads(a.threads)
        knee(datasets=(a.dataset,) if a.dataset else ("ulb", "banksim"),
             draws=a.knee_draws,
             seeds=[int(s) for s in a.seeds.split(",")] if a.knee_seeds
                   else [int(a.seeds.split(",")[0])],
             rows=tuple(int(r) for r in a.rows.split(",")) if a.rows else KNEE_ROWS)
    else:
        torch.set_num_threads(a.threads)
        main(datasets=(a.dataset,) if a.dataset else ("ulb", "banksim"),
             draws=a.draws, seeds=tuple(int(s) for s in a.seeds.split(",")))
