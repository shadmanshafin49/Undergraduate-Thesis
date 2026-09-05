"""
experiments/obj13_surrogate_repair.py
=====================================
Did repairing the DB-BOA surrogate change what the search chooses?

Why this exists
---------------
`objective_noise_audit.py` established that the pre-repair fitness was a random
function of its input: a fixed configuration re-evaluated 25x on ULB spanned
Obf2 3.4497-5.0000 and touched the 5.0000 ceiling once **with no search
involved**, which is why all five optimisers "found" exactly 5.0000.  OBJ-13
repaired it in `models/adtcn.py` behind `eval_mode`:

    legacy         fresh unstratified split + fresh torch seed per call
                   (the pre-repair behaviour, bit-for-bit)
    deterministic  split and seed frozen at construction; fitness is a pure
                   function of the candidate
    averaged       k draws pre-drawn once and SHARED by every candidate
                   (common random numbers); fitness is their mean

OBJ-13's decision was to run the repaired modes **side by side with legacy**,
because the gap between them is what measures how much of the original DB-BOA
behaviour was noise-chasing rather than optimisation.  This script is that run.

The measurement problem, and how it is handled
----------------------------------------------
**The three modes' own "best Obf2" numbers are not comparable to each other**,
and reporting them in one column would be the central dishonesty available here.
Legacy's best is a maximum over ~600 noisy draws — a best-of-N order statistic.
Deterministic's is a maximum over candidates on one frozen draw.  Averaged's is a
maximum over candidates of a k-mean.  Those are three different quantities and
the noisiest one is biased *upward* by construction, so a naive table would show
legacy "winning".

So every returned configuration is re-scored on a **common held-out yardstick**:
an `averaged` objective built at a different `random_state`, whose subsample and
whose draws no search ever saw.  That is the only number in this script that may
be compared across modes, and it is the one the verdict rests on.  Two reference
rows are scored on the same yardstick so the spread has a scale:

  * the hand-set default (`ADTCN_CONFIG`: 128 filters / 150 steps-per-epoch)
  * the shipped DB-BOA configuration (142 / 76) — the one behind
    `db_boa_results.json` and hardcoded in `detector_multiseed.py`

**Pre-registered expectations are in TASK.md under OBJ-13, written before this
script was run.**  In short: the ceiling is expected to survive `deterministic`
and die under `averaged`; the chosen configuration is expected to keep scattering
across objective seeds in every mode; and the three modes are expected to be
separated on the yardstick by *less* than the seed-to-seed spread inside any one
of them — i.e. the repair fixes the mechanism without producing a better model.
That is the honest expected outcome and rule 3 covers it: **this is diagnosis of
a negative result, not a rescue attempt.**

Usage
-----
    python experiments/obj13_surrogate_repair.py --dataset banksim --scout
    python experiments/obj13_surrogate_repair.py --dataset banksim
    python experiments/obj13_surrogate_repair.py --dataset banksim --redraft

`--scout` runs one mode at one seed on a small budget and prints a projected
cost for the full matrix.  Run it first; a full run is hours and the projection
costs minutes.
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

from config import RESULTS_DIR, DB_BOA_CONFIG, ADTCN_CONFIG
from models.adtcn import _ADTCNObjective
from algorithms.db_boa import DBBOA
from experiments.basepaper_comparison import (prepare_ulb, prepare_banksim,
                                              _eval_subset)

MODES = ("legacy", "deterministic", "averaged")

#: Reference configurations, scored on the same yardstick as every search result
#: so the mode-to-mode spread has something to be compared against.
REFERENCE_CFGS = {
    "hand_set_default": (ADTCN_CONFIG["hidden_neurons"], ADTCN_CONFIG["steps_per_epoch"]),
    "shipped_dbboa":    (142, 76),   # db_boa_results.json / detector_multiseed.py
}

#: The yardstick.  A different random_state from every search seed, so its
#: subsample AND its draws are disjoint from anything a search optimised on.
YARDSTICK_SEED = 20260904
YARDSTICK_K    = 7


def build_pools(dataset, pool_rows, verbose=True):
    """
    Two **row-disjoint** pools: one the searches draw their surrogates from, one
    the yardstick draws from.

    A different `random_state` alone would not be enough.  Every objective
    subsamples from whatever pool it is handed, so a single shared pool would
    leave the yardstick's rows overlapping the rows a search tuned on — at
    2,000 of 20,000 that is ~10 % shared by chance, and a configuration that
    happened to fit those rows would be rewarded twice.  Splitting the pool
    first makes "held out" mean held out.

    The split is stratified, because the fraud rows are the scarce resource: an
    unstratified halving of a pool whose fraud count is already small can leave
    one side short enough that `_MIN_FRAUD_ROWS` starts sampling with
    replacement, which would quietly change what the yardstick measures.
    """
    prep = prepare_ulb if dataset == "ulb" else prepare_banksim
    kw = {"ordering": "time"} if dataset == "ulb" else {"ordering": "customer"}
    d = prep(verbose=verbose, **kw)
    X, y = _eval_subset(d, n=pool_rows * 2)

    rs = np.random.RandomState(0)
    a_idx, b_idx = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        idx = idx[rs.permutation(len(idx))]
        cut = len(idx) // 2
        a_idx.append(idx[:cut])
        b_idx.append(idx[cut:])
    a = np.concatenate(a_idx); b = np.concatenate(b_idx)
    rs.shuffle(a); rs.shuffle(b)
    if verbose:
        print(f"  search pool    : {len(a):,} rows, {int(y[a].sum())} fraud", flush=True)
        print(f"  yardstick pool : {len(b):,} rows, {int(y[b].sum())} fraud  "
              f"(row-disjoint from the search pool)", flush=True)
    return d, (X[a], y[a]), (X[b], y[b])


def make_objective(X_opt, y_opt, n_raw, seed, mode, k, rows, arch):
    return _ADTCNObjective(X_opt, y_opt, random_state=seed, architecture=arch,
                           n_raw=n_raw, eval_mode=mode, k_repeats=k,
                           surrogate_rows=rows)


def run_search(X_opt, y_opt, n_raw, seed, mode, k, rows, arch, pop, iters,
               lb, ub, verbose=False):
    """One DB-BOA search under one surrogate protocol.  Returns the record."""
    obj = make_objective(X_opt, y_opt, n_raw, seed, mode, k, rows, arch)
    t0 = time.time()
    opt = DBBOA(objective_fn=obj, lb=lb, ub=ub, n_pop=pop, max_iter=iters,
                task_name=f"OBJ-13 {mode} seed {seed}",
                cfg={**DB_BOA_CONFIG, "population_size": pop,
                     "max_iterations": iters},
                seed=seed)
    best_pos, best_fit, history = opt.optimise(verbose=verbose)
    dt = time.time() - t0
    n_filters = max(8, int(round(float(best_pos[0]))))
    spe = max(20, int(round(float(best_pos[1]))))
    return {
        "eval_mode": mode, "search_seed": seed,
        "k": obj._k, "surrogate_rows": int(len(obj.y)),
        "surrogate_fraud_rows": int(obj.surrogate_fraud_rows),
        "chosen_filters": n_filters, "chosen_spe": spe,
        # ⚠ NOT comparable across modes — see the module docstring.  Kept
        # because it is what each search believed it was maximising.
        "own_best_obf2": float(-best_fit),
        "own_best_is_ceiling": bool(abs(-best_fit - 5.0) < 1e-6),
        "n_evals": int(obj._call_count),
        "search_seconds": dt,
    }


def score_on_yardstick(yard, n_filters, spe):
    """
    Score one configuration on the shared held-out objective.

    `_score_once` is called directly, on each of the yardstick's pre-drawn
    draws, so the per-draw values are available and not just their mean — the
    spread is what says whether a difference between two configurations is
    real or is the yardstick's own noise.
    """
    vals = []
    for tr, vl, tseed in yard._draws:
        s = yard._score_once(n_filters, spe, tr, vl, tseed)
        vals.append(float("nan") if s is None else float(s))
    arr = np.array(vals, dtype=float)
    return {"yardstick_mean": float(np.nanmean(arr)),
            "yardstick_std": float(np.nanstd(arr)),
            "yardstick_draws": vals}


def main(dataset="banksim", seeds=(42, 7, 123), modes=MODES, k=3,
         rows=None, arch="cnn", pop=None, iters=None, out_name=None,
         filt_hi=None, pool_rows=20000, threads=2):
    pop = pop or DB_BOA_CONFIG["population_size"]
    iters = iters or DB_BOA_CONFIG["max_iterations"]
    lo_f, hi_f = DB_BOA_CONFIG["filter_count_bounds"]
    hi_f = filt_hi or hi_f
    lo_s, hi_s = DB_BOA_CONFIG["steps_per_epoch_bounds"]
    lb = np.array([float(lo_f), float(lo_s)])
    ub = np.array([float(hi_f), float(hi_s)])
    out_name = out_name or f"obj13_surrogate_repair_{dataset}.json"
    path = os.path.join(RESULTS_DIR, out_name)

    print("=" * 78, flush=True)
    print(f"  OBJ-13 SURROGATE REPAIR — side-by-side, dataset={dataset}", flush=True)
    print(f"  modes={list(modes)}  seeds={list(seeds)}  k={k}", flush=True)
    print(f"  space: filters [{lo_f},{hi_f}]  spe [{lo_s},{hi_s}]  "
          f"pop={pop} x {iters} iters", flush=True)
    print(f"  yardstick: averaged objective, seed {YARDSTICK_SEED}, "
          f"k={YARDSTICK_K}, unseen by every search", flush=True)
    print("=" * 78, flush=True)

    d, (X_opt, y_opt), (X_yard, y_yard) = build_pools(dataset, pool_rows)
    n_raw = int(d["X"].shape[1])

    yard = make_objective(X_yard, y_yard, n_raw, YARDSTICK_SEED, "averaged",
                          YARDSTICK_K, rows, arch)
    print(f"  yardstick built: {len(yard.y)} rows, "
          f"{yard.surrogate_fraud_rows} fraud, {len(yard._draws)} draws", flush=True)

    summary = {
        "task": "OBJ-13 side-by-side: does repairing the DB-BOA surrogate change "
                "which configuration the search returns?",
        "dataset": dataset, "architecture": arch,
        "split_note": d["split_note"],
        "search_space": {"filters": [lo_f, hi_f], "steps_per_epoch": [lo_s, hi_s],
                         "population": pop, "iterations": iters},
        "surrogate_rows": rows or _ADTCNObjective._SURROGATE_ROWS,
        "search_pool_rows": int(len(y_opt)),
        "search_pool_fraud": int(y_opt.sum()),
        "seeds": list(seeds), "modes": list(modes), "k": k,
        "yardstick": {
            "seed": YARDSTICK_SEED, "k": YARDSTICK_K,
            "rows": int(len(yard.y)), "fraud_rows": int(yard.surrogate_fraud_rows),
            "pool_rows": int(len(y_yard)), "pool_fraud": int(y_yard.sum()),
            "note": "an `averaged` objective drawn from a ROW-DISJOINT pool at a "
                    "random_state no search used, so neither its rows nor its "
                    "draws were ever seen by a search. A different seed on a "
                    "shared pool would not have been enough — the subsamples "
                    "would still overlap by chance. "
                    "THIS is the only column comparable across modes: each mode's "
                    "own best-Obf2 is a different statistic (legacy's is a "
                    "best-of-N over noisy draws and is biased upward by "
                    "construction), so putting those in one column would hand "
                    "legacy a win it did not earn.",
        },
        "environment": {
            "torch": torch.__version__, "threads": threads,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "runs": [], "references": {},
    }

    def _flush(complete=False):
        summary["checkpoint"] = {"complete": complete,
                                 "written": time.strftime("%H:%M:%S")}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        os.replace(tmp, path)      # atomic — never a half-written file

    # Reference rows first: they are seconds, and having them on disk before the
    # hours-long part means a crashed run still leaves something interpretable.
    for name, (nf, spe) in REFERENCE_CFGS.items():
        r = score_on_yardstick(yard, nf, spe)
        r.update({"filters": nf, "spe": spe})
        summary["references"][name] = r
        print(f"  reference {name:>18}: {nf:>3}f/{spe:>3}spe  "
              f"yardstick {r['yardstick_mean']:.4f} ± {r['yardstick_std']:.4f}",
              flush=True)
    _flush()

    t_start = time.time()
    for mode in modes:
        for sd in seeds:
            print(f"\n  --- {mode} / seed {sd} ---", flush=True)
            rec = run_search(X_opt, y_opt, n_raw, sd, mode, k, rows, arch,
                             pop, iters, lb, ub)
            rec.update(score_on_yardstick(yard, rec["chosen_filters"],
                                          rec["chosen_spe"]))
            summary["runs"].append(rec)
            print(f"      chose {rec['chosen_filters']}f / {rec['chosen_spe']}spe  "
                  f"own-best {rec['own_best_obf2']:.4f}"
                  f"{'  [CEILING]' if rec['own_best_is_ceiling'] else ''}  "
                  f"| yardstick {rec['yardstick_mean']:.4f} ± "
                  f"{rec['yardstick_std']:.4f}  "
                  f"| {rec['n_evals']} evals, {rec['search_seconds']:.0f} s",
                  flush=True)
            _flush()

    summary["wall_clock_seconds"] = time.time() - t_start
    _flush(complete=True)
    print(f"\n  wrote {path}", flush=True)
    export_extra_configs(summary, dataset)
    write_draft(summary)
    return summary


#: The production training seed.  `ADTCN_CONFIG["random_state"]` is 42 and
#: `optimise_hyperparams` passes it straight through, so seed 42 is the
#: configuration the shipped pipeline would actually receive.
PRODUCTION_SEED = 42


def export_extra_configs(s, dataset):
    """
    Hand the repaired configurations to `detector_multiseed.py`.

    **Selection rule, fixed before the run and not negotiable afterwards: take
    the configuration each repaired mode returns at the PRODUCTION SEED (42).**
    Not the best of the nine by yardstick score.  Selecting a configuration
    because it scored well on the yardstick and then reporting that same
    yardstick score would be a best-of-N statistic dressed as a measurement —
    which is the exact defect this whole objective exists to remove, reappearing
    one level up. Seed 42 is what `ADTCN_CONFIG["random_state"]` feeds the real
    pipeline, so it is the configuration that would actually ship.
    """
    out = {}
    for r in s["runs"]:
        if r["eval_mode"] == "legacy" or r["search_seed"] != PRODUCTION_SEED:
            continue
        out[f"dbboa_repaired_{r['eval_mode']}"] = {
            "hidden_neurons": r["chosen_filters"],
            "steps_per_epoch": r["chosen_spe"],
            "source": (f"obj13_surrogate_repair_{dataset}.json — "
                       f"{r['eval_mode']} at the production seed "
                       f"{PRODUCTION_SEED}; NOT selected on the yardstick"),
        }
    if not out:
        return
    p = os.path.join(RESULTS_DIR, "_multiseed_runs", "extra_configs.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {p}", flush=True)
    for name, spec in out.items():
        print(f"    {name}: {spec['hidden_neurons']}f / "
              f"{spec['steps_per_epoch']}spe", flush=True)
    print("  -> detector_multiseed.py picks these up automatically; re-run it "
          "at the pinned thread count to score them on the test set.", flush=True)


def scout(dataset="banksim", mode="legacy", seed=42, pop=4, iters=3, k=3,
          rows=None, arch="cnn", pool_rows=20000, filt_hi=None):
    """Time one small search, then project the full matrix.  Minutes, not hours."""
    lo_f, hi_f = DB_BOA_CONFIG["filter_count_bounds"]
    hi_f = filt_hi or hi_f
    lo_s, hi_s = DB_BOA_CONFIG["steps_per_epoch_bounds"]
    d, (X_opt, y_opt), _ = build_pools(dataset, pool_rows)
    n_raw = int(d["X"].shape[1])
    print(f"\n  scouting {mode} at pop={pop} x {iters} iters "
          f"(filters [{lo_f},{hi_f}]) ...", flush=True)
    rec = run_search(X_opt, y_opt, n_raw, seed, mode, k, rows, arch, pop, iters,
                     np.array([float(lo_f), float(lo_s)]),
                     np.array([float(hi_f), float(hi_s)]))
    per_eval = rec["search_seconds"] / max(rec["n_evals"], 1)
    print(f"  {rec['n_evals']} evals in {rec['search_seconds']:.0f} s "
          f"= {per_eval:.2f} s/eval  (chose {rec['chosen_filters']}f/"
          f"{rec['chosen_spe']}spe, own-best {rec['own_best_obf2']:.4f})",
          flush=True)

    full_pop = DB_BOA_CONFIG["population_size"]
    full_it  = DB_BOA_CONFIG["max_iterations"]
    scale = (full_pop * full_it) / max(pop * iters, 1)
    evals_full = rec["n_evals"] * scale
    print(f"\n  PROJECTION at pop={full_pop} x {full_it} "
          f"({scale:.0f}x this scout), 3 seeds:", flush=True)
    total = 0.0
    for m in MODES:
        mult = k if m == "averaged" else 1
        hrs = evals_full * per_eval * mult * 3 / 3600
        total += hrs
        print(f"    {m:>14}: ~{evals_full:.0f} evals x {mult} draw(s) x 3 seeds "
              f"= {hrs:.1f} h", flush=True)
    print(f"    {'TOTAL':>14}: ~{total:.1f} h", flush=True)
    print(f"\n  NOTE: per-eval cost scales with the filter count a candidate "
          f"proposes, and this scout's {rec['chosen_filters']} filters is one "
          f"draw from that distribution — treat the projection as an order of "
          f"magnitude, not a promise.", flush=True)
    return rec, per_eval, total


def write_draft(s):
    ds = s["dataset"]
    runs = s["runs"]
    L = [f"# Does repairing the DB-BOA surrogate change what the search picks? — {ds.upper()}\n",
         "_Auto-generated by `experiments/obj13_surrogate_repair.py`._\n",
         f"**Setup.** {s['split_note']}. Surrogate {s['surrogate_rows']:,} rows "
         f"drawn from a {s['search_pool_rows']:,}-row search pool "
         f"({s['search_pool_fraud']} fraud), architecture "
         f"`{s['architecture']}`, search space filters "
         f"{s['search_space']['filters']} x steps/epoch "
         f"{s['search_space']['steps_per_epoch']}, population "
         f"{s['search_space']['population']} x "
         f"{s['search_space']['iterations']} iterations, seeds {s['seeds']}. "
         f"torch {s['environment']['torch']} at "
         f"{s['environment']['threads']} threads.\n",
         "## How to read this table — one column is comparable, one is not\n",
         "> ⚠ **`own best Obf2` may NOT be compared across modes.** Legacy's is a "
         "maximum over ~600 noisy draws — a best-of-N order statistic, biased "
         "*upward* by construction. Deterministic's is a maximum over candidates "
         "on one frozen draw. Averaged's is a maximum over candidates of a "
         f"{s['k']}-draw mean. Putting them in one column and picking the largest "
         "would hand legacy a win it did not earn; that is exactly the artefact "
         "OBJ-13 exists to remove.\n",
         f"> ✅ **`yardstick` is the comparable column.** Every configuration "
         f"below — searched or reference — is re-scored on one shared `averaged` "
         f"objective at seed {s['yardstick']['seed']}, k={s['yardstick']['k']}, "
         f"drawn from a **row-disjoint pool** ({s['yardstick']['pool_rows']:,} "
         f"rows, {s['yardstick']['pool_fraud']} fraud) that no search touched — "
         f"so neither its {s['yardstick']['rows']:,} rows nor its draws were ever "
         f"seen. A different seed on a *shared* pool would not have been enough: "
         f"the subsamples would still have overlapped by chance, and a "
         f"configuration that happened to fit those rows would be rewarded "
         f"twice. The verdict rests on this column alone.\n",
         "| Mode | Seed | Chose (filters / spe) | own best Obf2 ⚠ | **yardstick mean ± std** ✅ | evals | search s |",
         "|---|---|---|---|---|---|---|"]
    for r in runs:
        ceil = " **(ceiling)**" if r["own_best_is_ceiling"] else ""
        L.append(f"| `{r['eval_mode']}` | {r['search_seed']} | "
                 f"{r['chosen_filters']} / {r['chosen_spe']} | "
                 f"{r['own_best_obf2']:.4f}{ceil} | "
                 f"**{r['yardstick_mean']:.4f} ± {r['yardstick_std']:.4f}** | "
                 f"{r['n_evals']} | {r['search_seconds']:.0f} |")
    for name, r in s["references"].items():
        L.append(f"| _reference_ | — | {r['filters']} / {r['spe']} "
                 f"({name.replace('_', ' ')}) | — | "
                 f"**{r['yardstick_mean']:.4f} ± {r['yardstick_std']:.4f}** | — | — |")
    L.append("")

    # ── the three scored questions ───────────────────────────────────────────
    by_mode = {m: [r for r in runs if r["eval_mode"] == m] for m in s["modes"]}

    L += ["## 1. Did the ceiling survive?\n"]
    for m, rs in by_mode.items():
        if not rs:
            continue
        hits = sum(r["own_best_is_ceiling"] for r in rs)
        best = max(r["own_best_obf2"] for r in rs)
        L.append(f"- **`{m}`** — {hits}/{len(rs)} seeds returned exactly 5.0000; "
                 f"highest own-best {best:.4f}.")
    L += ["",
          "Pre-registered: the ceiling survives `deterministic` (freezing a draw "
          "does not add fraud rows — the surrogate still holds ~30, leaving ~9 in "
          "validation, and perfectly classifying nine rows is luck) and dies "
          f"under `averaged` (k={s['k']} draws must be perfect at once). If "
          "`deterministic` also collapsed, the redraw was the whole story and the "
          "\"n≈9 positives\" mechanism is wrong.\n"]

    L += ["## 2. Does the chosen configuration still scatter across seeds?\n",
          "| Mode | filters chosen | spread | spe chosen | spread |",
          "|---|---|---|---|---|"]
    for m, rs in by_mode.items():
        if not rs:
            continue
        f = [r["chosen_filters"] for r in rs]
        p = [r["chosen_spe"] for r in rs]
        L.append(f"| `{m}` | {', '.join(map(str, f))} | {max(f) - min(f)} | "
                 f"{', '.join(map(str, p))} | {max(p) - min(p)} |")
    L += ["",
          "Pre-registered: scatter persists in **every** mode, because "
          "determinism removes *within-seed* variance and the subsample — which "
          "changes with the seed — is what sets the difficulty. Convergence under "
          "`deterministic` would mean the between-seed spread was itself an "
          "artefact of the redraw.\n"]

    L += ["## 3. On the held-out yardstick, is any mode actually better?\n",
          "| Mode | yardstick mean over seeds | seed-to-seed spread |",
          "|---|---|---|"]
    mode_means = {}
    worst_within = 0.0
    for m, rs in by_mode.items():
        if not rs:
            continue
        v = [r["yardstick_mean"] for r in rs]
        mode_means[m] = float(np.mean(v))
        worst_within = max(worst_within, max(v) - min(v))
        L.append(f"| `{m}` | {np.mean(v):.4f} | {max(v) - min(v):.4f} |")
    L.append("")
    if len(mode_means) > 1:
        between = max(mode_means.values()) - min(mode_means.values())
        best_mode = max(mode_means, key=mode_means.get)
        verdict = ("**smaller than**" if between < worst_within else
                   "**larger than**")
        L += [f"Spread between modes: **{between:.4f}**. Largest seed-to-seed "
              f"spread inside a single mode: **{worst_within:.4f}**. The "
              f"between-mode difference is {verdict} the noise within a mode.\n"]
        if between < worst_within:
            L += ["> **So the repair did not buy a better configuration, and that "
                  "is the pre-registered outcome, not a disappointment.** "
                  "`noise_to_signal` (0.74 ULB / 0.94 BankSim) says the "
                  "candidates barely span a rankable band, so a search that now "
                  "ranks correctly is ranking within noise. What the repair buys "
                  "is that the *mechanism* is sound and the result is "
                  "reproducible — rule 3: this is diagnosis of a negative "
                  "result, not a rescue attempt.\n"]
        else:
            L += [f"> The between-mode gap exceeds the within-mode noise, with "
                  f"`{best_mode}` ahead. **That is a positive result and it was "
                  f"pre-registered as the less likely one** — it must be reported "
                  f"alongside the pre-repair tie (+0.047 MCC, p=0.34), not "
                  f"instead of it, and it is not established until it survives on "
                  f"the test set via `detector_multiseed.py`.\n"]

        refs = s["references"]
        if refs:
            L += ["### Against the two reference configurations\n"]
            for name, r in refs.items():
                better = [m for m, v in mode_means.items() if v > r["yardstick_mean"]]
                L.append(f"- **{name.replace('_', ' ')}** "
                         f"({r['filters']}f / {r['spe']}spe): "
                         f"{r['yardstick_mean']:.4f}. Modes beating it: "
                         f"{', '.join(f'`{m}`' for m in better) if better else '**none**'}.")
            L += ["",
                  "> The hand-set default is on this table for a reason. If no "
                  "searched configuration beats it on a yardstick none of them "
                  "were tuned against, the measured tie stands after the repair "
                  "as it stood before it.\n"]

    # ── paired analysis on the shared draws ──────────────────────────────────
    #
    # The yardstick scores every configuration on the SAME draws (common random
    # numbers), so comparing configurations by their means ± spreads throws away
    # most of the available power: the draw-to-draw variation is shared and
    # cancels in a paired difference.  A ±0.45 spread around each mean does NOT
    # mean two configurations 0.15 apart are indistinguishable — that inference
    # would be correct only for independent samples, which these are not.
    ref = s["references"].get("hand_set_default")
    if ref and ref.get("yardstick_draws"):
        base = np.asarray(ref["yardstick_draws"], dtype=float)
        L += ["## The paired comparison — the yardstick's draws are shared\n",
              "Every row below is scored on the **same 7 draws** as every other "
              "row (common random numbers), so the honest comparison is the "
              "**paired per-draw difference**, not the gap between two means "
              "each carrying a ±0.4 spread. That spread is draw-to-draw "
              "variation which is *common to both sides* and cancels here.\n",
              "Baseline is the hand-set default (128f / 150spe). `wins` counts "
              "draws where the searched configuration scores higher.\n",
              "| Mode | Seed | Chose | mean paired Δ vs default | SE | wins / 7 |",
              "|---|---|---|---|---|---|"]
        for r in runs:
            d = np.asarray(r.get("yardstick_draws", []), dtype=float)
            if d.size != base.size:
                continue
            diff = d - base
            se = float(diff.std(ddof=1) / math.sqrt(diff.size)) if diff.size > 1 else float("nan")
            L.append(f"| `{r['eval_mode']}` | {r['search_seed']} | "
                     f"{r['chosen_filters']}f/{r['chosen_spe']}spe | "
                     f"**{diff.mean():+.4f}** | {se:.4f} | "
                     f"{int((diff > 0).sum())} / {diff.size} |")
        for name, rr in s["references"].items():
            if name == "hand_set_default":
                continue
            d = np.asarray(rr.get("yardstick_draws", []), dtype=float)
            if d.size != base.size:
                continue
            diff = d - base
            se = float(diff.std(ddof=1) / math.sqrt(diff.size)) if diff.size > 1 else float("nan")
            L.append(f"| _{name.replace('_', ' ')}_ | — | "
                     f"{rr['filters']}f/{rr['spe']}spe | "
                     f"**{diff.mean():+.4f}** | {se:.4f} | "
                     f"{int((diff > 0).sum())} / {diff.size} |")
        L.append("")
        allp = []
        for r in runs:
            d = np.asarray(r.get("yardstick_draws", []), dtype=float)
            if d.size == base.size:
                allp.append(float((d - base).mean()))
        if allp:
            n_beat = sum(1 for x in allp if x > 0)
            L += [f"**{n_beat} of {len(allp)} searched configurations beat the "
                  f"hand-set default** on the shared draws; the mean paired "
                  f"difference across all of them is **{np.mean(allp):+.4f}**.\n"]
            if n_beat == 0:
                L += ["> **Not one searched configuration beats a hand-set "
                      "128 filters / 150 steps-per-epoch — under any surrogate "
                      "protocol, at any seed.** This is the pre-registered "
                      "outcome and rule 3 governs it: the repair is diagnosis of "
                      "a negative result, not a rescue attempt. It also sharpens "
                      "the OBJ-4 tie (+0.047 MCC, p=0.34) rather than "
                      "overturning it — a search that could not rank its "
                      "candidates was returning arbitrary configurations, and "
                      "repairing the ranking does not help if the candidates "
                      "genuinely differ by less than the objective can "
                      "resolve.\n"]

    L += ["## What is NOT established here\n",
          "This script scores configurations on the **surrogate's** held-out "
          "objective, not on the test set. A yardstick win is a statement about "
          "the search, not about the detector. Carrying it to a claim about "
          "fraud-detection performance requires re-running "
          "`experiments/detector_multiseed.py` with the chosen configuration as "
          "an additional arm, at the same seeds and the same pinned thread count "
          "— which is the last unticked box in OBJ-13.\n"]

    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                       f"OBJ13_surrogate_repair_{ds}.md"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ulb", "banksim"], default="banksim")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rows", type=int, default=None,
                    help="surrogate rows; default _ADTCNObjective._SURROGATE_ROWS")
    ap.add_argument("--arch", default="cnn",
                    help="cnn matches the shipped detector; dilated_attn is ADTCN")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--filters-max", type=int, default=None,
                    help="cap the filter axis to fit a CPU budget; reported")
    ap.add_argument("--pool-rows", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scout", action="store_true",
                    help="time one small search and project the full cost")
    ap.add_argument("--redraft", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    if a.redraft:
        name = a.out or f"obj13_surrogate_repair_{a.dataset}.json"
        with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
            write_draft(json.load(f))
        sys.exit(0)
    if a.scout:
        scout(dataset=a.dataset, rows=a.rows, arch=a.arch,
              pool_rows=a.pool_rows, filt_hi=a.filters_max, k=a.k)
        sys.exit(0)
    main(dataset=a.dataset,
         seeds=tuple(int(x) for x in a.seeds.split(",")),
         modes=tuple(m.strip() for m in a.modes.split(",")),
         k=a.k, rows=a.rows, arch=a.arch, pop=a.pop, iters=a.iters,
         out_name=a.out, filt_hi=a.filters_max, pool_rows=a.pool_rows,
         threads=a.threads)
