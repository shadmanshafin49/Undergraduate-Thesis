"""
experiments/basepaper_comparison.py
===================================
The base paper's own comparison, re-run on our data.

Prabanand & Thanabal (2025, Sci. Rep. 15:6764) evaluate their DB-BOA-ADTCN
against two families of baseline:

    classifiers  EfficientNet, ResNet, DenseNet, DTCN        (their Table 4)
    optimisers   MBO, WSA, DBOA, BOA                         (their Tables 2-3)

Their published numbers are MATLAB runs on their own data.  Divergence D4
records that copying that table into our report was the largest fabrication in
the draft; it has been purged, and **no number from that paper is reproduced
here**.  Instead both families are re-implemented
(`models/basepaper_models.py`, `algorithms/mbo.py`, `algorithms/wsa.py`) and run
by this script on our datasets, under our protocol, so the comparison is ours
end to end and every cell traces to a JSON in `results/`.

Two tracks
----------
**classifier** — the four base-paper classifiers plus our CNN, ADTCN and an
LSTM, all trained on identical windows with identical hyperparameters and
seeds.  The only thing that varies is the architecture.

**optimiser** — the four base-paper metaheuristics plus DB-BOA, each searching
the *same* ADTCN hyperparameter space under the *same* evaluation budget with
the *same* Obf2 fitness, then the winning configuration trained and scored on
the test set.  Reports the base paper's Table-2 statistics (best / worst /
median / mean / std of the cost function) and Table-5 style wall-clock, both
computed from our runs.

Datasets
--------
`--dataset ulb`      284,807 tx, 0.17 % fraud, no customer IDs.  Windows are
                     necessarily global; `--ordering time|random` selects the
                     causal time-ordered stream or the shipped stratified
                     shuffle.
`--dataset banksim`  594,643 tx, 1.21 % fraud, 4,112 customers.  `--ordering
                     customer|global` selects entity-linked or global windows.

Usage
-----
    python experiments/basepaper_comparison.py --dataset banksim --quick
    python experiments/basepaper_comparison.py --dataset banksim --track both
    python experiments/basepaper_comparison.py --dataset ulb --track classifier
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import RESULTS_DIR, BANKSIM_CONFIG, DB_BOA_CONFIG, ADTCN_CONFIG
from data.banksim_loader import load_banksim_frame, encode_banksim, order_rows
from models.adtcn import ARCH_LABELS, SEQ_LEN, _ADTCNObjective
from models.basepaper_models import BASEPAPER_CLASSIFIERS
from experiments._seqtrain import window_index, train_eval, aggregate

from algorithms.mbo import MBO
from algorithms.wsa import WSA
from algorithms.boa import BOA
from algorithms.dboa import DBOA
from algorithms.db_boa import DBBOA

ULB_CSV = os.path.join(os.path.dirname(ROOT), "creditcard.csv")

#: base-paper classifiers first, then ours — the order the tables read in.
CLASSIFIER_TRACK = BASEPAPER_CLASSIFIERS + ["lstm", "cnn", "dilated_attn"]

#: base-paper optimisers first, then ours.
OPTIMISERS = {
    "MBO":    MBO,      # Mine Blast Optimisation      (Sadollah et al. 2013)
    "WSA":    WSA,      # Water Strider Algorithm      (Kaveh & Dadras Eslamlou 2020)
    "DBOA":   DBOA,     # Dynamic Butterfly OA         (Tubishat et al. 2020)
    "BOA":    BOA,      # Billiards OA                 (Givi & Hubalovska 2023)
    "DB-BOA": DBBOA,    # proposed hybrid
}

# Shared search space for the optimiser track.
#
# The base paper's Eq. 11 bounds are filters [5, 255] and steps/epoch [50, 250].
# The filter ceiling is lowered to 64 here for one reason only: five optimisers
# x three seeds, each followed by a full training run on 418k rows, has to fit
# a CPU budget, and the cost of a run grows roughly with the square of the
# filter count.  The space, the population, the iteration budget and the fitness
# are IDENTICAL for all five optimisers, so the comparison itself stays fair —
# only its absolute scale is reduced, and that reduction is reported alongside
# the numbers rather than hidden.
SEARCH_LB = np.array([8.0, 50.0])
SEARCH_UB = np.array([64.0, 250.0])


# ── data preparation ─────────────────────────────────────────────────────────

def prepare_banksim(ordering, cfg=None, verbose=True):
    """BankSim rows in `ordering`, scaled on the training period, with windows."""
    cfg = cfg or dict(BANKSIM_CONFIG)
    df = load_banksim_frame(cfg["dataset_path"], verbose=False)
    df = df.iloc[order_rows(df, ordering, seed=cfg["order_seed"])].reset_index(drop=True)
    X, _ = encode_banksim(df, cfg)
    y = df["fraud"].values.astype(int)
    step = df["step"].values
    groups = df["customer"].values if ordering == "customer" else None
    tr, te = step < cfg["val_step"], step >= cfg["split_step"]
    X = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)
    idx = window_index(len(y), SEQ_LEN, groups=groups)
    if verbose:
        print(f"  BankSim [{ordering}] train={int(tr.sum()):,} "
              f"test={int(te.sum()):,} features={X.shape[1]}", flush=True)
    return {"X": X, "y": y, "idx": idx, "tr": tr, "te": te,
            "split_note": f"temporal past->future, train step<{cfg['val_step']}, "
                          f"test step>={cfg['split_step']}; {ordering} windows"}


def prepare_ulb(ordering="time", verbose=True):
    """
    ULB rows for the same protocol.

    ULB has no account IDs, so entity-linked windows are impossible — that is
    the whole reason OBJ-1 exists.  `ordering="time"` builds causal windows over
    the time-sorted stream with a past->future split; `ordering="random"`
    reproduces the shipped pipeline, which windows *after* a stratified shuffle
    and therefore has no temporal structure at all.
    """
    if not os.path.exists(ULB_CSV):
        raise FileNotFoundError(f"ULB dataset not found at {ULB_CSV}")
    df = pd.read_csv(ULB_CSV).sort_values("Time").reset_index(drop=True)
    cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    X = df[cols].values.astype(np.float32)
    y = df["Class"].values.astype(int)

    n = len(y)
    cut = int(0.8 * n)
    if ordering == "random":
        perm = np.random.default_rng(0).permutation(n)
        X, y = X[perm], y[perm]
    tr = np.zeros(n, bool); tr[:cut] = True
    te = ~tr
    X = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)
    idx = window_index(n, SEQ_LEN, groups=None)
    if verbose:
        print(f"  ULB [{ordering}] train={int(tr.sum()):,} test={int(te.sum()):,} "
              f"features={X.shape[1]}", flush=True)
    return {"X": X, "y": y, "idx": idx, "tr": tr, "te": te,
            "split_note": (f"first 80 % by Time -> train, last 20 % -> test; "
                           f"{ordering}-ordered global windows "
                           f"(ULB has no customer IDs)")}


def prepare(dataset, ordering, verbose=True):
    if dataset == "banksim":
        return prepare_banksim(ordering, verbose=verbose)
    if dataset == "ulb":
        return prepare_ulb(ordering, verbose=verbose)
    raise ValueError(f"unknown dataset {dataset!r}")


# ── track 1: classifiers ─────────────────────────────────────────────────────

def run_classifier_track(d, archs, epochs, n_filters, seeds, batch_size,
                         checkpoint=None, sink=None):
    """
    Train every architecture on identical windows; return {arch: aggregate}.

    `checkpoint`, when given, is called after each architecture completes so the
    partial results reach disk.  The ULB track ran 9.2 hours; losing all of it to
    a crash in the last cell is not an acceptable failure mode.

    `sink`, when given, is the dict results are written into — pass the very dict
    the checkpoint serialises, or the checkpoint flushes an empty placeholder
    while the real results sit in a local that is only returned at the end.
    """
    print("\n  --- CLASSIFIER TRACK (base-paper baselines + ours) ---", flush=True)
    out = sink if sink is not None else {}
    for arch in archs:
        runs = [train_eval(arch, d["X"], d["y"][d["tr"]], d["idx"][d["tr"]],
                           d["X"], d["y"][d["te"]], d["idx"][d["te"]],
                           epochs=epochs, n_filters=n_filters, seed=sd,
                           batch_size=batch_size)
                for sd in seeds]
        c = aggregate(runs)
        out[arch] = c
        tag = "  <- base paper" if arch in BASEPAPER_CLASSIFIERS else ""
        print(f"    {ARCH_LABELS.get(arch, arch):>16}  "
              f"MCC={c['MCC']:+.4f} +/- {c['MCC_std']:.3f}  "
              f"Prec={c['Precision']:5.1f}  Rec={c['Sensitivity']:5.1f}  "
              f"FP={c['FP']:.0f}  ({c['train_seconds']:.0f}s/seed){tag}", flush=True)
        if checkpoint:
            checkpoint()
    return out


# ── track 2: optimisers ──────────────────────────────────────────────────────

def _eval_subset(d, n=6000, seed=42):
    """Stratified subsample of the training split for the surrogate objective."""
    rng = np.random.RandomState(seed)
    tr_pos = np.where(d["tr"])[0]
    y = d["y"][d["tr"]]
    f, nf = tr_pos[y == 1], tr_pos[y == 0]
    n_f = min(len(f), max(30, int(n * y.mean())))
    n_n = min(len(nf), n - n_f)
    sel = np.concatenate([rng.choice(f, n_f, replace=False),
                          rng.choice(nf, n_n, replace=False)])
    rng.shuffle(sel)
    return d["X"][sel], d["y"][sel]


def run_optimiser_track(d, seeds, pop, iters, epochs, batch_size, architecture,
                        checkpoint=None, sink=None, eval_mode="legacy",
                        surrogate_k=3):
    """
    Give every optimiser the same space, the same budget and the same fitness,
    then train and score the configuration each one returns.

    ⚠ `eval_mode` defaults to **"legacy"**, not to the repo default.  The two
    optimiser JSONs already on disk (`basepaper_optimisers_ulb.json`,
    `basepaper_optimisers_banksim.json`) were produced before OBJ-13's repair,
    so re-running on the new default would quietly replace them with numbers
    that mean something different under the same filename.  The mode is stamped
    into the output and into the draft; pass `--eval-mode deterministic` (or
    `averaged`) deliberately, to a different `--out`.
    """
    print(f"\n  --- OPTIMISER TRACK (tuning "
          f"{ARCH_LABELS.get(architecture, architecture)}, pop={pop} x {iters} "
          f"iters, filters {int(SEARCH_LB[0])}-{int(SEARCH_UB[0])}) ---", flush=True)
    X_opt, y_opt = _eval_subset(d)
    # The surrogate must see the same feature width as the final model, or the
    # search tunes a different problem from the one it hands over.  Every
    # column here is a real per-transaction feature (no PTC/NTC tail).
    n_raw = int(d["X"].shape[1])
    out = sink if sink is not None else {}
    for name, cls in OPTIMISERS.items():
        seed_records = []
        for sd in seeds:
            obj = _ADTCNObjective(X_opt, y_opt, random_state=sd,
                                  architecture=architecture, n_raw=n_raw,
                                  eval_mode=eval_mode, k_repeats=surrogate_k)
            t0 = time.time()
            opt = cls(objective_fn=obj, lb=SEARCH_LB, ub=SEARCH_UB,
                      n_pop=pop, max_iter=iters,
                      cfg={**DB_BOA_CONFIG, "population_size": pop,
                           "max_iterations": iters},
                      seed=sd)
            best_pos, best_fit, history = opt.optimise(verbose=False)
            search_s = time.time() - t0

            n_filt = max(8, int(round(float(best_pos[0]))))
            spe = max(20, int(round(float(best_pos[1]))))
            m = train_eval(architecture, d["X"], d["y"][d["tr"]], d["idx"][d["tr"]],
                           d["X"], d["y"][d["te"]], d["idx"][d["te"]],
                           epochs=epochs, n_filters=n_filt, seed=sd,
                           batch_size=batch_size)
            m["search_seconds"] = search_s
            m["n_filters"] = n_filt
            m["steps_per_epoch"] = spe
            m["best_fitness"] = float(best_fit)          # -Obf2 (minimised)
            m["obf2"] = float(-best_fit)
            m["n_evals"] = int(getattr(obj, "_call_count", 0))
            seed_records.append(m)

        c = aggregate(seed_records)
        obf2s = np.array([r["obf2"] for r in seed_records])
        # base paper Table 2 reports these five statistics of the cost function
        c["obf2_stats"] = {"best": float(obf2s.max()), "worst": float(obf2s.min()),
                           "median": float(np.median(obf2s)),
                           "mean": float(obf2s.mean()),
                           "std": float(obf2s.std(ddof=1)) if len(obf2s) > 1 else 0.0}
        c["search_seconds"] = float(np.mean([r["search_seconds"] for r in seed_records]))
        c["n_evals"] = int(np.mean([r["n_evals"] for r in seed_records]))
        c["chosen_filters"] = [r["n_filters"] for r in seed_records]
        c["chosen_steps_per_epoch"] = [r["steps_per_epoch"] for r in seed_records]
        out[name] = c
        tag = "  <- proposed" if name == "DB-BOA" else "  <- base paper"
        print(f"    {name:>7}-ADTCN  Obf2={c['obf2_stats']['mean']:.4f}"
              f" (best {c['obf2_stats']['best']:.4f})  "
              f"MCC={c['MCC']:+.4f} +/- {c['MCC_std']:.3f}  "
              f"filters={c['chosen_filters']}  "
              f"search={c['search_seconds']:.0f}s{tag}", flush=True)
        if checkpoint:
            checkpoint()
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def _guard_eval_mode(path, out_name, track, eval_mode):
    """
    Refuse to overwrite an optimiser JSON produced under a different surrogate
    protocol.  A file with no stamp predates OBJ-13 and is therefore `legacy`.
    """
    if track not in ("optimiser", "optimizer", "both") or not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            prev = json.load(fh)
    except Exception:
        return                       # unreadable: not our business to adjudicate
    if "optimiser_search_space" not in prev:
        return                       # classifier-only file, no surrogate involved
    prev_mode = prev["optimiser_search_space"].get("surrogate_eval_mode", "legacy")
    if prev_mode != eval_mode:
        raise SystemExit(
            f"\nREFUSING TO OVERWRITE {path}\n"
            f"  on disk : surrogate_eval_mode = {prev_mode!r}\n"
            f"  this run: surrogate_eval_mode = {eval_mode!r}\n"
            f"These are not the same experiment. Write the new one beside it:\n"
            f"  --out {os.path.splitext(out_name)[0]}_{eval_mode}.json\n")


def main(dataset="banksim", ordering=None, track="both", quick=False,
         epochs=None, n_filters=None, seeds=None, batch_size=2048,
         pop=None, iters=None, opt_arch="dilated_attn", out_name=None,
         eval_mode="legacy", surrogate_k=3):
    ordering = ordering or ("customer" if dataset == "banksim" else "time")
    epochs = epochs if epochs else (2 if quick else 10)
    n_filters = n_filters if n_filters else (16 if quick else 32)
    seeds = seeds if seeds else ([42] if quick else [42, 7, 123])
    pop = pop if pop else (5 if quick else 10)
    iters = iters if iters else (5 if quick else 8)
    out_name = out_name or f"basepaper_comparison_{dataset}.json"

    # OBJ-13 guard, checked BEFORE the loader runs so it costs seconds rather
    # than a data load.  The optimiser JSONs on disk were produced under
    # `legacy`; writing post-repair numbers over them under the same filename
    # would leave no way to tell the two apart, and the side-by-side comparison
    # IS the result.  Refuse rather than warn — a warning scrolls off the top of
    # a six-hour run.  (Same class of mistake as OBJ-15's unsuffixed figures.)
    _guard_eval_mode(os.path.join(RESULTS_DIR, out_name), out_name,
                     track, eval_mode)

    print("=" * 78, flush=True)
    print(f"  BASE-PAPER COMPARISON — dataset={dataset}  ordering={ordering}", flush=True)
    print(f"  epochs={epochs} filters={n_filters} seeds={seeds} batch={batch_size}",
          flush=True)
    print("  No number from Prabanand & Thanabal (2025) is reproduced here; their",
          flush=True)
    print("  baselines are re-implemented and re-run on this data (see D4).", flush=True)
    print("=" * 78, flush=True)

    t0 = time.time()
    d = prepare(dataset, ordering)

    summary = {
        "dataset": dataset,
        "ordering": ordering,
        "split_note": d["split_note"],
        "protocol": {"seq_len": SEQ_LEN, "epochs": epochs, "n_filters": n_filters,
                     "seeds": seeds, "batch_size": batch_size, "quick": quick,
                     "n_features": int(d["X"].shape[1]),
                     "n_train": int(d["tr"].sum()), "n_test": int(d["te"].sum()),
                     "train_fraud": int(d["y"][d["tr"]].sum()),
                     "test_fraud": int(d["y"][d["te"]].sum())},
        "provenance": "base-paper baselines re-implemented in "
                      "models/basepaper_models.py, algorithms/mbo.py, "
                      "algorithms/wsa.py and run here; the paper's own published "
                      "numbers are never quoted (divergence D4)",
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, out_name)

    def _checkpoint():
        """Flush whatever is finished so far; a crash costs one cell, not the run."""
        summary["checkpoint"] = {"complete": False, "written": time.strftime("%H:%M:%S")}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        os.replace(tmp, path)      # atomic — never leaves a half-written file

    if track in ("classifier", "both"):
        summary["base_paper_classifiers"] = BASEPAPER_CLASSIFIERS
        summary["classifiers"] = {}
        run_classifier_track(
            d, CLASSIFIER_TRACK, epochs, n_filters, seeds, batch_size,
            checkpoint=_checkpoint, sink=summary["classifiers"])

    if track in ("optimiser", "optimizer", "both"):
        summary["optimiser_search_space"] = {
            "filters": [float(SEARCH_LB[0]), float(SEARCH_UB[0])],
            "steps_per_epoch": [float(SEARCH_LB[1]), float(SEARCH_UB[1])],
            "population": pop, "iterations": iters,
            "fitness": "Obf2 = 2*MCC + Spec + Pre + NPV (bounded form)",
            "tuned_architecture": opt_arch,
            "budget_note": "population and iteration count are identical for all "
                           "five optimisers; the NUMBER OF OBJECTIVE EVALUATIONS "
                           "is not, because each algorithm calls the fitness a "
                           "different number of times per iteration. See "
                           "optimisers[*].n_evals.",
            # OBJ-13: which surrogate protocol produced these numbers.  Without
            # this stamp a legacy file and a post-repair file are indistinguishable
            # once they are on disk, and the whole point of keeping `legacy` alive
            # is that the two can be compared.
            "surrogate_eval_mode": eval_mode,
            "surrogate_k": surrogate_k if eval_mode == "averaged" else 1,
            "surrogate_note": (
                "legacy = pre-OBJ-13 objective (fresh unstratified split and "
                "torch seed per call, batch_size = max(32, n_train // spe) so "
                "the steps_per_epoch axis is dead). Numbers produced under "
                "'legacy' are a best-of-N over a noisy fitness, not a ranking."
                if eval_mode == "legacy" else
                "post-OBJ-13 repaired objective: stratified split, draws "
                "pre-drawn once and shared across candidates (common random "
                "numbers), batch_size = ceil(n_train / spe) so the "
                "steps_per_epoch axis is live."),
        }
        summary["optimisers"] = {}
        run_optimiser_track(
            d, seeds, pop, iters, epochs, batch_size, opt_arch,
            checkpoint=_checkpoint, sink=summary["optimisers"],
            eval_mode=eval_mode, surrogate_k=surrogate_k)

    summary["wall_clock_seconds"] = time.time() - t0
    summary["checkpoint"] = {"complete": True, "written": time.strftime("%H:%M:%S")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {path}", flush=True)
    write_draft(summary, dataset, out_name)
    return summary


def write_draft(s, dataset, out_name=None):
    L = [f"# Base-paper comparison — {dataset.upper()}\n",
         "_Auto-generated by `experiments/basepaper_comparison.py`._\n",
         "**Provenance.** Prabanand & Thanabal (2025) compare their DB-BOA-ADTCN "
         "against EfficientNet, ResNet, DenseNet and DTCN (classifiers) and "
         "MBO, WSA, DBOA and BOA (optimisers), in MATLAB, on their own data. "
         "None of their published numbers appear below. Every baseline is "
         "re-implemented in this repository and re-run on our data under our "
         "protocol, so the whole table is reproducible from "
         "`results/`.\n",
         f"**Protocol.** {s['split_note']}. "
         f"{s['protocol']['n_train']:,} training rows "
         f"({s['protocol']['train_fraud']:,} fraud), "
         f"{s['protocol']['n_test']:,} test rows "
         f"({s['protocol']['test_fraud']:,} fraud), "
         f"{s['protocol']['n_features']} features, window {s['protocol']['seq_len']}, "
         f"{s['protocol']['epochs']} epochs, {s['protocol']['n_filters']} filters, "
         f"seeds {s['protocol']['seeds']}.\n"]

    if "classifiers" in s:
        L += ["## Classifier comparison (base paper Table 4 equivalent)\n",
              "| Model | Source | MCC | Prec % | Rec % | FP | Params |",
              "|---|---|---|---|---|---|---|"]
        for a, c in sorted(s["classifiers"].items(), key=lambda kv: -kv[1]["MCC"]):
            src = "base paper" if a in s.get("base_paper_classifiers", []) else "ours"
            L.append(f"| {ARCH_LABELS.get(a, a)} | {src} | "
                     f"{c['MCC']:+.4f} ± {c['MCC_std']:.3f} | {c['Precision']:.1f} | "
                     f"{c['Sensitivity']:.1f} | {c['FP']:.0f} | {c['n_params']:,} |")
        L.append("")

    if "optimisers" in s:
        sp = s["optimiser_search_space"]
        opt = s["optimisers"]
        mccs = [c["MCC"] for c in opt.values()]
        spread = max(mccs) - min(mccs)
        worst_std = max(c["MCC_std"] for c in opt.values())
        evals = {k: c["n_evals"] for k, c in opt.items()}
        best_obf2 = max(opt, key=lambda k: opt[k]["obf2_stats"]["mean"])
        best_mcc = max(opt, key=lambda k: opt[k]["MCC"])

        L += ["## Optimiser comparison (base paper Tables 2-3 equivalent)\n",
              f"All five search the same space (filters "
              f"{int(sp['filters'][0])}-{int(sp['filters'][1])}, steps/epoch "
              f"{int(sp['steps_per_epoch'][0])}-{int(sp['steps_per_epoch'][1])}) "
              f"with the same population ({sp['population']}) and iteration "
              f"count ({sp['iterations']}), minimising the same fitness "
              f"({sp['fitness']}), tuning "
              f"{ARCH_LABELS.get(sp['tuned_architecture'], sp['tuned_architecture'])}.\n"]

        # OBJ-13: which surrogate protocol produced this table.  A file with no
        # stamp predates the repair and is therefore legacy — say so, rather
        # than let the table read as if the objective had always been sound.
        _mode = sp.get("surrogate_eval_mode", "legacy")
        if _mode == "legacy":
            L += [f"> ⚠ **Surrogate protocol: `legacy` — the pre-OBJ-13 "
                  f"objective.** Fitness redraws its 70/30 split and its torch "
                  f"seed on every call, so it is a *random function of its "
                  f"input*: one fixed configuration re-evaluated 25× on ULB "
                  f"spans Obf2 3.4497–5.0000 and touches the 5.0000 ceiling "
                  f"once **with no search involved**. `batch_size = max(32, "
                  f"n_train // spe)` also pins to 32 across the whole searched "
                  f"range, so the steps/epoch axis is dead and every "
                  f"steps/epoch optimum below is arbitrary. **Read the "
                  f"`Obf2 best` column as a best-of-N noise statistic, not as a "
                  f"ranking.** See `OBJECTIVE_noise_audit.md`.\n"]
        else:
            L += [f"> **Surrogate protocol: `{_mode}`** (post-OBJ-13 repair"
                  + (f", k={sp.get('surrogate_k', 1)} draws averaged per "
                     f"candidate, shared across candidates as common random "
                     f"numbers" if _mode == "averaged" else
                     ", split and torch seed frozen at construction so fitness "
                     "is a pure function of the candidate")
                  + "). The split is stratified and `batch_size = "
                    "ceil(n_train / spe)`, so the steps/epoch axis is live. "
                    "Not comparable cell-for-cell with a `legacy` table — the "
                    "objective being maximised is a different function.\n"]

        L += ["| Optimiser | Obf2 best | Obf2 mean ± std | Test MCC | Objective evals | Search s | Filters chosen |",
              "|---|---|---|---|---|---|---|"]
        for name, c in opt.items():
            o = c["obf2_stats"]
            L.append(f"| {name}-ADTCN | {o['best']:.4f} | {o['mean']:.4f} ± "
                     f"{o['std']:.4f} | {c['MCC']:+.4f} ± {c['MCC_std']:.3f} | "
                     f"{c['n_evals']} | {c['search_seconds']:.0f} | "
                     f"{c['chosen_filters']} |")
        L += ["",
              "### Three caveats that decide how this table may be read\n",
              f"**1. Equal population and iterations is NOT equal objective "
              f"evaluations.** Each algorithm calls the fitness a different "
              f"number of times per iteration by construction: "
              f"{', '.join(f'{k} {v}' for k, v in evals.items())}. That is a "
              f"{max(evals.values())/max(min(evals.values()),1):.1f}x range. The "
              f"budget is matched the way this literature matches it — population "
              f"and iteration count — not by evaluation count, and the counts are "
              f"printed above so the reader can judge rather than assume.\n",
              f"**2. The test-MCC differences are inside the noise.** The spread "
              f"across all five optimisers is {spread:.4f}, while the largest "
              f"per-optimiser spread across seeds is {worst_std:.4f} — larger than "
              f"the effect being claimed. On the metric that actually matters, "
              f"**these optimisers are not distinguishable from one another.** "
              f"{best_obf2}-ADTCN has the best mean search fitness and "
              f"{best_mcc}-ADTCN the best test MCC, and on this evidence neither "
              f"lead is meaningful.\n",
              "**3. Search wall-clock is confounded, not an efficiency measure.** "
              "The surrogate's cost scales with the filter count a candidate "
              "proposes, so an optimiser that explores wide-filter regions pays "
              "more per evaluation regardless of its own efficiency. Seconds-per-"
              "evaluation therefore varies several-fold across the table for "
              "reasons that have nothing to do with optimiser quality. It is "
              "reported for transparency and should not be quoted as a speed "
              "comparison — which is precisely what the base paper's Table 5 "
              "does.\n"]

    # The draft name is derived from the RESULTS FILE, not from the dataset.
    # It used to be f"BASEPAPER_comparison_{dataset}.md" for every run, so the
    # classifier track and the optimiser track — separate JSONs, separate
    # experiments — wrote to one filename and silently overwrote each other;
    # whichever ran last was the only one on disk.  Same failure as OBJ-15's
    # unsuffixed figures.  basepaper_comparison_ulb.json still maps to
    # BASEPAPER_comparison_ulb.md, so no existing reference breaks.
    stem = os.path.splitext(os.path.basename(out_name or
                                             f"basepaper_comparison_{dataset}.json"))[0]
    if stem.startswith("basepaper_"):
        stem = stem[len("basepaper_"):]
    out = os.path.abspath(os.path.join(
        ROOT, "..", "final_report_data", f"BASEPAPER_{stem}.md"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ulb", "banksim"], default="banksim")
    ap.add_argument("--ordering", default=None,
                    help="banksim: customer|global   ulb: time|random")
    ap.add_argument("--track", choices=["classifier", "optimiser", "both"],
                    default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--filters", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--opt-arch", default="dilated_attn")
    ap.add_argument("--eval-mode", choices=["legacy", "deterministic", "averaged"],
                    default="legacy",
                    help="DB-BOA surrogate protocol (OBJ-13). Defaults to "
                         "'legacy' — the pre-repair objective — so that "
                         "re-running reproduces the JSONs already on disk. Pass "
                         "a repaired mode deliberately, and to a new --out.")
    ap.add_argument("--surrogate-k", type=int, default=3,
                    help="draws averaged per candidate when --eval-mode averaged")
    ap.add_argument("--out", default=None)
    ap.add_argument("--redraft", action="store_true",
                    help="Re-render the markdown draft from the saved JSON "
                         "without re-running. Use after editing write_draft().")
    a = ap.parse_args()
    if a.redraft:
        name = a.out or f"basepaper_comparison_{a.dataset}.json"
        with open(os.path.join(RESULTS_DIR, name)) as f:
            saved = json.load(f)
        write_draft(saved, a.dataset, name)
        sys.exit(0)
    main(dataset=a.dataset, ordering=a.ordering, track=a.track, quick=a.quick,
         epochs=a.epochs, n_filters=a.filters,
         seeds=[int(s) for s in a.seeds.split(",")] if a.seeds else None,
         batch_size=a.batch, pop=a.pop, iters=a.iters, opt_arch=a.opt_arch,
         out_name=a.out, eval_mode=a.eval_mode, surrogate_k=a.surrogate_k)
