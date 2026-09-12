"""
experiments/handbook_temporal_grid.py
=====================================
OBJ-16 (d) — does entity linkage, and does ADTCN, exploit time at one-second
resolution?

The question
------------
OBJ-1 (`banksim_temporal_grid.py`) found that entity-linked windows help every
architecture on BankSim, and that ADTCN still ranks last of seven.  BankSim's
clock resolves to one day, so it could only half-answer whether any
architecture genuinely exploits *time*.  The Fraud Detection Handbook stamps
every transaction to the second and carries two entity links, not one:

    ordering  in {global, customer, terminal}   x   architecture in {CNN, LSTM,
                                                    DTCN, ADTCN, ResNet-1D,
                                                    DenseNet-1D, EfficientNet-1D}

`customer` groups each cardholder's own history (fraud-adjacency lift 13.35x);
`terminal` groups each point-of-sale terminal's (71.65x — the generator's
scenario 2 compromises a terminal for 28 days and is 62 % of all fraud).

Why this is a separate script
-----------------------------
The protocol is copied from the BankSim grid on purpose — same training helper
(`_seqtrain.train_eval`), same epochs / filters / seeds / batch, windows built
on the full ordered stream and then split, the same per-transaction reference —
so the two grids are comparable cell for cell.  It is not an edit of that
script because `banksim_temporal_grid.py` produced numbers already in the
report, and it must stay re-runnable byte for byte.

What differs, and only because the data differs
-----------------------------------------------
* Three orderings instead of two, so the verdict reports a gain per linked
  ordering plus the terminal-vs-customer comparison.
* The split is by day (train day < `val_day`, test day >= `split_day`; see
  HANDBOOK_CONFIG) where BankSim's was by step.
* `--resume`.  At ~1.9x BankSim's per-epoch cost and 1.5x its arms this grid
  runs ~16 h, so a crash must cost a cell, not the run.  Resuming is exact:
  `train_eval` seeds every cell itself, so a resumed cell is bit-identical to
  an uninterrupted one at the same thread count.  A saved file produced under
  a different protocol is refused rather than mixed.

Pre-registration: TASK.md, OBJ-16, "PRE-REGISTRATION — Handbook", item (d).
This script reports **counts, not verdicts against thresholds** — the scoring
is done in TASK.md, against text written before the run.

Usage
-----
    python experiments/handbook_temporal_grid.py --quick        # smoke test
    python experiments/handbook_temporal_grid.py --resume       # full run (restartable)
    python experiments/handbook_temporal_grid.py --redraft      # markdown from saved JSON
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import RESULTS_DIR, HANDBOOK_CONFIG, DATASETS
from data.handbook_loader import load_handbook_frame, encode_handbook, order_rows
from models.adtcn import ARCH_LABELS, SEQ_LEN
from experiments._seqtrain import window_index, train_eval, aggregate
from experiments._dataset import environment

DEFAULT_ARCHS = ["cnn", "lstm", "dtcn", "dilated_attn",
                 "resnet", "densenet", "efficientnet"]
ORDERINGS = ["global", "customer", "terminal"]
LINKED = ["customer", "terminal"]
GROUP_COL = {"customer": "CUSTOMER_ID", "terminal": "TERMINAL_ID"}
DEFAULT_OUT = "handbook_temporal_grid.json"
QUICK_OUT = "handbook_temporal_grid_quick.json"
DRAFT_PATH = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                          "OBJ16_handbook_temporal_grid.md"))


def prepare(ordering, cfg, df, verbose=True):
    """
    Build one arm: rows ordered for `ordering`, scaled on the training period
    only, plus the window index matrix and the split masks.  Mirrors
    `banksim_temporal_grid.prepare` step for step; only the column names and
    the day-based split differ.
    """
    order = order_rows(df, ordering, seed=cfg["order_seed"])
    d = df.iloc[order].reset_index(drop=True)

    X, names = encode_handbook(d, cfg)
    y = d["TX_FRAUD"].values.astype(int)
    day = d["TX_TIME_DAYS"].values
    groups = d[GROUP_COL[ordering]].values if ordering in GROUP_COL else None

    tr = day < cfg["val_day"]
    te = day >= cfg["split_day"]

    scaler = StandardScaler().fit(X[tr])
    X = scaler.transform(X).astype(np.float32)

    idx = window_index(len(y), SEQ_LEN, groups=groups)

    # diagnostic: how much does the previous row in THIS ordering tell us?
    # Same definition as the BankSim grid (self-padding at a group's first row
    # included), so the two datasets' figures are comparable.
    prev = y[idx[:, -2]]
    cond = float(y[prev == 1].mean()) if (prev == 1).any() else float("nan")

    if verbose:
        print(f"  [{ordering:>8}] rows={len(y):,}  features={X.shape[1]}  "
              f"P(fraud | prev row fraud)={cond:.4f}  base={y.mean():.4f}",
              flush=True)

    return {"X": X, "y": y, "idx": idx, "tr": tr, "te": te,
            "cond_fraud": cond, "n_features": X.shape[1],
            "feature_names": names}


def _protocol(cfg, epochs, n_filters, seeds, batch_size, n_features, quick):
    """Everything a resumed cell must share with the cells already on disk."""
    return {
        "split": (f"temporal past->future; train day<{cfg['val_day']}, "
                  f"test day>={cfg['split_day']}"),
        "seq_len": SEQ_LEN,
        "epochs": epochs, "n_filters": n_filters, "seeds": list(seeds),
        "batch_size": batch_size, "n_features": int(n_features),
        "orderings": list(ORDERINGS), "order_seed": cfg["order_seed"],
        "windows_built_on_full_stream_then_split": True,
        "within_second_tie_break": "seeded shuffle (never raw file order)",
        "entity_derived_row_features": False,
        "quick": bool(quick),
    }


def verdict(cells, archs):
    """
    Counts the pre-registration is scored against — deliberately no thresholds.

    Two questions, kept apart exactly as in the BankSim grid:
      Q1  Do entity-linked windows carry signal that global windows do not?
          Answered by the gain columns, across ALL architectures, per ordering.
      Q2  Is ADTCN the architecture to exploit it?  Answered by its rank, and
          by ADTCN - DTCN (its own no-attention ablation) per ordering.
    """
    def mcc(a, o):
        return cells[f"{a}|{o}"]["MCC"]

    gains = {o: {a: mcc(a, o) - mcc(a, "global") for a in archs} for o in LINKED}
    ranked = {o: sorted(archs, key=lambda a: -mcc(a, o)) for o in ORDERINGS}
    out = {
        "gains_mcc": gains,
        "n_architectures": len(archs),
        "n_positive_gains": {o: int(sum(v > 0 for v in gains[o].values()))
                             for o in LINKED},
        "gain_range": {o: [min(gains[o].values()), max(gains[o].values())]
                       for o in LINKED},
        "terminal_gain_exceeds_customer_gain": [
            a for a in archs if gains["terminal"][a] > gains["customer"][a]],
        "ranking": {o: [[a, mcc(a, o)] for a in ranked[o]] for o in ORDERINGS},
        "best": {o: ranked[o][0] for o in ORDERINGS},
    }
    if "dilated_attn" in archs:
        out["adtcn_rank"] = {o: ranked[o].index("dilated_attn") + 1
                             for o in ORDERINGS}
        if "dtcn" in archs:
            out["adtcn_minus_dtcn"] = {o: mcc("dilated_attn", o) - mcc("dtcn", o)
                                       for o in ORDERINGS}
        if "cnn" in archs:
            out["adtcn_minus_cnn"] = {o: mcc("dilated_attn", o) - mcc("cnn", o)
                                      for o in ORDERINGS}
    return out


def main(quick=False, epochs=None, seeds=None, archs=None, n_filters=None,
         batch_size=2048, threads=None, out_name=None, resume=False,
         skip_reference=False):
    if threads:
        torch.set_num_threads(threads)
    epochs = epochs if epochs else (2 if quick else 10)
    n_filters = n_filters if n_filters else (16 if quick else 32)
    seeds = seeds if seeds else ([42] if quick else [42, 7, 123])
    archs = archs or (["cnn", "dilated_attn"] if quick else DEFAULT_ARCHS)
    out_name = out_name or (QUICK_OUT if quick else DEFAULT_OUT)
    cfg = dict(HANDBOOK_CONFIG)

    print("=" * 78, flush=True)
    print("  OBJ-16 (d) — HANDBOOK ENTITY-LINKED TEMPORAL GRID", flush=True)
    print(f"  ordering x architecture   |  epochs={epochs}  filters={n_filters}  "
          f"seeds={seeds}  batch={batch_size}  threads={torch.get_num_threads()}",
          flush=True)
    print("=" * 78, flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, out_name)
    if os.path.exists(path) and not resume:
        raise SystemExit(f"{path} already exists. Pass --resume to continue it, "
                         f"or --out to write elsewhere — this script never "
                         f"overwrites a grid silently.")

    df = load_handbook_frame(cfg["dataset_path"], cfg["raw_dir"], verbose=True)
    arms = {o: prepare(o, cfg, df) for o in ORDERINGS}
    del df

    # integrity check: the arms must differ ONLY in window composition
    g = arms["global"]
    for o in LINKED:
        a = arms[o]
        assert a["tr"].sum() == g["tr"].sum() and a["te"].sum() == g["te"].sum(), \
            f"split sizes differ between global and {o}"
        assert int(a["y"][a["tr"]].sum()) == int(g["y"][g["tr"]].sum()), \
            f"training fraud counts differ between global and {o}"
        assert int(a["y"][a["te"]].sum()) == int(g["y"][g["te"]].sum()), \
            f"test fraud counts differ between global and {o}"
    print(f"  integrity OK: all three arms train on {int(g['tr'].sum()):,} rows "
          f"({int(g['y'][g['tr']].sum()):,} fraud) and test on "
          f"{int(g['te'].sum()):,} rows ({int(g['y'][g['te']].sum()):,} fraud)\n",
          flush=True)

    protocol = _protocol(cfg, epochs, n_filters, seeds, batch_size,
                         g["n_features"], quick)
    cells, reference, elapsed_before = {}, None, 0.0
    if resume and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved.get("protocol") != protocol:
            raise SystemExit(
                f"--resume: {path} was produced under a different protocol; "
                f"refusing to mix cells.\n  saved: {saved.get('protocol')}\n"
                f"  now:   {protocol}")
        if saved.get("complete"):
            print(f"  {path} is already complete — nothing to resume.", flush=True)
            return saved
        cells = saved.get("cells", {})
        reference = saved.get("reference_per_transaction_no_window")
        elapsed_before = float(saved.get("wall_clock_seconds_so_far", 0.0))
        print(f"  resuming: {len(cells)} cell(s) already on disk"
              f"{', reference done' if reference else ''}", flush=True)

    t_start = time.time()
    cond = {o: arms[o]["cond_fraud"] for o in ORDERINGS}
    cond["base_rate"] = float(g["y"].mean())

    def _save(complete=False, extra=None):
        """Temp file + rename, so the JSON on disk is never half-written."""
        doc = {
            "objective": "OBJ-16 (d) — Fraud Detection Handbook entity-linked "
                         "temporal grid",
            "dataset": DATASETS["handbook"]["label"],
            "protocol": protocol,
            "archs": list(archs),
            "complete": complete,
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_clock_seconds_so_far": elapsed_before + (time.time() - t_start),
            "environment": environment(),
            "conditional_fraud_probability": cond,
            "reference_per_transaction_no_window": reference,
            "cells": cells,
        }
        if extra:
            doc.update(extra)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, path)
        return doc

    # per-transaction reference (seq_len = 1): identical in every arm, so once.
    # --skip-reference leaves it out of a part file: when the grid is split one
    # architecture per job, exactly one part carries it and `merge` refuses two.
    if reference is None and not skip_reference:
        print("  --- reference: per-transaction, no window (seq_len=1) ---",
              flush=True)
        idx1 = window_index(len(g["y"]), 1, groups=None)
        ref_runs = []
        for sd in seeds:
            # the FULL row matrix is passed; only labels and window indices are
            # subset (window indices address full-stream positions)
            ref_runs.append(train_eval(
                "cnn", g["X"], g["y"][g["tr"]], idx1[g["tr"]],
                g["X"], g["y"][g["te"]], idx1[g["te"]],
                epochs=epochs, n_filters=n_filters, seed=sd,
                batch_size=batch_size))
        reference = aggregate(ref_runs)
        _save()
    if reference is None:
        print("      reference: skipped (--skip-reference; another part carries it)\n",
              flush=True)
    else:
        print(f"      reference MCC={reference['MCC']:+.4f} +/- "
              f"{reference['MCC_std']:.3f}  Prec={reference['Precision']:.1f}  "
              f"Rec={reference['Sensitivity']:.1f}\n", flush=True)

    for arch in archs:
        for ordering in ORDERINGS:
            key = f"{arch}|{ordering}"
            label = ARCH_LABELS.get(arch, arch)
            if key in cells:
                print(f"  {label:>16} | {ordering:>8}-window  (on disk — skipped)",
                      flush=True)
                continue
            d = arms[ordering]
            runs = []
            for sd in seeds:
                runs.append(train_eval(
                    arch,
                    d["X"], d["y"][d["tr"]], d["idx"][d["tr"]],
                    d["X"], d["y"][d["te"]], d["idx"][d["te"]],
                    epochs=epochs, n_filters=n_filters, seed=sd,
                    batch_size=batch_size))
            c = aggregate(runs)
            cells[key] = c
            _save()
            print(f"  {label:>16} | {ordering:>8}-window  "
                  f"MCC={c['MCC']:+.4f} +/- {c['MCC_std']:.3f}  "
                  f"Prec={c['Precision']:5.1f}  Rec={c['Sensitivity']:5.1f}  "
                  f"FP={c['FP']:.0f}  ({c['train_seconds']:.0f}s/seed)", flush=True)

    missing = [f"{a}|{o}" for a in archs for o in ORDERINGS if f"{a}|{o}" not in cells]
    if missing:
        raise RuntimeError(f"cells missing after the loop: {missing}")

    v = verdict(cells, archs)
    doc = _save(complete=True, extra={
        "verdict": v,
        "wall_clock_seconds": elapsed_before + (time.time() - t_start)})

    print("\n" + "-" * 78, flush=True)
    for o in LINKED:
        print(f"  {o:>8}-linked beats global for {v['n_positive_gains'][o]} of "
              f"{v['n_architectures']} architectures  "
              f"(gain {v['gain_range'][o][0]:+.4f} to {v['gain_range'][o][1]:+.4f})",
              flush=True)
    print(f"  terminal gain > customer gain for "
          f"{len(v['terminal_gain_exceeds_customer_gain'])} of "
          f"{v['n_architectures']}", flush=True)
    if "adtcn_rank" in v:
        print(f"  ADTCN rank  global/customer/terminal: "
              f"{v['adtcn_rank']['global']}/{v['adtcn_rank']['customer']}/"
              f"{v['adtcn_rank']['terminal']} of {v['n_architectures']}", flush=True)
    print("-" * 78, flush=True)
    print(f"\n  wrote {path}", flush=True)
    # A part file has neither the reference nor every architecture, so it cannot
    # render the draft: `merge` writes it once the parts are assembled.
    if not quick and reference is not None and set(archs) == set(DEFAULT_ARCHS):
        write_draft(doc)
    return doc


def merge(allow_partial=False):
    """
    Assemble per-architecture part files into the one grid JSON.

    The parts come from the Kaggle split of 2026-09-12: one job per
    architecture, three orderings each.  The refusals mirror the entity grid's
    merge -- an incomplete part, a part from a different protocol, a cell
    claimed twice, or two parts carrying the reference each stop the merge
    rather than produce a grid that only looks whole.  The result is written
    where `--resume` looks for it and marked complete, so the laptop runner's
    grid step finds the work already done and returns in seconds.
    """
    import glob
    target = os.path.join(RESULTS_DIR, DEFAULT_OUT)
    stem = DEFAULT_OUT[:-len(".json")]
    parts = sorted(p for p in glob.glob(os.path.join(RESULTS_DIR, f"{stem}_*.json"))
                   if os.path.basename(p) != QUICK_OUT)
    if not parts:
        raise SystemExit(f"no {stem}_<arch>.json part files to merge")
    protocol = cond = dataset = reference = None
    cells, envs = {}, {}
    for p in parts:
        name = os.path.basename(p)
        with open(p, encoding="utf-8") as fh:
            s = json.load(fh)
        if not s.get("complete"):
            raise SystemExit(f"{name} is incomplete")
        if protocol is None:
            protocol, cond, dataset = (s["protocol"],
                                       s["conditional_fraud_probability"],
                                       s["dataset"])
        elif s["protocol"] != protocol:
            raise SystemExit(f"{name} ran a different protocol; refusing to merge")
        for k, c in s["cells"].items():
            if k in cells:
                raise SystemExit(f"cell {k} appears in two parts")
            cells[k] = c
        if s.get("reference_per_transaction_no_window"):
            if reference is not None:
                raise SystemExit("two parts carry the per-transaction reference")
            reference = s["reference_per_transaction_no_window"]
        envs[name] = s.get("environment")
    archs = [a for a in DEFAULT_ARCHS if all(f"{a}|{o}" in cells for o in ORDERINGS)]
    missing = [a for a in DEFAULT_ARCHS if a not in archs]
    if (missing or reference is None) and not allow_partial:
        raise SystemExit(f"incomplete grid: missing architectures {missing}, "
                         f"reference {'present' if reference else 'MISSING'}")
    # The cells were computed wherever the parts ran, not here: report that
    # environment, and keep the merging machine's separately.
    distinct = {json.dumps(e, sort_keys=True) for e in envs.values() if e}
    env = json.loads(distinct.pop()) if len(distinct) == 1 else {"mixed": envs}
    doc = {
        "objective": "OBJ-16 (d) — Fraud Detection Handbook entity-linked "
                     "temporal grid",
        "dataset": dataset,
        "protocol": protocol,
        "archs": archs,
        "complete": not missing and reference is not None,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env,
        "merged_from": envs,
        "merged_on": environment(),
        "conditional_fraud_probability": cond,
        "reference_per_transaction_no_window": reference,
        "cells": cells,
        "verdict": verdict(cells, archs),
    }
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, target)
    if doc["complete"]:
        write_draft(doc)
    print(f"merged {len(parts)} part(s) -> {os.path.basename(target)}"
          f"{'' if doc['complete'] else '  (PARTIAL)'}", flush=True)
    return doc


def write_draft(s):
    """Report-ready markdown (drafts-before-tex).  Counts only — scored in TASK.md."""
    c, v, archs = s["cells"], s["verdict"], s["archs"]
    cond = s["conditional_fraud_probability"]
    base = cond["base_rate"]
    L = [
        "# OBJ-16 (d) — Fraud Detection Handbook entity-linked temporal grid\n",
        "_Auto-generated by `experiments/handbook_temporal_grid.py`. Every number "
        "here was produced by that script. It reports counts; the verdict against "
        "the pre-registration (TASK.md, OBJ-16) is scored there, not here._\n",
        f"**Dataset.** {s['dataset']}. Split: {s['protocol']['split']}. "
        f"{s['protocol']['n_features']} per-row features, none entity-derived, so "
        f"the three arms differ only in which nine rows precede each transaction.\n",
        f"**Protocol.** {s['protocol']['epochs']} epochs, "
        f"{s['protocol']['n_filters']} filters, batch {s['protocol']['batch_size']}, "
        f"seeds {s['protocol']['seeds']} — identical to the BankSim grid "
        f"(OBJ-1). Environment: torch {s['environment'].get('torch')}, "
        f"{s['environment'].get('torch_threads')} threads.\n",
        "## Why the arms differ\n",
        "| Ordering | P(fraud \\| previous row fraud) | Lift over base rate |",
        "|---|---|---|",
    ]
    for o in ORDERINGS:
        L.append(f"| {o} | {cond[o]:.4f} | {cond[o] / base:.2f}x |")
    L += [f"\nBase rate {base:.4f}.\n", "## Grid (test MCC, mean ± sd over seeds)\n",
          "| Model | Global | Customer | Terminal | Customer gain | Terminal gain |",
          "|---|---|---|---|---|---|"]
    for a in archs:
        gl, cu, te = (c[f"{a}|{o}"] for o in ORDERINGS)
        L.append(f"| {ARCH_LABELS.get(a, a)} | {gl['MCC']:+.4f} ± {gl['MCC_std']:.3f} "
                 f"| {cu['MCC']:+.4f} ± {cu['MCC_std']:.3f} "
                 f"| {te['MCC']:+.4f} ± {te['MCC_std']:.3f} "
                 f"| **{v['gains_mcc']['customer'][a]:+.4f}** "
                 f"| **{v['gains_mcc']['terminal'][a]:+.4f}** |")
    r = s["reference_per_transaction_no_window"]
    L.append(f"| _per-transaction reference (no window)_ | {r['MCC']:+.4f} ± "
             f"{r['MCC_std']:.3f} | — | — | — | — |")
    L += ["", "## Full metrics\n",
          "| Model | Ordering | MCC | Prec % | Rec % | FP | Params |",
          "|---|---|---|---|---|---|---|"]
    for a in archs:
        for o in ORDERINGS:
            k = c[f"{a}|{o}"]
            L.append(f"| {ARCH_LABELS.get(a, a)} | {o} | {k['MCC']:+.4f} | "
                     f"{k['Precision']:.1f} | {k['Sensitivity']:.1f} | "
                     f"{k['FP']:.0f} | {k['n_params']:,} |")
    L += ["", "## Counts the pre-registration is scored against\n"]
    for o in LINKED:
        L.append(f"- **{o}-linked vs global:** positive for "
                 f"{v['n_positive_gains'][o]} of {v['n_architectures']} "
                 f"architectures; gain {v['gain_range'][o][0]:+.4f} to "
                 f"{v['gain_range'][o][1]:+.4f}.")
    tg = v["terminal_gain_exceeds_customer_gain"]
    L.append(f"- **Terminal gain > customer gain:** {len(tg)} of "
             f"{v['n_architectures']} ({', '.join(ARCH_LABELS.get(a, a) for a in tg) or 'none'}).")
    for o in ORDERINGS:
        rk = " · ".join(f"{ARCH_LABELS.get(a, a)} {m:.4f}" for a, m in v["ranking"][o])
        L.append(f"- **Ranking, {o} windows:** {rk}.")
    if "adtcn_rank" in v:
        L.append(f"- **ADTCN rank** (global / customer / terminal): "
                 f"{v['adtcn_rank']['global']} / {v['adtcn_rank']['customer']} / "
                 f"{v['adtcn_rank']['terminal']} of {v['n_architectures']}.")
    if "adtcn_minus_dtcn" in v:
        d = v["adtcn_minus_dtcn"]
        L.append(f"- **ADTCN − DTCN** (the attention, isolated): global "
                 f"{d['global']:+.4f}, customer {d['customer']:+.4f}, terminal "
                 f"{d['terminal']:+.4f}.")
    L += ["", "## Scope limits\n",
          "One simulated dataset; three seeds; one shared hyperparameter budget "
          "for every architecture, so this is an equal-budget comparison, not a "
          "per-architecture-tuned one; a fixed 0.5 decision threshold. The "
          "per-row features are deliberately thinner than the Handbook's own "
          "published baseline (no entity aggregates, which would contaminate the "
          "global-vs-linked comparison), so **absolute MCC is not comparable to "
          "the Handbook's published numbers and must never be compared to them.**\n"]
    os.makedirs(os.path.dirname(DRAFT_PATH), exist_ok=True)
    with open(DRAFT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {DRAFT_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--filters", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--archs", type=str, default=None,
                    help=f"comma-separated; default {','.join(DEFAULT_ARCHS)}")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help=f"default {DEFAULT_OUT} ({QUICK_OUT} with --quick)")
    ap.add_argument("--resume", action="store_true",
                    help="continue a partial run of the same protocol")
    ap.add_argument("--redraft", action="store_true",
                    help="re-render the markdown draft from the saved JSON, "
                         "without training anything")
    ap.add_argument("--skip-reference", action="store_true",
                    help="do not compute the per-transaction reference -- for a "
                         "part file, so exactly one part carries it")
    ap.add_argument("--merge", action="store_true",
                    help=f"assemble {DEFAULT_OUT[:-5]}_<arch>.json part files "
                         f"into {DEFAULT_OUT}")
    ap.add_argument("--allow-partial", action="store_true",
                    help="--merge: accept a grid missing architectures or the "
                         "reference (marked incomplete)")
    a = ap.parse_args()
    if a.merge:
        merge(allow_partial=a.allow_partial)
        sys.exit(0)
    if a.redraft:
        with open(os.path.join(RESULTS_DIR, a.out or DEFAULT_OUT),
                  encoding="utf-8") as f:
            saved = json.load(f)
        if not saved.get("complete"):
            raise SystemExit("--redraft needs a complete grid")
        write_draft(saved)
        sys.exit(0)
    main(quick=a.quick, epochs=a.epochs, n_filters=a.filters,
         seeds=[int(s) for s in a.seeds.split(",")] if a.seeds else None,
         archs=a.archs.split(",") if a.archs else None,
         batch_size=a.batch, threads=a.threads, out_name=a.out, resume=a.resume,
         skip_reference=a.skip_reference)
