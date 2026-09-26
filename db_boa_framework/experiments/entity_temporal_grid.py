"""
experiments/entity_temporal_grid.py
===================================
The OBJ-1 / OBJ-16 (d) temporal grid for the datasets that run on Kaggle
(PaySim now; AMLSim once its loader lands), split per architecture so every
Kaggle job fits one 12 h session, then merged into one result.

The protocol is copied, not reinvented: `_seqtrain.train_eval`, the same
epochs / filters / seeds / batch, windows built on the full ordered stream and
then split, the same per-transaction reference as `banksim_temporal_grid.py`
and `handbook_temporal_grid.py`.  Only the dataset adapter differs — which
orderings exist, which column links them, where the temporal cuts fall.  The
Handbook keeps its own script because it is already approved and queued.

A dataset's loader module provides LABEL, TIME, GROUP_COLS, load_frame(cfg),
encode(df, cfg) and order_rows(df, mode, seed); its config provides the two
temporal cuts named in the adapter.

The per-part files report cells only.  `--merge` checks that every part ran
the same protocol, that each cell appears exactly once and that the reference
exists, then writes the merged JSON with counts (not verdicts — the scoring is
done in TASK.md against the pre-registration) and a markdown draft.

Usage
-----
    python experiments/entity_temporal_grid.py --dataset paysim --archs cnn --with-reference --out paysim_temporal_grid_cnn.json
    python experiments/entity_temporal_grid.py --dataset paysim --archs lstm --out paysim_temporal_grid_lstm.json
    python experiments/entity_temporal_grid.py --merge paysim
"""

import argparse
import glob
import importlib
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

import config                                                           # noqa: E402
from config import RESULTS_DIR, DATASETS                                # noqa: E402
from models.adtcn import ARCH_LABELS, SEQ_LEN                           # noqa: E402
from experiments._seqtrain import window_index, train_eval, aggregate   # noqa: E402
from experiments._dataset import environment                            # noqa: E402

DEFAULT_ARCHS = ["cnn", "lstm", "dtcn", "dilated_attn",
                 "resnet", "densenet", "efficientnet"]

ADAPTERS = {
    "paysim": {"module": "data.paysim_loader", "config": "PAYSIM_CONFIG",
               "orderings": ["global", "receiver"], "cuts": ("val_step", "split_step"),
               "caveat": "the global arm is NOT a no-signal control on PaySim: "
                         "time order carries a 151.6x fraud-clustering artefact "
                         "(data/paysim_loader.py, design decision 4)"},
    "amlsim": {"module": "data.amlsim_loader", "config": "AMLSIM_CONFIG",
               "orderings": ["global", "sender", "receiver"], "cuts": ("val_day", "split_day"),
               "caveat": "the label is laundering (is_sar), not fraud; per-row features "
                         "sit at the floor, so any signal is relational"},
}


def _adapter(dataset):
    if dataset not in ADAPTERS:
        raise SystemExit(f"no grid adapter for {dataset!r}; have {sorted(ADAPTERS)}")
    ad = ADAPTERS[dataset]
    return ad, importlib.import_module(ad["module"]), dict(getattr(config, ad["config"]))


def prepare(ordering, cfg, df, mod, cuts, verbose=True):
    """One arm — the same steps as the BankSim/Handbook grids' `prepare`."""
    order = mod.order_rows(df, ordering, seed=cfg["order_seed"])
    d = df.iloc[order].reset_index(drop=True)
    X, names = mod.encode(d, cfg)
    y = d[mod.LABEL].to_numpy().astype(int)
    t = d[mod.TIME].to_numpy()
    groups = d[mod.GROUP_COLS[ordering]].to_numpy() if ordering in mod.GROUP_COLS else None
    tr, te = t < cfg[cuts[0]], t >= cfg[cuts[1]]
    X = StandardScaler().fit(X[tr]).transform(X).astype(np.float32)
    idx = window_index(len(y), SEQ_LEN, groups=groups)
    prev = y[idx[:, -2]]
    cond = float(y[prev == 1].mean()) if (prev == 1).any() else float("nan")
    if verbose:
        print(f"  [{ordering:>8}] rows={len(y):,}  features={X.shape[1]}  "
              f"P(fraud | prev row fraud)={cond:.4f}  base={y.mean():.4f}", flush=True)
    return {"X": X, "y": y, "idx": idx, "tr": tr, "te": te, "cond_fraud": cond,
            "n_features": X.shape[1]}


def run(dataset, archs, with_reference, out_name, quick=False, epochs=None,
        seeds=None, n_filters=None, batch_size=2048, threads=None, resume=False):
    if threads:
        torch.set_num_threads(threads)
    ad, mod, cfg = _adapter(dataset)
    epochs = epochs or (2 if quick else 10)
    n_filters = n_filters or (16 if quick else 32)
    seeds = seeds or ([42] if quick else [42, 7, 123])
    os.makedirs(RESULTS_DIR, exist_ok=True)     # a fresh checkout (or a Kaggle copy) may lack it
    path = os.path.join(RESULTS_DIR, out_name)
    if os.path.exists(path) and not resume:
        raise SystemExit(f"{path} exists; pass --resume, or --out elsewhere")

    print("=" * 78, flush=True)
    print(f"  ENTITY TEMPORAL GRID — {dataset}   archs={archs}   reference={with_reference}", flush=True)
    print(f"  epochs={epochs}  filters={n_filters}  seeds={seeds}  batch={batch_size}  "
          f"threads={torch.get_num_threads()}", flush=True)
    print("=" * 78, flush=True)

    df = mod.load_frame(cfg, verbose=True)
    orderings = ad["orderings"]
    arms = {o: prepare(o, cfg, df, mod, ad["cuts"]) for o in orderings}
    del df
    g = arms[orderings[0]]
    for o in orderings[1:]:
        a = arms[o]
        assert a["tr"].sum() == g["tr"].sum() and a["te"].sum() == g["te"].sum(), \
            f"split sizes differ between {orderings[0]} and {o}"
        assert int(a["y"][a["tr"]].sum()) == int(g["y"][g["tr"]].sum()) and \
            int(a["y"][a["te"]].sum()) == int(g["y"][g["te"]].sum()), \
            f"fraud counts differ between {orderings[0]} and {o}"
    print(f"  integrity OK: every arm trains on {int(g['tr'].sum()):,} rows "
          f"({int(g['y'][g['tr']].sum()):,} fraud), tests on {int(g['te'].sum()):,} "
          f"({int(g['y'][g['te']].sum()):,} fraud)\n", flush=True)

    protocol = {"dataset": dataset, "orderings": orderings,
                "split": f"temporal; train {ad['cuts'][0]}<{cfg[ad['cuts'][0]]}, "
                         f"test {ad['cuts'][1]}>={cfg[ad['cuts'][1]]}",
                "seq_len": SEQ_LEN, "epochs": epochs, "n_filters": n_filters,
                "seeds": list(seeds), "batch_size": batch_size,
                "n_features": int(g["n_features"]), "order_seed": cfg["order_seed"],
                "windows_built_on_full_stream_then_split": True,
                "entity_derived_row_features": False, "quick": bool(quick)}
    cells, reference = {}, None
    if resume and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved.get("protocol") != protocol:
            raise SystemExit(f"--resume: {path} has a different protocol; refusing to mix")
        if saved.get("complete"):
            print("  already complete", flush=True)
            return saved
        cells, reference = saved.get("cells", {}), saved.get("reference_per_transaction_no_window")

    cond = {o: arms[o]["cond_fraud"] for o in orderings}
    cond["base_rate"] = float(g["y"].mean())
    t0 = time.time()

    def _save(complete=False):
        doc = {"objective": f"entity temporal grid — {DATASETS[dataset]['label']}",
               "protocol": protocol, "archs": list(archs), "complete": complete,
               "caveat": ad.get("caveat"), "environment": environment(),
               "written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seconds_this_process": round(time.time() - t0, 1),
               "conditional_fraud_probability": cond,
               "reference_per_transaction_no_window": reference, "cells": cells}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, path)
        return doc

    if with_reference and reference is None:
        print("  --- reference: per-transaction, no window (seq_len=1) ---", flush=True)
        idx1 = window_index(len(g["y"]), 1, groups=None)
        reference = aggregate([train_eval(
            "cnn", g["X"], g["y"][g["tr"]], idx1[g["tr"]], g["X"], g["y"][g["te"]],
            idx1[g["te"]], epochs=epochs, n_filters=n_filters, seed=sd,
            batch_size=batch_size) for sd in seeds])
        _save()
        print(f"      reference MCC={reference['MCC']:+.4f} +/- {reference['MCC_std']:.3f}", flush=True)

    for arch in archs:
        for o in orderings:
            key = f"{arch}|{o}"
            if key in cells:
                continue
            d = arms[o]
            c = aggregate([train_eval(
                arch, d["X"], d["y"][d["tr"]], d["idx"][d["tr"]], d["X"],
                d["y"][d["te"]], d["idx"][d["te"]], epochs=epochs, n_filters=n_filters,
                seed=sd, batch_size=batch_size) for sd in seeds])
            cells[key] = c
            _save()
            print(f"  {ARCH_LABELS.get(arch, arch):>16} | {o:>8}-window  MCC={c['MCC']:+.4f} "
                  f"+/- {c['MCC_std']:.3f}  ({c['train_seconds']:.0f}s/seed)", flush=True)
    doc = _save(complete=True)
    print(f"\n  wrote {path}", flush=True)
    return doc


def verdict(cells, archs, orderings):
    """Counts only; scored against the pre-registration in TASK.md."""
    base, linked = orderings[0], orderings[1:]
    mcc = lambda a, o: cells[f"{a}|{o}"]["MCC"]                          # noqa: E731
    gains = {o: {a: mcc(a, o) - mcc(a, base) for a in archs} for o in linked}
    ranked = {o: sorted(archs, key=lambda a: -mcc(a, o)) for o in orderings}
    v = {"gains_mcc_vs_" + base: gains,
         "n_positive_gains": {o: int(sum(x > 0 for x in gains[o].values())) for o in linked},
         "n_architectures": len(archs),
         "ranking": {o: [[a, mcc(a, o)] for a in ranked[o]] for o in orderings},
         "best": {o: ranked[o][0] for o in orderings}}
    if "dilated_attn" in archs:
        v["adtcn_rank"] = {o: ranked[o].index("dilated_attn") + 1 for o in orderings}
        for other in ("dtcn", "cnn"):
            if other in archs:
                v[f"adtcn_minus_{other}"] = {o: mcc("dilated_attn", o) - mcc(other, o)
                                             for o in orderings}
    return v


def merge(dataset, allow_partial=False):
    ad, _, _ = _adapter(dataset)
    target = f"{dataset}_temporal_grid.json"
    parts = sorted(p for p in glob.glob(os.path.join(RESULTS_DIR, f"{dataset}_temporal_grid_*.json"))
                   if not p.endswith("_quick.json"))
    if not parts:
        raise SystemExit("no part files to merge")
    protocol, cells, reference, envs, cond = None, {}, None, {}, None
    for p in parts:
        with open(p, encoding="utf-8") as fh:
            s = json.load(fh)
        if not s.get("complete"):
            raise SystemExit(f"{p} is incomplete")
        if protocol is None:
            protocol, cond = s["protocol"], s["conditional_fraud_probability"]
        elif s["protocol"] != protocol:
            raise SystemExit(f"{p} ran a different protocol; refusing to merge")
        for k, c in s["cells"].items():
            if k in cells:
                raise SystemExit(f"cell {k} appears in two parts")
            cells[k] = c
        if s.get("reference_per_transaction_no_window"):
            if reference is not None:
                raise SystemExit("two parts carry a reference")
            reference = s["reference_per_transaction_no_window"]
        envs[os.path.basename(p)] = s.get("environment")
    archs = [a for a in DEFAULT_ARCHS if all(f"{a}|{o}" in cells for o in ad["orderings"])]
    missing = [a for a in DEFAULT_ARCHS if a not in archs]
    if (missing or reference is None) and not allow_partial:
        raise SystemExit(f"incomplete grid: missing archs {missing}, reference "
                         f"{'present' if reference else 'MISSING'}")
    out = {"objective": f"entity temporal grid — {DATASETS[dataset]['label']}",
           "protocol": protocol, "archs": archs, "complete": not missing and reference is not None,
           "caveat": ad.get("caveat"), "parts": envs,
           "conditional_fraud_probability": cond,
           "reference_per_transaction_no_window": reference, "cells": cells,
           "verdict": verdict(cells, archs, ad["orderings"])}
    with open(os.path.join(RESULTS_DIR, target), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    write_draft(dataset, out)
    print(f"merged {len(parts)} part(s) -> {target}", flush=True)
    return out


def write_draft(dataset, s):
    c, v, archs, ords = s["cells"], s["verdict"], s["archs"], s["protocol"]["orderings"]
    base = ords[0]
    L = [f"# Entity temporal grid — {dataset}\n",
         "_Auto-generated by `experiments/entity_temporal_grid.py --merge`. Counts only; "
         "scored against the pre-registration in TASK.md._\n",
         f"**Protocol.** {s['protocol']['split']}; {s['protocol']['epochs']} epochs, "
         f"{s['protocol']['n_filters']} filters, seeds {s['protocol']['seeds']} — identical "
         f"to the BankSim and Handbook grids.\n"]
    if s.get("caveat"):
        L.append(f"⚠ **Caveat fixed before the run:** {s['caveat']}.\n")
    cond = s["conditional_fraud_probability"]
    L += ["| Ordering | P(fraud \\| previous row fraud) | Lift |", "|---|---|---|"]
    L += [f"| {o} | {cond[o]:.4f} | {cond[o] / cond['base_rate']:.2f}x |" for o in ords]
    L += ["", "| Model | " + " | ".join(ords) + " | " + " | ".join(f"{o} gain" for o in ords[1:]) + " |",
          "|---" * (1 + len(ords) + len(ords) - 1) + "|"]
    for a in archs:
        row = [f"{c[f'{a}|{o}']['MCC']:+.4f} ± {c[f'{a}|{o}']['MCC_std']:.3f}" for o in ords]
        row += [f"**{v['gains_mcc_vs_' + base][o][a]:+.4f}**" for o in ords[1:]]
        L.append(f"| {ARCH_LABELS.get(a, a)} | " + " | ".join(row) + " |")
    r = s.get("reference_per_transaction_no_window")
    if r:
        L.append(f"| _per-transaction reference_ | {r['MCC']:+.4f} ± {r['MCC_std']:.3f} |"
                 + " — |" * (len(ords) - 1 + len(ords) - 1))
    L += ["", "## Counts\n"]
    for o in ords[1:]:
        L.append(f"- **{o} vs {base}:** positive for {v['n_positive_gains'][o]} of {v['n_architectures']}.")
    for o in ords:
        L.append(f"- **Ranking, {o}:** " + " · ".join(
            f"{ARCH_LABELS.get(a, a)} {m:.4f}" for a, m in v["ranking"][o]) + ".")
    if "adtcn_rank" in v:
        L.append("- **ADTCN rank:** " + ", ".join(f"{o} {v['adtcn_rank'][o]}" for o in ords) + ".")
    if "adtcn_minus_dtcn" in v:
        L.append("- **ADTCN − DTCN:** " + ", ".join(f"{o} {v['adtcn_minus_dtcn'][o]:+.4f}" for o in ords) + ".")
    path = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                        f"OBJ16_{dataset}_temporal_grid.md"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  wrote draft -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--archs", type=str, default=None)
    ap.add_argument("--with-reference", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--filters", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge", metavar="DATASET")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge, allow_partial=a.allow_partial)
        sys.exit(0)
    if not a.dataset:
        ap.error("--dataset is required unless --merge")
    archs = a.archs.split(",") if a.archs else (["cnn"] if a.quick else DEFAULT_ARCHS)
    out = a.out or f"{a.dataset}_temporal_grid_{'quick' if a.quick else '_'.join(archs)}.json"
    run(a.dataset, archs, a.with_reference, out, quick=a.quick, epochs=a.epochs,
        seeds=[int(s) for s in a.seeds.split(",")] if a.seeds else None,
        n_filters=a.filters, batch_size=a.batch, threads=a.threads, resume=a.resume)
