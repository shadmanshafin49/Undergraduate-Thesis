"""
experiments/detector_multiseed.py
=================================
OBJ-2 (kill the MCC contradiction) + OBJ-4 (multi-seed the detector).

The problem
-----------
Two result files report the *same* ADTCN configuration on the *same* ULB split
(142 filters / 76 steps-per-epoch / 30 epochs, test n=56,962, 98 fraud) with
irreconcilable numbers:

    results/db_boa_results.json     MCC 0.677   Acc 99.85 %   FP  70
    results/dbboa_vs_default.json   MCC 0.313   Acc 98.77 %   FP 688

and `dbboa_vs_default.json` claims in its own note to "cross-check
db_boa_results.json (MCC 0.677)".  It does not.  This script settles which
number — if either — is reproducible, and replaces the single-seed headline
with mean ± std over several training seeds.

What was already established (mode `--determinism`, recorded in the output JSON)
-------------------------------------------------------------------------------
ADTCN training is bitwise-reproducible at a *fixed* CPU thread count: two runs
of the tuned config at 4 threads produced identical weight hashes.  Changing the
thread count changes the float reduction order and therefore the trained
weights.  `main.py` leaves torch at its default thread count; the older
`experiments/dbboa_vs_default.py` calls `torch.set_num_threads(os.cpu_count())`.
The two contradicting JSONs were therefore never running the same arithmetic —
which is a mechanism for the divergence, not an excuse for it.  A configuration
whose reported MCC moves from 0.677 to 0.313 because of a thread count is a
configuration with no stable operating point, and that is what the seed sweep
below measures.

Design
------
* The train/val/test split is held fixed (`DATA_CONFIG["random_state"] = 42`),
  exactly the split both original runs used, so the only thing varying is the
  training seed — model init and mini-batch order.  This isolates optimisation
  variance from split variance.
* Thread count is pinned and recorded for every run.
* Both configurations get identical treatment: same data, same seeds, same
  budget.  No configuration gets a seed the other does not.

Runs are one-per-process and each writes its own JSON as soon as it finishes, so
the sweep is restartable and a crashed worker costs one run rather than all of
them.

Usage
-----
    python experiments/detector_multiseed.py --build-cache
    python experiments/detector_multiseed.py --run --config tuned --seed 42 --threads 2
    python experiments/detector_multiseed.py --determinism
    python experiments/detector_multiseed.py --collect
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
RUN_DIR     = os.path.join(RESULTS_DIR, "_multiseed_runs")
CACHE_PATH  = os.path.join(RESULTS_DIR, "_multiseed_runs", "ulb_split_cache.npz")
OUT_PATH    = os.path.join(RESULTS_DIR, "detector_multiseed.json")

# The two configurations under test.  `hidden_neurons` is the conv filter count.
CONFIGS = {
    # ⚠ OBJ-13: this configuration was chosen by the PRE-REPAIR surrogate, whose
    # fitness was a random function of its input — `db_boa_stats` in
    # db_boa_results.json maxes at -4.999999999991326, the Obf2 ceiling, which is
    # reachable by redrawing alone with no search involved.  The +0.047 MCC /
    # p=0.34 tie this script measured is therefore a tie against a
    # *noise-selected* config.  That does not make the tie wrong (rule 3 — it
    # stands as measured), but a post-repair configuration belongs beside it,
    # which is what EXTRA_CONFIGS below is for.
    "dbboa_tuned"     : {"hidden_neurons": 142, "steps_per_epoch": 76},
    "hand_set_default": {"hidden_neurons": 128, "steps_per_epoch": 150},
}

# Extra arms, registered from a file rather than by editing this list.
#
# The post-repair configuration is not known until `obj13_surrogate_repair.py`
# has run, and hardcoding it afterwards would put a number in the source that no
# script in the repo produced — rule 2.  Instead the search writes it here and
# every consumer (`--config`, `collect()`, the draft) picks it up automatically,
# so the arm and its provenance travel together.
#
#   results/_multiseed_runs/extra_configs.json
#   {"dbboa_repaired_averaged": {"hidden_neurons": 40, "steps_per_epoch": 88,
#                                "source": "obj13_surrogate_repair_banksim.json"}}
EXTRA_CONFIGS_PATH = os.path.join(RESULTS_DIR, "_multiseed_runs",
                                  "extra_configs.json")
if os.path.exists(EXTRA_CONFIGS_PATH):
    with open(EXTRA_CONFIGS_PATH, encoding="utf-8") as _fh:
        for _name, _spec in json.load(_fh).items():
            if _name in CONFIGS:
                raise SystemExit(
                    f"{EXTRA_CONFIGS_PATH} redefines built-in config {_name!r}. "
                    f"Rename it — silently shadowing an arm would make two "
                    f"different configurations share one label in the output.")
            CONFIGS[_name] = {"hidden_neurons": int(_spec["hidden_neurons"]),
                              "steps_per_epoch": int(_spec["steps_per_epoch"])}
            if _spec.get("source"):
                CONFIGS[_name]["source"] = _spec["source"]

# Thread count the seed sweep is pinned to.  Runs at other thread counts are
# treated as reproduction attempts, not as extra seeds (see collect()).
PRIMARY_THREADS = 2

# Which historical file each reproduction thread count is trying to reproduce.
REPRO_TARGET = {
    4: ("db_boa_results.json",   "main.py leaves torch at its 4-thread default"),
    8: ("dbboa_vs_default.json", "dbboa_vs_default.py calls torch.set_num_threads(os.cpu_count()) = 8"),
}

# What the two contradicting files claimed for these configs, carried into the
# output so the comparison is legible without opening three JSONs.
HISTORICAL = {
    "dbboa_tuned": {
        "db_boa_results.json"  : {"MCC": 0.6772, "Accuracy": 99.8508, "FP": 70,  "TP": 83, "FN": 15},
        "dbboa_vs_default.json": {"MCC": 0.3133, "Accuracy": 98.7729, "FP": 688, "TP": 87, "FN": 11},
    },
    "hand_set_default": {
        "dbboa_vs_default.json": {"MCC": 0.7849, "Accuracy": 99.9192, "FP": 31,  "TP": 83, "FN": 15},
    },
}


def _set_threads(n: int):
    """Must run before torch does any work, hence before the torch import."""
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)


# ─── data cache ───────────────────────────────────────────────────────────────

def build_cache(verbose: bool = True):
    """
    Run the real ULB loader once and cache what the detector actually consumes.

    `ADTCN._make_sequences` reads only the leading `n_raw` (=33) columns — the
    268 PTC/NTC columns are engineered but never fed to the network — so caching
    the leading block is numerically identical to caching all 301 columns, and
    lets four workers hold the data at once inside a 6 GB budget.  Scaling is
    per-column (StandardScaler), so slicing after the scaler is exact.
    """
    import numpy as np
    from data.data_loader import FinancialDataLoader

    os.makedirs(RUN_DIR, exist_ok=True)
    t0 = time.time()
    loader = FinancialDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=verbose)

    n_raw = min(X_train.shape[1], 33)
    X_full = np.vstack([X_train, X_val])[:, :n_raw].astype(np.float32)
    y_full = np.concatenate([y_train, y_val])
    X_test = X_test[:, :n_raw].astype(np.float32)

    np.savez_compressed(CACHE_PATH, X_full=X_full, y_full=y_full,
                        X_test=X_test, y_test=y_test, n_raw=n_raw)
    if verbose:
        print(f"[CACHE] {CACHE_PATH}", flush=True)
        print(f"[CACHE] train+val={X_full.shape}  test={X_test.shape}  "
              f"test fraud={int(y_test.sum())}  n_raw={n_raw}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return CACHE_PATH


def _load_cache():
    import numpy as np
    if not os.path.exists(CACHE_PATH):
        build_cache()
    d = np.load(CACHE_PATH)
    return d["X_full"], d["y_full"], d["X_test"], d["y_test"], int(d["n_raw"])


# ─── a single training run ────────────────────────────────────────────────────

def _weight_hash(model) -> str:
    return hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in model.parameters())
    ).hexdigest()[:16]


def run_one(config_name: str, seed: int, threads: int, epochs: int,
            save: bool = True) -> dict:
    import numpy as np
    import torch
    torch.set_num_threads(threads)

    from config        import ADTCN_CONFIG
    from models.adtcn  import ADTCN
    from utils.metrics import compute_all_metrics

    X_full, y_full, X_test, y_test, n_raw = _load_cache()
    params = CONFIGS[config_name]

    # `random_state` is what ADTCN.fit feeds to torch.manual_seed — model init
    # and mini-batch order.  The data split stays at 42 (baked into the cache).
    cfg = dict(ADTCN_CONFIG)
    cfg["random_state"]   = seed
    cfg["epoch_count"]    = epochs
    cfg["n_raw_features"] = n_raw

    print(f"[RUN] {config_name} seed={seed} threads={threads} "
          f"filters={params['hidden_neurons']} spe={params['steps_per_epoch']} "
          f"epochs={epochs}", flush=True)

    t0 = time.time()
    m = ADTCN(cfg=cfg)
    m.optimal_params = {"hidden_neurons" : params["hidden_neurons"],
                        "epoch_count"    : epochs,
                        "steps_per_epoch": params["steps_per_epoch"]}
    m.fit(X_full, y_full, verbose=False)
    met = compute_all_metrics(y_test, m.predict(X_test))
    secs = time.time() - t0

    rec = {
        "config"        : config_name,
        "seed"          : int(seed),
        "threads"       : int(torch.get_num_threads()),
        "epochs"        : int(epochs),
        "hidden_neurons": params["hidden_neurons"],
        "steps_per_epoch": params["steps_per_epoch"],
        "architecture"  : cfg.get("architecture", "cnn"),
        "n_raw_features": n_raw,
        "split_random_state": 42,
        "test_n"        : int(len(y_test)),
        "test_fraud"    : int(y_test.sum()),
        "metrics"       : {k: float(v) for k, v in met.items()
                           if isinstance(v, (int, float, np.floating))},
        "weight_sha256_16": _weight_hash(m.model),
        "torch_version" : torch.__version__,
        "secs"          : round(secs, 1),
        "finished"      : datetime.datetime.now().isoformat(timespec="seconds"),
    }

    print(f"[RUN] {config_name} seed={seed}  MCC={met['MCC']:.4f}  "
          f"Acc={met['Accuracy']:.4f}%  TP={int(met['TP'])} FP={int(met['FP'])} "
          f"FN={int(met['FN'])}  ({secs/60:.1f} min)", flush=True)

    if save:
        os.makedirs(RUN_DIR, exist_ok=True)
        path = os.path.join(RUN_DIR, f"{config_name}_seed{seed}_t{threads}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"[SAVE] {path}", flush=True)
    return rec


# ─── determinism probe ────────────────────────────────────────────────────────

def determinism_probe(epochs: int = 2) -> dict:
    """
    Reproduces the finding recorded in this module's docstring, cheaply.

    Run in-process for the repeat (same thread count) and as a subprocess for
    the differing thread count, because torch's thread count cannot be changed
    once the pool has been used.
    """
    import subprocess
    here = os.path.abspath(__file__)
    out = {}
    for tag, threads in [("a1_t4", 4), ("a2_t4_repeat", 4), ("a3_t8", 8)]:
        proc = subprocess.run(
            [sys.executable, here, "--run", "--config", "dbboa_tuned",
             "--seed", "42", "--threads", str(threads), "--epochs", str(epochs),
             "--no-save", "--emit-json"],
            capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(proc.stdout, proc.stderr, flush=True)
            raise RuntimeError(f"probe {tag} produced no result")
        rec = json.loads(line[-1])
        out[tag] = {"threads": rec["threads"], "MCC": rec["metrics"]["MCC"],
                    "TP": rec["metrics"]["TP"], "FP": rec["metrics"]["FP"],
                    "FN": rec["metrics"]["FN"],
                    "weight_sha256_16": rec["weight_sha256_16"]}
        print(f"[PROBE] {tag}: {out[tag]}", flush=True)

    out["epochs"] = epochs
    out["deterministic_at_fixed_thread_count"] = (
        out["a1_t4"]["weight_sha256_16"] == out["a2_t4_repeat"]["weight_sha256_16"])
    out["thread_count_changes_weights"] = (
        out["a1_t4"]["weight_sha256_16"] != out["a3_t8"]["weight_sha256_16"])
    out["note"] = (
        "Identical weight hashes across a repeat at the same thread count show "
        "training is bitwise-reproducible; a different hash at a different "
        "thread count shows the float reduction order — not the seed — differs "
        "between main.py (torch default threads) and the older "
        "experiments/dbboa_vs_default.py (torch.set_num_threads(cpu_count()))."
    )
    return out


# ─── collection ───────────────────────────────────────────────────────────────

def collect() -> dict:
    import numpy as np

    all_runs = []
    if os.path.isdir(RUN_DIR):
        for fn in sorted(os.listdir(RUN_DIR)):
            if fn.endswith(".json") and not fn.startswith("_"):
                with open(os.path.join(RUN_DIR, fn)) as f:
                    all_runs.append(json.load(f))
    all_runs = [r for r in all_runs if r.get("epochs", 0) >= 30]

    # The seed sweep is run at one pinned thread count.  Runs at any *other*
    # thread count are reproduction attempts against the two historical files
    # (main.py used torch's 4-thread default; dbboa_vs_default.py forced 8), and
    # must not be pooled into the seed statistics — thread count changes the
    # float reduction order and so is a second, confounded factor.
    runs  = [r for r in all_runs if r["threads"] == PRIMARY_THREADS]
    repro = [r for r in all_runs if r["threads"] != PRIMARY_THREADS]

    summary = {}
    for cname in CONFIGS:
        rs = sorted([r for r in runs if r["config"] == cname], key=lambda r: r["seed"])
        if not rs:
            continue
        entry = {
            "hidden_neurons" : CONFIGS[cname]["hidden_neurons"],
            "steps_per_epoch": CONFIGS[cname]["steps_per_epoch"],
            "n_seeds"        : len(rs),
            "seeds"          : [r["seed"] for r in rs],
            "per_seed"       : {str(r["seed"]): {
                                    "MCC"     : round(r["metrics"]["MCC"], 4),
                                    "Accuracy": round(r["metrics"]["Accuracy"], 4),
                                    "Precision": round(r["metrics"]["Precision"], 4),
                                    "Sensitivity": round(r["metrics"]["Sensitivity"], 4),
                                    "TP": int(r["metrics"]["TP"]),
                                    "FP": int(r["metrics"]["FP"]),
                                    "FN": int(r["metrics"]["FN"]),
                                } for r in rs},
        }
        for metric in ("MCC", "Accuracy", "Precision", "Sensitivity", "F1_Score", "FP"):
            vals = np.array([r["metrics"][metric] for r in rs], dtype=float)
            entry[metric] = {
                "mean"  : round(float(vals.mean()), 4),
                "std"   : round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
                "min"   : round(float(vals.min()), 4),
                "max"   : round(float(vals.max()), 4),
                "spread": round(float(vals.max() - vals.min()), 4),
            }
        entry["historical_claims"] = HISTORICAL.get(cname, {})
        summary[cname] = entry

    # Reproduction attempts: same seed (42) as both historical runs, but at the
    # thread count each historical script actually used.  Training is bitwise
    # deterministic at a fixed thread count, so a matching run should land on
    # the historical number exactly if nothing else has changed.
    repro_out = []
    for r in sorted(repro, key=lambda r: (r["threads"], r["config"])):
        target_file, why = REPRO_TARGET.get(r["threads"], ("(unknown)", ""))
        claimed = HISTORICAL.get(r["config"], {}).get(target_file)
        got = round(r["metrics"]["MCC"], 4)
        entry = {"config": r["config"], "seed": r["seed"], "threads": r["threads"],
                 "reproducing": target_file, "why_this_thread_count": why,
                 "claimed_MCC": claimed["MCC"] if claimed else None,
                 "observed_MCC": got,
                 "observed_FP": int(r["metrics"]["FP"]),
                 "claimed_FP": claimed["FP"] if claimed else None}
        if claimed:
            entry["delta_MCC"] = round(got - claimed["MCC"], 4)
            entry["reproduced"] = bool(abs(got - claimed["MCC"]) < 0.005)
        repro_out.append(entry)

    out = {
        "task": "OBJ-2/OBJ-4 — reproducibility and seed variance of the ADTCN "
                "detector on ULB (DB-BOA-tuned vs hand-set default)",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": "ULB creditcard.csv (284,807 tx, 492 fraud)",
        "split": {"test_size": 0.20, "val_size": 0.10, "stratified": True,
                  "random_state": 42,
                  "note": "split held fixed across all runs; only the training "
                          "seed (torch.manual_seed -> init + batch order) varies"},
        "epochs": 30,
        "architecture": "cnn",
        "sweep_threads": PRIMARY_THREADS,
        "n_runs": len(runs),
        "summary": summary,
        "reproduction_attempts": repro_out,
        "runs": all_runs,
    }

    probe_path = os.path.join(RUN_DIR, "_determinism.json")
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            out["determinism_probe"] = json.load(f)

    # Verdict on the contradiction, computed rather than asserted.
    if "dbboa_tuned" in summary:
        t = summary["dbboa_tuned"]["MCC"]
        lo, hi = t["min"], t["max"]
        claims = HISTORICAL["dbboa_tuned"]
        out["obj2_verdict"] = {
            "tuned_mcc_range_observed": [lo, hi],
            "db_boa_results_0677_within_range"  : bool(lo <= claims["db_boa_results.json"]["MCC"]   <= hi),
            "dbboa_vs_default_0313_within_range": bool(lo <= claims["dbboa_vs_default.json"]["MCC"] <= hi),
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] {OUT_PATH}  ({len(runs)} runs)", flush=True)

    for cname, e in summary.items():
        print(f"\n  {cname}  (F={e['hidden_neurons']}, spe={e['steps_per_epoch']}, "
              f"n={e['n_seeds']} seeds)", flush=True)
        print(f"    MCC       {e['MCC']['mean']:.4f} ± {e['MCC']['std']:.4f}   "
              f"[{e['MCC']['min']:.4f}, {e['MCC']['max']:.4f}]", flush=True)
        print(f"    Accuracy  {e['Accuracy']['mean']:.4f} ± {e['Accuracy']['std']:.4f} %", flush=True)
        print(f"    FP        {e['FP']['mean']:.1f} ± {e['FP']['std']:.1f}", flush=True)

    if repro_out:
        print("\n  Reproduction attempts (seed 42, at each historical script's "
              "own thread count):", flush=True)
        for e in repro_out:
            mark = "REPRODUCED" if e.get("reproduced") else "DID NOT REPRODUCE"
            print(f"    {e['config']:>16} @ {e['threads']}t vs {e['reproducing']:<22} "
                  f"claimed {e['claimed_MCC']}  observed {e['observed_MCC']}  "
                  f"-> {mark}", flush=True)
    return out


# ─── cli ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--run",         action="store_true")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--collect",     action="store_true")
    ap.add_argument("--config",  choices=list(CONFIGS), default="dbboa_tuned")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs",  type=int, default=30)
    ap.add_argument("--no-save",   action="store_true")
    ap.add_argument("--emit-json", action="store_true",
                    help="print the run record as one JSON line (for probes)")
    args = ap.parse_args()

    _set_threads(args.threads)

    if args.build_cache:
        build_cache()
    elif args.run:
        rec = run_one(args.config, args.seed, args.threads, args.epochs,
                      save=not args.no_save)
        if args.emit_json:
            print(json.dumps(rec), flush=True)
    elif args.determinism:
        out = determinism_probe()
        os.makedirs(RUN_DIR, exist_ok=True)
        path = os.path.join(RUN_DIR, "_determinism.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[SAVE] {path}", flush=True)
    elif args.collect:
        collect()
    else:
        ap.error("pick one of --build-cache / --run / --determinism / --collect")


if __name__ == "__main__":
    main()
