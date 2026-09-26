"""
experiments/check_detector_repro.py
===================================
Does the ULB reproduction failure reach the detector track?
(TASK.md follow-up A; WORK_REPORT open question Q2.)

Why this exists
---------------
OBJ-17 found that ULB results generated before 2026-09-01 do not reproduce:
13 of 18 private-incentive cells moved at identical seeds, cause unexplained.
Only that one sweep was ever re-checked.  Every stored `detector_multiseed`
run was trained on 2026-08-31 — inside the window — so before any new arm is
paired against them (OBJ-13 pre-registration 5), they have to be shown to
still reproduce in today's environment.  Pairing a new run against a stored
one that no longer reproduces would compare two environments, not two
configurations.

How
---
The re-runs are produced by

    detector_multiseed.py --run --config <cfg> --seed 42 --threads 2 \\
                          --no-save --emit-json

(`--no-save`, so a re-run can never overwrite the record it is checked
against) with stdout captured to a log.  This script reads the JSON line each
log ends with and compares it to the stored record at the same
(config, seed, threads): bitwise on the trained-weight hash, and on the test
confusion counts and MCC.  Training is bitwise-reproducible at a fixed thread
count (`_determinism.json`), so anything short of an identical hash is a
change in conditions, not noise.

Usage
-----
    python experiments/check_detector_repro.py --logs LOG [LOG ...]
"""

import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")
RUN_DIR = os.path.join(RESULTS_DIR, "_multiseed_runs")
OUT_PATH = os.path.join(RESULTS_DIR, "detector_repro_check.json")


def _last_json_line(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip().startswith("{")]
    if not lines:
        raise SystemExit(f"{path}: no JSON record — did the run finish? "
                         f"(it must be launched with --emit-json)")
    return json.loads(lines[-1])


def compare(log_path):
    now = _last_json_line(log_path)
    cfg, seed, thr = now["config"], now["seed"], now["threads"]
    stored_path = os.path.join(RUN_DIR, f"{cfg}_seed{seed}_t{thr}.json")
    if not os.path.exists(stored_path):
        raise SystemExit(f"no stored record to compare against: {stored_path}")
    with open(stored_path, encoding="utf-8") as fh:
        old = json.load(fh)

    counts = ("TP", "FP", "FN", "TN")
    row = {
        "config": cfg, "seed": seed, "threads": thr,
        "stored_record": os.path.relpath(stored_path, ROOT).replace("\\", "/"),
        "stored_finished": old.get("finished"),
        "stored_torch": old.get("torch_version"),
        "now_torch": now.get("torch_version"),
        "weight_hash_stored": old["weight_sha256_16"],
        "weight_hash_now": now["weight_sha256_16"],
        "weights_bitwise_identical": now["weight_sha256_16"] == old["weight_sha256_16"],
        "MCC_stored": old["metrics"]["MCC"],
        "MCC_now": now["metrics"]["MCC"],
        "MCC_delta": now["metrics"]["MCC"] - old["metrics"]["MCC"],
        "confusion_identical": all(int(now["metrics"][k]) == int(old["metrics"][k])
                                   for k in counts),
        "confusion_stored": {k: int(old["metrics"][k]) for k in counts},
        "confusion_now": {k: int(now["metrics"][k]) for k in counts},
        "secs_now": now.get("secs"),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="stdout logs of --no-save --emit-json re-runs")
    args = ap.parse_args()

    rows = [compare(p) for p in args.logs]
    all_same = all(r["weights_bitwise_identical"] for r in rows)
    out = {
        "task": "Does the ULB reproduction failure (OBJ-17) reach the detector "
                "track? Re-run stored 2026-08-31 detector_multiseed records "
                "in today's environment and compare.",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "all_bitwise_identical": all_same,
        "verdict": ("REPRODUCED — the stored detector runs are valid in today's "
                    "environment, so new arms may be paired against them"
                    if all_same else
                    "DID NOT REPRODUCE — the stored detector runs sit on a "
                    "condition that no longer exists; any new arm must be paired "
                    "only against arms re-run in the same environment"),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    for r in rows:
        mark = "IDENTICAL" if r["weights_bitwise_identical"] else "DIFFERENT"
        print(f"  {r['config']:>18} seed {r['seed']} @ {r['threads']}t  "
              f"weights {mark}  MCC stored {r['MCC_stored']:.4f} "
              f"now {r['MCC_now']:.4f}  (delta {r['MCC_delta']:+.4f})", flush=True)
    print(f"  verdict: {out['verdict']}", flush=True)
    print(f"[SAVE] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
