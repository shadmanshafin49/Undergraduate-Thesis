r"""
experiments/obj13_shipped_path_check.py
=======================================
What does the surrogate look like on the path that actually SHIPS?

Why this exists
---------------
`objective_noise_audit.py` measures the surrogate built from
`basepaper_comparison._eval_subset(d, n=6000)`.  The deployed pipeline does not
use that.  `main.py:144` calls `loader.get_eval_subset(X_train, y_train)`, which
returns a **stratified `eval_subset` = 3,000-row** pool (`config.py:31`), and
`_ADTCNObjective` then subsamples its 2,000 rows from *that*.

Those two pools do not contain the same number of unique fraud rows, and the
difference lands squarely on this branch in `_ADTCNObjective.__init__`:

    n_f       = max(self._MIN_FRAUD_ROWS, int(n_rows * fraud_rate))
    replace_f = len(fraud_idx) < n_f      # sample WITH REPLACEMENT
    idx       = rng.choice(fraud_idx, n_f, replace=replace_f)

If the pool holds fewer than 30 unique fraud rows, the surrogate's 30 "fraud
rows" are a handful of unique transactions **repeated**.  The unstratified 70/30
split then puts copies of the same transaction on both sides, so the surrogate is
validated on rows it memorised — and `Obf2 = 5.0000` stops meaning "classified
nine rare rows correctly" and starts meaning "recognised five rows it had already
seen".

This script does not assume that; it measures it, for both datasets, on both
pool constructions, with no training at all (seconds).

    python experiments/obj13_shipped_path_check.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import RESULTS_DIR, DATA_CONFIG, BANKSIM_CONFIG
from models.adtcn import _ADTCNObjective

MIN_FRAUD = _ADTCNObjective._MIN_FRAUD_ROWS
SURR_ROWS = _ADTCNObjective._SURROGATE_ROWS


def describe(label, y_pool, n_rows=SURR_ROWS):
    """Replay __init__'s fraud-selection arithmetic on a given pool."""
    n_pool = len(y_pool)
    n_fraud_unique = int(np.sum(y_pool == 1))
    rate = n_fraud_unique / max(n_pool, 1)
    n_f = max(MIN_FRAUD, int(n_rows * rate))
    replace = n_fraud_unique < n_f
    # Expected DISTINCT fraud rows after sampling n_f from n_fraud_unique with
    # replacement: n*(1 - (1-1/n)^k).  Without replacement it is just n_f.
    if replace and n_fraud_unique > 0:
        distinct = n_fraud_unique * (1 - (1 - 1 / n_fraud_unique) ** n_f)
    else:
        distinct = float(n_f)
    return {
        "pool_label": label,
        "pool_rows": n_pool,
        "pool_fraud_unique": n_fraud_unique,
        "pool_fraud_rate": rate,
        "surrogate_n_f": n_f,
        "samples_with_replacement": bool(replace),
        "expected_distinct_fraud_in_surrogate": round(float(distinct), 2),
        "duplication_factor": round(n_f / max(distinct, 1e-9), 2),
        "approx_val_fraud_rows": int(round(n_f * 0.3)),
    }



# ── stage 2: does the duplication actually leak across the split? ────────────

def _replay_subsample(y_pool, random_state, n_rows=SURR_ROWS):
    """
    Reproduce `_ADTCNObjective.__init__`'s index construction exactly, so each
    surrogate row can be traced back to the POOL row it came from.

    The objective keeps only `y_sub`, not `idx`, so the identity information
    needed to detect duplicates is not recoverable from the object.  Replaying
    is safe only if it is verified: the caller checks `y_sub` against the live
    object's `.y` element-wise before trusting anything below.
    """
    rng = np.random.RandomState(random_state)
    fraud_idx = np.where(y_pool == 1)[0]
    normal_idx = np.where(y_pool == 0)[0]
    fraud_rate = len(fraud_idx) / (len(fraud_idx) + len(normal_idx))
    n_f = max(MIN_FRAUD, int(n_rows * fraud_rate))
    n_n = min(len(normal_idx), n_rows - n_f)
    replace_f = len(fraud_idx) < n_f
    idx = np.concatenate([
        rng.choice(fraud_idx, n_f, replace=replace_f),
        rng.choice(normal_idx, n_n, replace=False),
    ])
    rng.shuffle(idx)
    return idx, y_pool[idx]


def measure_leak(label, X_pool, y_pool, mode, seed=42, n_raw=None):
    """
    Build the real objective in `mode`, take the draw it would score on, and
    count how many validation fraud ROWS are copies of a transaction that is
    also in the training half.
    """
    obj = _ADTCNObjective(X_pool, y_pool, random_state=seed,
                          architecture="dilated_attn", n_raw=n_raw,
                          eval_mode=mode, k_repeats=3)
    idx, y_sub = _replay_subsample(y_pool, seed)

    # Verify the replay before drawing any conclusion from it.
    exact = bool(len(y_sub) == len(obj.y) and np.array_equal(y_sub, obj.y))

    tr, vl, _ = (obj._legacy_draw() if mode == "legacy" else obj._draws[0])
    tr_f = idx[tr][y_sub[tr] == 1]      # pool identities of training fraud
    vl_f = idx[vl][y_sub[vl] == 1]      # pool identities of validation fraud
    leaked = int(np.sum(np.isin(vl_f, tr_f)))
    return {
        "pool_label": label,
        "eval_mode": mode,
        "replay_verified_bitwise": exact,
        "surrogate_fraud_rows": int(y_sub.sum()),
        "unique_fraud_transactions": int(len(np.unique(idx[y_sub == 1]))),
        "train_fraud_rows": int(len(tr_f)),
        "val_fraud_rows": int(len(vl_f)),
        "val_fraud_unique": int(len(np.unique(vl_f))),
        "val_fraud_rows_also_in_train": leaked,
        "leak_fraction": round(leaked / max(len(vl_f), 1), 4),
    }


def run_stage2(pools):
    """`pools` maps dataset -> (label, X, y, n_raw) for the SHIPPED path."""
    rows = []
    for ds, (label, X, y, n_raw) in pools.items():
        for mode in ("legacy", "deterministic"):
            r = measure_leak(label, X, y, mode, n_raw=n_raw)
            r["dataset"] = ds
            rows.append(r)
    print("\n  === STAGE 2: does the duplication leak across the 70/30 split? ===",
          flush=True)
    for r in rows:
        verdict = ("⛔ LEAK" if r["leak_fraction"] > 0 else "ok, disjoint")
        print(f"    [{r['dataset']}/{r['eval_mode']:13s}] "
              f"{r['unique_fraud_transactions']:3d} unique fraud -> "
              f"{r['train_fraud_rows']:2d} train / {r['val_fraud_rows']:2d} val rows; "
              f"{r['val_fraud_rows_also_in_train']:2d} of {r['val_fraud_rows']:2d} "
              f"val positives are copies of a TRAIN row  {verdict}", flush=True)
        if not r["replay_verified_bitwise"]:
            print("      ⚠ replay did NOT match the live object — row above is "
                  "not trustworthy", flush=True)
    return rows

def main():
    out = {
        "note": "No training. Replays _ADTCNObjective.__init__'s fraud-selection "
                "arithmetic on the pool each caller actually supplies.",
        "min_fraud_rows": MIN_FRAUD, "surrogate_rows": SURR_ROWS,
        "datasets": {},
    }

    # ── ULB, shipped path — exactly what main.py:143-144 does ────────────────
    from config import get_loader
    loader = get_loader("ulb")
    X_train, _, _, y_train, _, _ = loader.load(verbose=False)
    X_opt, y_opt = loader.get_eval_subset(X_train, y_train)
    ulb_shipped = describe(
        f"ULB shipped: loader.get_eval_subset -> eval_subset={DATA_CONFIG['eval_subset']}",
        y_opt)

    # ── ULB, the pool the audit uses ─────────────────────────────────────────
    from experiments.basepaper_comparison import prepare_ulb, _eval_subset
    d = prepare_ulb(verbose=False)
    _, y_audit = _eval_subset(d, n=6000)
    ulb_audit = describe("ULB audit: basepaper _eval_subset(n=6000)", y_audit)

    out["datasets"]["ulb"] = [ulb_shipped, ulb_audit]

    # ── BankSim, both ────────────────────────────────────────────────────────
    bs_loader = get_loader("banksim")
    Xb_train, _, _, yb_train, _, _ = bs_loader.load(verbose=False)
    Xb_opt, yb_opt = bs_loader.get_eval_subset(Xb_train, yb_train)
    bs_shipped = describe(
        f"BankSim shipped: loader.get_eval_subset -> "
        f"eval_subset={BANKSIM_CONFIG.get('eval_subset', 3000)}", yb_opt)

    from experiments.basepaper_comparison import prepare_banksim
    db = prepare_banksim(ordering="customer", verbose=False)
    _, yb_audit = _eval_subset(db, n=6000)
    bs_audit = describe("BankSim audit: basepaper _eval_subset(n=6000)", yb_audit)
    out["datasets"]["banksim"] = [bs_shipped, bs_audit]

    for ds, rows in out["datasets"].items():
        print(f"\n  === {ds.upper()} ===", flush=True)
        for r in rows:
            flag = ("  ⛔ WITH REPLACEMENT" if r["samples_with_replacement"]
                    else "  ok, all distinct")
            print(f"    {r['pool_label']}", flush=True)
            print(f"      pool {r['pool_rows']:,} rows, "
                  f"{r['pool_fraud_unique']} unique fraud "
                  f"({r['pool_fraud_rate']*100:.3f} %)  ->  surrogate draws "
                  f"n_f={r['surrogate_n_f']}{flag}", flush=True)
            print(f"      expected DISTINCT fraud in surrogate: "
                  f"{r['expected_distinct_fraud_in_surrogate']}  "
                  f"(duplication {r['duplication_factor']}x), "
                  f"~{r['approx_val_fraud_rows']} fraud rows in validation",
                  flush=True)


    # ── stage 2 ──────────────────────────────────────────────────────────────
    out["leak_check"] = run_stage2({
        "ulb": (ulb_shipped["pool_label"], X_opt, y_opt,
                int(loader.raw_feature_count)
                if hasattr(loader, "raw_feature_count") else None),
        "banksim": (bs_shipped["pool_label"], Xb_opt, yb_opt,
                    int(bs_loader.raw_feature_count)
                    if hasattr(bs_loader, "raw_feature_count") else None),
    })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = os.path.join(RESULTS_DIR, "obj13_shipped_path_check.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {p}", flush=True)
    return out


if __name__ == "__main__":
    main()
