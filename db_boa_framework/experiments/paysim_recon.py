"""
experiments/paysim_recon.py
===========================
OBJ-11 — reconnaissance on PaySim, before any loader decision or any training.

PaySim (Lopez-Rojas, Elmir & Axelsson, 2016; Kaggle `ealaxi/paysim1`) is a
mobile-money simulator: ~6.36 M transactions over hourly steps, with sender and
receiver account IDs and pre/post balances.  The plan ranked it last because it
has no bank IDs and because sending accounts may appear only once or twice.
Rule 8 is suspended, so it is being run anyway — which makes it more important,
not less, to measure exactly what it *can* test before a single model trains.

What this measures (all from the file, none quoted from the paper)
------------------------------------------------------------------
1. Shape — rows, fraud, isFlaggedFraud, steps, fraud by transaction type.
2. Repeat structure — does any account have a history to window?  Per sender
   (`nameOrig`) and per receiver (`nameDest`).
3. The file-order trap (BankSim's) — adjacent fraud pairs in raw file order
   against a seeded within-step shuffle.  A step is one hour, so there is no
   real intra-step order to preserve.
4. Entity-link strength — P(fraud_t | fraud_{t-1}) under global,
   sender-linked and receiver-linked orderings: the number that decided the
   Handbook's windowing arms.
5. The balance columns — whether they give the label away.  A ceiling would
   make every architecture and every federation method tie, and a tie at a
   ceiling is not evidence (the 5.0000 lesson).  Three logistic references on a
   temporal split measure it: row features without balances, plus the four
   balance columns, plus the two per-row balance-error terms.

**None of these references is a result, and none is compared with anything.**
They exist to detect a ceiling, exactly as BankSim's "per-transaction logistic
reference reaches MCC 0.574" did before its loader design was fixed.

Usage
-----
    python experiments/paysim_recon.py        # a few minutes, no torch
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import RESULTS_DIR
from experiments._dataset import environment

PAYSIM_PATH = os.path.join(os.path.dirname(ROOT), "datasets", "paysim",
                           "PS_20174392719_1491204439457_log.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "paysim_recon.json")
SEED = 0
TRAIN_SUBSAMPLE = 1_500_000


def adjacency(y, same_group=None):
    """
    P(y_t = 1 | y_{t-1} = 1) over consecutive rows — only where row t-1 belongs
    to the same group as row t, if `same_group` is given (so a group's first row
    has no predecessor, rather than borrowing another account's).
    """
    prev, cur = y[:-1], y[1:]
    ok = np.ones(len(cur), bool) if same_group is None else same_group
    m = ok & (prev == 1)
    p = float(cur[m].mean()) if m.any() else float("nan")
    base = float(y.mean())
    return {"conditioning_rows": int(m.sum()),
            "adjacent_fraud_pairs": int((m & (cur == 1)).sum()),
            "p_fraud_given_prev_fraud": p,
            "lift_over_base": (p / base) if base > 0 else float("nan")}


def repeat_stats(codes, y):
    counts = np.bincount(codes)
    per_row = counts[codes]
    live = counts[counts > 0]
    return {
        "unique_accounts": int(live.size),
        "share_rows_account_appears_once": float((per_row == 1).mean()),
        "share_rows_account_has_ge10": float((per_row >= 10).mean()),
        "median_tx_per_account": float(np.median(live)),
        "max_tx_per_account": int(live.max()),
        "share_fraud_rows_account_appears_once": float((per_row[y == 1] == 1).mean()),
    }


def ordered_adjacency(y, step, tie, group=None):
    if group is None:
        order = np.lexsort((tie, step))
        return adjacency(y[order])
    order = np.lexsort((tie, step, group))
    g = group[order]
    return adjacency(y[order], same_group=(g[1:] == g[:-1]))


def features(df, which):
    """Row-level only: every column is computable from its own row."""
    amt = df["amount"].to_numpy(np.float64)
    cols = [np.log1p(amt), amt]
    cols += [(df["type"] == t).to_numpy(np.float64) for t in sorted(df["type"].unique())]
    hour = (df["step"].to_numpy() % 24)
    cols += [(hour == h).astype(np.float64) for h in range(24)]
    if which in ("balances", "balances+errors"):
        for c in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
            cols.append(np.log1p(df[c].to_numpy(np.float64).clip(min=0)))
    if which == "balances+errors":
        cols.append((df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).to_numpy(np.float64))
        cols.append((df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).to_numpy(np.float64))
    return np.column_stack(cols)


def logistic_reference(df, tr, te, which, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef
    from sklearn.preprocessing import StandardScaler

    tr_idx = np.flatnonzero(tr)
    if len(tr_idx) > TRAIN_SUBSAMPLE:
        tr_idx = np.sort(rng.choice(tr_idx, TRAIN_SUBSAMPLE, replace=False))
    te_idx = np.flatnonzero(te)
    Xtr = features(df.iloc[tr_idx], which)
    Xte = features(df.iloc[te_idx], which)
    ytr = df["isFraud"].to_numpy()[tr_idx]
    yte = df["isFraud"].to_numpy()[te_idx]
    sc = StandardScaler().fit(Xtr)
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(Xtr), ytr)
    pred = lr.predict(sc.transform(Xte))
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    return {"features": which, "n_features": int(Xtr.shape[1]),
            "train_rows": int(len(tr_idx)), "train_fraud": int(ytr.sum()),
            "test_rows": int(len(te_idx)), "test_fraud": int(yte.sum()),
            "MCC": float(matthews_corrcoef(yte, pred)), "TP": tp, "FP": fp, "FN": fn,
            "fit_seconds": round(time.time() - t0, 1)}


def main():
    t_all = time.time()
    if not os.path.exists(PAYSIM_PATH):
        raise SystemExit(f"PaySim not found at {PAYSIM_PATH}\n"
                         "  kaggle datasets download ealaxi/paysim1 -p datasets/paysim --unzip")
    df = pd.read_csv(PAYSIM_PATH, dtype={"type": "category"})
    y = df["isFraud"].to_numpy().astype(np.int8)
    step = df["step"].to_numpy()
    rng = np.random.default_rng(SEED)
    tie = rng.permutation(len(df))
    orig = pd.factorize(df["nameOrig"])[0]
    dest = pd.factorize(df["nameDest"])[0]
    out = {"task": "OBJ-11 PaySim reconnaissance — measured before any loader "
                   "decision; no result here is compared with anything",
           "file": os.path.basename(PAYSIM_PATH),
           "file_bytes": os.path.getsize(PAYSIM_PATH),
           "columns": list(df.columns)}

    # 1. shape
    by_type = df.groupby("type", observed=True)["isFraud"].agg(["size", "sum"])
    rows_per_step = np.bincount(step)[np.unique(step)]
    out["shape"] = {
        "rows": int(len(df)), "fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "isFlaggedFraud": int(df["isFlaggedFraud"].sum()),
        "flagged_and_fraud": int(((df["isFlaggedFraud"] == 1) & (y == 1)).sum()),
        "step_min": int(step.min()), "step_max": int(step.max()),
        "distinct_steps": int(np.unique(step).size),
        "rows_per_step_min_median_max": [int(rows_per_step.min()),
                                         float(np.median(rows_per_step)),
                                         int(rows_per_step.max())],
        "by_type": {str(t): {"rows": int(r["size"]), "fraud": int(r["sum"]),
                             "fraud_rate": float(r["sum"] / r["size"])}
                    for t, r in by_type.iterrows()},
    }

    # 2. repeat structure
    out["repeat_structure"] = {
        "sender_nameOrig": repeat_stats(orig, y),
        "receiver_nameDest": repeat_stats(dest, y),
        "share_rows_receiver_is_merchant_M": float(df["nameDest"].str.startswith("M").mean()),
        "share_fraud_rows_receiver_is_merchant_M":
            float(df.loc[y == 1, "nameDest"].str.startswith("M").mean()),
    }

    # 3. file-order trap
    typ = df["type"].astype(str).to_numpy()
    amt = df["amount"].to_numpy()
    pair = ((typ[:-1] == "TRANSFER") & (y[:-1] == 1) & (typ[1:] == "CASH_OUT")
            & (y[1:] == 1) & np.isclose(amt[:-1], amt[1:]))
    out["file_order_trap"] = {
        "raw_file_order": adjacency(y),
        "within_step_seeded_shuffle": ordered_adjacency(y, step, tie),
        "raw_adjacent_fraud_TRANSFER_then_CASH_OUT_same_amount": int(pair.sum()),
    }

    # 4. entity-link strength (after the shuffle; a group's first row has no
    #    predecessor, so it cannot borrow another account's label)
    out["entity_links"] = {
        "global": ordered_adjacency(y, step, tie),
        "sender_linked": ordered_adjacency(y, step, tie, orig),
        "receiver_linked": ordered_adjacency(y, step, tie, dest),
    }

    # 5. balance columns
    err_o = (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).abs().to_numpy()
    err_d = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).abs().to_numpy()
    zero_after = (df["newbalanceOrig"] == 0).to_numpy()
    out["balances"] = {
        cls: {"share_sender_balance_error_gt_0.01": float((err_o[y == v] > 0.01).mean()),
              "share_receiver_balance_error_gt_0.01": float((err_d[y == v] > 0.01).mean()),
              "share_sender_emptied_newbalance_0": float(zero_after[y == v].mean())}
        for cls, v in (("fraud", 1), ("legit", 0))}
    cut = int(np.quantile(step, 0.8))
    tr, te = step < cut, step >= cut
    out["logistic_references"] = {
        "split": f"temporal: train step<{cut}, test step>={cut}; train subsampled "
                 f"to {TRAIN_SUBSAMPLE:,} rows (seed {SEED}); class_weight=balanced; "
                 f"threshold 0.5",
        "runs": [logistic_reference(df, tr, te, w, np.random.default_rng(SEED))
                 for w in ("row_only", "balances", "balances+errors")],
    }
    out["environment"] = environment()
    out["wall_clock_seconds"] = round(time.time() - t_all, 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[SAVE] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
