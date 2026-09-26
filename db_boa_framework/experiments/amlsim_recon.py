"""
experiments/amlsim_recon.py
===========================
OBJ-12 — reconnaissance on the AMLSim data this project generated
(`datasets/amlsim/10K_3banks`, produced by `data/generate_amlsim.py`), before
any loader decision or any training.  The twin of `paysim_recon.py`.

What it measures
----------------
1. Shape — rows, laundering (`is_sar`) positives, days, amounts by class.
2. Label aliases — columns that are the label under another name and must
   never become features (`alert_id`; the account table's `prior_sar_count`).
3. Banks — positives per sender bank, and the cross-bank share by class.
4. Repeat structure of senders and receivers — is there a history to window?
5. The file-order trap (BankSim's) — raw order against a seeded within-day
   shuffle; the timestamp resolves to a day.
6. Entity-link strength — P(sar_t | sar_{t-1}) under global, sender-linked and
   receiver-linked orderings.
7. Temporal cuts at the 70 / 80 % row-mass day quantiles, and the positives
   each split would hold.
8. Logistic references on row-only features (amount, day-of-week), with and
   without bank columns.  A **floor** detector as much as a ceiling one: if the
   rows carry no signal, every federated comparison on this dataset is noise,
   and that has to be known before it is run, not discovered after.

None of these references is a result, and none is compared with anything.

Usage
-----
    python experiments/amlsim_recon.py        # seconds; 198 k rows
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

from config import RESULTS_DIR                                          # noqa: E402
from experiments._dataset import environment                            # noqa: E402

DATA = os.path.join(os.path.dirname(ROOT), "datasets", "amlsim", "10K_3banks")
OUT_PATH = os.path.join(RESULTS_DIR, "amlsim_recon.json")
SEED = 0


def adjacency(y, same=None):
    prev, cur = y[:-1], y[1:]
    ok = np.ones(len(cur), bool) if same is None else same
    m = ok & (prev == 1)
    p = float(cur[m].mean()) if m.any() else float("nan")
    return {"conditioning_rows": int(m.sum()), "adjacent_positive_pairs": int((m & (cur == 1)).sum()),
            "p_pos_given_prev_pos": p, "lift_over_base": p / float(y.mean())}


def ordered(y, day, tie, group=None):
    if group is None:
        return adjacency(y[np.lexsort((tie, day))])
    o = np.lexsort((tie, day, group))
    g = group[o]
    return adjacency(y[o], g[1:] == g[:-1])


def repeats(codes, y):
    counts = np.bincount(codes)
    per_row = counts[codes]
    live = counts[counts > 0]
    return {"unique": int(live.size),
            "share_rows_once": float((per_row == 1).mean()),
            "share_rows_ge10": float((per_row >= 10).mean()),
            "median_tx": float(np.median(live)), "max_tx": int(live.max()),
            "share_positive_rows_once": float((per_row[y == 1] == 1).mean())}


def logistic(t, day, y, tr, te, which, sb, rb):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef
    from sklearn.preprocessing import StandardScaler
    amt = t["base_amt"].to_numpy(np.float64)
    cols = [np.log1p(amt), amt] + [((day % 7) == k).astype(float) for k in range(7)]
    if which == "row+banks":
        for b in sorted(sb.unique()):
            cols += [(sb == b).to_numpy(float), (rb == b).to_numpy(float)]
        cols.append((sb != rb).to_numpy(float))
    X = np.column_stack(cols)
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    p = lr.predict(sc.transform(X[te]))
    return {"features": which, "n_features": int(X.shape[1]),
            "MCC": float(matthews_corrcoef(y[te], p)),
            "TP": int(((p == 1) & (y[te] == 1)).sum()), "FP": int(((p == 1) & (y[te] == 0)).sum()),
            "FN": int(((p == 0) & (y[te] == 1)).sum())}


def main():
    t0 = time.time()
    t = pd.read_csv(os.path.join(DATA, "transactions.csv"))
    a = pd.read_csv(os.path.join(DATA, "accounts.csv"),
                    usecols=["acct_id", "bank_id", "prior_sar_count"])
    y = t["is_sar"].astype(int).to_numpy()
    ts = pd.to_datetime(t["tran_timestamp"], utc=True)
    day = ((ts - ts.min()).dt.days).to_numpy()
    bank = a.set_index("acct_id")["bank_id"]
    sb, rb = t["orig_acct"].map(bank), t["bene_acct"].map(bank)
    sar_acct = a.set_index("acct_id")["prior_sar_count"].astype(bool)
    amt = t["base_amt"].to_numpy()
    out = {"task": "OBJ-12 AMLSim reconnaissance — measured before any loader decision; "
                   "nothing here is compared with anything",
           "data": os.path.relpath(DATA, os.path.dirname(ROOT)).replace("\\", "/"),
           "columns": t.columns.tolist()}

    out["shape"] = {
        "rows": int(len(t)), "positives": int(y.sum()), "rate": float(y.mean()),
        "tx_types": t["tx_type"].value_counts().to_dict(),
        "days": [int(day.min()), int(day.max())], "distinct_days": int(np.unique(day).size),
        "amount_by_class": {c: {"min": float(amt[y == v].min()), "median": float(np.median(amt[y == v])),
                                "max": float(amt[y == v].max())}
                            for c, v in (("sar", 1), ("normal", 0))},
        "share_in_100_200_band": {"sar": float(((amt >= 100) & (amt <= 200))[y == 1].mean()),
                                  "normal": float(((amt >= 100) & (amt <= 200))[y == 0].mean())},
    }
    out["label_aliases"] = {
        "alert_id_ne_minus1_equals_is_sar": bool(((t["alert_id"] != -1).to_numpy() == (y == 1)).all()),
        "accounts_prior_sar_count_true": int(sar_acct.sum()),
        "share_sar_tx_sender_flagged": float(t["orig_acct"].map(sar_acct)[y == 1].mean()),
        "share_normal_tx_sender_flagged": float(t["orig_acct"].map(sar_acct)[y == 0].mean()),
    }
    out["banks"] = {
        "per_sender_bank": {b: {"rows": int((sb == b).sum()), "positives": int(y[(sb == b).to_numpy()].sum())}
                            for b in sorted(sb.unique())},
        "cross_bank_share": {"all": float((sb != rb).mean()),
                             "sar": float((sb != rb)[y == 1].mean()),
                             "normal": float((sb != rb)[y == 0].mean())},
    }
    orig = pd.factorize(t["orig_acct"])[0]
    bene = pd.factorize(t["bene_acct"])[0]
    out["repeat_structure"] = {"sender": repeats(orig, y), "receiver": repeats(bene, y)}
    tie = np.random.default_rng(SEED).permutation(len(t))
    out["file_order_trap"] = {"raw_file_order": adjacency(y),
                              "within_day_seeded_shuffle": ordered(y, day, tie)}
    out["entity_links"] = {"global": ordered(y, day, tie),
                           "sender_linked": ordered(y, day, tie, orig),
                           "receiver_linked": ordered(y, day, tie, bene)}
    q70, q80 = int(np.quantile(day, 0.70)), int(np.quantile(day, 0.80))
    tr, va, te = day < q70, (day >= q70) & (day < q80), day >= q80
    out["cuts"] = {"val_day_q70": q70, "split_day_q80": q80,
                   **{nm: {"rows": int(m.sum()), "positives": int(y[m].sum())}
                      for nm, m in (("train", tr), ("val", va), ("test", te))}}
    trr, tee = day < q80, day >= q80
    out["logistic_references"] = {
        "split": f"temporal: train day<{q80}, test day>={q80}; class_weight=balanced; threshold 0.5",
        "runs": [logistic(t, day, y, trr, tee, w, sb, rb) for w in ("row_only", "row+banks")]}
    out["environment"] = environment()
    out["wall_clock_seconds"] = round(time.time() - t0, 1)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[SAVE] {OUT_PATH}")


if __name__ == "__main__":
    main()
