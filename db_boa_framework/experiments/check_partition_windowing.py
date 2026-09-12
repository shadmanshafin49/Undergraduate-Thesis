"""
experiments/check_partition_windowing.py
========================================
How entity-linked are the windows the system sweeps TRAIN on, per partition?

Why this exists (found 2026-09-11)
----------------------------------
None of the four system sweeps passes window groups to `ADTCN.fit`, so the
model builds sliding windows over each org's rows *in whatever order the
partition leaves them*.  The stratified split (`FinancialDataLoader.
split_for_orgs` -> `train_test_split`, shuffle=True) leaves them in random
order; an entity split (boolean masks) keeps them grouped by entity, in time
order.  So changing the partition also changes what a training window
contains.  OBJ-15's control was written up as "both partitions window
identically; the partition is the only factor moving" — this measures by how
much that was wrong, per dataset, without training anything.

Method: every loader's `split_for_orgs` depends on the labels, the entity
groups and the seed — never on the feature values — so passing a column of row
indices in place of X returns, per org, exactly the row order that org holds.
For each org stream we report the share of training windows whose ten rows all
belong to the target row's own entity (build_sequences clamps at the stream
start, so the first rows repeat row 0), and the fraud-adjacency lift inside
the stream.

Usage
-----
    python experiments/check_partition_windowing.py      # a few minutes, no training
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import RESULTS_DIR                          # noqa: E402
from experiments._dataset import resolve                # noqa: E402
from models.adtcn import SEQ_LEN                        # noqa: E402

CONDITIONS = [("banksim", "stratified"), ("banksim", "customer"),
              ("handbook", "stratified"), ("handbook", "customer"),
              ("amlsim", "stratified"), ("amlsim", "bank")]


def stream_stats(ent, y):
    """Share of windows single-entity, and P(y_t | y_{t-1}) / base, for one org stream."""
    n = len(ent)
    idx = np.arange(n)[:, None] - np.arange(SEQ_LEN - 1, -1, -1)[None, :]
    np.maximum(idx, 0, out=idx)                        # build_sequences' clamp at the start
    pure = (ent[idx] == ent[:, None]).all(axis=1)
    prev = y[:-1] == 1
    lift = float(y[1:][prev].mean() / y.mean()) if prev.any() and y.mean() > 0 else float("nan")
    return float(pure.mean()), lift


def main():
    t0 = time.time()
    out = {"task": "share of sweep TRAINING windows that are single-entity, per partition "
                   "(no training; see module docstring)", "seq_len": SEQ_LEN, "conditions": {}}
    for ds, part in CONDITIONS:
        ld = resolve(ds, part, verbose=False)
        Xtr, _, _, ytr, _, _ = ld.load(verbose=False)
        ent = np.asarray(ld.groups_train)
        orgs = ld.split_for_orgs(np.arange(len(ytr))[:, None], ytr)
        per_org, w_pure, w_n = {}, 0.0, 0
        for org, (ix, yy) in orgs.items():
            rows = ix[:, 0].astype(np.int64)
            pure, lift = stream_stats(ent[rows], yy)
            per_org[org] = {"rows": int(len(rows)), "single_entity_window_share": pure,
                            "adjacency_lift": lift}
            w_pure += pure * len(rows)
            w_n += len(rows)
        entity_name = {"banksim": "customer", "handbook": "customer", "amlsim": "sender"}[ds]
        out["conditions"][f"{ds}/{part}"] = {
            "entity": entity_name, "loader_ordering": ld.cfg.get("ordering"),
            "single_entity_window_share": w_pure / w_n, "per_org": per_org}
        print(f"  {ds + '/' + part:<22} single-{entity_name} windows {100 * w_pure / w_n:6.2f} %  "
              + "  ".join(f"{o} lift {v['adjacency_lift']:.1f}x" for o, v in per_org.items()),
              flush=True)
        del Xtr
    out["wall_clock_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS_DIR, "partition_windowing.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[SAVE] {path}")


if __name__ == "__main__":
    main()
