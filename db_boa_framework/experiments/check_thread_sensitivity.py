"""
experiments/check_thread_sensitivity.py
=======================================
Is ULB org-model training sensitive to the CPU thread count?

Why this exists
---------------
OBJ-17 re-ran the three private-incentive conditions on an extended eps grid.
Two of them reproduced their stored results bitwise; **ULB did not** — its
ground-truth Shapley split moved from [0.619, 0.250, 0.131] to
[0.554, 0.336, 0.110] and 13 of 18 inversion cells moved with it.  ULB then
reproduced *itself* exactly when re-run the same day, so the training is
deterministic under fixed conditions and something about the conditions changed.

An audit of every diff on the ULB path since the stored result found nothing
that should affect it: the `build_sequences` rewrite is provably equivalent for
`groups=None`, `n_raw` resolves to 33 either way, `get_loader("ulb")` is the old
bare constructor, and the `obf2_value` / `coalition_score` edits touch the DB-BOA
*search* objective, not the balanced-accuracy coalition score this sweep uses.

That leaves the mechanism `requirements.txt` already documents:

    ADTCN training is bitwise reproducible only for a fixed
    (torch version, CPU thread count) pair ... Changing either changes the
    float reduction order inside the convolutions, and for the large-batch
    DB-BOA-tuned configuration that is enough to move test MCC by ~0.09.

Thread count is pinned nowhere in the code — it defaults to the machine's core
count and can vary with load.  If that is the cause, it is not a bug in this
objective; it is an un-pinned dependency that silently invalidates any stored
ULB result, which is worth knowing before anything else is re-derived from one.

What this measures
------------------
Trains ONE org model (the largest ULB split) twice, at two thread counts,
everything else identical, and compares the extracted weights exactly.  One
model is enough: if training is thread-sensitive at all, the federation built on
three of them cannot be reproducible.

    differ  -> thread count is the mechanism. Pin it, and treat every stored ULB
               result whose thread count is unrecorded as unreproducible.
    same    -> thread count is NOT the mechanism and the divergence is still
               unexplained. Do not close it; say so.

Usage
-----
    python experiments/check_thread_sensitivity.py
    python experiments/check_thread_sensitivity.py --threads 1 8 --org BankA
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch

from config                 import ADTCN_CONFIG
from experiments._dataset   import resolve, apply_to_model_cfg
from models.federated_adtcn import FederatedADTCN

# Match private_incentive_sweep.run_sweep exactly, or this measures a different
# model than the one whose reproducibility is in question.
EPOCH_CNT       = 12
ORG_LABEL_NOISE = {"BankA": 0.00, "BankB": 0.10, "BankC": 0.25}


def _corrupt_labels(y, rate, seed=123):
    """Verbatim from private_incentive_sweep.py — same labels, same seed."""
    if rate <= 0:
        return y
    rng = np.random.RandomState(seed)
    y = y.copy()
    flip = rng.rand(len(y)) < rate
    y[flip] = 1 - y[flip]
    return y


def train_once(X_org, y_org, cfg_model, n_threads):
    torch.set_num_threads(n_threads)
    m = FederatedADTCN(cfg=cfg_model)
    m.optimal_params = {"hidden_neurons":  cfg_model["hidden_neurons"],
                        "epoch_count":     EPOCH_CNT,
                        "steps_per_epoch": cfg_model["steps_per_epoch"]}
    m.fit(X_org, y_org, verbose=False)
    return [np.asarray(w, dtype=np.float64) for w in m.extract_weights()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs=2, default=None, metavar=("A", "B"),
                    help="the two thread counts to compare (default: 1 and the "
                         "machine default)")
    ap.add_argument("--org", default="BankA", help="which org split to train")
    args = ap.parse_args()

    default_threads = torch.get_num_threads()
    ta, tb = args.threads if args.threads else (1, default_threads)
    if ta == tb:
        raise SystemExit(f"the two thread counts are identical ({ta}); nothing to compare")

    print("=" * 78)
    print("  ULB training - CPU thread-count sensitivity")
    print(f"  torch {torch.__version__}   default threads on this machine: {default_threads}")
    print(f"  comparing {ta} vs {tb} threads, org={args.org}, epochs={EPOCH_CNT}")
    print("=" * 78, flush=True)

    loader = resolve("ulb", None, verbose=False)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=False)
    org_splits = loader.split_for_orgs(X_train, y_train)
    if args.org not in org_splits:
        raise SystemExit(f"unknown org {args.org!r}; have {list(org_splits)}")
    X_org, y_org = org_splits[args.org]
    y_org = _corrupt_labels(y_org, ORG_LABEL_NOISE.get(args.org, 0.0))

    cfg_model = dict(ADTCN_CONFIG)
    cfg_model["epoch_count"] = EPOCH_CNT
    apply_to_model_cfg(cfg_model, loader)
    print(f"  {args.org}: {len(y_org):,} samples, n_raw_features="
          f"{cfg_model['n_raw_features']}\n", flush=True)

    print(f"  training at {ta} thread(s) ...", flush=True)
    wa = train_once(X_org, y_org, cfg_model, ta)
    print(f"  training at {tb} thread(s) ...", flush=True)
    wb = train_once(X_org, y_org, cfg_model, tb)

    print("\n  tensor | shape | max |A-B| | identical")
    print("  " + "-" * 56)
    worst, all_same = 0.0, True
    for i, (a, b) in enumerate(zip(wa, wb)):
        d = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
        same = (a.shape == b.shape) and bool(np.array_equal(a, b))
        all_same &= same
        worst = max(worst, d if d != float("inf") else worst)
        print(f"  {i:>6} | {str(a.shape):>16} | {d:.3e} | {'yes' if same else 'NO'}")

    print("\n" + "=" * 78)
    if all_same:
        print("  IDENTICAL - thread count is NOT the mechanism.")
        print("  The ULB divergence from the stored result remains UNEXPLAINED.")
        print("  Do not close it; the next suspects are the torch/numpy versions")
        print("  themselves, or a change in a file not yet diffed.")
    else:
        print(f"  DIFFERENT - max |A-B| = {worst:.3e} across {len(wa)} weight tensors.")
        print("  Thread count changes ULB training, exactly as requirements.txt")
        print("  documents. It is pinned nowhere in the code, so ANY stored ULB")
        print("  result whose thread count was not recorded is unreproducible -")
        print("  including every private-incentive number generated before today.")
    print("=" * 78)


if __name__ == "__main__":
    main()
