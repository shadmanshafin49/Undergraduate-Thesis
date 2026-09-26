"""
experiments/_seqtrain.py
========================
Shared window-training helper for the dataset and model-comparison experiments.

Why this exists
---------------
BankSim's training split is 417,848 rows x 79 features.  Materialising its
10-step windows as a dense (n, 10, 79) float32 tensor costs 1.3 GB *per arm*,
and the grid runs 14 arms.  Instead we keep the row matrix once and an
(n, seq_len) int32 **index** matrix (17 MB), gathering each batch's windows on
the fly.  Same arithmetic, ~80x less memory, and the gather is cheap next to the
convolutions.

Everything here is deliberately shared between
`experiments/banksim_temporal_grid.py` and
`experiments/basepaper_comparison.py`, so an architecture is never trained under
two subtly different protocols and then compared.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from models.adtcn import make_temporal_model, group_starts, SEQ_LEN
from utils.metrics import compute_all_metrics


def window_index(n: int, seq_len: int = SEQ_LEN,
                 groups: np.ndarray = None) -> np.ndarray:
    """
    (n, seq_len) int32 matrix of row indices forming each causal window.

    `groups` clamps every window to its own group's first row, so a window never
    spans two customers.  Without it the whole matrix is one stream — the ULB
    behaviour, and the "global windows" arm of the OBJ-1 grid.
    """
    idx = np.arange(n, dtype=np.int64)[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
    floor = group_starts(groups)[:, None] if groups is not None else 0
    np.maximum(idx, floor, out=idx)
    return idx.astype(np.int32)


def _batches(n, batch_size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for s in range(0, n, batch_size):
        yield order[s:s + batch_size]


def train_eval(architecture, X_tr, y_tr, idx_tr, X_te, y_te, idx_te,
               epochs=12, n_filters=64, seed=42, batch_size=1024,
               lr=1e-3, dropout=0.1, verbose=False, threshold=0.5):
    """
    Train one architecture on windowed data and score it on the test split.

    Parameters
    ----------
    X_tr, X_te : (N, n_features) float32 row matrices, already scaled.
        These address the rows that `idx_tr` / `idx_te` point into.  When
        windows are built on a full stream and the *split* is applied to rows
        (as the OBJ-1 grid does), pass the FULL matrix here and subset only
        `y` and `idx` — slicing X would mis-address every window.
    idx_tr, idx_te : (n, seq_len) window index matrices from `window_index`,
        already restricted to the rows of this split.
    y_tr, y_te : (n,) int labels for those same rows.

    Class imbalance is handled by weighted cross-entropy — the same treatment
    the deployed ADTCN uses (`models/adtcn.py::fit`), never by resampling, so
    the test distribution stays the real one.

    Returns
    -------
    dict — `compute_all_metrics` output plus `train_seconds` and `n_params`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    Xtr_t = torch.from_numpy(np.ascontiguousarray(X_tr, dtype=np.float32))
    Xte_t = torch.from_numpy(np.ascontiguousarray(X_te, dtype=np.float32))
    Itr = torch.from_numpy(idx_tr.astype(np.int64))
    Ite = torch.from_numpy(idx_te.astype(np.int64))
    ytr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64))

    if len(idx_tr) != len(y_tr) or len(idx_te) != len(y_te):
        raise ValueError("window-index and label arrays must describe the same rows")
    if int(idx_tr.max()) >= len(X_tr) or int(idx_te.max()) >= len(X_te):
        raise ValueError(
            "window indices address rows outside the given matrix — pass the "
            "FULL row matrix when windows were built on the full stream")

    n_features = X_tr.shape[1]
    net = make_temporal_model(n_features=n_features, n_filters=n_filters,
                              architecture=architecture, dropout=dropout)
    n_params = sum(p.numel() for p in net.parameters())

    n_fraud = int(np.sum(y_tr))
    w_fraud = (len(y_tr) - n_fraud) / max(n_fraud, 1)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, w_fraud], dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    t0 = time.time()
    net.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for b in _batches(len(y_tr), batch_size, rng):
            bt = torch.from_numpy(b)
            xb = Xtr_t[Itr[bt]]                     # (B, seq_len, n_features)
            opt.zero_grad()
            loss = crit(net(xb), ytr_t[bt])
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        if verbose:
            print(f"      epoch {ep+1:>2}/{epochs}  loss={tot/max(nb,1):.4f}",
                  flush=True)
    train_seconds = time.time() - t0

    net.eval()
    preds = np.empty(len(y_te), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(y_te), 4096):
            sl = slice(s, min(s + 4096, len(y_te)))
            xb = Xte_t[Ite[sl]]
            p = torch.softmax(net(xb), dim=1)[:, 1]
            preds[sl] = (p >= threshold).long().numpy()

    m = compute_all_metrics(np.asarray(y_te), preds)
    m["train_seconds"] = float(train_seconds)
    m["n_params"] = int(n_params)
    return m


def aggregate(runs, keys=("MCC", "Precision", "Sensitivity", "Accuracy",
                          "Specificity", "FPR")):
    """
    Mean and sample std across seeds.

    One seed is not evidence — OBJ-2 measured the same tuned config spanning
    MCC 0.660–0.792 across five seeds, and 0.708 → 0.800 on CPU thread count
    alone — so every cell in these experiments is a multi-seed mean with its
    spread reported next to it.

    (An earlier version of this note cited a 0.677 ↔ 0.313 swing. 0.313 is
    withdrawn: OBJ-2 could not reproduce it under any condition tested, and
    the lowest tuned MCC in 7 runs is 0.6598.)
    """
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in runs if k in r]
        if not vals:
            continue
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    for k in ("TP", "FP", "FN", "TN"):
        vals = [float(r[k]) for r in runs if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    out["train_seconds"] = float(np.mean([r["train_seconds"] for r in runs]))
    out["n_params"] = int(runs[0]["n_params"])
    out["seeds"] = len(runs)
    out["MCC_runs"] = [float(r["MCC"]) for r in runs]
    return out
