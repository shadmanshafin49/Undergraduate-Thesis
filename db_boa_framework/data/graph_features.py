"""
data/graph_features.py
======================
Temporal-amount recurrence feature extraction.

The ULB Credit Card Fraud dataset has no account identifiers — all PII was
removed before PCA projection, so there are no sender→receiver edges and no
true transaction graph can be constructed.

Instead, three sliding-window frequency features are computed from the Amount
column, capturing how often the same discretised Amount bucket recurs in the
neighbourhood of each transaction:

  amount_recurrence_before  — count of same-bucket transactions in the
                              preceding rolling window of `window_size` rows
                              (normalised to [0, 1]).  Captures "has this
                              amount pattern appeared recently?"
  amount_recurrence_after   — same for the following window ("does this pattern
                              repeat after?").
  degree_ratio              — amount_recurrence_before / (before + after + ε),
                              an asymmetry score: values near 1 indicate the
                              pattern mostly preceded this transaction; near 0
                              mostly followed it.

Implementation
--------------
Fully vectorised using cumulative-sum passes per bucket — O(n_bins × n) time,
no Python loops over rows.

Note on prior citation
----------------------
Liu et al. (WWW 2021) use these feature concepts on real account-to-account
graphs (PaySim dataset) where genuine sender→receiver edges exist.  The ULB
dataset lacks account IDs, so these features are Amount-recurrence proxies,
not graph-derived features.  The implementation is inspired by the idea but
the data substrate is different.
"""

import numpy as np


def extract_graph_features(
    amounts:     np.ndarray,
    n_bins:      int = 50,
    window_size: int = 100,
    verbose:     bool = True,
) -> np.ndarray:
    """
    Compute per-transaction temporal-amount recurrence features.

    Parameters
    ----------
    amounts     : (n,) float array — transaction amounts in temporal order
    n_bins      : number of percentile-equal Amount buckets
    window_size : rolling window width (number of rows)
    verbose     : print progress info

    Returns
    -------
    (n, 3) float32  — [amount_recurrence_before, amount_recurrence_after, degree_ratio]
    """
    n = len(amounts)
    amounts = np.asarray(amounts, dtype=np.float32)

    if verbose:
        print(f"[GRAPH] Building temporal-amount recurrence features "
              f"(n={n:,}, bins={n_bins}, window={window_size}) …", flush=True)

    # Discretise Amount into n_bins percentile-equal buckets
    bin_edges  = np.percentile(amounts, np.linspace(0, 100, n_bins + 1))
    bin_edges  = np.unique(bin_edges)
    bucket_ids = np.digitize(amounts, bin_edges[1:-1])   # integer 0 … n_bins-1
    n_b        = int(bucket_ids.max()) + 1

    recur_before = np.zeros(n, dtype=np.float32)
    recur_after  = np.zeros(n, dtype=np.float32)

    for b in range(n_b):
        mask = (bucket_ids == b).astype(np.float32)
        if mask.sum() == 0:
            continue

        cum = np.cumsum(mask)

        # ── recurrence_before: same-bucket txns in [i-window, i) ────────────
        shift              = np.empty_like(cum)
        shift[:window_size] = 0.0
        shift[window_size:] = cum[:-window_size]
        recur_before += np.maximum(0.0, cum - mask - shift)

        # ── recurrence_after: same-bucket txns in (i, i+window] ─────────────
        rev_mask             = mask[::-1]
        rev_cum              = np.cumsum(rev_mask)
        rev_shift            = np.empty_like(rev_cum)
        rev_shift[:window_size] = 0.0
        rev_shift[window_size:] = rev_cum[:-window_size]
        recur_after += np.maximum(0.0, rev_cum - rev_mask - rev_shift)[::-1]

    # ── degree_ratio: before / (before + after) ∈ [0, 1] ───────────────────
    degree_ratio = (recur_before / (recur_before + recur_after + 1e-8)).astype(np.float32)

    # ── Normalise recurrence counts to [0, 1] ────────────────────────────────
    recur_before = (recur_before / (recur_before.max() + 1e-8)).astype(np.float32)
    recur_after  = (recur_after  / (recur_after.max()  + 1e-8)).astype(np.float32)

    if verbose:
        print(f"[GRAPH] Features computed — "
              f"recur_before max={recur_before.max():.4f}  "
              f"recur_after max={recur_after.max():.4f}  "
              f"degree_ratio mean={degree_ratio.mean():.4f}", flush=True)

    return np.stack([recur_before, recur_after, degree_ratio], axis=1)   # (n, 3)
