"""
data/graph_features.py
======================
Transaction graph feature extraction.

Constructs a directed temporal-amount similarity graph over the transaction
sequence and extracts three node-level features per transaction:

  in_degree_norm   — fraction of preceding transactions in the rolling window
                     that share the same Amount bucket  (→ "how often has this
                     card/amount pattern appeared recently?")
  out_degree_norm  — same for subsequent transactions   (→ "recurrence after")
  pagerank_norm    — in_degree / (in_degree + out_degree), a node-centrality
                     proxy derived from the graph topology

Implementation note
-------------------
The ULB Credit Card Fraud dataset has no explicit sender/receiver account IDs
(all PII was removed before PCA projection). We therefore proxy the
sender→receiver edge relationship using temporal-amount co-occurrence:
transactions with the same discretised Amount bucket within a rolling window
of `window_size` rows are treated as graph neighbours. This captures the
"same card used at repeated merchants" pattern that is strongly predictive
of fraud (Liu et al., 2021).

The algorithm is fully vectorised — a single cumulative-sum pass per bucket,
O(n_bins × n) time, no Python loops over rows.

Reference
---------
Liu et al., "Pick and Choose: A GNN-based Imbalanced Learning Approach for
Fraud Detection", WWW 2021.
"""

import numpy as np


def extract_graph_features(
    amounts:     np.ndarray,
    n_bins:      int = 50,
    window_size: int = 100,
    verbose:     bool = True,
) -> np.ndarray:
    """
    Compute per-transaction graph features from the Amount column.

    Parameters
    ----------
    amounts     : (n,) float array — transaction amounts in temporal order
    n_bins      : number of percentile-equal Amount buckets
    window_size : rolling window width (number of rows) for edge construction
    verbose     : print progress info

    Returns
    -------
    (n, 3) float32  — [in_degree_norm, out_degree_norm, pagerank_norm]
    """
    n = len(amounts)
    amounts = np.asarray(amounts, dtype=np.float32)

    if verbose:
        print(f"[GRAPH] Building temporal-amount similarity graph "
              f"(n={n:,}, bins={n_bins}, window={window_size}) …", flush=True)

    # Discretise Amount into n_bins percentile-equal buckets
    bin_edges  = np.percentile(amounts, np.linspace(0, 100, n_bins + 1))
    bin_edges  = np.unique(bin_edges)
    bucket_ids = np.digitize(amounts, bin_edges[1:-1])   # integer 0 … n_bins-1
    n_b        = int(bucket_ids.max()) + 1

    in_deg  = np.zeros(n, dtype=np.float32)
    out_deg = np.zeros(n, dtype=np.float32)

    for b in range(n_b):
        mask = (bucket_ids == b).astype(np.float32)
        if mask.sum() == 0:
            continue

        cum = np.cumsum(mask)

        # ── in_degree: count of same-bucket txns in [i-window, i) ──────────
        # = cumsum[i-1] - cumsum[i-window-1]  (with boundary clamping)
        shift              = np.empty_like(cum)
        shift[:window_size] = 0.0
        shift[window_size:] = cum[:-window_size]
        in_deg  += np.maximum(0.0, cum - mask - shift)

        # ── out_degree: count of same-bucket txns in (i, i+window] ─────────
        # Same logic on the reversed sequence
        rev_mask             = mask[::-1]
        rev_cum              = np.cumsum(rev_mask)
        rev_shift            = np.empty_like(rev_cum)
        rev_shift[:window_size] = 0.0
        rev_shift[window_size:] = rev_cum[:-window_size]
        out_deg += np.maximum(0.0, rev_cum - rev_mask - rev_shift)[::-1]

    # ── PageRank proxy: in / (in + out) ∈ [0, 1] ────────────────────────────
    pr_norm = (in_deg / (in_deg + out_deg + 1e-8)).astype(np.float32)

    # ── Normalise degrees to [0, 1] ──────────────────────────────────────────
    in_deg  = (in_deg  / (in_deg.max()  + 1e-8)).astype(np.float32)
    out_deg = (out_deg / (out_deg.max() + 1e-8)).astype(np.float32)

    if verbose:
        print(f"[GRAPH] Features computed — "
              f"in_deg max={in_deg.max():.4f}  "
              f"out_deg max={out_deg.max():.4f}  "
              f"pr mean={pr_norm.mean():.4f}", flush=True)

    return np.stack([in_deg, out_deg, pr_norm], axis=1)   # (n, 3)
