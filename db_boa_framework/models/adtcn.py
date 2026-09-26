"""
models/adtcn.py
===============
Adaptive Deep Temporal Context Networks (ADTCN)

Architecture (per paper §VI, with true temporal modelling):
  ┌───────────────────────────────────────────────────────────────────┐
  │  MJE   Multi-modal Joint Embedding  — raw 30-feature input         │
  │  TCL   Temporal Context Learning    — 1D-CNN over seq_len=10 steps │
  │  MTTA  GlobalMaxPool — selects the most anomalous time-step        │
  │         activation across the SEQ_LEN window  (pooling, not        │
  │         attention; the paper's "MTTA" label is re-used here)       │
  │  OUT   Classifier head  Linear(n_filters×2 → 2)                    │
  └───────────────────────────────────────────────────────────────────┘

The 1D-CNN processes 10 consecutive transactions as an ordered sequence
(Conv1d(30, F, 3) → ReLU → Conv1d(F, 2F, 3) → GlobalMaxPool → Linear),
so temporal order is genuinely exploited and the "TCL" claim is defensible.

DB-BOA optimises (n_filters, steps/epoch) via Eq.11.  Epoch count is fixed
at ADTCN_CONFIG["epoch_count"] (not searched) because the surrogate cap
makes the epoch dimension flat above 5 — see _ADTCNObjective docstring.
Class imbalance (~0.17% fraud) is handled with weighted cross-entropy loss.

References
----------
Prabanand & Thanabal (2025) Scientific Reports 15, 6764.
"""

import math
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config        import ADTCN_CONFIG, DB_BOA_CONFIG, LOG_WIDTH
from utils.metrics import compute_all_metrics, obf2_value
from algorithms.db_boa import DBBOA

warnings.filterwarnings("ignore")

N_RAW_FEATURES = 30   # V1-V28 + Amount + Time (first 30 cols of engineered matrix)
SEQ_LEN        = 10   # 10-step window chosen empirically; ablation over {5,10,20} is left for future work


# ─── helpers ──────────────────────────────────────────────────────────────────

def _print(msg: str):
    print(f"[ADTCN] {msg}", flush=True)


def _sep():
    print("-" * LOG_WIDTH, flush=True)


# ─── windowing ────────────────────────────────────────────────────────────────

def group_starts(groups: np.ndarray) -> np.ndarray:
    """
    For each row, the index of the first row of its group.

    Requires each group's rows to be **contiguous** — which the BankSim loader
    guarantees under `ordering="customer"`.  Contiguity is checked rather than
    assumed, because a silently non-contiguous `groups` array would produce
    windows that look entity-linked but are not, and the whole OBJ-1 comparison
    would be measuring nothing.
    """
    groups = np.asarray(groups)
    if len(groups) == 0:
        return np.zeros(0, dtype=np.int64)
    new = np.empty(len(groups), dtype=bool)
    new[0] = True
    new[1:] = groups[1:] != groups[:-1]
    n_runs = int(new.sum())
    n_uniq = len(np.unique(groups))
    if n_runs != n_uniq:
        raise ValueError(
            f"`groups` is not contiguous: {n_runs} runs for {n_uniq} distinct "
            "groups. Sort rows by (group, time) before building sequences — "
            "otherwise the windows are not entity-linked.")
    return np.maximum.accumulate(np.where(new, np.arange(len(groups)), 0))


def build_sequences(X: np.ndarray, seq_len: int = None,
                    groups: np.ndarray = None) -> np.ndarray:
    """
    Causal sliding windows: ``out[i] = rows[i-seq_len+1 .. i]``.

    Parameters
    ----------
    X : (n, n_features)
        Rows already in the order the window should follow.
    seq_len : int, default SEQ_LEN
    groups : (n,) array, optional
        Group key per row (e.g. customer ID).  When given, a window never
        crosses a group boundary: the start of each group is left-padded by
        repeating that group's own first row.  When ``None`` the whole matrix is
        one stream and only row 0 pads — the ULB behaviour, kept as the default
        so existing call sites are unchanged.

    Returns
    -------
    (n, seq_len, n_features)
    """
    seq_len = seq_len or SEQ_LEN
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n == 0:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32)

    idx = np.arange(n)[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
    floor = group_starts(groups)[:, None] if groups is not None else 0
    np.maximum(idx, floor, out=idx)
    return X[idx]


# ─── 1D-CNN temporal classifier ───────────────────────────────────────────────

class _Conv1dClassifier(nn.Module):
    """
    Temporal 1D-CNN that exploits transaction order (TCL layer).

    Input : (batch, seq_len, n_features) — SEQ_LEN consecutive transactions
    Output: (batch, 2)                   — logits for [normal, fraud]

    Conv1d(30, F, kernel=3) → ReLU → Conv1d(F, 2F, kernel=3) → GlobalMaxPool
    → Linear(2F, 2)

    Activation: ReLU (hardcoded). TanH was the paper's claimed best activation
    but was not tested in this implementation; no activation ablation was run.
    """

    def __init__(self, n_features: int = N_RAW_FEATURES, n_filters: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, n_filters,     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(n_filters * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → transpose for Conv1d
        x = x.permute(0, 2, 1)   # (batch, n_features, seq_len)
        x = self.conv(x)          # (batch, n_filters*2, seq_len)
        x = x.amax(dim=-1)        # global max pool → (batch, n_filters*2)
        return self.head(x)       # (batch, 2)


# ─── dilated-causal-conv + temporal-attention classifier (B3) ──────────────────

class _DilatedBlock(nn.Module):
    """
    One residual TCN block (Bai et al. 2018), lean variant: a single dilated
    causal Conv1d → BatchNorm → ReLU → dropout, plus a 1×1 residual projection.
    "Causal" = left-pad by (kernel-1)·dilation so step t only sees ≤ t.

    Single conv (vs two) keeps the param budget close to the plain CNN and
    reduces over-fitting.  (BatchNorm was tried and removed: under the 0.17%
    fraud rate its batch statistics are dominated by the normal class and it
    destabilised training, hurting precision.)
    """
    def __init__(self, c_in, c_out, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))   # left-pad only
        y = self.drop(self.relu(y))
        res = x if self.down is None else self.down(x)
        return y + res


class _DilatedAttnClassifier(nn.Module):
    """
    Adaptive Deep Temporal Context Network — the architecture the report
    describes (and the prior _Conv1dClassifier did not implement):

      stack of lean dilated causal conv blocks (dilations 1,2,4 → receptive
      field 1+(k-1)·Σd = 1+2·7 = 15 ≥ SEQ_LEN, so the model sees the entire
      window, unlike the k=3 ×2 CNN whose receptive field is only 5)
        → HYBRID temporal pool: softmax-attention context (learned
          "most-anomalous-step" weighting — the report's MTTA) CONCATENATED with
          the global-max activation.  Pure attention averages over steps and
          dilutes the single-step anomaly peak the max-pool captures; keeping
          both restores precision while adding the adaptive weighting.
        → Linear classifier head over [attn_ctx ‖ max_ctx].

    Input : (batch, seq_len, n_features)  Output: (batch, 2) logits.
    n_filters keeps the same DB-BOA-tuned meaning as in _Conv1dClassifier.
    """
    def __init__(self, n_features=N_RAW_FEATURES, n_filters=64,
                 dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        chans = [n_features] + [n_filters] * len(dilations)
        self.blocks = nn.ModuleList([
            _DilatedBlock(chans[i], chans[i + 1], kernel=3,
                          dilation=d, dropout=dropout)
            for i, d in enumerate(dilations)
        ])
        # temporal attention: score each step, softmax over time, weighted sum
        self.attn = nn.Conv1d(n_filters, 1, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_filters * 2, 2),       # [attn_ctx ‖ max_ctx]
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)             # (batch, n_features, seq_len)
        for blk in self.blocks:
            x = blk(x)                     # (batch, n_filters, seq_len)
        scores  = self.attn(x)             # (batch, 1, seq_len)
        w       = torch.softmax(scores, dim=-1)
        ctx_att = (x * w).sum(dim=-1)      # attention-weighted context
        ctx_max = x.amax(dim=-1)           # peak-anomaly context
        return self.head(torch.cat([ctx_att, ctx_max], dim=-1))


def make_temporal_model(n_features, n_filters, architecture="cnn", dropout=0.1):
    """
    Factory: select the temporal classifier by name.

    Ours
    ----
    "cnn"          — plain 2-layer Conv1d + global max pool (deployed model).
    "dilated_attn" — ADTCN: dilated causal stack + hybrid attention/max pool.

    Base-paper baselines (models/basepaper_models.py, re-implemented for 1-D;
    the base paper's own published numbers are never quoted — see D4)
    ----------------------------------------------------------------------
    "dtcn" | "resnet" | "densenet" | "efficientnet"  and "lstm" for the OBJ-1 grid.
    """
    if architecture == "dilated_attn":
        return _DilatedAttnClassifier(n_features=n_features, n_filters=n_filters,
                                      dropout=dropout)
    if architecture == "cnn":
        return _Conv1dClassifier(n_features=n_features, n_filters=n_filters)

    from models.basepaper_models import BASEPAPER_ARCHITECTURES, build
    if architecture in BASEPAPER_ARCHITECTURES:
        return build(architecture, n_features=n_features,
                     n_filters=n_filters, dropout=dropout)
    raise ValueError(f"unknown architecture {architecture!r}")


#: Every architecture the grid and comparison experiments can name.
ALL_ARCHITECTURES = ["cnn", "lstm", "dtcn", "dilated_attn",
                     "resnet", "densenet", "efficientnet"]

#: Display names, so tables read the way the base paper's do.
ARCH_LABELS = {
    "cnn":          "CNN",
    "lstm":         "LSTM",
    "dtcn":         "DTCN",
    "dilated_attn": "ADTCN (ours)",
    "resnet":       "ResNet-1D",
    "densenet":     "DenseNet-1D",
    "efficientnet": "EfficientNet-1D",
}


# ─── fitness wrapper for DB-BOA ───────────────────────────────────────────────

class _ADTCNObjective:
    """
    Wraps ADTCN evaluation as a callable fitness function for DB-BOA.

    Uses the same _Conv1dClassifier architecture as the final model, trained on
    a stratified subsample of `_SURROGATE_ROWS` rows for _SURROGATE_EPOCHS
    epochs.  DB-BOA minimises, so we return −Obf2.

    Epoch count is NOT part of the search space: the surrogate hard-caps at 5
    epochs regardless of what DB-BOA proposes, making that dimension flat.

    Hyperparameter mapping
    ----------------------
    params[0]  →  n_filters          (Conv1d channel count)
    params[1]  →  steps_per_epoch    (gradient steps per epoch → batch_size)
    epoch count is fixed at _SURROGATE_EPOCHS for the surrogate; the final
    model uses ADTCN_CONFIG["epoch_count"].

    OBJ-13 — what was wrong with this objective, and what changed
    -------------------------------------------------------------
    `objective_noise_audit.py` established three defects, all of which made the
    *search* meaningless while leaving the final training perfectly
    deterministic (that was OBJ-2's separate finding, and it still holds):

    1.  **Fitness was a random function of its input.**  `__call__` redrew both
        the 70/30 split permutation and the torch seed from `self.rng` on every
        call, and `self.rng` advances, so evaluating the *same* candidate twice
        gave different scores.  A fixed config re-evaluated 25× on ULB spanned
        Obf2 3.4497–5.0000 and touched the 5.0000 ceiling once **with no search
        involved** — which is why all five optimisers "found" exactly 5.0000.
    2.  **The validation split was not stratified.**  `_MIN_FRAUD_ROWS = 30`
        fires on both datasets, so the surrogate holds exactly 30 fraud rows and
        an unstratified 70/30 leaves ≈9 of them in validation — varying, and in
        principle zero.  Obf2 = 5.0000 means classifying about nine rows.
    3.  **One search axis was arithmetically dead.**  `batch_size = max(32,
        n_train // spe)` with n_train = 1400 and spe ∈ [50, 250] gives 28 … 5,
        every one of them below the 32 floor — so batch_size was **32 for every
        spe in the search range** and the advertised 2-D search was 1-D.

    The repair
    ----------
    `eval_mode` selects between the three behaviours, and running the first two
    side by side **is the result**: the gap between them measures how much of
    the original DB-BOA behaviour was noise-chasing rather than optimisation.

    ``"legacy"``
        The pre-repair behaviour, bit-for-bit.  Kept so the old numbers remain
        reproducible and the comparison is against a live baseline rather than
        a remembered one.  Never the default.
    ``"deterministic"``
        Split permutation and torch seed are fixed at construction, so fitness
        is a **pure function of the candidate**.  1× cost, ranking becomes
        reproducible — but it optimises one arbitrary draw, and that limitation
        is the reason "averaged" exists rather than something to hide.
    ``"averaged"``
        `k_repeats` draws, fixed at construction and **shared by every
        candidate** (common random numbers), returning the mean.  Optimising an
        expectation kills the best-of-N ceiling artefact: a lucky draw can no
        longer win, because every candidate is scored on the same draws.  k×
        cost.

    Two details that matter for honesty:

    *   The draws are pre-drawn in `__init__` from a dedicated RandomState, so
        candidate *i* and candidate *j* are compared on identical data and
        identical initialisation.  Re-drawing per candidate would leave the
        search comparing a good config on an easy draw against a good config on
        a hard one, which is the original defect wearing a mean.
    *   Both repaired modes stratify the 70/30 split, so fraud appears in both
        halves at every surrogate size.  Without this, "more rows" and "fraud
        actually present in validation" would move together and the size knee
        below would not be attributable.

    **Pre-registered expectation (rule 4, written before running).**  A repaired
    surrogate is a *better-behaved* DB-BOA, not necessarily a winning one.  The
    five-optimiser tie already survives on BankSim, where the objective's mean is
    at least unbiased.  If DB-BOA still ties the hand-set default after the
    repair, that is the honest outcome and rule 3 applies — **this is diagnosis
    of a negative result, not a rescue attempt.**
    """

    _SURROGATE_ROWS   = 2_000   # rows per evaluation (speed vs. fidelity trade-off)
    _SURROGATE_EPOCHS = 5       # surrogate training epochs (not searched)
    _MIN_FRAUD_ROWS   = 30      # minimum fraud samples; guards against near-zero count at 0.17% rate

    EVAL_MODES = ("legacy", "deterministic", "averaged")

    def __init__(self, X_opt, y_opt, random_state: int = 42,
                 architecture: str = "cnn", n_raw: int = None,
                 eval_mode: str = "deterministic", k_repeats: int = 3,
                 surrogate_rows: int = None):
        if eval_mode not in self.EVAL_MODES:
            raise ValueError(f"eval_mode must be one of {self.EVAL_MODES}, got {eval_mode!r}")
        self._eval_mode = eval_mode
        self._k = 1 if eval_mode in ("legacy", "deterministic") else max(1, int(k_repeats))
        n_rows = int(surrogate_rows or self._SURROGATE_ROWS)
        self._n_rows = n_rows

        rng   = np.random.RandomState(random_state)
        # `n_raw` must match what the FINAL model will see, or the search tunes
        # a different problem from the one it hands over.  The default caps at
        # 33, which is right for ULB (the tail of its matrix is the PTC/NTC
        # block the CNN never consumes) and wrong for any dataset whose columns
        # are all real features — BankSim's 79 would be cut to 33.  Callers on
        # such datasets pass the loader's `raw_feature_count`.
        n_raw = int(n_raw) if n_raw else min(X_opt.shape[1], N_RAW_FEATURES + 3)
        n_raw = min(n_raw, X_opt.shape[1])
        self._n_raw = n_raw
        self._architecture = architecture

        # Subsample preserving the real class distribution
        fraud_idx  = np.where(y_opt == 1)[0]
        normal_idx = np.where(y_opt == 0)[0]
        total      = len(fraud_idx) + len(normal_idx)
        fraud_rate = len(fraud_idx) / total
        # Minimum of _MIN_FRAUD_ROWS (30) ensures the CNN receives enough
        # positive examples for stable gradient estimates even when the input
        # has the real 0.17% fraud rate.  When the fraud pool is smaller than
        # n_f, we sample with replacement so the selection doesn't raise — see
        # the guard below, which is where that turned out to matter enormously.
        #
        # NOTE (OBJ-13), CORRECTED 2026-09-04: this comment used to claim "the
        # fraud *rate* is not what separates ULB from BankSim here; separability
        # at n≈9 validation positives is."  That is withdrawn.  The floor makes
        # both surrogates hold 30 fraud ROWS, but on the deployed 3,000-row pool
        # ULB's 30 rows were 5 unique transactions repeated ~6x while BankSim's
        # were 30 distinct ones.  The fraud rate IS what separates them — it is
        # what drops ULB below the `len(fraud_idx) < n_f` threshold on the next
        # line.  Counting rows instead of distinct transactions is what hid a
        # total train/validation leak for the whole project.
        n_f = max(self._MIN_FRAUD_ROWS, int(n_rows * fraud_rate))
        n_n = min(len(normal_idx), n_rows - n_f)
        replace_f = len(fraud_idx) < n_f   # allow repetition when pool is too small

        # ⛔ This branch is not a harmless convenience — it is a train/validation
        # leak, and it fired unnoticed on the shipped ULB path for the whole
        # project (OBJ-13, found 2026-09-04).  With 5 unique fraud rows in the
        # pool and n_f = 30, the surrogate's positives are ~6 copies each; the
        # 70/30 split then puts copies of the same transaction on both sides and
        # **9 of 9 validation positives were copies of training rows**.  Obf2
        # pins at its 5.0000 ceiling because the model is recognising rows it
        # memorised, and every optimiser "finds" that ceiling.
        #
        # Stratifying the split does NOT fix this (the repaired modes leak
        # identically); only a pool with enough unique fraud does.  So the
        # condition is recorded on the object and announced loudly, because the
        # thing that made this expensive was that it happened in silence.
        self.fraud_sampled_with_replacement = bool(replace_f)
        self.unique_fraud_available = int(len(fraud_idx))
        if replace_f:
            warnings.warn(
                f"_ADTCNObjective: pool has only {len(fraud_idx)} unique fraud "
                f"rows but n_f={n_f} are needed, so fraud is being sampled WITH "
                f"REPLACEMENT (~{n_f / max(len(fraud_idx), 1):.1f}x duplication). "
                f"The validation split will then contain copies of training rows "
                f"and Obf2 becomes meaningless. Enlarge the caller's pool "
                f"(config `eval_subset`) instead of ignoring this.",
                RuntimeWarning, stacklevel=2)
        idx = np.concatenate([
            rng.choice(fraud_idx,  n_f, replace=replace_f),
            rng.choice(normal_idx, n_n, replace=False),
        ])
        rng.shuffle(idx)
        X_sub = X_opt[idx, :n_raw].astype(np.float32)
        y_sub = y_opt[idx]

        # Pre-build sequences once so __call__ only trains the CNN
        pad   = np.repeat(X_sub[:1], SEQ_LEN - 1, axis=0)
        X_pad = np.vstack([pad, X_sub])
        self.X_seq = np.stack(
            [X_pad[i : i + SEQ_LEN] for i in range(len(X_sub))], axis=0
        ).astype(np.float32)   # (n_sub, SEQ_LEN, n_raw)
        self.y   = y_sub
        self.rng = rng
        self._call_count = 0
        self.surrogate_fraud_rows = int(y_sub.sum())

        # Pre-draw the evaluation draws ONCE.  Every candidate is then scored on
        # the same splits and the same initialisation — common random numbers —
        # so a difference in fitness is a difference in the candidate.
        self._draws = []
        if eval_mode != "legacy":
            drng = np.random.RandomState(random_state + 1)
            for _ in range(self._k):
                tr, vl = self._stratified_split(drng)
                self._draws.append((tr, vl, int(drng.randint(0, 2 ** 31))))
        self.last_scores = None

    # ── draws ────────────────────────────────────────────────────────────────

    def _stratified_split(self, rs):
        """
        70/30 split that keeps the class ratio in BOTH halves.

        The pre-repair split was a plain permutation over ~2,000 rows holding
        exactly 30 fraud, so validation received ≈9 positives — a count that
        varied per call and could in principle be zero, at which point MCC is
        undefined and Obf2 is meaningless.  Stratifying removes that as a source
        of between-call variance, which is a precondition for the size sweep
        below meaning anything.
        """
        out_tr, out_vl = [], []
        for cls in (0, 1):
            idx = np.where(self.y == cls)[0]
            idx = idx[rs.permutation(len(idx))]
            cut = int(round(0.7 * len(idx)))
            # Guarantee at least one row of each class on each side where the
            # class exists at all, so the metric is always defined.
            cut = min(max(cut, 1), len(idx) - 1) if len(idx) >= 2 else len(idx)
            out_tr.append(idx[:cut])
            out_vl.append(idx[cut:])
        tr = np.concatenate(out_tr)
        vl = np.concatenate(out_vl)
        return tr[rs.permutation(len(tr))], vl[rs.permutation(len(vl))]

    def _legacy_draw(self):
        """The pre-repair draw: a fresh unstratified permutation every call."""
        n       = len(self.X_seq)
        n_train = int(0.7 * n)
        idx     = self.rng.permutation(n)
        return idx[:n_train], idx[n_train:], int(self.rng.randint(0, 2 ** 31))

    # ── scoring ──────────────────────────────────────────────────────────────

    def _score_once(self, n_filters, spe, tr_idx, vl_idx, torch_seed):
        """Train one surrogate on one draw and return Obf2 (higher is better)."""
        X_tr_t = torch.tensor(self.X_seq[tr_idx], dtype=torch.float32)
        y_tr_t = torch.tensor(self.y[tr_idx],     dtype=torch.long)
        X_vl_t = torch.tensor(self.X_seq[vl_idx], dtype=torch.float32)
        y_vl   = self.y[vl_idx]

        n_train = len(tr_idx)
        if self._eval_mode == "legacy":
            # The dead axis, preserved exactly: with n_train = 1400 and
            # spe ∈ [50, 250] this is 32 for every spe in the search range.
            batch_size = max(32, n_train // spe)
        else:
            # `steps_per_epoch` means what it says: spe gradient steps per
            # epoch, so the axis is alive across the whole search range
            # (spe=50 → batch 28, spe=250 → batch 6).  Removing the floor is
            # the fix; the floor WAS the bug.
            batch_size = int(max(1, math.ceil(n_train / max(1, spe))))

        n_fraud  = int(y_tr_t.numpy().sum())
        n_normal = len(y_tr_t) - n_fraud
        w_fraud  = n_normal / max(n_fraud, 1)
        cw       = torch.tensor([1.0, w_fraud], dtype=torch.float32)

        try:
            torch.manual_seed(torch_seed)
            net       = make_temporal_model(n_features=self._n_raw,
                                            n_filters=n_filters,
                                            architecture=self._architecture)
            criterion = nn.CrossEntropyLoss(weight=cw)
            optimizer = optim.Adam(net.parameters(), lr=1e-3)
            loader    = DataLoader(
                TensorDataset(X_tr_t, y_tr_t),
                batch_size=batch_size, shuffle=True,
            )
            net.train()
            for _ in range(self._SURROGATE_EPOCHS):
                for Xb, yb in loader:
                    optimizer.zero_grad()
                    criterion(net(Xb), yb).backward()
                    optimizer.step()
            net.eval()
            with torch.no_grad():
                y_pred = net(X_vl_t).argmax(dim=1).numpy()
        except Exception:
            return None

        # Bounded fitness  Obf2 = 2*MCC + Spec + Pre + NPV, defined once in
        # utils.metrics.obf2_value (which also records why the paper's unbounded
        # 1/FPR form was replaced).
        m = compute_all_metrics(y_vl, y_pred)
        # Keep the component metrics reachable so the variance in Obf2 can be
        # attributed to its four terms without retraining anything
        # (objective_noise_audit.py --components).  Purely additive: the return
        # value and every RNG draw are unchanged, so `legacy` stays bit-for-bit.
        self.last_metrics = m
        return obf2_value(m)

    def __call__(self, params: np.ndarray) -> float:
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 10.0

        n_filters = max(8,  int(round(float(params[0]))))
        spe       = max(10, int(round(float(params[1]))))

        draws = [self._legacy_draw()] if self._eval_mode == "legacy" else self._draws

        scores = []
        for tr, vl, seed in draws:
            s = self._score_once(n_filters, spe, tr, vl, seed)
            if s is None:
                return 10.0          # a failed train is a failed candidate
            scores.append(s)

        self.last_scores = scores
        self._call_count += 1
        # DB-BOA minimises, so return the negation of the mean Obf2.
        return -float(np.mean(scores))


# ─── main ADTCN class ─────────────────────────────────────────────────────────

class ADTCN:
    """
    Adaptive Deep Temporal Context Network.

    Workflow
    --------
    1. DB-BOA searches for optimal (n_filters, steps/epoch) — 2D; epoch fixed.
    2. Final _Conv1dClassifier is trained on 10-step transaction sequences.
    3. All paper metrics are computed on the held-out test set.

    Parameters
    ----------
    cfg : dict  — override ADTCN_CONFIG
    """

    def __init__(self, cfg: dict = None):
        self.cfg            = cfg or ADTCN_CONFIG
        self.model          = None
        self.optimal_params = None
        self.opt_history    = None
        self._n_raw         = N_RAW_FEATURES   # updated in fit() when graph features present

    # ── public API ────────────────────────────────────────────────────────────

    def optimise_hyperparams(self, X_opt, y_opt, verbose: bool = True):
        """Run DB-BOA to find optimal (n_filters, steps/epoch) — 2D search."""
        if verbose:
            _print("Starting DB-BOA hyperparameter search (2D) …")
            _print(f"Search space: filters∈{DB_BOA_CONFIG['filter_count_bounds']}  "
                   f"SeD∈{DB_BOA_CONFIG['steps_per_epoch_bounds']}  "
                   f"epoch_count fixed={self.cfg['epoch_count']}")
            _sep()

        # OBJ-13: the surrogate's evaluation protocol is now explicit and
        # configurable rather than implicitly random.  "deterministic" is the
        # default because a search whose fitness is a random function of its
        # input cannot rank candidates at all; "averaged" additionally kills the
        # best-of-N ceiling artefact at k x the cost.  "legacy" reproduces the
        # pre-repair behaviour and exists so the old numbers stay reproducible.
        objective = _ADTCNObjective(X_opt, y_opt,
                                    random_state=self.cfg["random_state"],
                                    architecture=self.cfg.get("architecture", "cnn"),
                                    n_raw=self.cfg.get("n_raw_features"),
                                    eval_mode=self.cfg.get("surrogate_eval_mode",
                                                           "deterministic"),
                                    k_repeats=self.cfg.get("surrogate_k", 3),
                                    surrogate_rows=self.cfg.get("surrogate_rows"))
        if verbose:
            _print(f"Surrogate protocol          : {objective._eval_mode} "
                   f"(k={objective._k}, rows={objective._n_rows}, "
                   f"fraud rows={objective.surrogate_fraud_rows})")

        lb = np.array([DB_BOA_CONFIG["filter_count_bounds"][0],
                       DB_BOA_CONFIG["steps_per_epoch_bounds"][0]], dtype=float)
        ub = np.array([DB_BOA_CONFIG["filter_count_bounds"][1],
                       DB_BOA_CONFIG["steps_per_epoch_bounds"][1]], dtype=float)

        optimizer = DBBOA(
            objective_fn = objective,
            lb           = lb,
            ub           = ub,
            n_pop        = DB_BOA_CONFIG["population_size"],
            max_iter     = DB_BOA_CONFIG["max_iterations"],
            task_name    = "ADTCN Hyperparameter Optimisation",
            cfg          = DB_BOA_CONFIG,
            seed         = self.cfg["random_state"],
        )

        best_pos, best_fit, history = optimizer.optimise(verbose=verbose)

        self.optimal_params = {
            "hidden_neurons"  : max(8,  int(round(best_pos[0]))),   # → n_filters
            "epoch_count"     : self.cfg["epoch_count"],            # fixed, not searched
            "steps_per_epoch" : max(20, int(round(best_pos[1]))),
        }
        self.opt_history = history
        self.opt_stats   = optimizer.summary_stats()

        if verbose:
            _print(f"Optimal Conv filters (F)    : {self.optimal_params['hidden_neurons']}")
            _print(f"Epoch count (fixed)         : {self.optimal_params['epoch_count']}")
            _print(f"Optimal steps/epoch  (SeD)  : {self.optimal_params['steps_per_epoch']}")
            _print(f"Best Obf2 (negated)         : {best_fit:.6f}")

        return self.optimal_params

    def fit(self, X_train, y_train, verbose: bool = True, groups: np.ndarray = None):
        """
        Train 1D-CNN on SEQ_LEN-step transaction sequences.
        Class imbalance handled via weighted cross-entropy (no oversampling).
        """
        if self.optimal_params is None:
            _print("WARNING: No optimal params found — using defaults.")
            self.optimal_params = {
                "hidden_neurons" : self.cfg["hidden_neurons"],
                "epoch_count"    : self.cfg["epoch_count"],
                "steps_per_epoch": self.cfg["steps_per_epoch"],
            }

        n_filters  = self.optimal_params["hidden_neurons"]
        epochs     = self.optimal_params["epoch_count"]
        spe        = self.optimal_params["steps_per_epoch"]
        batch_size = max(32, len(X_train) // spe)

        # How many leading columns are real per-transaction features?
        #
        # On ULB the engineered matrix is ~301 columns wide: 30 base + 3
        # recurrence features, then a large PTC/NTC block that the CNN does not
        # consume (see _make_sequences), so the count is capped at 33.
        # On BankSim there is no PTC/NTC block — all 79 columns are real
        # features — and that cap would silently discard 46 of them.  Loaders
        # therefore declare the true width via `raw_feature_count`, which
        # main.py / run_baselines.py pass in as cfg["n_raw_features"].
        override = self.cfg.get("n_raw_features")
        self._n_raw = (int(override) if override
                       else min(X_train.shape[1], N_RAW_FEATURES + 3))

        if verbose:
            _sep()
            _print(f"Training 1D-CNN  |  in={self._n_raw}  filters={n_filters}×{n_filters*2}  "
                   f"seq_len={SEQ_LEN}  epochs={epochs}  batch={batch_size}")

        # Build temporal sequences: (n, SEQ_LEN, _n_raw)
        X_seq = self._make_sequences(X_train, groups=groups)

        if verbose:
            _print(f"Sequence tensor  : {X_seq.shape}"
                   f"{'  (customer-linked windows)' if groups is not None else ''}")

        # Class-weighted loss — avoids memory explosion from oversampling
        n_fraud  = int(y_train.sum())
        n_normal = len(y_train) - n_fraud
        w_fraud  = n_normal / max(n_fraud, 1)
        class_weight = torch.tensor([1.0, w_fraud], dtype=torch.float32)

        if verbose:
            _print(f"Class weights    : normal=1.0  fraud={w_fraud:.1f}")

        torch.manual_seed(self.cfg["random_state"])
        arch = self.cfg.get("architecture", "cnn")
        self.model = make_temporal_model(
            n_features=self._n_raw, n_filters=n_filters,
            architecture=arch, dropout=self.cfg.get("dropout_rate", 0.2),
        )
        if verbose:
            _print(f"Architecture     : {arch}")
        criterion  = nn.CrossEntropyLoss(weight=class_weight)
        optimizer  = optim.Adam(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
        )

        dataset = TensorDataset(
            torch.tensor(X_seq,    dtype=torch.float32),
            torch.tensor(y_train,  dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for ep in range(epochs):
            ep_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            if verbose and (ep == 0 or (ep + 1) % max(1, epochs // 6) == 0):
                _print(f"  epoch {ep+1:>3}/{epochs}  "
                       f"loss={ep_loss / len(loader):.4f}")

        self.model.eval()
        if verbose:
            _print("Training complete.")

        return self

    def predict(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        X_seq = self._make_sequences(X, groups=groups)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32))
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        X_seq = self._make_sequences(X, groups=groups)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def evaluate(self, X_test, y_test, verbose: bool = True, groups: np.ndarray = None):
        """Compute all metrics from the paper (Tables 3 & 4)."""
        y_pred  = self.predict(X_test, groups=groups)
        metrics = compute_all_metrics(y_test, y_pred)

        if verbose:
            _sep()
            _print("Evaluation results:")
            for k, v in metrics.items():
                unit = "" if k == "MCC" else " %"
                print(f"    {k:<20} : {v:.5f}{unit}", flush=True)
            _sep()

        return metrics

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
        """
        Convert (n, n_engineered) flat features → (n, SEQ_LEN, n_raw) sequences.

        Entity linkage (OBJ-1)
        ----------------------
        `groups` carries one key per row — a customer ID on BankSim.  When it is
        given, a window never crosses a customer boundary, so each window is one
        cardholder's own recent history.  When it is ``None`` (always, on ULB,
        which has no account IDs) the whole matrix is treated as one stream and
        a window stitches together *unrelated* cardholders.  That difference is
        the entire subject of `experiments/banksim_temporal_grid.py`.

        Extracts the leading n_raw columns (30 base features + optional 3 graph
        features) and builds sliding windows of SEQ_LEN consecutive transactions.
        The first SEQ_LEN-1 rows are padded by repeating the first row so that
        output length always equals input length.

        Design note — PTC/NTC features are intentionally discarded (Q30)
        ------------------------------------------------------------------
        The input matrix X has shape (n, ~301): leading 33 columns are raw
        features; columns 33+ are PTC rolling-statistics and NTC differences
        computed by _engineer_temporal_features().  This method extracts only
        the leading n_raw (≤33) columns.  The PTC/NTC block is available in X
        but the 1D-CNN receives its own temporal context implicitly by sliding
        over SEQ_LEN consecutive raw-feature vectors.  Adding the 268 PTC/NTC
        columns to the CNN input would change the channel count and was not
        evaluated; they serve as documentation of the TCL concept rather than
        live model inputs.

        Boundary condition: the first SEQ_LEN-1 predictions use a padded context
        (row 0 repeated). This affects ~0.003% of the 284,807-row dataset and
        does not meaningfully bias aggregate metrics.
        """
        n_raw = getattr(self, "_n_raw", N_RAW_FEATURES)
        X_raw = X[:, :n_raw].astype(np.float32)
        return build_sequences(X_raw, SEQ_LEN, groups=groups)   # (n, SEQ_LEN, n_raw)

