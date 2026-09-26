"""
models/basepaper_models.py
==========================
The base paper's classifier baselines, re-implemented for 1-D transaction
sequences, plus an LSTM for the OBJ-1 temporal grid.

Why re-implement instead of quoting
-----------------------------------
Prabanand & Thanabal (2025, Sci. Rep. 15:6764) §"Simulation settings" compares
its ADTCN against **EfficientNet, ResNet, DenseNet and DTCN**, in MATLAB 2020a,
on its own data.  Those published numbers are *not* comparable with anything we
run: different data, different preprocessing, different split.  Divergence D4
records that copying that table into our report was the single largest
fabrication in the draft, and it has been purged.

So the only honest way to keep the base paper's comparison is to re-implement
its baselines and run them on *our* data, under *our* protocol, ourselves.
That is what this file is for.  Nothing here reproduces a number from the base
paper; every figure that ships comes from `experiments/basepaper_comparison.py`.

Honest scoping of the adaptation
--------------------------------
EfficientNet, ResNet and DenseNet are 2-D image networks.  A transaction window
is (SEQ_LEN=10 steps x n_features), not an image, so each is adapted to 1-D by
replacing Conv2d with Conv1d over the time axis and treating features as
channels.  The *architectural idea* is preserved in each case — residual
identity shortcuts (ResNet), concatenating feature reuse with transitions
(DenseNet), inverted-residual depthwise-separable MBConv with squeeze-excitation
(EfficientNet) — but these are our 1-D adaptations, not the published networks,
and the report must say so.  The base paper does not state its own adaptation,
so an exact match to what it ran is not recoverable.

`DTCN` is the interesting one: it is the plain dilated Temporal Convolutional
Network of Bai et al. (2018) that the base paper's ADTCN claims to improve on.
Here it is exactly `_DilatedAttnClassifier` minus the adaptive part — same
dilated causal stack, but a last-timestep readout instead of the hybrid
attention+max pool.  That makes DTCN vs ADTCN a clean single-factor ablation of
the "adaptive temporal attention" claim rather than a comparison of two
unrelated networks.

References
----------
Bai, Kolter & Koltun (2018)  An Empirical Evaluation of Generic Convolutional
    and Recurrent Networks for Sequence Modeling.  arXiv:1803.01271.
He et al. (2016)             Deep Residual Learning for Image Recognition. CVPR.
Huang et al. (2017)          Densely Connected Convolutional Networks. CVPR.
Tan & Le (2019)              EfficientNet: Rethinking Model Scaling for CNNs. ICML.
Hochreiter & Schmidhuber (1997)  Long Short-Term Memory.  Neural Computation 9(8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- LSTM --------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    Two-layer LSTM over the SEQ_LEN window, classified from the final hidden
    state.  The recurrent point of comparison for the OBJ-1 grid: if entity
    linkage is what matters, an LSTM should gain from customer-linked windows
    for the same reason a TCN does.

    n_filters is reused as the hidden width so every model in the grid is tuned
    by the same single width knob.
    """

    def __init__(self, n_features, n_filters=64, dropout=0.1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=n_filters,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_filters, 2))

    def forward(self, x):                     # (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])       # last timestep


# --- DTCN (Bai et al. 2018) --------------------------------------------------

class _TCNBlock(nn.Module):
    """
    Standard TCN residual block: two dilated causal convolutions with weight
    sharing of the dilation, ReLU and dropout between, plus a 1x1 projection
    shortcut.  "Causal" = left-pad by (kernel-1)*dilation so step t sees only
    steps <= t.
    """

    def __init__(self, c_in, c_out, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)


class DTCNClassifier(nn.Module):
    """
    Plain dilated TCN — the base paper's `DTCN` baseline, and the exact
    architecture its ADTCN claims to improve upon.

    Identical dilated causal stack to `_DilatedAttnClassifier` but with a
    **last-timestep readout**: no attention, no max pool, no adaptive weighting.
    ADTCN minus "adaptive", so the head-to-head isolates that one factor.
    """

    def __init__(self, n_features, n_filters=64, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        chans = [n_features] + [n_filters] * len(dilations)
        self.blocks = nn.ModuleList([
            _TCNBlock(chans[i], chans[i + 1], kernel=3, dilation=d, dropout=dropout)
            for i, d in enumerate(dilations)
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_filters, 2))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, :, -1])         # last timestep, no pooling


# --- ResNet (He et al. 2016), 1-D adaptation ---------------------------------

class _ResBlock1d(nn.Module):
    """Basic ResNet block adapted to 1-D: conv-BN-ReLU-conv-BN + identity."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out))

    def forward(self, x):
        r = x if self.short is None else self.short(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + r)


class ResNet1dClassifier(nn.Module):
    """
    1-D ResNet baseline: stem conv, three residual stages with width doubling,
    global average pool, linear head.  A shallow ResNet-ish depth is used
    because the sequence is only 10 steps long — stacking a full ResNet-50 over
    a length-10 axis would spend its depth on padding.
    """

    def __init__(self, n_features, n_filters=64, dropout=0.1):
        super().__init__()
        w = max(8, n_filters // 2)
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, w, 3, padding=1, bias=False),
            nn.BatchNorm1d(w), nn.ReLU())
        self.stages = nn.Sequential(
            _ResBlock1d(w, w),
            _ResBlock1d(w, w * 2),
            _ResBlock1d(w * 2, w * 2))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(w * 2, 2))

    def forward(self, x):
        x = self.stem(x.permute(0, 2, 1))
        x = self.stages(x)
        return self.head(x.mean(dim=-1))


# --- DenseNet (Huang et al. 2017), 1-D adaptation ----------------------------

class _DenseLayer1d(nn.Module):
    """BN-ReLU-Conv layer whose output is concatenated onto its input."""

    def __init__(self, c_in, growth):
        super().__init__()
        self.bn = nn.BatchNorm1d(c_in)
        self.conv = nn.Conv1d(c_in, growth, 3, padding=1, bias=False)

    def forward(self, x):
        return torch.cat([x, self.conv(F.relu(self.bn(x)))], dim=1)


class DenseNet1dClassifier(nn.Module):
    """
    1-D DenseNet baseline: two dense blocks of concatenating layers separated by
    a 1x1 transition that halves the channel count, then global average pool.

    Transitions do **not** downsample in time here: with SEQ_LEN=10 a single
    stride-2 pool would leave 5 steps and a second would leave 2, which destroys
    the temporal axis the experiment is about.  Feature-dimension compression is
    kept; spatial compression is dropped, and that departure is deliberate.
    """

    def __init__(self, n_features, n_filters=64, growth=None, dropout=0.1,
                 block_layers=(3, 3)):
        super().__init__()
        growth = growth or max(4, n_filters // 8)
        c = max(8, n_filters // 2)
        self.stem = nn.Conv1d(n_features, c, 3, padding=1, bias=False)
        layers = []
        for bi, n_layers in enumerate(block_layers):
            for _ in range(n_layers):
                layers.append(_DenseLayer1d(c, growth))
                c += growth
            if bi < len(block_layers) - 1:       # transition (channel compression)
                c_out = max(8, c // 2)
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(c), nn.ReLU(),
                    nn.Conv1d(c, c_out, 1, bias=False)))
                c = c_out
        self.body = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(c)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, 2))

    def forward(self, x):
        x = self.body(self.stem(x.permute(0, 2, 1)))
        return self.head(F.relu(self.norm(x)).mean(dim=-1))


# --- EfficientNet (Tan & Le 2019), 1-D adaptation ----------------------------

class _SE1d(nn.Module):
    """Squeeze-and-excitation over the time axis."""

    def __init__(self, c, r=4):
        super().__init__()
        h = max(1, c // r)
        self.fc1, self.fc2 = nn.Conv1d(c, h, 1), nn.Conv1d(h, c, 1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return x * s


class _MBConv1d(nn.Module):
    """
    Inverted-residual MBConv adapted to 1-D: 1x1 expand -> depthwise conv ->
    squeeze-excitation -> 1x1 project, with a residual when shapes match.
    SiLU activations, as in the original.
    """

    def __init__(self, c_in, c_out, expand=4, kernel=3, dropout=0.1):
        super().__init__()
        h = c_in * expand
        self.expand = nn.Sequential(
            nn.Conv1d(c_in, h, 1, bias=False), nn.BatchNorm1d(h), nn.SiLU())
        self.depthwise = nn.Sequential(
            nn.Conv1d(h, h, kernel, padding=kernel // 2, groups=h, bias=False),
            nn.BatchNorm1d(h), nn.SiLU())
        self.se = _SE1d(h)
        self.project = nn.Sequential(
            nn.Conv1d(h, c_out, 1, bias=False), nn.BatchNorm1d(c_out))
        self.drop = nn.Dropout(dropout)
        self.residual = (c_in == c_out)

    def forward(self, x):
        y = self.project(self.se(self.depthwise(self.expand(x))))
        return x + self.drop(y) if self.residual else y


class EfficientNet1dClassifier(nn.Module):
    """
    1-D EfficientNet baseline: stem, a compound-scaled stack of MBConv blocks
    with widening channels, 1x1 head conv, global average pool.

    Kept deliberately shallow (4 MBConv blocks) for the same reason as ResNet1d:
    the time axis is 10 steps.  Depth/width follow the EfficientNet-B0 *pattern*
    scaled down, not its published coefficients.
    """

    def __init__(self, n_features, n_filters=64, dropout=0.1):
        super().__init__()
        w = max(8, n_filters // 4)
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, w, 3, padding=1, bias=False),
            nn.BatchNorm1d(w), nn.SiLU())
        self.blocks = nn.Sequential(
            _MBConv1d(w, w, expand=1, kernel=3, dropout=dropout),
            _MBConv1d(w, w * 2, expand=4, kernel=3, dropout=dropout),
            _MBConv1d(w * 2, w * 2, expand=4, kernel=5, dropout=dropout),
            _MBConv1d(w * 2, w * 4, expand=4, kernel=3, dropout=dropout))
        self.top = nn.Sequential(
            nn.Conv1d(w * 4, w * 4, 1, bias=False),
            nn.BatchNorm1d(w * 4), nn.SiLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(w * 4, 2))

    def forward(self, x):
        x = self.top(self.blocks(self.stem(x.permute(0, 2, 1))))
        return self.head(x.mean(dim=-1))


# --- registry ----------------------------------------------------------------

#: Architectures contributed by this module, keyed by the name used in
#: `ADTCN_CONFIG["architecture"]` and on experiment command lines.
BASEPAPER_ARCHITECTURES = {
    "lstm":         LSTMClassifier,
    "dtcn":         DTCNClassifier,
    "resnet":       ResNet1dClassifier,
    "densenet":     DenseNet1dClassifier,
    "efficientnet": EfficientNet1dClassifier,
}

#: The base paper's four classifier baselines, in the order its Table 4 uses.
BASEPAPER_CLASSIFIERS = ["efficientnet", "resnet", "densenet", "dtcn"]


def build(architecture, n_features, n_filters=64, dropout=0.1):
    """Instantiate one of this module's architectures by name."""
    if architecture not in BASEPAPER_ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; "
                         f"expected one of {sorted(BASEPAPER_ARCHITECTURES)}")
    return BASEPAPER_ARCHITECTURES[architecture](
        n_features=n_features, n_filters=n_filters, dropout=dropout)
