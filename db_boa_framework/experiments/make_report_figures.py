"""Report figures for WORK_REPORT.tex.

Reads only files that other scripts produced (results/*.json, results/_obj13_logs/*.log)
and writes results/figures/*.png plus a generated LaTeX score-index table.

No training, no sampling, no new numbers: every value plotted here is read from a
stored result file, and the provenance of each panel is printed to stdout so the
figure can be traced back to the script that produced its input.

Run:  python -m experiments.make_report_figures      (from db_boa_framework/)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
LOGS = RESULTS / "_obj13_logs"

# ---------------------------------------------------------------- palette
# Validated categorical palette (dataviz reference instance, light surface).
# Slot order is fixed and never cycled.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# Status tokens -- used only where the colour *means* broken / repaired, and
# always paired with a written label so hue never carries the meaning alone.
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "figure.dpi": 160,
    }
)


def style(ax, ygrid=True, xgrid=False):
    ax.set_axisbelow(True)
    if xgrid:
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    else:
        ax.grid(axis="y", visible=ygrid)
        ax.grid(axis="x", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.title.set_color(INK)


def load(name):
    with open(RESULTS / name, encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, name, provenance):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")
    for line in provenance:
        print(f"      <- {line}")


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.0, fontsize=8.2, colour=INK2):
    """Direct value labels -- the relief the palette's contrast WARN requires."""
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        off = dy if h >= 0 else -dy
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h + off),
            ha="center",
            va=va,
            fontsize=fontsize,
            color=colour,
        )


# ============================================================ FIGURE 1
# Memorisation -> learning: the DB-BOA surrogate before and after OBJ-13.
def parse_shipped_path_log(path):
    """Pull the shipped-pool row of each dataset out of an obj13 log."""
    text = path.read_text(encoding="utf-8-sig")
    out = {}
    for ds, tag in (("ulb", "ULB shipped"), ("banksim", "BankSim shipped")):
        block = text.split(tag, 1)[1].split("audit:", 1)[0]
        m = re.search(r"pool ([\d,]+) rows, (\d+) unique fraud", block)
        d = re.search(r"duplication ([\d.]+)x", block)
        out[ds] = {
            "pool_rows": int(m.group(1).replace(",", "")),
            "unique_fraud": int(m.group(2)),
            "duplication": float(d.group(1)),
        }
    leaks = {}
    for m in re.finditer(
        r"\[(\w+)/(\w+)\s*\]\s+(\d+) unique fraud -> (\d+) train /\s*(\d+) val rows;\s*"
        r"(\d+) of\s*(\d+) val positives",
        text,
    ):
        ds, mode = m.group(1), m.group(2)
        leaks.setdefault(ds, {})[mode] = {
            "leaked": int(m.group(6)),
            "val": int(m.group(7)),
        }
    return out, leaks


def figure_memorisation():
    before, before_leak = parse_shipped_path_log(LOGS / "shipped_path.log")
    after_json = load("obj13_shipped_path_check.json")

    after = {}
    for ds, rows in after_json["datasets"].items():
        row = next(r for r in rows if "shipped" in r["pool_label"])
        after[ds] = {
            "pool_rows": row["pool_rows"],
            "unique_fraud": row["pool_fraud_unique"],
            "duplication": row["duplication_factor"],
        }
    after_leak = {}
    for r in after_json["leak_check"]:
        after_leak.setdefault(r["dataset"], {})[r["eval_mode"]] = {
            "leaked": r["val_fraud_rows_also_in_train"],
            "val": r["val_fraud_rows"],
        }

    labels = ["ULB", "BankSim"]
    keys = ["ulb", "banksim"]
    x = np.arange(2)
    w = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9))

    # (a) distinct fraud transactions the surrogate could learn from
    ax = axes[0]
    b1 = ax.bar(x - w / 2 - 0.01, [before[k]["unique_fraud"] for k in keys], w,
                color=CRITICAL, label="before (3,000-row pool)")
    b2 = ax.bar(x + w / 2 + 0.01, [after[k]["unique_fraud"] for k in keys], w,
                color=GOOD, label="after (36,000-row pool)")
    ax.set_yscale("log")
    ax.set_ylim(1, 9000)
    ax.axhline(30, color=MUTED, lw=0.9, zorder=1)
    for b in list(b1) + list(b2):
        ax.annotate(f"{int(b.get_height())}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8.2, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_title("(a) Distinct fraud transactions\nin the surrogate pool")
    ax.set_ylabel("unique fraud rows (log)")
    ax.legend(handles=[Line2D([0], [0], color=CRITICAL, lw=6, label="before (3,000-row pool)"),
                       Line2D([0], [0], color=GOOD, lw=6, label="after (36,000-row pool)"),
                       Line2D([0], [0], color=MUTED, lw=1, label="30 = what the draw asks for")],
              loc="upper left", fontsize=7.6)
    style(ax)

    # (b) duplication factor of the fraud draw
    ax = axes[1]
    b1 = ax.bar(x - w / 2 - 0.01, [before[k]["duplication"] for k in keys], w, color=CRITICAL)
    b2 = ax.bar(x + w / 2 + 0.01, [after[k]["duplication"] for k in keys], w, color=GOOD)
    ax.axhline(1.0, color=MUTED, lw=0.9)
    bar_labels(ax, list(b1) + list(b2), fmt="{:.2f}x", dy=0.09, fontsize=8.2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 9.8)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_title("(b) Fraud-row duplication\nin the drawn surrogate")
    ax.set_ylabel("mean copies per distinct fraud row")
    ax.legend(handles=[Line2D([0], [0], color=CRITICAL, lw=6, label="before repair"),
                       Line2D([0], [0], color=GOOD, lw=6, label="after repair"),
                       Line2D([0], [0], color=MUTED, lw=1,
                              label="1.00x = every drawn row distinct")],
              loc="upper right", fontsize=7.6)
    style(ax)

    # (c) validation positives that are copies of a training row
    ax = axes[2]
    modes = ["legacy", "deterministic"]
    xs, heights, colours, notes = [], [], [], []
    pos = 0.0
    ticks, ticklabels = [], []
    for k, lab in zip(keys, labels):
        centre = []
        for src, colour in ((before_leak, CRITICAL), (after_leak, GOOD)):
            for mode in modes:
                cell = src[k][mode]
                xs.append(pos)
                heights.append(100.0 * cell["leaked"] / cell["val"])
                colours.append(colour)
                notes.append(f"{cell['leaked']} of {cell['val']}")
                centre.append(pos)
                pos += 1.0
            pos += 0.35
        ticks.append(float(np.mean(centre)))
        ticklabels.append(lab)
        pos += 0.9
    bars = ax.bar(xs, heights, 0.82, color=colours)
    for b, note in zip(bars, notes):
        ax.annotate(note, (b.get_x() + b.get_width() / 2, b.get_height() + 3),
                    ha="center", va="bottom", fontsize=7.6, color=INK2, rotation=90)
    ax.set_xticks(ticks, ticklabels)
    ax.set_ylim(0, 148)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("(c) Validation positives that are\ncopies of a training row")
    ax.set_ylabel("% of validation fraud rows")
    ax.set_xlabel("bars: legacy / deterministic mode")
    ax.legend(handles=[Line2D([0], [0], color=CRITICAL, lw=6, label="before repair"),
                       Line2D([0], [0], color=GOOD, lw=6, label="after repair")],
              loc="upper right")
    style(ax)

    fig.tight_layout(w_pad=2.6)
    fig.suptitle(
        "OBJ-13  —  the DB-BOA surrogate stopped validating on rows it had trained on",
        fontsize=12.0, color=INK, fontweight="bold", y=1.05,
    )
    save(fig, "fig_memorisation_before_after.png",
         ["_obj13_logs/shipped_path.log (before)",
          "obj13_shipped_path_check.json (after)"])
    return before, after, before_leak, after_leak


# ============================================================ FIGURE 2
# Every condition the federated ablation can run on, in reading order.  A
# condition gets a column only once its result file exists (Rule 2): nothing is
# drawn for a run that has not happened.  PaySim's all-rows arm is a sensitivity
# variant of PaySim, reported in the text -- not a condition, so not a column.
ALL_CONDITIONS = [
    ("ULB / stratified", "ULB\nstratified", "baselines.json"),
    ("BankSim / stratified", "BankSim\nstratified", "baselines_banksim_stratified.json"),
    ("BankSim / entity-disjoint", "BankSim\nentity-disj.", "baselines_banksim_customer.json"),
    ("Handbook / stratified", "Handbook\nstratified", "baselines_handbook_stratified.json"),
    ("Handbook / entity-disjoint", "Handbook\nentity-disj.",
     "baselines_handbook_customer.json"),
    ("PaySim / stratified", "PaySim\nstratified", "baselines_paysim_stratified.json"),
    ("AMLSim / stratified", "AMLSim\nstratified", "baselines_amlsim_stratified.json"),
    ("AMLSim / native banks", "AMLSim\nnative banks", "baselines_amlsim_bank.json"),
]
CONDITIONS = [c for c, _, f in ALL_CONDITIONS if (RESULTS / f).exists()]
BASELINE_FILES = {c: f for c, _, f in ALL_CONDITIONS if c in CONDITIONS}
_SHORT = {c: s for c, s, _ in ALL_CONDITIONS}
METHODS = ["FedAvg", "FedAvg+Krum", "FedAvg+DP", "DB-BOA-ADTCN"]
METHOD_COLOURS = [C1, C2, C3, C4]
# FedAvg MCC below this leaves nothing to attack, reward or protect -- the floor
# the AMLSim pre-registration fixed as (f).  A condition there is drawn, marked,
# and never read as a result.
FLOOR_MCC = 0.05
MARK_NOTE = ("† run on Kaggle (Python 3.12, the laptop's library pins, 4 threads)     "
             "* FedAvg below the MCC 0.05 floor: untestable there, shown not scored")


def cond_flags():
    """{condition: (ran on Kaggle, FedAvg below the floor)}, read off each
    condition's own baselines file -- never assumed from its name."""
    out = {}
    for c in CONDITIONS:
        d = load(BASELINE_FILES[c])
        plat = str((d.get("environment") or {}).get("platform", ""))
        out[c] = (plat.startswith("Linux"), d["results"]["FedAvg"]["MCC"] < FLOOR_MCC)
    return out


# Tick labels for panels too narrow for the full names (8 columns once the
# Handbook lands).
_TINY = {"ULB / stratified": "ULB\nstrat", "BankSim / stratified": "BankSim\nstrat",
         "BankSim / entity-disjoint": "BankSim\nentity", "Handbook / stratified": "Handbook\nstrat",
         "Handbook / entity-disjoint": "Handbook\nentity", "PaySim / stratified": "PaySim\nstrat",
         "AMLSim / stratified": "AMLSim\nstrat", "AMLSim / native banks": "AMLSim\nbanks"}


def cond_labels(tiny=False):
    flags = cond_flags()
    names = _TINY if tiny else _SHORT
    return [names[c] + ("†" if flags[c][0] else "") + ("*" if flags[c][1] else "")
            for c in CONDITIONS]


def figure_cross_dataset():
    fx = load("federated_cross_dataset.json")
    conds = [c for c in CONDITIONS if c in fx["mcc"]]
    lab = dict(zip(CONDITIONS, cond_labels()))
    labels = [lab[c] for c in conds]
    tiny = dict(zip(CONDITIONS, cond_labels(tiny=True)))
    tlabels = [tiny[c] for c in conds]
    dpc = fx["dp_cost"]
    test = [dpc[c]["testable"] for c in conds]
    n = len(conds)
    x = np.arange(n)

    fig = plt.figure(figsize=(10.0, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.62, wspace=0.24)

    # (a) MCC by method across every condition on record
    ax = fig.add_subplot(gs[0, :])
    w = 0.19
    for i, (m, colour) in enumerate(zip(METHODS, METHOD_COLOURS)):
        vals = [fx["mcc"][c][m] for c in conds]
        bars = ax.bar(x + (i - 1.5) * (w + 0.012), vals, w, color=colour, label=m)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, max(h, 0) + 0.015),
                        ha="center", va="bottom", fontsize=6.8, color=INK2, rotation=90)
    ax.axhline(FLOOR_MCC, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.06, 1.12)
    ax.set_ylabel("MCC on the held-out test set")
    ax.set_title(f"(a) Four federated methods, {n} conditions")
    # The floor goes in the legend: a label on the line collides with the bar values.
    handles, names = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=MUTED, lw=0.8, ls="--"))
    names.append(f"floor (MCC {FLOOR_MCC})")
    ax.legend(handles, names, ncol=5, loc="upper right", fontsize=8.2)
    style(ax)

    # (b) what DP does to accuracy -- the metric that hides the failure.  Plotted
    # as DP minus FedAvg, so a rise reads as a rise.
    ax = fig.add_subplot(gs[1, 0])
    d = [-dpc[c]["delta_acc"] for c in conds]
    cols = [MUTED if not t else (CRITICAL if v >= 0 else C1) for v, t in zip(d, test)]
    bars = ax.bar(x, d, 0.6, color=cols)
    ax.axhline(0, color=AXIS, lw=0.9)
    for b, v in zip(bars, d):
        ax.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v + (1.5 if v >= 0 else -1.5)),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.0, color=INK2)
    ax.set_xticks(x, tlabels, fontsize=7.0)
    lo, hi = min(d + [0.0]), max(d + [0.0])
    ax.set_ylim(lo - 0.12 * (hi - lo) - 4, hi + 0.5 * (hi - lo))
    ax.set_ylabel("accuracy under DP minus FedAvg (pp)")
    ax.set_title("(b) The DP collapse, as accuracy reports it")
    k_col = sum(1 for c in conds if dpc[c]["testable"] and dpc[c]["collapsed"])
    n_test = sum(test)
    ax.annotate(f"MCC falls below the floor in {k_col} of {n_test} testable\n"
                "conditions. Red: accuracy did not fall at all.\nGrey: FedAvg at the floor.",
                (0.02, 0.97), xycoords="axes fraction", ha="left", va="top",
                fontsize=7.4, color=INK2)
    style(ax)

    # (c) what Krum costs with no attacker, per condition
    ax = fig.add_subplot(gs[1, 1])
    kd = [fx["krum_delta"][c] for c in conds]
    cols = [MUTED if not t else (C1 if v >= 0 else CRITICAL) for v, t in zip(kd, test)]
    bars = ax.bar(x, kd, 0.6, color=cols)
    ax.axhline(0, color=AXIS, lw=0.9)
    for b, v in zip(bars, kd):
        ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v + (0.005 if v >= 0 else -0.005)),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.0, color=INK2)
    ax.set_xticks(x, tlabels, fontsize=7.0)
    lo, hi = min(kd + [0.0]), max(kd + [0.0])
    ax.set_ylim(lo - 0.2 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.set_ylabel("MCC(Krum) − MCC(FedAvg)")
    ax.set_title("(c) Krum without an attacker")
    k_neg = sum(1 for v, t in zip(kd, test) if t and v < 0)
    ax.annotate(f"Costs MCC in {k_neg} of {n_test} conditions above\nthe floor. "
                "Grey: FedAvg at the floor.", (0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=7.4, color=INK2)
    style(ax)

    fig.suptitle("The federated ablation in every condition measured so far",
                 fontsize=12.0, color=INK, fontweight="bold", y=0.975)
    fig.text(0.5, 0.005, MARK_NOTE, ha="center", fontsize=7.8, color=MUTED)
    save(fig, "fig_cross_dataset_comparison.png",
         ["federated_cross_dataset.json (collated from baselines*.json)"]
         + [f"{BASELINE_FILES[c]} ({c})" for c in conds])


# ============================================================ FIGURE 3
ARCH_LABEL = {
    "cnn": "CNN", "lstm": "LSTM", "dtcn": "DTCN (no attn.)",
    "dilated_attn": "ADTCN (ours)", "resnet": "ResNet-1D",
    "densenet": "DenseNet-1D", "efficientnet": "EfficientNet-1D",
}
ARCH_ORDER = ["dilated_attn", "dtcn", "cnn", "lstm", "resnet", "densenet", "efficientnet"]


def _bold_ours(ax, order):
    for tick, a in zip(ax.get_yticklabels(), order):
        if a == "dilated_attn":
            tick.set_color(INK)
            tick.set_fontweight("bold")


def figure_architecture_scoreboard():
    bs = load("banksim_temporal_grid.json")
    ulb = load("basepaper_comparison_ulb.json")["classifiers"]
    cells = bs["cells"]

    order = sorted(ARCH_ORDER, key=lambda a: cells[f"{a}|customer"]["MCC"])
    y = np.arange(len(order))
    h = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.1),
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})

    # (a) BankSim: global vs entity-linked windows
    ax = axes[0]
    g = [cells[f"{a}|global"]["MCC"] for a in order]
    gs = [cells[f"{a}|global"]["MCC_std"] for a in order]
    c = [cells[f"{a}|customer"]["MCC"] for a in order]
    cs = [cells[f"{a}|customer"]["MCC_std"] for a in order]
    ax.barh(y - h / 2 - 0.01, g, h, xerr=gs, color=C1, label="global windows",
            error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
    ax.barh(y + h / 2 + 0.01, c, h, xerr=cs, color=C2, label="entity-linked windows",
            error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
    for yy, v, e in zip(y - h / 2 - 0.01, g, gs):
        ax.annotate(f"{v:.3f}", (v + e + 0.02, yy), va="center", fontsize=7.6, color=INK2)
    for yy, v, e in zip(y + h / 2 + 0.01, c, cs):
        ax.annotate(f"{v:.3f}", (v + e + 0.02, yy), va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(y, [ARCH_LABEL[a] for a in order])
    _bold_ours(ax, order)
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("MCC (mean of 3 seeds, ±1 s.d.)")
    ax.set_title("(a) BankSim — seven architectures,\ntwo window types", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.2)
    style(ax, xgrid=True)

    # (b) ULB, same architectures, time-ordered global windows
    ax = axes[1]
    uorder = sorted(ARCH_ORDER, key=lambda a: ulb[a]["MCC"])
    uy = np.arange(len(uorder))
    uv = [ulb[a]["MCC"] for a in uorder]
    us = [ulb[a]["MCC_std"] for a in uorder]
    ax.barh(uy, uv, 0.55, xerr=us, color=[C4 if a == "dilated_attn" else C1 for a in uorder],
            error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
    for yy, v, e in zip(uy, uv, us):
        ax.annotate(f"{v:.3f}", (v + e + 0.015, yy), va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(uy, [ARCH_LABEL[a] for a in uorder])
    _bold_ours(ax, uorder)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("MCC (mean of 3 seeds, ±1 s.d.)")
    ax.set_title("(b) ULB — the same seven,\nglobal windows only")
    # one series, ADTCN emphasised -- a legend box would name nothing the
    # bold tick label does not already say.
    ax.annotate("gold = the detector we built", (0.90, 0.35), ha="right",
                fontsize=7.6, color=C4, fontweight="bold")
    style(ax, xgrid=True)

    # (c) entity-linkage gain
    ax = axes[2]
    gain = bs["entity_linkage_gain_mcc"]
    gorder = sorted(ARCH_ORDER, key=lambda a: gain[a])
    gy = np.arange(len(gorder))
    gv = [gain[a] for a in gorder]
    ax.barh(gy, gv, 0.55, color=[C4 if a == "dilated_attn" else C3 for a in gorder])
    for yy, v in zip(gy, gv):
        ax.annotate(f"+{v:.4f}", (v + 0.004, yy), va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(gy, [ARCH_LABEL[a] for a in gorder])
    _bold_ours(ax, gorder)
    ax.set_xlim(0, 0.245)
    ax.set_xlabel("MCC gained by entity linkage")
    ax.set_title("(c) The window helped\nevery architecture")
    style(ax, xgrid=True)

    fig.tight_layout(w_pad=2.4)
    fig.suptitle(
        "Model evaluation — the detector is a measured component, not a winning one",
        fontsize=12.0, color=INK, fontweight="bold", y=1.05,
    )
    save(fig, "fig_architecture_scoreboard.png",
         ["banksim_temporal_grid.json", "basepaper_comparison_ulb.json"])


# ============================================================ FIGURE 4 + tables
SCORE_METRICS = [
    ("MCC", "MCC", "{:.3f}", 1.0),
    ("Accuracy", "Accuracy (%)", "{:.2f}", 100.0),
    ("Precision", "Precision (%)", "{:.2f}", 100.0),
    ("Sensitivity", "Recall / Sensitivity (%)", "{:.2f}", 100.0),
    ("Specificity", "Specificity (%)", "{:.2f}", 100.0),
    ("F1_Score", "F1 score (%)", "{:.2f}", 100.0),
]
# BASELINE_FILES is built with the condition list (FIGURE 2), from the files on disk.


def figure_score_index():
    data = {c: load(f)["results"] for c, f in BASELINE_FILES.items()}
    labels = cond_labels(tiny=True)
    n = len(CONDITIONS)

    # 3 rows x 2 columns: at \linewidth on A4 this scales down far less than a
    # 2x3 strip, which is what makes the tick labels survive the page.
    # Values are deliberately NOT printed on the bars -- the score-index table
    # sits directly under this figure and is the table view for every number.
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 9.4))
    x = np.arange(n)
    w = 0.19
    for ax, (key, label, fmt, top) in zip(axes.ravel(), SCORE_METRICS):
        for i, (m, colour) in enumerate(zip(METHODS, METHOD_COLOURS)):
            vals = [data[c][m][key] for c in CONDITIONS]
            ax.bar(x + (i - 1.5) * (w + 0.012), vals, w, color=colour, label=m)
        ax.set_xticks(x, labels, fontsize=7.4)
        ax.set_ylim(-top * 0.03, top * 1.06)
        ax.set_title(label)
        style(ax)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 0.945), h_pad=2.2)
    fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.972),
               ncol=4, fontsize=9.5)
    fig.suptitle(
        f"Score index — every metric, four federated methods, all {n} conditions on record",
        fontsize=12.0, color=INK, fontweight="bold", y=0.995,
    )
    n_ds = len({c.split(" / ")[0] for c in CONDITIONS})
    missing = [c for c, _, _ in ALL_CONDITIONS if c not in CONDITIONS]
    note = (f"{n_ds} datasets in {n} conditions."
            + (f" Not on disk yet, so not drawn: {', '.join(missing)}." if missing else "")
            + "\n" + MARK_NOTE)
    fig.text(0.5, 0.008, note, ha="center", fontsize=8.4, color=MUTED)
    save(fig, "fig_score_index.png", [f"{f} ({c})" for c, f in BASELINE_FILES.items()])
    return data


TABLE_METRICS = ["MCC", "Accuracy", "Precision", "Sensitivity", "Specificity",
                 "NPV", "F1_Score", "FPR", "TP", "FP", "FN"]
TEX_NAME = {"F1_Score": "F1", "Sensitivity": "Recall"}


def write_score_tables(data):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGDIR / "score_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["condition", "method"] + TABLE_METRICS)
        for c in CONDITIONS:
            for m in METHODS:
                wr.writerow([c, m] + [data[c][m][k] for k in TABLE_METRICS])
    print(f"  wrote {csv_path.relative_to(ROOT)}")

    ncol = 1 + len(TABLE_METRICS)
    lines = [
        "% GENERATED by experiments/make_report_figures.py -- do not hand-edit.",
        "% Regenerate with:  python -m experiments.make_report_figures",
        "\\begin{tabular}{@{}l" + "r" * len(TABLE_METRICS) + "@{}}",
        "\\toprule",
        "Method & " + " & ".join(TEX_NAME.get(k, k) for k in TABLE_METRICS) + " \\\\",
    ]
    flags = cond_flags()
    for ci, c in enumerate(CONDITIONS):
        mark = (("\\textsuperscript{\\dag}" if flags[c][0] else "")
                + ("\\textsuperscript{*}" if flags[c][1] else ""))
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}"
                     f"{{\\itshape {c}{mark}}} \\\\[1pt]")
        for m in METHODS:
            row = data[c][m]
            cells = []
            for k in TABLE_METRICS:
                v = row[k]
                if k == "MCC":
                    cells.append(f"{v:.3f}")
                elif k in ("TP", "FP", "FN"):
                    cells.append(f"{int(v)}")
                else:
                    cells.append(f"{v:.2f}")
            lines.append(f"\\quad {m} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tex_path = FIGDIR / "score_index_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {tex_path.relative_to(ROOT)}")


# ============================================================ FIGURE 5
def figure_process_optimisation():
    knee = load("objective_size_knee.json")
    rep = load("obj13_surrogate_repair_banksim.json")
    opt_ulb = load("basepaper_optimisers_ulb.json")["optimisers"]
    opt_bs = load("basepaper_optimisers_banksim.json")["optimisers"]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.0))

    # (a) the objective's noise, against surrogate size
    ax = axes[0]
    for ds, colour, lab in (("ulb", C1, "ULB"), ("banksim", C2, "BankSim")):
        rows = [r["surrogate_rows"] for r in knee["datasets"][ds]]
        n2s = [r["noise_to_signal"] for r in knee["datasets"][ds]]
        ax.plot(rows, n2s, "o-", color=colour, label=lab, markersize=5)
    ax.axhline(1.0, color=MUTED, lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([2000, 5000, 10000, 20000], ["2k", "5k", "10k", "20k"])
    ax.set_xlim(1700, 28000)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("surrogate training rows")
    ax.set_ylabel("noise-to-signal of the objective")
    ax.set_title("(a) The objective stays noisy at\nevery size we can afford")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C1, label="ULB", markersize=5),
                       Line2D([0], [0], marker="o", color=C2, label="BankSim", markersize=5),
                       Line2D([0], [0], color=MUTED, lw=1,
                              label="1.0 = noise as large as the\nwhole range being searched")],
              loc="lower left", fontsize=7.6)
    style(ax)

    # (b) the 16.3-hour side-by-side
    ax = axes[1]
    modes = rep["modes"]
    mode_colour = {"legacy": C1, "deterministic": C2, "averaged": C3}
    xs, hs, es, cols = [], [], [], []
    ticks, ticklabels = [], []
    pos = 0.0
    for mode in modes:
        runs = [r for r in rep["runs"] if r["eval_mode"] == mode]
        centre = []
        for r in runs:
            xs.append(pos)
            hs.append(r["yardstick_mean"])
            es.append(r["yardstick_std"])
            cols.append(mode_colour[mode])
            centre.append(pos)
            pos += 1.0
        ticks.append(float(np.mean(centre)))
        ticklabels.append(mode)
        pos += 0.8
    ax.bar(xs, hs, 0.82, yerr=es, color=cols,
           error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
    dflt = rep["references"]["hand_set_default"]["yardstick_mean"]
    ship = rep["references"]["shipped_dbboa"]["yardstick_mean"]
    ax.axhline(dflt, color=CRITICAL, lw=1.4)
    ax.axhline(ship, color=MUTED, lw=1.0, zorder=1)
    ax.set_xticks(ticks, ticklabels)
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_ylim(2.75, 4.45)
    ax.set_ylabel("held-out objective (7 shared draws)")
    ax.set_title("(b) Nine searches, 16.3 h —\nnone beat the default")
    ax.set_xlabel("three search seeds per surrogate mode")
    ax.legend(handles=[
        Line2D([0], [0], color=CRITICAL, lw=1.6, label=f"hand-set default  {dflt:.3f}"),
        Line2D([0], [0], color=MUTED, lw=1.2, label=f"the configuration we shipped  {ship:.3f}"),
    ], loc="upper left", fontsize=7.6)
    style(ax)

    # (c) what each search chose
    ax = axes[2]
    for i, mode in enumerate(modes):
        runs = [r for r in rep["runs"] if r["eval_mode"] == mode]
        vals = [r["chosen_filters"] for r in runs]
        ax.plot([i, i], [min(vals), max(vals)], color=mode_colour[mode], lw=1.6, alpha=0.45)
        ax.scatter([i] * len(vals), vals, s=48, color=mode_colour[mode], zorder=3,
                   edgecolors=SURFACE, linewidths=1.2)
        ax.annotate(f"spread {max(vals) - min(vals)}", (i, max(vals) + 9), ha="center",
                    fontsize=8.2, color=INK2)
    ax.axhline(rep["references"]["hand_set_default"]["filters"], color=CRITICAL, lw=1.4)
    ax.set_xticks(range(len(modes)), modes)
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 235)
    ax.set_ylabel("filter count chosen by the search")
    ax.set_title("(c) Shared draws converge the search;\ndeterminism does not")
    ax.legend(handles=[Line2D([0], [0], color=CRITICAL, lw=1.6,
                              label="hand-set default = 128 filters")],
              loc="upper left", fontsize=7.6)
    style(ax)

    fig.tight_layout(w_pad=2.6)
    fig.suptitle(
        "Process optimisation — auditing the search the framework is named after",
        fontsize=12.0, color=INK, fontweight="bold", y=1.05,
    )
    save(fig, "fig_process_optimisation.png",
         ["objective_size_knee.json", "obj13_surrogate_repair_banksim.json"])

    ceiling = {
        "ULB": [v["obf2_stats"]["best"] for v in opt_ulb.values()],
        "BankSim": [v["obf2_stats"]["best"] for v in opt_bs.values()],
    }
    print("      optimiser best-objective (theoretical ceiling 5.0000): "
          f"ULB {['%.4f' % v for v in ceiling['ULB']]}, "
          f"BankSim {['%.4f' % v for v in ceiling['BankSim']]}")
    return ceiling


# ============================================================ FIGURE 6
def figure_significance():
    ms = load("detector_multiseed.json")["summary"]
    rep = load("obj13_surrogate_repair_banksim.json")

    tuned = ms["dbboa_tuned"]["per_seed"]
    default = ms["hand_set_default"]["per_seed"]
    seeds = sorted(tuned, key=int)
    t = np.array([tuned[s]["MCC"] for s in seeds])
    d = np.array([default[s]["MCC"] for s in seeds])
    diff = t - d
    tstat, pval = stats.ttest_rel(t, d)
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    ci = (diff.mean() - crit * se, diff.mean() + crit * se)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 0.85, 1.35]})

    # (a) paired per-seed MCC
    ax = axes[0]
    for i in range(n):
        ax.plot([0, 1], [d[i], t[i]], color=MUTED, lw=1.0, zorder=1)
    ax.scatter([0] * n, d, s=54, color=C2, zorder=3, edgecolors=SURFACE, linewidths=1.2,
               label="hand-set default (128 / 150)")
    ax.scatter([1] * n, t, s=54, color=C1, zorder=3, edgecolors=SURFACE, linewidths=1.2,
               label="DB-BOA tuned (142 / 76)")
    ax.plot([-0.16, 0.16], [d.mean()] * 2, color=C2, lw=2.6)
    ax.plot([0.84, 1.16], [t.mean()] * 2, color=C1, lw=2.6)
    ax.annotate(f"mean {d.mean():.3f}", (0.0, 0.845), ha="center", fontsize=8.2,
                color=C2, fontweight="bold")
    ax.annotate(f"mean {t.mean():.3f}", (1.0, 0.845), ha="center", fontsize=8.2,
                color=C1, fontweight="bold")
    ax.set_xticks([0, 1], ["default", "tuned"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.56, 0.90)
    ax.set_ylabel("MCC on ULB (5 training seeds)")
    ax.set_title("(a) The detector headline is\na distribution, not a number")
    ax.set_xlabel("each grey line is one training seed")
    ax.legend(loc="lower center", fontsize=7.6)
    style(ax)

    # (b) the paired difference and its interval
    ax = axes[1]
    ax.axhline(0, color=CRITICAL, lw=1.2)
    ax.annotate("no difference", (0.54, 0.005), ha="right", fontsize=8.2, color=CRITICAL)
    ax.scatter(np.full(n, 0.18), diff, s=44, color=MUTED, zorder=3,
               edgecolors=SURFACE, linewidths=1.2, label="per-seed difference")
    ax.errorbar([-0.18], [diff.mean()], yerr=[[diff.mean() - ci[0]], [ci[1] - diff.mean()]],
                fmt="o", color=C1, markersize=8, capsize=5, elinewidth=1.6,
                label="mean, 95% CI")
    ax.annotate(f"{diff.mean():+.3f}", (-0.18, diff.mean()), textcoords="offset points",
                xytext=(-9, 0), ha="right", va="center", fontsize=8.8, color=C1,
                fontweight="bold")
    ax.annotate(f"paired t-test\nt = {tstat:.2f},  p = {pval:.2f}   (n = {n})",
                (-0.55, 0.205), ha="left", fontsize=8.2, color=INK2)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.12, 0.25)
    ax.set_ylabel("MCC(tuned) − MCC(default)")
    ax.set_title("(b) The interval\ncontains zero")
    ax.legend(loc="lower left", fontsize=7.6)
    style(ax)

    # (c) OBJ-13 paired differences, per search, on shared draws
    ax = axes[2]
    ref = np.array(rep["references"]["hand_set_default"]["yardstick_draws"])
    mode_colour = {"legacy": C1, "deterministic": C2, "averaged": C3}
    labels, means, errs, cols = [], [], [], []
    for r in rep["runs"]:
        dr = np.array(r["yardstick_draws"]) - ref
        labels.append(f"{r['eval_mode'][:4]}/{r['search_seed']}")
        means.append(dr.mean())
        errs.append(stats.t.ppf(0.975, len(dr) - 1) * dr.std(ddof=1) / np.sqrt(len(dr)))
        cols.append(mode_colour[r["eval_mode"]])
    xs = np.arange(len(means))
    ax.axhline(0, color=CRITICAL, lw=1.2)
    for xi, m, e, c in zip(xs, means, errs, cols):
        ax.errorbar([xi], [m], yerr=[[e], [e]], fmt="o", color=c, markersize=6,
                    capsize=3, elinewidth=1.3)
    ax.set_ylim(-0.95, 0.62)
    ax.annotate(f"mean of all nine:  {np.mean(means):+.4f}", (-0.4, -0.88),
                ha="left", fontsize=8.2, color=INK2, fontweight="bold")
    ax.set_xticks(xs, labels, rotation=45, ha="right")
    ax.set_ylabel("paired Δ objective vs default")
    ax.set_title("(c) 0 of 9 searches beat it —\npaired on 7 shared draws")
    ax.legend(handles=[Line2D([0], [0], color=CRITICAL, lw=1.4,
                              label="0 = the hand-set default")]
              + [Line2D([0], [0], marker="o", color=mode_colour[m], lw=0, label=m,
                        markersize=6) for m in rep["modes"]],
              loc="upper center", fontsize=7.6, ncol=2)
    style(ax)

    fig.tight_layout(w_pad=2.6)
    fig.suptitle(
        "Statistical significance — both negative results, tested rather than asserted",
        fontsize=12.0, color=INK, fontweight="bold", y=1.05,
    )
    save(fig, "fig_statistical_significance.png",
         ["detector_multiseed.json", "obj13_surrogate_repair_banksim.json"])
    print(f"      paired t-test tuned vs default: t={tstat:.4f} p={pval:.4f} "
          f"mean diff={diff.mean():+.4f} CI=({ci[0]:+.4f}, {ci[1]:+.4f})")
    print(f"      obj13 mean paired delta over nine searches: {np.mean(means):+.4f}")
    return {"t": float(tstat), "p": float(pval), "mean_diff": float(diff.mean()),
            "ci": [float(ci[0]), float(ci[1])],
            "obj13_mean_paired_delta": float(np.mean(means))}


# ============================================================ FIGURE 7
def figure_system_performance():
    fab = load("fabric_consensus_measured.json")
    sc = load("scalability_sweep.json")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.0))

    # (a) commit latency
    ax = axes[0]
    lat = fab["latency_ms"]
    names = ["min", "p50", "mean", "p95", "max"]
    vals = [lat["min_ms"], lat["p50_ms"], lat["mean_ms"], lat["p95_ms"], lat["max_ms"]]
    bars = ax.bar(names, vals, 0.55, color=[C1, C1, C4, C1, C1])
    bar_labels(ax, bars, fmt="{:.0f}", dy=22, fontsize=8.2)
    ax.axhline(300, color=CRITICAL, lw=1.3)
    ax.set_ylim(0, 3600)
    ax.set_ylabel("commit latency (ms)")
    ax.set_title(f"(a) Fabric commit latency\n(n = {lat['n']} transactions)")
    ax.legend(handles=[Line2D([0], [0], color=CRITICAL, lw=1.5,
                              label="300 ms — the bonus threshold\nthe incentive scheme pays for")],
              loc="upper left", fontsize=7.6)
    style(ax)

    # (b) goodput
    ax = axes[1]
    conc = [t["concurrency"] for t in fab["throughput"]]
    tps = [t["tps"] for t in fab["throughput"]]
    ax.plot(conc, tps, "o-", color=C1, markersize=5)
    peak = fab["peak_tps_concurrency"]
    pi = conc.index(peak)
    ax.scatter([peak], [tps[pi]], s=95, color=C4, zorder=4, edgecolors=SURFACE, linewidths=1.4)
    ax.annotate(f"peak {tps[pi]:.1f} TPS\nat concurrency {peak}", (peak, tps[pi]),
                textcoords="offset points", xytext=(4, -26), fontsize=8.2, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(conc, [str(c) for c in conc])
    ax.set_ylim(0, 48)
    ax.set_xlabel("client concurrency")
    ax.set_ylabel("goodput (committed TPS)")
    ax.set_title("(b) Goodput peaks, then falls")
    style(ax)

    # (c) committed vs failed -- the knee, in transactions
    ax = axes[2]
    xs = np.arange(len(conc))
    comm = [t["committed"] for t in fab["throughput"]]
    fail = [t["failed"] for t in fab["throughput"]]
    ax.bar(xs, comm, 0.6, color=C3, label="committed")
    ax.bar(xs, fail, 0.6, bottom=comm, color=CRITICAL, label="failed (MVCC conflict)")
    for xi, committed, f in zip(xs, comm, fail):
        if f:
            ax.annotate(f"{f}% fail", (xi, committed + f / 2), ha="center", va="center",
                        rotation=90, fontsize=8.4, color="white", fontweight="bold")
    ax.set_xticks(xs, [str(c) for c in conc])
    ax.set_ylim(0, 152)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("client concurrency")
    ax.set_ylabel("transactions out of 100")
    ax.set_title("(c) The knee is read-write conflict,\nnot queueing")
    ax.legend(loc="upper left", fontsize=8.0)
    style(ax)

    fig.tight_layout(w_pad=2.6)
    fig.suptitle(
        "System performance — measured on a real Hyperledger Fabric network, not simulated",
        fontsize=12.0, color=INK, fontweight="bold", y=1.05,
    )
    save(fig, "fig_system_performance.png", ["fabric_consensus_measured.json"])

    # ---- Shapley cost, its own figure (different unit -- never a second y-axis)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
    rt = sc["runtime"]
    ns = [r["n_orgs"] for r in rt]
    ax = axes[0]
    ex = [(r["n_orgs"], r["exact_time_sec"]) for r in rt if r.get("exact_time_sec") is not None]
    ax.plot([p[0] for p in ex], [p[1] for p in ex], "o-", color=C1, markersize=4.5,
            label="exact Shapley")
    ax.plot(ns, [r["mc_time_sec"] for r in rt], "s-", color=C2, markersize=4.5,
            label="Monte-Carlo estimate")
    ax.set_yscale("log")
    ax.set_xlabel("organisations in the federation")
    ax.set_ylabel("wall-clock seconds (log)")
    ax.set_title("(a) Attribution cost is structural")
    ax.legend(loc="upper left")
    style(ax)

    ax = axes[1]
    # only the sizes where an exact reference exists can be scored for agreement
    scored = [r for r in rt if "fidelity" in r]
    sn = [r["n_orgs"] for r in scored]
    top1 = [1.0 if r["fidelity"]["top1_match"] else 0.0 for r in scored]
    ax.bar(sn, top1, 0.62, color=[C3 if v else CRITICAL for v in top1])
    for x_, v in zip(sn, top1):
        if not v:
            ax.annotate("✗", (x_, 0.03), ha="center", fontsize=8.8, color=CRITICAL)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 1], ["disagrees", "agrees"])
    ax.set_xticks(sn)
    ax.set_xlabel("organisations in the federation")
    ax.set_title("(b) Monte-Carlo top-1 agreement is\nintermittent, not a threshold")
    ax.annotate("the failures do not start at one federation size\nand stay — so \"top-1 fails "
                "from n ≈ 6\" is withdrawn",
                (max(sn) + 0.4, 1.24), ha="right", fontsize=7.6, color=MUTED)
    style(ax)
    fig.tight_layout(w_pad=2.6)
    fig.suptitle("Scalability of contribution attribution", fontsize=10, color=INK,
                 fontweight="bold", y=1.05)
    save(fig, "fig_scalability.png", ["scalability_sweep.json"])


# ============================================================ main
def main():
    print(f"figures -> {FIGDIR}")
    figure_memorisation()
    figure_cross_dataset()
    figure_architecture_scoreboard()
    data = figure_score_index()
    write_score_tables(data)
    figure_process_optimisation()
    figure_significance()
    figure_system_performance()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
