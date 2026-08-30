"""
experiments/plot_federated_confusion.py
=======================================
Renders the confusion matrices of the four federated-ablation configurations
(FedAvg / FedAvg+Krum / FedAvg+DP / proposed Krum+DP+Shapley) directly from
the saved TP/TN/FP/FN counts in results/baselines.json (Chapter 5, §5.4).
No model is re-run: this is a pure visualisation of the persisted run.

Run:  python3 experiments/plot_federated_confusion.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family"   : "DejaVu Sans",
    "font.size"     : 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi"    : 100,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

PANELS = [
    ("FedAvg",       "FedAvg"),
    ("FedAvg+Krum",  "FedAvg + Krum"),
    ("FedAvg+DP",    r"FedAvg + DP ($\epsilon$=1.0)"),
    ("DB-BOA-ADTCN", r"Proposed (Krum+DP+Shapley, $\epsilon$=1.0)"),
]


def main():
    with open(os.path.join(RESULTS_DIR, "baselines.json")) as fh:
        results = json.load(fh)["results"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (key, title) in zip(axes.flat, PANELS):
        m = results[key]
        TP, TN, FP, FN = (int(m[k]) for k in ("TP", "TN", "FP", "FN"))
        matrix = np.array([[TN, FP], [FN, TP]])
        im = ax.imshow(matrix, cmap="RdPu", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}",
                        ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Normal", "Predicted Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Normal", "Actual Fraud"])
        ax.set_title(f"{title}\nMCC = {m['MCC']:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Federated FL-ADTCN Confusion Matrices — ULB Test Set (n = 56,962)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(RESULTS_DIR, "confusion_matrix_federated.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Saved → {path}", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
