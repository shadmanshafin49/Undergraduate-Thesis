"""
utils/metrics.py
================
Performance metrics used in the paper (§VII, Tables 3 & 4).

All formulas are implemented exactly as stated in the paper:

    Notation: Tap=TP, Tan=TN, Fbp=FP, Fbn=FN

    Accuracy    (Eq.12): (TP+TN) / (TP+TN+FP+FN)
    Precision   (Eq.13): TP / (TP+FP)
    NPV         (Eq.14): TN / (TN+FN)
    MCC         (Eq.15): (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    FPR         (Eq.16): FP / (FP+TN)
    Sensitivity (Eq.17): TP / (TP+FN)          [Recall / TPR]
    FNR         (Eq.18): FN / (FN+TP)
    Specificity (Eq.19): TN / (TN+FP)          [corrected from paper typo]
    FDR         (Eq.20): FP / (FP+TP)
    F1-Score    (Eq.21): 2*TP / (2*TP+FP+FN)   [standard harmonic mean]
"""

import numpy as np
from typing import Dict


def confusion_components(y_true: np.ndarray,
                         y_pred: np.ndarray) -> tuple:
    """
    Extract raw TP, TN, FP, FN from binary label arrays.
    Positive class = 1 (fraud).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return TP, TN, FP, FN


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute the full metric suite from the paper.

    Returns
    -------
    dict with keys:
        Accuracy, Precision, Sensitivity, Specificity,
        NPV, FPR, FNR, FDR, F1_Score, MCC
    Values are percentages (×100) EXCEPT MCC which is [-1, +1].
    """
    TP, TN, FP, FN = confusion_components(y_true, y_pred)
    eps = 1e-10   # avoid division by zero

    accuracy    = 100.0 * (TP + TN)  / (TP + TN + FP + FN + eps)
    precision   = 100.0 * TP         / (TP + FP + eps)
    npv         = 100.0 * TN         / (TN + FN + eps)
    sensitivity = 100.0 * TP         / (TP + FN + eps)   # recall / TPR
    specificity = 100.0 * TN         / (TN + FP + eps)
    fpr         = 100.0 * FP         / (FP + TN + eps)
    fnr         = 100.0 * FN         / (FN + TP + eps)
    fdr         = 100.0 * FP         / (FP + TP + eps)
    f1          = 100.0 * 2*TP       / (2*TP + FP + FN + eps)

    denom_mcc = np.sqrt(
        float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + eps
    )
    mcc = (TP * TN - FP * FN) / denom_mcc

    return {
        "Accuracy"   : accuracy,
        "Precision"  : precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "NPV"        : npv,
        "FPR"        : fpr,
        "FNR"        : fnr,
        "FDR"        : fdr,
        "F1_Score"   : f1,
        "MCC"        : mcc,
        "TP"         : TP,
        "TN"         : TN,
        "FP"         : FP,
        "FN"         : FN,
    }


def obf2_value(metrics: dict) -> float:
    """
    Compute the DB-BOA objective Obf2 from a metrics dict.

    Obf2 = Acc + Pre + NPV + MCC + 1/FPR   (Eq.11)
    (Values are normalised to [0,1] range before summing.)
    """
    eps = 1e-8
    return (metrics["Accuracy"]   / 100.0 +
            metrics["Precision"]  / 100.0 +
            metrics["NPV"]        / 100.0 +
            metrics["MCC"]              +
            1.0 / (metrics["FPR"] / 100.0 + eps))


def coalition_score(metrics: dict) -> float:
    """
    Bounded scalar quality score for federated aggregation / Shapley attribution.

    Returns **balanced accuracy** = (Sensitivity + Specificity) / 2, in [0, 1].

    Why not obf2_value() here?
    -------------------------
    obf2_value() (Eq.11) contains an unbounded 1/FPR term that explodes to ~1e8
    whenever a coalition reaches FPR=0.  Using it as a Shapley coalition value
    makes marginal contributions numerically degenerate and the resulting
    aggregation weights meaningless.  Balanced accuracy is bounded, robust to the
    0.17% class imbalance, and gives interpretable contribution attribution:
    a model that catches fraud (high sensitivity) without over-flagging normal
    transactions (high specificity) scores near 1.0, while an always-fraud
    attacker scores ~0.5 (sensitivity 1.0, specificity 0.0) and *lowers* any
    coalition it joins — so Shapley correctly assigns it a near-zero weight.
    """
    return (metrics["Sensitivity"] / 100.0 + metrics["Specificity"] / 100.0) / 2.0


def print_metrics_table(metrics: dict, model_name: str = "DB-BOA-ADTCN"):
    """Pretty-print a metrics table matching Table 4 of the paper."""
    border = "═" * 55
    print(f"\n{border}")
    print(f"  Performance Metrics — {model_name}")
    print(border)
    rows = [
        ("Accuracy",    metrics["Accuracy"],    "%"),
        ("Precision",   metrics["Precision"],   "%"),
        ("Sensitivity", metrics["Sensitivity"], "%"),
        ("Specificity", metrics["Specificity"], "%"),
        ("NPV",         metrics["NPV"],         "%"),
        ("FPR",         metrics["FPR"],         "%"),
        ("FNR",         metrics["FNR"],         "%"),
        ("FDR",         metrics["FDR"],         "%"),
        ("F1-Score",    metrics["F1_Score"],    "%"),
        ("MCC",         metrics["MCC"],         ""),
    ]
    for name, val, unit in rows:
        print(f"  {name:<20}  {val:>10.5f} {unit}")
    print(f"  {'TP':<20}  {metrics['TP']:>10d}")
    print(f"  {'TN':<20}  {metrics['TN']:>10d}")
    print(f"  {'FP':<20}  {metrics['FP']:>10d}")
    print(f"  {'FN':<20}  {metrics['FN']:>10d}")
    print(border, flush=True)


def baseline_metrics():
    """
    Baseline metrics for comparison with DB-BOA-ADTCN on the ULB dataset.

    The original paper's hardcoded numbers (MBO-ADTCN, WSA-ADTCN, etc.) were
    produced on synthetic data and cannot be validly compared against ULB results.
    They have been removed.

    Populate these dicts by running the following configurations on the ULB
    dataset and recording their test-set metrics:
      algo_results — FedAvg (McMahan et al., AISTATS 2017);
                     FedAvg + Krum (use_krum=True, use_dp=False);
                     FedAvg + DP   (use_krum=False, use_dp=True)
      clf_results  — same models with different classifier architectures

    Until those runs are complete, comparison plots will show only the proposed
    model.
    """
    algo_results: dict = {}   # populate with ULB-evaluated algorithm baselines
    clf_results:  dict = {}   # populate with ULB-evaluated classifier baselines
    return algo_results, clf_results
