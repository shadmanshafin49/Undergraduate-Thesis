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
    Reference values from Table 3 & 4 of the paper for comparison.
    Used in visualiser to draw comparison bars.
    """
    # Table 3: algorithm comparison
    algo_results = {
        "MBO-ADTCN" : {
            "Accuracy":88.4, "Precision":88.35, "Sensitivity":88.17,
            "Specificity":88.63,"NPV":88.63,"FPR":11.37,"FNR":11.83,
            "FDR":11.65,"F1_Score":88.26,"MCC":0.768
        },
        "WSA-ADTCN" : {
            "Accuracy":91.85,"Precision":91.47,"Sensitivity":92.11,
            "Specificity":91.59,"NPV":91.59,"FPR":8.41,"FNR":7.89,
            "FDR":8.53,"F1_Score":91.79,"MCC":0.837
        },
        "DBOA-ADTCN": {
            "Accuracy":90.0, "Precision":90.0, "Sensitivity":89.79,
            "Specificity":90.21,"NPV":90.21,"FPR":9.79,"FNR":10.21,
            "FDR":10.03,"F1_Score":89.88,"MCC":0.800
        },
        "BOA-ADTCN" : {
            "Accuracy":93.8, "Precision":93.47,"Sensitivity":94.03,
            "Specificity":93.57,"NPV":93.57,"FPR":6.43,"FNR":5.97,
            "FDR":6.53,"F1_Score":93.75,"MCC":0.876
        },
    }
    # Table 4: classifier comparison
    clf_results = {
        "EfficientNet": {
            "Accuracy":89.05,"Precision":88.89,"Sensitivity":88.98,
            "Specificity":89.12,"NPV":89.12,"FPR":10.88,"FNR":11.02,
            "FDR":11.11,"F1_Score":88.93,"MCC":0.781
        },
        "ResNet"      : {
            "Accuracy":92.55,"Precision":92.34,"Sensitivity":92.62,
            "Specificity":92.48,"NPV":92.48,"FPR":7.52,"FNR":7.38,
            "FDR":7.66,"F1_Score":92.48,"MCC":0.851
        },
        "DenseNet"    : {
            "Accuracy":90.65,"Precision":90.51,"Sensitivity":90.60,
            "Specificity":90.70,"NPV":90.70,"FPR":9.30,"FNR":9.40,
            "FDR":9.49,"F1_Score":90.55,"MCC":0.813
        },
        "DTCN"        : {
            "Accuracy":94.8, "Precision":94.74,"Sensitivity":94.74,
            "Specificity":94.86,"NPV":94.86,"FPR":5.14,"FNR":5.26,
            "FDR":5.26,"F1_Score":94.74,"MCC":0.896
        },
    }
    return algo_results, clf_results
