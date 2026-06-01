"""
run_baselines.py
================
Runs the four ULB-comparable baseline configurations and prints their test-set
metrics in the format needed to populate baseline_metrics() in utils/metrics.py.

Usage
-----
    python3 run_baselines.py

Output
------
For each run the script prints a Python dict literal you can paste into
baseline_metrics() in utils/metrics.py.

Runs performed
--------------
| # | use_krum | use_dp | use_shapley | Label         |
|---|----------|--------|-------------|---------------|
| 1 | False    | False  | False       | FedAvg        |
| 2 | True     | False  | False       | FedAvg+Krum   |
| 3 | False    | True   | False       | FedAvg+DP     |
| 4 | True     | True   | True        | DB-BOA-ADTCN  |

All four runs use the same dataset, split, and DB-BOA hyperparameter search so
the results are directly comparable.

References
----------
McMahan et al., "Communication-Efficient Learning of Deep Networks from
Decentralized Data", AISTATS 2017.  (FedAvg baseline)
"""

import copy
import json
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                  import ADTCN_CONFIG, FEDERATION_CONFIG
from data.data_loader        import FinancialDataLoader
from models.adtcn            import ADTCN
from models.federated_adtcn  import FederatedADTCN
from models.federation_manager import FederationManager
from utils.metrics           import compute_all_metrics, print_metrics_table


BASELINE_CONFIGS = [
    {"label": "FedAvg",        "use_krum": False, "use_dp": False, "use_shapley": False},
    {"label": "FedAvg+Krum",   "use_krum": True,  "use_dp": False, "use_shapley": False},
    {"label": "FedAvg+DP",     "use_krum": False, "use_dp": True,  "use_shapley": False},
    {"label": "DB-BOA-ADTCN",  "use_krum": True,  "use_dp": True,  "use_shapley": True},
]


def _avg_weights(weights_list: list, counts: list = None) -> list:
    """
    FedAvg aggregation.  When counts is provided, uses data-size-weighted
    averaging (McMahan et al., AISTATS 2017: w_global ← Σ_k (n_k/n)·w_k).
    Without counts, falls back to equal-weight averaging (unweighted FedAvg).
    With the 50/30/20 split, the correct McMahan weights are [0.5, 0.3, 0.2].
    """
    n = len(weights_list)
    if counts is not None and len(counts) == n:
        total = sum(counts)
        frac  = [c / total for c in counts]
    else:
        frac  = [1.0 / n] * n   # equal weights (unweighted FedAvg)

    n_arrays = len(weights_list[0])
    return [
        sum(frac[i] * weights_list[i][a] for i in range(n))
        for a in range(n_arrays)
    ]


def run_one_baseline(loader, X_train, X_val, X_test, y_train, y_val, y_test,
                     adtcn_base, cfg_override: dict) -> dict:
    """
    Train all three org models, run one federated round, evaluate on test set.
    Returns the test-set metrics dict.
    """
    fed_cfg = copy.deepcopy(FEDERATION_CONFIG)
    fed_cfg.update(cfg_override)

    org_splits = loader.split_for_orgs(X_train, y_train)
    org_models = {}
    org_counts = []   # sample counts per org, for McMahan size-weighted FedAvg
    for org_name, (X_org, y_org) in org_splits.items():
        m = FederatedADTCN(cfg=ADTCN_CONFIG)
        m.optimal_params = adtcn_base.optimal_params
        m.fit(X_org, y_org, verbose=False)
        org_models[org_name] = m
        org_counts.append(len(y_org))

    # Federation round
    use_dp   = fed_cfg.get("use_dp", False)
    dp_eps   = fed_cfg.get("dp_epsilon", 1.0)
    dp_delta = fed_cfg.get("dp_delta",   1e-5)

    if use_dp:
        weights_list = [m.extract_weights_with_dp(epsilon=dp_eps, delta=dp_delta)
                        for m in org_models.values()]
    else:
        weights_list = [m.extract_weights() for m in org_models.values()]

    use_krum = fed_cfg.get("use_krum", False)
    # Note: use_shapley is not evaluated here because Shapley affects only
    # the token-distribution incentive weights, not the global model itself.
    # The aggregated model is always Krum-selected (use_krum=True) or
    # FedAvg-averaged (use_krum=False), so accuracy comparison is valid.
    if use_krum:
        fed_mgr       = FederationManager(n_orgs=3, cfg=fed_cfg)
        global_w, _, _ = fed_mgr._krum_aggregate(weights_list)
    else:
        # McMahan et al. size-weighted FedAvg (n_k/n weights)
        global_w = _avg_weights(weights_list, counts=org_counts)

    # Load global model into first org and evaluate
    eval_model = list(org_models.values())[0]
    eval_model.load_weights(global_w)

    y_pred = eval_model.predict(X_test)
    return compute_all_metrics(y_test, y_pred)


def main():
    print("Loading data …", flush=True)
    loader = FinancialDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=False)
    X_opt, y_opt = loader.get_eval_subset(X_train, y_train)

    print("Running DB-BOA hyperparameter search (shared across all runs) …",
          flush=True)
    adtcn_base = ADTCN()
    adtcn_base.optimise_hyperparams(X_opt, y_opt, verbose=False)
    print(f"  Optimal params: {adtcn_base.optimal_params}", flush=True)

    results = {}
    for bc in BASELINE_CONFIGS:
        label = bc["label"]
        print(f"\n─── {label} ────────────────────────────────────────────",
              flush=True)
        override = {k: bc[k] for k in ("use_krum", "use_dp", "use_shapley")}
        m = run_one_baseline(
            loader, X_train, X_val, X_test, y_train, y_val, y_test,
            adtcn_base, override,
        )
        results[label] = m
        print_metrics_table(m, model_name=label)

    # ── DP accuracy cost (answers Q38: "how much does DP cost at ε=1.0?") ──────
    if "FedAvg" in results and "FedAvg+DP" in results:
        dp_cost_acc = results["FedAvg"]["Accuracy"] - results["FedAvg+DP"]["Accuracy"]
        dp_cost_mcc = results["FedAvg"]["MCC"]       - results["FedAvg+DP"]["MCC"]
        print("\n" + "─" * 70, flush=True)
        print("DP ACCURACY COST  (ε=1.0, δ=1e-5, basic Gaussian mechanism)",
              flush=True)
        print(f"  FedAvg (no DP)  Accuracy={results['FedAvg']['Accuracy']:.5f}%  "
              f"MCC={results['FedAvg']['MCC']:.5f}", flush=True)
        print(f"  FedAvg+DP       Accuracy={results['FedAvg+DP']['Accuracy']:.5f}%  "
              f"MCC={results['FedAvg+DP']['MCC']:.5f}", flush=True)
        sign_a = "▼" if dp_cost_acc > 0 else "▲"
        sign_m = "▼" if dp_cost_mcc > 0 else "▲"
        print(f"  DP cost:  Accuracy {sign_a}{abs(dp_cost_acc):.5f}%  "
              f"MCC {sign_m}{abs(dp_cost_mcc):.5f}", flush=True)
        print("─" * 70, flush=True)

    print("\n\n" + "=" * 70, flush=True)
    print("PASTE THE FOLLOWING INTO baseline_metrics() in utils/metrics.py")
    print("=" * 70, flush=True)
    print("    algo_results = {")
    for label, m in results.items():
        row = {k: round(float(v), 5) for k, v in m.items()
               if k not in ("TP", "TN", "FP", "FN")}
        print(f'        "{label}": {json.dumps(row)},')
    print("    }")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
