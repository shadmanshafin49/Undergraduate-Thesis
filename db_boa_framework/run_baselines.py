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


def _avg_weights(weights_list: list) -> list:
    """Plain FedAvg: equal-weight average of all org weight lists."""
    n_arrays = len(weights_list[0])
    return [
        np.mean([weights_list[i][a] for i in range(len(weights_list))], axis=0)
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
    for org_name, (X_org, y_org) in org_splits.items():
        m = FederatedADTCN(cfg=ADTCN_CONFIG)
        m.optimal_params = adtcn_base.optimal_params
        m.fit(X_org, y_org, verbose=False)
        org_models[org_name] = m

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
    if use_krum:
        fed_mgr       = FederationManager(n_orgs=3, cfg=fed_cfg)
        global_w, _, _ = fed_mgr._krum_aggregate(weights_list)
    else:
        global_w = _avg_weights(weights_list)

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
