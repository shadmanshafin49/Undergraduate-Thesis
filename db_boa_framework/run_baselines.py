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

import argparse
import copy
import json
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                  import (ADTCN_CONFIG, FEDERATION_CONFIG,
                                     RESULTS_DIR, DATASETS, ENTITY_PARTITIONS,
                                     get_loader)
from experiments._dataset     import environment
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
                     adtcn_base, cfg_override: dict, test_groups=None) -> dict:
    """
    Train all three org models, run one federated round, evaluate on test set.
    Returns the test-set metrics dict.
    """
    fed_cfg = copy.deepcopy(FEDERATION_CONFIG)
    fed_cfg.update(cfg_override)

    org_splits = loader.split_for_orgs(X_train, y_train)
    # Entity-linked windows inside the federation, when the loader can supply
    # them (BankSim under partition="customer").  None everywhere else, which
    # falls back to global windows — see BankSimDataLoader.split_for_orgs.
    org_groups = getattr(loader, "last_org_groups", {}) or {}
    org_models = {}
    org_counts = []   # sample counts per org, for McMahan size-weighted FedAvg
    for org_name, (X_org, y_org) in org_splits.items():
        m = FederatedADTCN(cfg=ADTCN_CONFIG)
        m.optimal_params = adtcn_base.optimal_params
        m.fit(X_org, y_org, verbose=False, groups=org_groups.get(org_name))
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

    y_pred = eval_model.predict(X_test, groups=test_groups)
    return compute_all_metrics(y_test, y_pred)


def main(dataset="ulb", partition=None, out_name=None, filters=None, epochs=None):
    print(f"Loading data … ({DATASETS[dataset]['label']})", flush=True)
    loader = get_loader(dataset)
    entity = tuple(ENTITY_PARTITIONS.get(dataset, ()))
    if partition:
        # ULB takes no partition at all; BankSim has customers; the Handbook has
        # customers and terminals.  config.ENTITY_PARTITIONS is the one place
        # that mapping lives (see experiments/_dataset.resolve).
        if dataset == "ulb" or partition not in ("stratified",) + entity:
            raise SystemExit(f"--partition {partition} is not available on "
                             f"--dataset {dataset}")
        loader.cfg["partition"] = partition
        # An entity split deals whatever the rows are ordered by, so the
        # ordering must match it.  A no-op for BankSim/customer (its default).
        # ...but only when the partition is also an ordering (AMLSim's `bank` is not).
        if partition in entity and partition in getattr(loader, "ENTITY_ORDERINGS", (partition,)):
            loader.cfg["ordering"] = partition
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=False)
    # Tell the detector how many leading columns are real features — see
    # ADTCN.fit. Without this BankSim's 79 features are truncated to 33.
    ADTCN_CONFIG["n_raw_features"] = loader.raw_feature_count
    X_opt, y_opt = loader.get_eval_subset(X_train, y_train)

    if epochs:
        ADTCN_CONFIG["epoch_count"] = int(epochs)

    adtcn_base = ADTCN()
    if filters:
        # Skip the DB-BOA search and pin the detector width.
        #
        # Why this option exists: experiments/objective_noise_audit.py shows the
        # DB-BOA surrogate fitness is a *random function of its input* on a
        # low-fraud dataset — a fixed configuration re-scored 25 times spans most
        # of the objective's range, and between-seed variation is as large as
        # between-configuration variation.  A width it returns is therefore not
        # meaningfully "optimal".  For an ablation whose subject is
        # FedAvg vs Krum vs DP, the detector width is a nuisance parameter, and
        # pinning it removes a confound (and hours of compute) rather than
        # hiding one.  The pinned value is recorded in the output artifact.
        adtcn_base.optimal_params = {
            "hidden_neurons": int(filters),
            "epoch_count": ADTCN_CONFIG["epoch_count"],
            "steps_per_epoch": ADTCN_CONFIG["steps_per_epoch"],
        }
        print(f"  Detector width PINNED (DB-BOA search skipped): "
              f"{adtcn_base.optimal_params}", flush=True)
    else:
        print("Running DB-BOA hyperparameter search (shared across all runs) ...",
              flush=True)
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
            test_groups=(getattr(loader, "groups_test", None)
                         if partition in entity else None),
        )
        results[label] = m
        print_metrics_table(m, model_name=label)

    # ── Persist a verifiable artifact (provenance for the §6.2 ablation table) ─
    out_path = os.path.join(RESULTS_DIR,
                            out_name or ("baselines.json" if dataset == "ulb"
                                         else f"baselines_{dataset}.json"))
    artifact = {
        "task": f"federated ablation (FedAvg / +Krum / +DP / proposed) on the "
                f"{dataset} test set",
        "dataset": DATASETS[dataset]["label"],
        "partition": partition or getattr(loader, "cfg", {}).get("partition", "stratified"),
        "ordering": getattr(loader, "cfg", {}).get("ordering"),
        "environment": environment(),
        "detector_width_source": ("pinned (DB-BOA search skipped; see "
                                  "experiments/objective_noise_audit.py)"
                                  if filters else "DB-BOA search"),
        # OBJ-13: which surrogate protocol chose the width.  Only meaningful when
        # the search actually ran — the pinned path never builds an objective, so
        # `baselines_banksim_*.json` are NOT affected by the repair even though
        # OBJ-13's downstream-cost list originally said they were.
        "surrogate_eval_mode": (None if filters
                                else ADTCN_CONFIG.get("surrogate_eval_mode",
                                                      "legacy")),
        "surrogate_k": (None if filters else ADTCN_CONFIG.get("surrogate_k")),
        "epoch_count": ADTCN_CONFIG["epoch_count"],
        "dp_epsilon": 1.0,
        "dp_delta": 1e-5,
        "optimal_params": getattr(adtcn_base, "optimal_params", None),
        "results": {label: {k: float(v) for k, v in m.items()} for label, m in results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved ablation artifact -> {out_path}", flush=True)

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
        print("-" * 70, flush=True)

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(DATASETS), default="ulb")
    ap.add_argument("--partition", choices=["stratified", "customer", "terminal", "bank"],
                    default=None,
                    help="'customer' / 'terminal' / 'bank' deal whole entities to banks, "
                         "so no entity's history sits at two banks (BankSim: customer; "
                         "Handbook: customer, terminal; AMLSim: bank, which is native). "
                         "ULB takes no partition.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--filters", type=int, default=None,
                    help="Pin the detector width and skip the DB-BOA search. "
                         "See the note in main() for why this is a legitimate "
                         "choice for an ablation about Krum/DP.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override ADTCN_CONFIG['epoch_count'].")
    a = ap.parse_args()
    main(dataset=a.dataset, partition=a.partition, out_name=a.out,
         filters=a.filters, epochs=a.epochs)
