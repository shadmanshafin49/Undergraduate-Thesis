"""
main.py
=======
DB-BOA Financial Security Framework — Main Orchestrator

Runs the full pipeline end-to-end:

  Phase 1 ── Blockchain Setup & Leader Block Selection (DB-BOA, Eq.10)
  Phase 2 ── ADTCN Hyperparameter Optimisation      (DB-BOA, Eq.11)
  Phase 3 ── ADTCN Training (with optimal params)
  Phase 4 ── Evaluation & Results
  Phase 5 ── Multi-round Consensus Simulation
  Phase 6 ── Plot Generation

Usage
-----
    python3 main.py
    python3 main.py --quick      # reduced pop/iter for fast demo
    python3 main.py --no-plots   # skip matplotlib output
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                  import (RESULTS_DIR, DB_BOA_CONFIG,
                                     LEADER_BLOCK_CONFIG, ADTCN_CONFIG,
                                     FEDERATION_CONFIG, INCENTIVE_CONFIG,
                                     LOG_WIDTH)
from data.data_loader        import FinancialDataLoader
from models.adtcn            import ADTCN
from models.federated_adtcn  import FederatedADTCN
from models.federation_manager import FederationManager
from blockchain.leader_block import ConsortiumBlockchain
from utils.metrics           import (compute_all_metrics,
                                     print_metrics_table, baseline_metrics)
from utils.visualizer        import generate_all_plots

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── banner helpers ───────────────────────────────────────────────────────────

def sep(char="═"):
    print(char * LOG_WIDTH, flush=True)

def header(title: str):
    sep("═")
    pad = (LOG_WIDTH - len(title) - 4) // 2
    print("  " + "─" * pad + f"  {title}  " + "─" * pad, flush=True)
    sep("═")


# ─── argument parser ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DB-BOA Financial Security Framework")
    p.add_argument("--quick",    action="store_true",
                   help="Use reduced population/iterations for fast demo")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    p.add_argument("--fed-only", action="store_true",
                   help="Run only the federated simulation (Phase 7)")
    p.add_argument("--round",    type=int, default=None,
                   help="Federation round number override")
    p.add_argument("--attack",   action="store_true",
                   help="Run Phase 8: attack resilience simulation (BankC always reports fraud)")
    return p.parse_args()


# ─── main pipeline ────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    t_total = time.time()

    print("\n", flush=True)
    sep("█")
    print("█" + " " * (LOG_WIDTH - 2) + "█", flush=True)
    title = "  DB-BOA FINANCIAL SECURITY FRAMEWORK  "
    pad   = (LOG_WIDTH - len(title)) // 2
    print("█" + " " * pad + title + " " * (LOG_WIDTH - pad - len(title) - 1) + "█", flush=True)
    subtitle = "  Prabanand & Thanabal (2025) — Replication  "
    pad2  = (LOG_WIDTH - len(subtitle)) // 2
    print("█" + " " * pad2 + subtitle + " " * (LOG_WIDTH - pad2 - len(subtitle) - 1) + "█", flush=True)
    print("█" + " " * (LOG_WIDTH - 2) + "█", flush=True)
    sep("█")
    print(flush=True)

    # ── quick-mode overrides ──────────────────────────────────────────────────
    if args.quick:
        print("[INFO]  QUICK MODE — reduced population and iterations", flush=True)
        DB_BOA_CONFIG["population_size"] = 10
        DB_BOA_CONFIG["max_iterations"]  = 15
        LEADER_BLOCK_CONFIG["population_size"] = 8
        LEADER_BLOCK_CONFIG["max_iterations"]  = 12

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 0 ── Load Data
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 0  ─  DATA LOADING & ENGINEERING")

    loader = FinancialDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=True)
    X_opt, y_opt = loader.get_eval_subset(X_train, y_train)
    print(f"[DATA]  Optimisation subset: {X_opt.shape[0]:,} samples "
          f"({X_opt.shape[1]} features)", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 ── Leader Block Selection
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 1  ─  BLOCKCHAIN SETUP & LEADER BLOCK SELECTION")

    bc = ConsortiumBlockchain(seed=42)
    bc.initialise_nodes(verbose=True)

    leader_node, leader_idx, leader_history = bc.select_leader(verbose=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 ── DB-BOA Hyperparameter Optimisation
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 2  ─  DB-BOA HYPERPARAMETER OPTIMISATION  (Eq.11)")

    adtcn = ADTCN()
    optimal_params = adtcn.optimise_hyperparams(X_opt, y_opt, verbose=True)

    print(f"\n[OPT]   ✔ Optimal hyperparameters found:", flush=True)
    print(f"         Hidden Neurons   (HnD) : {optimal_params['hidden_neurons']}", flush=True)
    print(f"         Epoch Count      (EpD) : {optimal_params['epoch_count']}", flush=True)
    print(f"         Steps per Epoch  (SeD) : {optimal_params['steps_per_epoch']}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 ── ADTCN Training
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 3  ─  ADTCN MODEL TRAINING")

    # Combine train + val for final training
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    adtcn.fit(X_full, y_full, verbose=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 ── Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 4  ─  MODEL EVALUATION")

    y_pred  = adtcn.predict(X_test)
    y_proba = adtcn.predict_proba(X_test)
    metrics = compute_all_metrics(y_test, y_pred)

    print_metrics_table(metrics, model_name="DB-BOA-ADTCN")

    # ── Statistical evaluation (Table 2) ─────────────────────────────────────
    if hasattr(adtcn, "opt_stats"):
        sep()
        print("[STAT]  DB-BOA Statistical Evaluation (Table 2):", flush=True)
        for k, v in adtcn.opt_stats.items():
            print(f"         {k:<10} : {v:.6f}", flush=True)

    # ── Comparison against baselines ─────────────────────────────────────────
    sep()
    print("[COMP]  Accuracy gain over baselines:", flush=True)
    algo_baselines, clf_baselines = baseline_metrics()
    for name, bm in {**algo_baselines, **clf_baselines}.items():
        delta = metrics["Accuracy"] - bm["Accuracy"]
        bar   = "▲" if delta > 0 else "▼"
        print(f"         {name:<20}  {bar} {abs(delta):+.2f} %", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5 ── Multi-Round Consensus Simulation
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 5  ─  CONSENSUS SIMULATION (5 ROUNDS)")

    n_rounds = 5
    for r in range(1, n_rounds + 1):
        # Rotate leader every 2 rounds to show node communication dynamics
        if r % 2 == 0:
            bc.select_leader(verbose=False)
        bc.simulate_consensus_round(
            round_num=r,
            n_transactions=50 + r * 10,
            verbose=True
        )
        sep("·")

    bc.print_node_status()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 6 ── Save Results & Plots
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 6  ─  RESULTS & PLOTS")

    # ── build results dict (JSON saved after Phase 7 adds fed_rounds) ────────
    results = {
        "optimal_hyperparams" : optimal_params,
        "metrics"             : {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))
        },
        "db_boa_stats"        : adtcn.opt_stats if hasattr(adtcn, "opt_stats") else {},
        "leader_node"         : leader_idx,
        "leader_cost"         : float(leader_node.cost),
        "leader_ct"           : float(leader_node.ct),
        "leader_cc"           : float(leader_node.cc),
        "leader_ms"           : float(leader_node.ms),
        "consensus_rounds"    : bc.rounds,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 7 ── Multi-Org Federated Simulation
    # ══════════════════════════════════════════════════════════════════════════
    header("PHASE 7  ─  FEDERATED MULTI-ORG SIMULATION")

    # 1. Split data across orgs
    org_splits = loader.split_for_orgs(X_train, y_train)

    # 2. Create one FederatedADTCN per org and train locally
    org_models = {}
    org_pre_fed_metrics = {}
    for org_name, (X_org, y_org) in org_splits.items():
        print(f"[FED]  Training {org_name} on {len(y_org):,} samples …",
              flush=True)
        m = FederatedADTCN(cfg=ADTCN_CONFIG)
        m.optimal_params = adtcn.optimal_params   # reuse DB-BOA Job 1 results
        m.fit(X_org, y_org, verbose=False)
        m._n_trained = len(y_org)
        m_metrics = m.evaluate(X_test, y_test, verbose=False)
        org_pre_fed_metrics[org_name] = m_metrics
        print(f"[FED]  {org_name} accuracy (pre-fed): "
              f"{m_metrics['Accuracy']:.2f}%", flush=True)
        org_models[org_name] = m

    # 3. Run N federated rounds
    # Federation triggers every fed_round_interval consensus rounds (§6.2 / §8.3).
    # With 5 consensus rounds already completed in Phase 5, the first trigger
    # fires at round 5.  Each subsequent trigger at rounds 10, 15, …
    # We demonstrate 3 triggers (corresponding to 15 total consensus rounds).
    fed_manager    = FederationManager(n_orgs=3)
    fed_interval   = FEDERATION_CONFIG["fed_round_interval"]   # = 5
    n_fed_rounds   = 3    # rounds 5, 10, 15 → 3 federation events
    print(f"[FED]  Federation triggers every {fed_interval} consensus rounds "
          f"— demonstrating {n_fed_rounds} events", flush=True)
    fed_results  = []

    for fed_round in range(1, n_fed_rounds + 1):
        header(f"FEDERATION ROUND {fed_round}")

        # Collect current org metrics for metadata
        org_metrics = {
            name: m.evaluate(X_test, y_test, verbose=False)
            for name, m in org_models.items()
        }

        # Shared anonymised validation set
        X_val_shared = X_test[:500]
        y_val_shared = y_test[:500]

        # Run DB-BOA Job 3
        fed_result = fed_manager.run_federation_round(
            org_models  = org_models,
            org_metrics = org_metrics,
            X_val       = X_val_shared,
            y_val       = y_val_shared,
            round_num   = fed_round,
            verbose     = True,
        )

        # Load global model into all orgs
        for m in org_models.values():
            m.load_weights(fed_result["global_weights"])

        # Evaluate all orgs after federation
        accuracy_deltas = {}
        print("[FED]  Post-federation accuracy:", flush=True)
        for name, m in org_models.items():
            post  = m.evaluate(X_test, y_test, verbose=False)
            delta = post["Accuracy"] - org_metrics[name]["Accuracy"]
            accuracy_deltas[name] = delta
            print(f"         {name}: {post['Accuracy']:.2f}%  "
                  f"(delta: {delta:+.2f}%)", flush=True)

            fed_result["accuracy_deltas"] = accuracy_deltas
        fed_results.append(fed_result)
        sep("·")

    # ── save final results JSON (includes fed_rounds) ─────────────────────────
    import hashlib

    def _hash_weights(wlist):
        """SHA-256 of the concatenated weight bytes — proves integrity."""
        h = hashlib.sha256()
        for arr in wlist:
            h.update(np.asarray(arr).tobytes())
        return h.hexdigest()

    fed_rounds_serialisable = []
    for fr in fed_results:
        entry = {k: v for k, v in fr.items() if k != "global_weights"}
        entry["global_weights_hash"] = _hash_weights(fr["global_weights"])
        fed_rounds_serialisable.append(entry)

    results["fed_rounds"] = fed_rounds_serialisable

    json_path = os.path.join(RESULTS_DIR, "db_boa_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE]  Results JSON → {json_path}", flush=True)

    # ── generate plots ────────────────────────────────────────────────────────
    if not args.no_plots:
        print(flush=True)
        generate_all_plots(
            y_true           = y_test,
            y_pred           = y_pred,
            y_proba          = y_proba,
            history_best     = adtcn.opt_history["best"],
            proposed_metrics = metrics,
            nodes            = bc.nodes,
            leader_idx       = leader_idx,
            rounds           = bc.rounds,
            fed_rounds       = fed_results,
            verbose          = True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 8 ── Attack Resilience Simulation  (§7.2 Contribution 5)
    # ══════════════════════════════════════════════════════════════════════════
    if args.attack:
        header("PHASE 8  ─  ATTACK RESILIENCE SIMULATION")

        print("[ATK]  Scenario: BankC is malicious — always reports isFraud=True",
              flush=True)
        print("[ATK]  Expected outcome: token depletion, reputation fall, "
              "low DB-BOA Job 3 weight", flush=True)
        sep()

        # ── Build attacker model (always predicts fraud) ──────────────────────
        # Deep-copy so Phase 7 models are unaffected
        attack_models = {k: copy.deepcopy(v) for k, v in org_models.items()}
        attacker_name = "BankC"
        # Override predict on the instance — Python checks __dict__ before class,
        # so self.predict(X) inside evaluate_on_validation also uses this.
        attack_models[attacker_name].predict = (
            lambda X: np.ones(len(X), dtype=int)
        )

        # ── Simulate N fraud-verdict rounds ───────────────────────────────────
        n_attack_rounds = 15
        attacker_tokens     = 100
        attacker_reputation = 1.0
        dispute_penalty     = INCENTIVE_CONFIG["dispute_penalty"]           # 2
        fraud_reward        = INCENTIVE_CONFIG["fraud_consensus_reward"]    # 10

        attack_log = []
        print(f"[ATK]  Simulating {n_attack_rounds} fraud-verdict rounds …",
              flush=True)
        sep()
        print(f"  {'Rnd':>3}  {'y_true':>6}  {'Consensus':>9}  "
              f"{'BankC':>5}  {'Outcome':<10}  {'Tokens':>7}  {'Rep':>6}",
              flush=True)
        sep("·")

        rng_atk = np.random.RandomState(99)
        for r in range(1, n_attack_rounds + 1):
            # Sample a random test transaction
            idx    = rng_atk.randint(0, len(X_test))
            x_s    = X_test[idx : idx + 1]
            y_true = int(y_test[idx])

            # BankA and BankB predict correctly
            v_a = int(org_models["BankA"].predict(x_s)[0])
            v_b = int(org_models["BankB"].predict(x_s)[0])
            v_c = 1   # attacker always says fraud

            # Majority of 3
            consensus = 1 if (v_a + v_b + v_c) >= 2 else 0

            if v_c == consensus:
                # Attacker's verdict matches consensus → earns reward
                attacker_tokens += fraud_reward
                outcome = "CORRECT"
            else:
                # Attacker disputed → penalty + reputation drop
                attacker_tokens     -= dispute_penalty
                attacker_reputation  = max(0.5, attacker_reputation - 0.05)
                outcome = "DISPUTED"

            entry = dict(round=r, y_true=y_true, consensus=consensus,
                         attacker_verdict=v_c, outcome=outcome,
                         tokens=attacker_tokens, reputation=attacker_reputation)
            attack_log.append(entry)

            print(f"  {r:>3}   {y_true:>6}      {consensus:>6}      "
                  f"{v_c:>5}  {outcome:<10}  {attacker_tokens:>7}  "
                  f"{attacker_reputation:>6.3f}", flush=True)

        sep()
        n_disputed  = sum(1 for e in attack_log if e["outcome"] == "DISPUTED")
        token_delta = attacker_tokens - 100
        rep_delta   = attacker_reputation - 1.0
        print(f"[ATK]  Rounds disputed   : {n_disputed}/{n_attack_rounds}",
              flush=True)
        print(f"[ATK]  Token change      : 100 → {attacker_tokens} "
              f"({token_delta:+d})", flush=True)
        print(f"[ATK]  Reputation change : 1.000 → {attacker_reputation:.3f} "
              f"({rep_delta:+.3f})", flush=True)

        # ── DB-BOA Job 3 with attacker present ───────────────────────────────
        sep()
        print("[ATK]  Running DB-BOA Job 3 with attacker in federation …",
              flush=True)
        print("[ATK]  Prediction: DB-BOA assigns BankC near-minimum weight",
              flush=True)

        atk_org_metrics = {
            name: m.evaluate(X_test, y_test, verbose=False)
            for name, m in attack_models.items()
        }
        atk_fed_manager = FederationManager(n_orgs=3)
        atk_fed_result  = atk_fed_manager.run_federation_round(
            org_models  = attack_models,
            org_metrics = atk_org_metrics,
            X_val       = X_test[:500],
            y_val       = y_test[:500],
            round_num   = 99,   # distinguished from normal rounds
            verbose     = False,
        )

        sep()
        print("[ATK]  DB-BOA Job 3 aggregation weights under attack:", flush=True)
        w = atk_fed_result["aggregation_weights"]
        for i, name in enumerate(["BankA", "BankB", "BankC"]):
            marker = "  ← attacker (suppressed)" if name == attacker_name else ""
            print(f"         {name}: {w[i]:.4f}{marker}", flush=True)

        print(f"[ATK]  Best fitness (Obf2): "
              f"{-atk_fed_result['best_fitness']:.6f}", flush=True)

        # ── Summary ───────────────────────────────────────────────────────────
        sep("═")
        print("[ATK]  ATTACK RESILIENCE VERDICT:", flush=True)
        w_attacker = w[["BankA", "BankB", "BankC"].index(attacker_name)]
        if w_attacker < 0.20:
            print(f"  ✔  BankC weight = {w_attacker:.4f} (<0.20) — "
                  f"attack SUPPRESSED by DB-BOA", flush=True)
        else:
            print(f"  ⚠  BankC weight = {w_attacker:.4f} — "
                  f"attack influence reduced but not fully suppressed", flush=True)
        if attacker_tokens < 100:
            print(f"  ✔  Token balance depleted: 100 → {attacker_tokens} "
                  f"— incentive penalty active", flush=True)
        if attacker_reputation < 1.0:
            print(f"  ✔  Reputation fell: 1.000 → {attacker_reputation:.3f} "
                  f"— future leader selection penalised", flush=True)
        sep("═")

        # Append attack results to JSON
        results["attack_simulation"] = {
            "attacker": attacker_name,
            "n_rounds": n_attack_rounds,
            "n_disputed": n_disputed,
            "final_tokens": attacker_tokens,
            "final_reputation": attacker_reputation,
            "db_boa_weight_under_attack": w_attacker,
            "attack_log": attack_log,
        }
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[ATK]  Attack results appended → {json_path}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    sep("═")
    elapsed = time.time() - t_total
    print(f"  ✔  DB-BOA Framework complete  ({elapsed:.1f}s total)", flush=True)
    print(f"  Accuracy   : {metrics['Accuracy']:.4f} %", flush=True)
    print(f"  MCC        : {metrics['MCC']:.4f}", flush=True)
    print(f"  Leader     : Node-{leader_idx:02d}  (cost={leader_node.cost:.4f})", flush=True)
    print(f"  Results in : {RESULTS_DIR}/", flush=True)
    sep("═")
    print(flush=True)


if __name__ == "__main__":
    main()
