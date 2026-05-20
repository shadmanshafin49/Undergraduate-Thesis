"""
config.py
=========
Central configuration for DB-BOA Financial Security Framework.
All hyperparameters, paths, and constants are defined here.

Reference: "Advanced financial security system using smart contract in
private ethereum consortium blockchain with hybrid optimization strategy"
Prabanand & Thanabal, Scientific Reports, 2025.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# ─── Dataset Configuration ───────────────────────────────────────────────────
DATA_CONFIG = {
    "n_samples"       : 20_000,   # total transactions to simulate
    "fraud_rate"      : 0.05,     # 5 % fraud (realistic for demo speed)
    "n_features"      : 30,       # V1-V28 + Amount + Time  (like CC-Fraud dataset)
    "sequence_length" : 10,       # look-back window for temporal context
    "test_size"       : 0.20,     # 80/20 train-test split
    "val_size"        : 0.10,     # 10 % of training set for validation
    "random_state"    : 42,
    "eval_subset"     : 3_000,    # samples used for fast DB-BOA fitness eval
}

# ─── DB-BOA Optimizer Configuration ──────────────────────────────────────────
DB_BOA_CONFIG = {
    # Population & iterations (paper uses 50; we use 30 for demo speed)
    "population_size" : 20,
    "max_iterations"  : 30,

    # Hyperparameter search bounds for ADTCN  (Paper Table / Eq. 11)
    "hidden_neurons_bounds"  : (5,   255),   # HnD
    "epoch_count_bounds"     : (5,   50),    # EpD
    "steps_per_epoch_bounds" : (50,  250),   # SeD

    # DBOA (Butterfly) parameters
    "sensory_modality"       : 0.01,   # d  in g = d * J^b
    "power_exponent"         : 0.1,    # b
    "switch_probability"     : 0.8,    # ρ  — controls global vs local search
    "mutation_rate"          : 0.1,    # LSAM mutation scale
    "num_lsam_iterations"    : 5,      # Num_itr in LSAM

    # BOA (Billiards) parameters
    "n_pockets"              : 8,      # billiards table pockets

    "random_state"           : 42,
}

# ─── Leader Block Selection Configuration ────────────────────────────────────
LEADER_BLOCK_CONFIG = {
    # Simulated blockchain nodes in the consortium
    "n_nodes"            : 10,

    # Objective weights (equal in the paper, adjustable here)
    "weight_ct"          : 1.0,   # computation time weight
    "weight_cc"          : 1.0,   # communication cost weight
    "weight_ms"          : 1.0,   # memory size weight

    # Node resource bounds (normalised 0-1 scale)
    "ct_bounds"          : (0.1,  1.0),  # computation time
    "cc_bounds"          : (0.05, 0.80), # communication cost
    "ms_bounds"          : (0.05, 0.90), # memory size

    # DB-BOA settings for leader block search
    "population_size"    : 15,
    "max_iterations"     : 25,
    "random_state"       : 7,
}

# ─── ADTCN Model Configuration ───────────────────────────────────────────────
ADTCN_CONFIG = {
    # These are overwritten by DB-BOA optimal values at runtime
    "hidden_neurons"     : 128,
    "epoch_count"        : 30,
    "steps_per_epoch"    : 150,

    # Architecture flags
    "activation"         : "tanh",    # paper's best activation (TanH)
    "dropout_rate"       : 0.3,
    "learning_rate"      : 0.001,

    # Temporal feature scales for PTC / NTC simulation
    "ptc_windows"        : [5, 10, 20],   # Periodic Temporal Context windows
    "ntc_diff_orders"    : [1, 2],        # Non-Periodic TC diff orders
    "random_state"       : 42,
}

# ─── Comparison Baselines ────────────────────────────────────────────────────
# Baseline algorithms for convergence comparison (Section VII of paper)
BASELINE_NAMES = [
    "MBO-ADTCN",
    "WSA-ADTCN",
    "DBOA-ADTCN",
    "BOA-ADTCN",
    "DB-BOA-ADTCN",   # ← proposed
]

# Baseline classifier names (Table 4 of paper)
CLASSIFIER_NAMES = [
    "EfficientNet",
    "ResNet",
    "DenseNet",
    "DTCN",
    "DB-BOA-ADTCN",   # ← proposed
]

# ─── Incentive Mechanism ─────────────────────────────────────────────────────
# All values match §6.2 token structure table (Research Approach Document).
# Chaincode enforces these rules on-chain; Python simulation mirrors them.
INCENTIVE_CONFIG = {
    # recordFraudResult chaincode function
    "fraud_consensus_reward" : 10,   # +10  fraud verdict confirmed by majority
    "dispute_penalty"        : 2,    # −2   verdict disputed by majority

    # recordConsensusRound chaincode function
    "leader_success_reward"  : 10,   # +10  selected as leader AND round succeeds
    "latency_bonus"          : 15,   # +15  confirmed verdict AND latency < 300ms
    "consensus_fail_penalty" : 2,    # −2   consensus round fails under leader

    # recordFederationRound chaincode function
    "federation_pool"        : 20,   # +20  shared by DB-BOA weight across orgs

    # Latency threshold for bonus award (ms)
    "latency_threshold_ms"   : 300,

    # Legacy aliases kept for backward compat with apply_incentives()
    "base_reward"            : 10,   # = leader_success_reward
    "performance_bonus"      : 15,   # = latency_bonus
    "penalty"                : 2,    # = consensus_fail_penalty
}

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_WIDTH = 70   # terminal separator width

# ─── Federation Configuration ─────────────────────────────────────────────────
FEDERATION_CONFIG = {
    'n_orgs'               : 3,                     # Bank A, Bank B, Bank C
    'org_names'            : ['BankA', 'BankB', 'BankC'],
    'fed_round_interval'   : 5,                     # trigger federation every N rounds
    'min_rounds_for_fed'   : 2,                     # minimum rounds before first federation
    'weight_bounds'        : (0.05, 0.70),          # min/max contribution weight per org
    'validation_fraction'  : 0.15,                  # fraction of local data used for fed eval
    'db_boa_fed_pop'       : 12,                    # DB-BOA population for weight search
    'db_boa_fed_iter'      : 20,                    # DB-BOA iterations for weight search
    'random_state'         : 42,
}

ORG_DATA_SPLITS = {
    'BankA': 0.50,   # Bank A has the most data (50%)
    'BankB': 0.30,   # Bank B has medium data (30%)
    'BankC': 0.20,   # Bank C has least data (20%)
}
