"""
config.py
=========
Central configuration for DB-BOA Financial Security Framework.
All hyperparameters, paths, and constants are defined here.

Reference: Prabanand & Thanabal (2025), "Advanced financial security system
using smart contract in private ethereum consortium blockchain with hybrid
optimization strategy", Scientific Reports 15, 6764.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# ULB Credit Card Fraud dataset (Kaggle, Lopez-Rojas et al. 2016)
# 284,807 rows, 0.17% fraud, 28 PCA features + Amount + Time
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "datasets", "creditcard.csv")

# ─── Dataset Configuration ───────────────────────────────────────────────────
DATA_CONFIG = {
    "dataset_path"      : DATASET_PATH,
    "n_features"        : 30,       # V1-V28 + Amount + Time (base raw features)
    "sequence_length"   : 10,       # look-back window; chosen empirically — ablation over {5,10,20} is left for future work
    "test_size"         : 0.20,     # 80/20 train-test split
    "val_size"          : 0.10,     # 10 % of training set for validation
    "random_state"      : 42,
    "eval_subset"       : 3_000,    # samples used for fast DB-BOA fitness eval
    # Temporal-amount recurrence features (inspired by Liu et al., WWW 2021)
    # NOTE: ULB has no account IDs, so these are sliding-window frequency
    # features (amount_recurrence_before, amount_recurrence_after, degree_ratio),
    # not true graph features.
    "use_graph_features": True,     # append 3 recurrence features to raw features
    "graph_n_bins"      : 50,       # Amount discretisation buckets
    "graph_window"      : 100,      # rolling window size for edge construction
}

# ─── DB-BOA Optimizer Configuration ──────────────────────────────────────────
DB_BOA_CONFIG = {
    # Population & iterations (paper uses 50; we use 30 for demo speed)
    "population_size" : 20,
    "max_iterations"  : 30,

    # Hyperparameter search bounds for ADTCN  (Paper Table / Eq. 11)
    "filter_count_bounds"    : (5,   255),   # CNN n_filters (HnD in paper)
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

    # Leader-selection method:
    #   "db_boa" — original myopic single-round optimiser (argmin CT+CC+MS)
    #   "rl"     — sequential RL agent (see RL_LEADER_CONFIG); learns a leader
    #              *policy* over rounds, optimising cumulative discounted reward
    #              rather than per-round cost.  This is the title's
    #              "Reinforcement Learning" component, integrated with the
    #              consensus / incentive mechanism.
    # Default "rl": the title claims Reinforcement Learning, so the headline run
    # elects leaders with the RL policy.  DB-BOA still performs the one-shot
    # cold-start pick in main.py Phase 1 (a single selection the RL agent cannot
    # learn yet); RL then governs the *sequential* multi-round election (Phase 5).
    # Set to "db_boa" to reproduce the pre-RL baseline.
    "leader_method"      : "rl",

    # Consensus latency/throughput source (B2):
    #   False — legacy *simulated* arithmetic (time.sleep + resource scores).
    #   True  — use REAL wall-clock numbers measured against the live Hyperledger
    #           Fabric test-network (Raft orderer, 2 orgs) by
    #           db_boa_fabric/api-server/measure_consensus.js, loaded from
    #           results/fabric_consensus_measured.json.  Per-round latency is
    #           drawn from the measured distribution; throughput is the measured
    #           sustained value.  This removes the fabricated 85 TPS / 180 ms
    #           figures (divergence D7) in favour of measured ones.
    "use_measured_consensus": True,
    "measured_consensus_file": os.path.join(RESULTS_DIR,
                                            "fabric_consensus_measured.json"),
}

# ─── RL Leader Selection Configuration ───────────────────────────────────────
# Linear-function-approximation Q-learning (Sutton & Barto, 2018, Ch. 9-10)
# applied to consortium leader selection.  The agent treats each consensus
# round as one step of a Markov Decision Process:
#
#   state  s_t  — the consortium's per-node resource + reputation + token +
#                 load profile (featurised in rl_leader.py::node_features)
#   action a_t  — which node is elected leader for round t
#   reward r_t  — the on-chain incentive payout to that leader
#                 (+leader_success_reward [+latency_bonus] on success,
#                  −consensus_fail_penalty on failure — INCENTIVE_CONFIG)
#   transition  — simulate_consensus_round() (reputation/endorsement dynamics)
#
# Q(s,a) = θ·φ(s,a) is linear in the node feature vector φ; θ is updated by the
# temporal-difference rule  θ ← θ + α[r + γ·max_a' Q(s',a') − Q(s,a)]·φ.
# This is genuine bootstrapped TD learning (γ>0), not a one-shot bandit, so the
# agent performs long-horizon credit assignment that the myopic DB-BOA objective
# (which re-solves argmin CT+CC+MS independently each round) cannot.
RL_LEADER_CONFIG = {
    "alpha"          : 0.10,    # TD learning rate
    "gamma"          : 0.90,    # discount factor (long-horizon credit assignment)
    "epsilon_start"  : 0.30,    # initial exploration probability (ε-greedy)
    "epsilon_min"    : 0.02,    # floor after decay
    "epsilon_decay"  : 0.97,    # ε ← ε·decay each round
    "reward_scale"   : 0.04,    # scales token reward into a stable TD target
    "random_state"   : 7,
}

# ─── ADTCN Model Configuration ───────────────────────────────────────────────
ADTCN_CONFIG = {
    # These are overwritten by DB-BOA optimal values at runtime
    "hidden_neurons"     : 128,
    "epoch_count"        : 30,
    "steps_per_epoch"    : 150,

    # Architecture flags
    # activation: ReLU is used (hardcoded in _Conv1dClassifier); TanH was the
    # paper's claimed best but was never tested in this implementation.
    "dropout_rate"       : 0.3,
    "learning_rate"      : 0.001,

    # Temporal model architecture (B3):
    #   "cnn"          — plain 2-layer Conv1d(k=3) + global-max-pool (receptive
    #                    field = 5 < SEQ_LEN=10, so it cannot see the whole
    #                    window; the original baseline, kept for reproducibility).
    #   "dilated_attn" — TCN-style stack of dilated causal convolutions
    #                    (dilations 1,2,4 → receptive field 15 ≥ SEQ_LEN) with
    #                    residual connections, then a softmax temporal-attention
    #                    pool over the SEQ_LEN steps (the report's "MTTA" /
    #                    Adaptive Deep Temporal Context claim, now actually
    #                    implemented).  See models/adtcn.py and
    #                    experiments/architecture_ablation.py for the head-to-head.
    # Default "cnn": the ablation (experiments/temporal_pipeline_ablation.py) shows
    # the dilated/attention model does NOT beat the plain CNN on this data — the
    # ULB set is tabular, and time-ordered windows make both models worse because
    # temporal context overfits period-specific fraud bursts. The simpler,
    # permutation-robust CNN is the evidence-justified deployed model; the
    # "dilated_attn" path exists for the report-faithful architecture ablation.
    "architecture"       : "cnn",

    # Temporal feature scales for PTC / NTC simulation
    "ptc_windows"        : [5, 10, 20],   # Periodic Temporal Context windows
    "ntc_diff_orders"    : [1, 2],        # Non-Periodic TC diff orders
    "random_state"       : 42,
}

# ─── Comparison Baselines ────────────────────────────────────────────────────
# ULB-evaluated algorithm baselines (run run_baselines.py to populate results)
BASELINE_NAMES = [
    "FedAvg",
    "FedAvg+Krum",
    "FedAvg+DP",
    "DB-BOA-ADTCN",   # ← proposed
]

# ULB-evaluated classifier baselines
CLASSIFIER_NAMES = [
    "FedAvg",
    "FedAvg+Krum",
    "FedAvg+DP",
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
    "federation_pool"        : 20,   # +20  shared by Shapley contribution weight across orgs

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
    # Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017)
    # Krum requires n ≥ 2f+3.  With n=3 orgs: f=0 is the largest value that
    # satisfies this (3 ≥ 2×0+3 = 3).  f=0 means no Byzantine adversary is
    # assumed; Krum still selects the most consensus-aligned org each round.
    'use_krum'             : True,                  # replace weighted average with Krum
    'byzantine_f'          : 0,                     # max Byzantine orgs; 0 is correct for n=3
    # Differential privacy for weight sharing (Dwork et al., 2006)
    # OFF by default.  Weight-channel DP at small ε adds Gaussian noise to the
    # ~1e5-dim weight vector (noise/signal ≈ k·√d/ε), which at ε=1 drives the
    # aggregated model to near-random AND destroys the Shapley incentive signal
    # (see experiments/privacy_incentive_sweep.py).  It is therefore an *explicitly
    # swept* knob, not the shipped operating point.  Incentive-channel privacy is
    # instead delivered by the low-sensitivity output-perturbation mechanism
    # (use_private_incentive below), which keeps on-chain rewards rank-faithful at
    # ε≤50 — a ~√(d/n) better budget than the weight channel.
    'use_dp'               : False,                 # weight-sharing DP: swept knob, not default
    'dp_epsilon'           : 1.0,                   # privacy budget ε for the weight channel (when swept)
    'dp_delta'             : 1e-5,                  # failure probability δ
    # Shapley-value contribution weights (Wang et al., FedSV 2020)
    'use_shapley'          : True,                  # replace DB-BOA Job 3 with exact Shapley
    # Shapley estimator — controls the cost of contribution attribution as n_orgs grows.
    #   'exact' : evaluate all 2^n-1 coalitions      → O(2^n), intractable beyond ~13 orgs
    #   'mc'    : Monte-Carlo permutation sampling    → O(samples·n), scalable
    # The default stays 'exact' so the n=3 pipeline is byte-for-byte unchanged; the
    # scalability sweep (experiments/scalability_sweep.py) flips this to 'mc'.
    'shapley_method'       : 'exact',
    'shapley_mc_samples'   : 200,                   # permutations for the MC estimator
    'shapley_mc_truncation': True,                  # TMC early-stop (Ghorbani & Zou, ICML 2019)
    'shapley_mc_tol'       : 1e-3,                  # |v_full - v(S)| below this → marginal≈0
    # ── Private-incentive mechanism (B1 contribution) ────────────────────────
    # Output-perturbation DP on the n-dim contribution vector φ instead of the
    # ~1e5-dim weight vector.  φ is computed from the CLEAN models, clipped to
    # L2 ≤ C, then perturbed with Gaussian noise σ = C·√(2ln(1.25/δ))/ε before it
    # becomes the on-chain token split.  Because the released statistic is n-dim,
    # noise/signal ≈ k·√n/ε instead of k·√d/ε — rewards stay rank-faithful at a
    # ~√(d/n) smaller (i.e. stronger) privacy budget (validated ε*≈50 vs ≈3000).
    # The privacy unit is the released contribution statistic (output perturbation,
    # cf. Chaudhuri et al. 2011); model-weight privacy remains the separate (off-
    # by-default) use_dp channel.  See experiments/private_incentive_sweep.py.
    'use_private_incentive': False,                 # privatise the φ/token split (output perturbation)
    'incentive_epsilon'    : 10.0,                  # ε for the contribution channel
    'incentive_delta'      : 1e-5,                  # δ for the contribution channel
    'incentive_clip'       : None,                  # L2 clip C for φ; None → adaptive C=‖φ‖₂
}

ORG_DATA_SPLITS = {
    'BankA': 0.50,   # Bank A has the most data (50%)
    'BankB': 0.30,   # Bank B has medium data (30%)
    'BankC': 0.20,   # Bank C has least data (20%)
}
# Limitation: the ULB dataset comes from one bank's transactions; all three
# "orgs" share the same customer population, time period, and fraud patterns.
# This is a controlled simulation, not a real cross-bank federated deployment.
# McMahan et al. (AISTATS 2017) assume i.i.d. data; this split is mildly
# non-i.i.d. (different volumes, same fraud rate).  For severely heterogeneous
# distributions FedProx (Li et al., MLSys 2020) would be more appropriate.


def make_org_splits(n_orgs: int) -> dict:
    """
    Generate an ORG_DATA_SPLITS-style dict for an arbitrary federation size.

    Returns {f'Bank{i:02d}': fraction} with descending volumes (the largest org
    holds the most data, mirroring the hand-set 50/30/20 split for n=3) that sum
    to 1.0.  Used by the scalability sweep to parameterise n_orgs end-to-end
    without touching the rest of the pipeline (data_loader.split_for_orgs accepts
    this dict directly).

    For n_orgs=3 the volumes are 3:2:1 → {0.50, 0.33, 0.17}; this is *close to* but
    not identical to the canonical 50/30/20 ORG_DATA_SPLITS above, so the default
    n=3 pipeline keeps using ORG_DATA_SPLITS and only the sweep uses this helper.
    """
    if n_orgs < 1:
        raise ValueError("n_orgs must be >= 1")
    raw   = [n_orgs - i for i in range(n_orgs)]   # n, n-1, …, 1  (descending)
    total = float(sum(raw))
    return {f"Bank{i+1:02d}": raw[i] / total for i in range(n_orgs)}
