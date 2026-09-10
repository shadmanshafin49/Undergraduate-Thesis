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
    # Pool the DB-BOA surrogate subsamples its 2,000 training rows FROM.
    #
    # ⛔ Was 3_000, and that silently broke the ULB search (OBJ-13, 2026-09-04).
    # At ULB's 0.167 % fraud a 3,000-row stratified pool holds only **5 unique
    # fraud transactions**, which is below `_ADTCNObjective._MIN_FRAUD_ROWS = 30`.
    # The objective then takes the `replace=True` branch and builds its 30 "fraud
    # rows" by repeating those 5 about six times each; the 70/30 split scatters
    # the copies, and **9 of 9 validation positives end up being copies of rows
    # in the training half** (measured — see obj13_shipped_path_check.json).
    # The surrogate was scoring itself on rows it had memorised, which is why
    # Obf2 pinned at its 5.0000 ceiling and why all five optimisers "found" it.
    #
    # Enlarging the pool costs **no training time at all** — the surrogate still
    # draws 2,000 rows; only the stratified draw it draws them from changes.
    # 36,000 rows yields ~60 unique fraud, i.e. 2x _MIN_FRAUD_ROWS, so the
    # replacement branch has headroom rather than sitting one row from firing.
    # BankSim never needed this (3,000 rows already held 38 unique fraud) but
    # gets the same treatment so the two datasets share one code path.
    "eval_subset"       : 36_000,   # samples used for fast DB-BOA fitness eval
    # Temporal-amount recurrence features (inspired by Liu et al., WWW 2021)
    # NOTE: ULB has no account IDs, so these are sliding-window frequency
    # features (amount_recurrence_before, amount_recurrence_after, degree_ratio),
    # not true graph features.
    "use_graph_features": True,     # append 3 recurrence features to raw features
    "graph_n_bins"      : 50,       # Amount discretisation buckets
    "graph_window"      : 100,      # rolling window size for edge construction
}

# ─── BankSim Dataset Configuration (OBJ-1) ───────────────────────────────────
# BankSim (Lopez-Rojas & Axelsson, 2014; Kaggle `ealaxi/banksim1`)
#   594,643 tx | 4,112 customers | 50 merchants | 180 daily steps | 1.211 % fraud
# The point of this dataset is the `customer` column: ULB has no account IDs, so
# its 10-step "sequences" stitch together unrelated cardholders.  BankSim lets a
# window be one customer's own history — see data/banksim_loader.py.
BANKSIM_PATH = os.path.join(os.path.dirname(BASE_DIR), "datasets",
                            "bs140513_032310.csv")

BANKSIM_CONFIG = {
    "dataset_path"   : BANKSIM_PATH,
    "label_col"      : "fraud",
    "group_col"      : "customer",
    "time_col"       : "step",

    # Row-level feature switches (see encode_banksim).  Both default on: the
    # per-transaction logistic reference reaches only MCC 0.574 with them, so
    # there is no ceiling effect hiding the temporal comparison.
    "use_category"   : True,
    "use_merchant"   : True,

    # Split.  "temporal" is the honest default: train on the past, test on the
    # future, no look-ahead.  Steps run 0-179; the 80 % row mass falls at 147.
    "split"          : "temporal",
    "split_step"     : 147,   # test  = steps >= 147   (122,278 rows, 1.080 % fraud)
    "val_step"       : 132,   # val   = 132 <= step < 147  (future-of-train)
    "test_size"      : 0.20,  # used only by split="stratified"
    "val_size"       : 0.10,
    "random_state"   : 42,
    # See the DATA_CONFIG note: 3,000 rows put ULB below _MIN_FRAUD_ROWS and made
    # its surrogate validate on memorised rows. BankSim was never affected (its
    # 3,000-row pool held 38 unique fraud against a floor of 30) but is raised
    # with it so both datasets run the same code path — and so the margin is not
    # 8 rows on a dataset whose fraud rate could change with a re-split.
    "eval_subset"    : 36_000,

    # Windowing arm: "customer" (entity-linked) or "global" (bank-wide stream).
    "ordering"       : "customer",
    # Tie-break seed for rows sharing a step.  BankSim's file order clusters
    # fraud rows adjacently (3,635 adjacent fraud pairs vs 84 after shuffling);
    # never window the raw file order.
    "order_seed"     : 0,

    # Federated partitioning: "stratified" (ULB-equivalent volume split) or
    # "customer" (entity-disjoint — no customer's history at two banks).
    "partition"      : "stratified",
}

# ─── Fraud Detection Handbook Configuration (OBJ-5, under OBJ-16) ────────────
# Le Borgne, Siblini, Lebichot & Bontempi, *Reproducible Machine Learning for
# Credit Card Fraud Detection* (ULB);  simulated-data-raw, 183 daily pickles.
#   1,754,155 tx | 4,990 customers | 10,000 terminals | 183 days | 0.837 % fraud
# The point of this dataset is `TERMINAL_ID` plus a one-second timestamp: BankSim
# resolves to one day and has no terminal, so this is the only place the "does
# any architecture exploit time?" question can be asked at fine resolution.
# See data/handbook_loader.py for the reconnaissance behind every value here.
HANDBOOK_PATH = os.path.join(os.path.dirname(BASE_DIR), "datasets",
                             "handbook_transactions.csv")
HANDBOOK_RAW_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets",
                                "handbook_raw", "data")

HANDBOOK_CONFIG = {
    "dataset_path"   : HANDBOOK_PATH,
    # Consolidated from here on first use if `dataset_path` is missing.
    "raw_dir"        : HANDBOOK_RAW_DIR,
    "label_col"      : "TX_FRAUD",
    "group_col"      : "CUSTOMER_ID",
    "time_col"       : "TX_TIME_SECONDS",

    # Row-level feature switches (see encode_handbook).  There is no category or
    # merchant analogue to switch here: 10,000 terminals and 4,990 customers
    # cannot be one-hot encoded, and doing so would be an entity-derived feature
    # anyway — so the entity signal is reachable only through the windowing arm.
    "use_hour"       : True,    # 24-column hour-of-day one-hot
    "use_dow"        : True,    #  7-column day-of-week one-hot

    # Split.  "temporal" is the honest default: train on the past, test on the
    # future, no look-ahead.  Days run 0-182; the 70 % and 80 % row-mass
    # quantiles fall at day 128 and day 146 (measured, not assumed).
    "split"          : "temporal",
    "split_day"      : 146,   # test = day >= 146   (354,643 rows, 0.896 % fraud)
    "val_day"        : 128,   # val  = 128 <= day < 146  (172,522 rows, 0.877 %)
    "test_size"      : 0.20,  # used only by split="stratified"
    "val_size"       : 0.10,
    "random_state"   : 42,

    # See the DATA_CONFIG note on the OBJ-13 memorised-rows leak.  36,000 rows at
    # this dataset's 0.814 % train fraud rate holds ~293 unique fraud against the
    # objective's floor of 30.  The inherited value is load-bearing here too, not
    # merely copied: at the old 3,000 the pool would hold ~24, *below* the floor.
    "eval_subset"    : 36_000,

    # Windowing arm: "customer", "terminal" (entity-linked) or "global"
    # (bank-wide stream).  Reconnaissance fraud-adjacency lift over base rate:
    # global 1.01x, customer 13.35x, terminal 71.65x — terminal is strongest
    # because scenario 2 compromises a terminal for 28 days and is 62 % of fraud.
    "ordering"       : "customer",
    # Tie-break seed for rows sharing a timestamp.  6.788 % of rows share one to
    # the second.  Unlike BankSim, raw file order carries **no** fraud adjacency
    # here (125 adjacent pairs, lift 1.02x, vs 124 after the shuffle) — the trap
    # was checked and does not fire, but the shuffle stays so that is a measured
    # property rather than an assumption.
    "order_seed"     : 0,

    # Federated partitioning: "stratified" (ULB-equivalent volume split),
    # "customer" or "terminal" (entity-disjoint — no entity at two banks).
    "partition"      : "stratified",
}

# Dataset registry — lets experiments and main.py take
# `--dataset {ulb,banksim,handbook}` without importing loader modules by hand.
DATASETS = {
    "ulb"      : {"loader": "data.data_loader:FinancialDataLoader",
                  "config": "DATA_CONFIG",
                  "label" : "ULB Credit Card (284,807 tx, 0.17 % fraud, no customer IDs)"},
    "banksim"  : {"loader": "data.banksim_loader:BankSimDataLoader",
                  "config": "BANKSIM_CONFIG",
                  "label" : "BankSim (594,643 tx, 1.21 % fraud, 4,112 customer IDs)"},
    "handbook" : {"loader": "data.handbook_loader:HandbookDataLoader",
                  "config": "HANDBOOK_CONFIG",
                  "label" : "Fraud Detection Handbook (1,754,155 tx, 0.84 % fraud, "
                            "4,990 customer IDs, 10,000 terminal IDs, 1 s resolution)"},
}

#: Which entity-disjoint federated partitions each dataset can support.  ULB has
#: no entity IDs at all — that is a property of the data, not a missing feature,
#: so `_dataset.resolve` fails loudly rather than falling back to stratified.
ENTITY_PARTITIONS = {
    "ulb"      : (),
    "banksim"  : ("customer",),
    "handbook" : ("customer", "terminal"),
}


def get_loader(dataset: str = "ulb", cfg: dict = None):
    """
    Return an instantiated loader for `dataset` ("ulb" or "banksim").

    Both loaders expose the same surface (`load`, `get_eval_subset`,
    `split_for_orgs`, `n_engineered_features`), so callers do not branch.
    The BankSim loader additionally exposes `groups_train/val/test`.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(DATASETS)}")
    mod_path, cls_name = DATASETS[dataset]["loader"].split(":")
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(cfg) if cfg else getattr(mod, cls_name)()


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

    # --- DB-BOA surrogate evaluation protocol (OBJ-13) -----------------------
    # Before the repair the surrogate redrew its 70/30 split and its torch seed
    # on every call, so fitness was a random function of its input: a fixed
    # config re-evaluated 25x on ULB spanned Obf2 3.4497-5.0000 and hit the
    # 5.0000 ceiling once with no search involved.  That is why all five
    # optimisers "found" exactly 5.0000.
    #   "deterministic" - split + seed fixed at construction; fitness is a pure
    #                     function of the candidate.  1x cost.  Default.
    #   "averaged"      - surrogate_k pre-drawn draws shared by every candidate
    #                     (common random numbers), fitness is their mean.  Kills
    #                     the best-of-N ceiling artefact.  k x cost.
    #   "legacy"        - the pre-repair behaviour, kept so the old numbers stay
    #                     reproducible.  Do not use for new results.
    "surrogate_eval_mode": "deterministic",
    "surrogate_k"        : 3,
    "surrogate_rows"     : None,   # None -> _ADTCNObjective._SURROGATE_ROWS (2000)

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
