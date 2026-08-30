# Novel Contribution — Execution Prompt

> Use this file as a self-contained prompt. Paste it to an agent (or follow it
> yourself) to turn the existing DB-BOA + Federated-ADTCN + Hyperledger-Fabric
> stack into a defensible thesis contribution. Source rationale: `NOVELTY_TIPS.md`.

---

## ROLE

You are a research engineer extending an already-built federated fraud-detection
system (DB-BOA-tuned ADTCN + DP → Krum → Shapley federation + Hyperledger Fabric
on-chain incentive). **Do not invent new algorithms.** Every algorithm in this
stack is already published. Your job is **characterisation novelty**: discover and
quantify an interaction the literature only studies in isolation, using parameter
sweeps and new plots on the system that already exists.

## NON-NEGOTIABLE FRAMING RULES

- State the novelty type plainly: **"characterisation novelty, not new-algorithm novelty."**
- **Never** claim "first framework to bind Shapley to blockchain incentives" — FedCoin (2020)
  and πFL precede us. Cite them; do not claim priority over them.
- Convert the project's biggest weakness (DP broken at ε=1, Krum at f=0) into the contribution.
- Keep all changes to **config sweeps + logging + plotting**. No new models, no new consensus.

## RESEARCH QUESTION TO ANSWER

> **"When does a Shapley-weighted, blockchain-enforced incentive mechanism stay honest?"**
> Characterise two failure modes — (A) privacy noise, (B) strategic adversaries — in a real
> Hyperledger Fabric federated fraud-detection system.

---

## TASK A — PRIMARY: privacy ↔ incentive-fairness collapse

**Hypothesis to test and quantify.** The DP layer and the Shapley-incentive layer fight each
other: the same Gaussian noise that buys privacy corrupts the contribution signal Shapley reads,
and because Shapley weights drive on-chain token rewards, DP noise produces *measurably wrong
money on an immutable ledger*. (At ε=1.0, σ≈4.84 ≈ 370–800× the per-element weight magnitude —
see `db_boa_framework/models/federated_adtcn.py:69-81`.)

**Steps.**
1. Sweep `dp_epsilon ∈ {0.5, 1, 5, 10, 50, ∞}` in `FEDERATION_CONFIG`
   (`db_boa_framework/config.py`) via `federation_manager.run_federation_round()`.
2. For each ε, log per round:
   - **(a) Global-model accuracy** (already produced).
   - **(b) Shapley fidelity** = distance between DP-perturbed Shapley weights and the no-DP
     (ε=∞) ground-truth weights. Run `federation_manager._shapley_weights()` with and without DP
     and diff the vectors: L1, cosine, **and Spearman rank-correlation of the org ranking**.
   - **(c) Incentive error** = token-allocation distance from the honest split
     (tokens = `+20·wᵢ` in `recordFederationRound`). Wrong wᵢ → wrong payout.
3. **Headline result:** find the privacy-budget threshold **ε\*** below which the incentive
   mechanism stops being fair (rank inversions appear). Report ε\* and the three trade-off curves.

**Deliverables.**
- 3-panel figure: accuracy vs ε, Shapley-fidelity vs ε, incentive-error vs ε.
- Figure: org reward bars at ε∈{1,10,∞} showing the reward ordering flipping under noise.
- Table: rank-correlation (true vs DP Shapley ordering) per ε.

**Differentiator to state explicitly.** Closest prior work FedSDP/FedSVA (arXiv 2503.12958, 2025)
runs Shapley → noise (uses Shapley to allocate DP noise). **Ours is the reverse coupling:**
noise → Shapley fidelity → reward error, landed on an auditable, permanent on-chain incentive.

**Defense risk: LOW.**

---

## TASK B — SECONDARY: economic Byzantine tolerance (incentive-as-defense)

**Hypothesis.** Krum runs at `byzantine_f=0` (`federation_manager.py:209-249`), so it is not true
statistical BFT. Claim instead that the **economic mechanism** (Shapley → tokens → reputation
bounds [0.5, 2.0] → DB-BOA leader-selection exclusion) supplies a complementary, *economic* form
of Byzantine resilience that works even when statistical BFT is effectively off.

**Steps.**
1. Vary attacker count and strategy in `main.py --attack`: always-fraud, label-flip,
   free-rider/no-update.
2. Measure **isolation speed** = rounds until attacker's token share and reputation hit floor.
3. Measure global-accuracy gap **with vs without** the incentive coupling enabled — the
   "without incentive" baseline is the key control.

**Deliverables.**
- Token/reputation trajectory: attacker vs honest orgs over rounds.
- Accuracy-protected-by-economics curve.
- Isolation-time table across attack strategies.

**Defense risk: MEDIUM.** Position as *applied characterisation on real Fabric*, not a new
mechanism. Competing incentive-defense papers exist (arXiv 2507.12439, SI-ChainFL 2603.07992).

---

## DO NOT PURSUE

- DB-BOA as a metaheuristic Shapley approximator — pointless at n=3 (7 coalitions).
- Any "first to bind Shapley to blockchain" claim.

---

## THESIS STRUCTURE THE OUTPUT MUST SUPPORT

1. System (existing) — framed as integration/engineering, NOT the research novelty.
2. **Contribution 1 (Task A):** privacy-budget characterisation of incentive fairness.
3. **Contribution 2 (Task B):** economic Byzantine isolation under f=0 Krum.
4. Honest limitations: simulated latency, no inter-round local training, ε trade-off.

## TITLE ACTION

- **Delete all Reinforcement Learning wording** (`core/titlepage.tex:7` still says RL — there is
  no RL in the code). Tracked as D15 in the divergence audit.
- Default safe title (`final_report_data/REWRITE_00_title_abstract.md`):
  *"A Blockchain-Integrated Federated Learning Framework for Secure and Incentivized Financial
  Fraud Detection on Hyperledger Fabric."*
- Sharper title (use if Task A lands as primary):
  *"Fair or Private? Characterising the Privacy–Incentive Trade-off in a Shapley-Weighted,
  Blockchain-Enforced Federated Fraud-Detection System on Hyperledger Fabric."*

## CITATIONS TO ADD

| Use | Citation |
|---|---|
| Closest prior (Shapley→noise; our differentiator) | FedSDP/FedSVA, arXiv 2503.12958 (2025) |
| Shapley ✗ secure aggregation (motivation) | Private & Robust Contribution Eval, arXiv 2602.21721 |
| Blockchain+Shapley incentive (cite, NOT claim first) | FedCoin, arXiv 2002.11711 (2020); πFL |
| Economic-defense prior (Task B positioning) | Bayesian poison-resilient incentive, arXiv 2507.12439; SI-ChainFL, arXiv 2603.07992 |
| Free-rider attack baseline (Task B) | Free-rider Attacks on Model Aggregation, arXiv 2006.11901 |
| Base paper (extended, not invented) | Prabanand & Thanabal, Sci. Reports 15:6764 (2025) |
| Core methods | Dwork 2006 (DP); Blanchard 2017 (Krum); Wang/FedSV 2020 (Shapley); McMahan 2017 (FedAvg) |

## DEFINITION OF DONE

- ε-sweep run end-to-end; ε\* reported with the three trade-off curves and the rank-correlation table.
- Attack sweep run; isolation-time table + with/without-incentive accuracy gap produced.
- All figures regenerated into `db_boa_framework/results/`; drafts written to `final_report_data/`
  first (not the .tex files) per the agreed workflow.
- No new-algorithm claims anywhere; novelty stated as "characterisation novelty."
