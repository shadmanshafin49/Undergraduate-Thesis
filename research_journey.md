# The Research Journey — FL-ADTCN

**A journal of the whole project, from the first idea to the latest commit.**

*Project:* A Blockchain-Integrated Framework for Secure, Incentivized, and Scalable Machine Learning:
Integrating Consensus Mechanisms and Reinforcement Learning. The system is called **FL-ADTCN**.
*Team T2430460, BRAC University, Department of CSE.* Md. Abrar-ud-doula Amiya (21301567),
Shadman Sakib (21301566), Nafi Alif Mahin (21201699), Muntasir Mahin Siam (21301708) and Mahima
Islam (21201143). *Supervisor:* Md. Golam Rabiul Alam.
*Covers:* the P2 proposal (before May 2026) → the last commit, `c66d1d1`, on **13 September 2026**.
*Compiled:* 26 September 2026.

---

## How to read this

This is a journal, not a paper. It is written in the order things happened, and each entry says
what we were thinking, what we did, what came out, and which files were made or changed. When
a decision was taken, the entry says who took it and why. When a belief turned out to be wrong,
the entry keeps the wrong belief next to the correction, because the corrections are the most
useful part of this project.

**Where it comes from.** Almost every entry is reconstructed from things in the repository: the
26 commits across both GitHub repositories, every revision of `NOVELTY_TIPS.md`, `build_log.md`,
`TASK.md` (4,291 lines), `WORK_REPORT.tex`, `SUPERVISOR_BRIEF.md`, the `final_report_data/`
drafts, `reports/report.txt` (the P2 report), and the result JSONs. A few details from 30 August
come from the working notes kept alongside the AI-assisted sessions: two remarks of Shadman's
quoted in Part V, and the check of an external status handbook. Those are marked where they
appear. Where nothing records something, such as the conversations that led to the original
proposal, this journal says so and does not fill the gap.

**Companion files.** `build_log.md` is the terse changelog: every change set, file by file.
`TASK.md` is the live mission board, with its pre-registrations and scorecards. `WORK_REPORT.tex`
is the supervisor-facing account of 30 August – 13 September. Numbers live only in
`db_boa_framework/results/*.json`. If this journal and a JSON disagree, the JSON wins.

**The shape of the document.**

- Part I — Before the code: where the idea came from.
- Part II — The first build (20 May – 2 June 2026).
- Part III — The audit and the turn (2 – 13 June 2026).
- Part IV — The quiet months (13 June – 29 August).
- Part V — The paper extension (30 August – 13 September 2026), day by day.
- Part VI — Where things stand at the last commit.
- Part VII — What we learned.
- Appendix A — Every file in the project and why it exists.
- Appendix B — Numbers that were withdrawn, and why.
- Appendix C — The decision register.
- Appendix D — Glossary.

---

## The whole story on one page

We set out to build a federated fraud-detection system on a blockchain. Several banks train a
shared model without sharing their data. A blockchain records who contributed what and pays
them for it. A privacy mechanism hides the banks' updates, and a robust-aggregation rule stops a
malicious bank from poisoning the model. The design came from our P2 proposal and from one base
paper, Prabanand & Thanabal (2025), whose **DB-BOA** optimiser and **ADTCN** detector we adopted.

The first working version (20 May 2026) ran end to end, but mostly on things that were not what
they claimed to be: synthetic data, an MLP labelled as a temporal network, a simulated
consensus, and plots of baselines that had never been run. Over the next two weeks, repeated
self-reviews replaced each of those with something real: the ULB credit-card dataset, a real
1-D CNN, Krum, differential privacy (DP), Shapley-value rewards, and a real Hyperledger Fabric
network.

In early June we audited our own thesis draft against the code and found **nine claims no code
had ever produced**, among them 97.38 % accuracy, MCC 0.966, 85 TPS, 180 ms and an eight-model
comparison table. We deleted them all. We also found the result that became the thesis: **the
privacy noise added to the banks' model weights destroys the Shapley calculation that decides
who gets paid, and moving that noise onto the published contribution score instead restores
honest rewards at a far more private budget.** The thesis was accepted on **13 June 2026**.

From 30 August we extended the work toward a paper. We added four more datasets (BankSim, the
Fraud Detection Handbook, PaySim, IBM AMLSim), wrote down every prediction before running it,
and ran controls designed to break our own claims. **Three of our headline claims died that way,
killed by our own tests rather than by a reviewer.** Along the way we also found that our
hyperparameter search had been validating on memorised rows, that a privacy "improvement factor"
grows the harder you look for it, and that on the Handbook two defences we had called universal
both fail on the same attack. By the last commit, 49 predictions had been scored and 20 were
wrong. The central privacy result survives on four datasets and fails on one. We report that
split as it is.

---

## Cast and places

| Who / what | Role in this story |
|---|---|
| The five authors | Team T2430460. **Shadman Sakib** is the team lead for the paper extension ("the operator" in `TASK.md`) and takes the scope decisions recorded there. |
| Md. Golam Rabiul Alam | Supervisor. `SUPERVISOR_BRIEF.*` and `WORK_REPORT.*` are written for him. |
| Prabanand & Thanabal (2025), *Scientific Reports* 15:6764 | The base paper: the DB-BOA optimiser and the ADTCN detector. Kept in `all papers/Prabanand_Thanabal_2025_DB-BOA-ADTCN.pdf`. |
| `shadmanshafin49/Undergraduate-Thesis` | The canonical repository (`origin`). Branch `obj13-surrogate-repair` holds the whole paper extension; `main` is still at `ccdd19f` (30 Aug). |
| `shadmanshafin49/Undergraduate-Thesis-v2` | A frozen archive of how the June work was built (6 commits, 6–11 June, author `root`). Read-only remote `v2`. It has no shared history with the canonical repository, so it must never be pushed to. |
| Machines | May: the local checkout. June: a Linux/WSL2 machine (Docker 29, Fabric 2.5.10). Aug–Sep: a Windows 11 laptop (Python 3.13.7, torch 2.12.0+cpu, 4 threads) for ULB, BankSim and the Handbook sweeps, plus **Kaggle** CPU sessions (Python 3.12, the same library pins, 4 threads) for PaySim, AMLSim and the Handbook's window grid. |
| AI assistance | Much of the implementation, auditing and drafting from June onward was done in sessions with an AI coding assistant (Claude, through Claude Code). Ten commits carry a `Co-Authored-By: Claude` trailer: v2's initial commit (6 June) and nine of the eleven commits from 30 August on. In September every pre-registration was *drafted by the assistant and approved by the team lead before any run*, as `TASK.md` records. The dataset-expansion plan began from a ChatGPT consultation that Shadman shared on 30 August. This is recorded here because the thesis's honesty standard applies to how the work was done as much as to its numbers. |

## Timeline at a glance

| Date | What happened |
|---|---|
| before May 2026 | P2 proposal and report: a Blockchain-Integrated Federated Learning (B-FL) framework. The title is fixed. |
| **20 May** | First commit. Six "novelty tips" implemented the same evening. The first self-review turns into an open-issues list. |
| 21 May – 2 Jun | Twenty-nine fixes and ten bug fixes, most of them honesty fixes: removing false claims and invented plots. |
| 2 – 6 Jun | Report-vs-code audit (D1–D17). Characterisation Tasks A–D. RL leader selection added to honour the title. |
| 6 – 7 Jun | **The key insight:** move the DP noise to the contribution channel (B1). Real Fabric measured (B2). Temporal ablation (B3). |
| 8 – 11 Jun | Report rebuilt from the drafts. Supervisor brief. Final audit fix batch. |
| **13 Jun** | **Thesis accepted.** |
| 13 Jun – 29 Aug | No commits. |
| **30 Aug** | Repository synced to the June state. `TASK.md` board created. Dataset plan: BankSim first. |
| 31 Aug | Loose ends closed. BankSim grid. Base-paper models re-implemented. Multi-seed detector. |
| 1 Sep | Direction review mistake, then corrected. System sweeps taken off ULB. Confound control launched. |
| **4 Sep** | Control scored and **our best new claim dies**. Privacy factor measured, then retired. Surrogate leak found and fixed. |
| 5 Sep | The 16.3 h search side-by-side: **0 of 9 beat the hand-set default**. Everything committed. First work report. |
| 8 – 10 Sep | Handbook loader. Report figures. |
| **11 Sep** | Rule 8 suspended: all five datasets to be run. PaySim reconnaissance. AMLSim generated, discarded, regenerated. Kaggle pipeline. |
| **12 Sep** | PaySim and AMLSim scored. ULB regenerated. **The Handbook breaks two defences.** |
| **13 Sep** | Separation-margin audit. AMLSim shown unable to test Krum. Per-dataset journey section written. **Last commit.** |

---

# PART I — BEFORE THE CODE: WHERE THE IDEA CAME FROM

### 📅 Before May 2026 — The P2 proposal

**What we were thinking.** The starting question was broad: *can blockchain make federated
learning trustworthy?* Federated learning lets several parties train one model without pooling
their data. It leaves four problems open: who aggregates, why anyone should contribute honestly,
how to keep the shared updates private, and how to stop a participant from poisoning the model.
Blockchain looked like an answer to the first two, because an immutable ledger can record
contributions and pay for them.

**What the P2 report proposed** (`reports/P2_REPORT_T2430460.pdf`, text in `reports/report.txt`,
poster `reports/P2_POSTER_T2430460.pdf`). A "Blockchain-Integrated Federated Learning (B-FL)
framework" using the ledger, Byzantine Fault Tolerance, Proof of Deep Learning, smart contracts,
Fragmented Federated Learning, Additive Homomorphic Encryption, Gaussian-mixture and
random-forest DDoS detection, and token/reputation incentives. Target domains: healthcare,
finance and IoT. The title, **"A Blockchain-Integrated Framework for Secure, Incentivized, and
Scalable Machine Learning: Integrating Consensus Mechanisms and Reinforcement Learning"**, was
fixed at this stage. Every later decision had to live inside it.

**How it narrowed.** By Chapter 4 of the P2 report the design had become specific: a **financial
consortium on Hyperledger Fabric**, where each bank trains an **ADTCN** fraud detector locally
and a **DB-BOA** optimiser does three jobs:

1. **Leader selection:** choose the validator that aggregates and creates the block
   (minimising latency, resources and network cost; Eq. 10 in the base paper);
2. **Hyperparameter tuning** of ADTCN (Eq. 11);
3. **Global optimisation** of the federated aggregation (what the code later called "Job 3").

Both components came from the base paper, **Prabanand & Thanabal (2025)**. In that paper, ADTCN
is a temporal network with three conceptual blocks: MJE (Multi-Modal Joint Embedding), TCL
(Temporal Context Learning) and MTTA (Multiple Time-scale Temporal Attention). DB-BOA, the
"Dynamic Butterfly–Billiards Optimisation Algorithm", is a hybrid of the Billiards Optimisation
Algorithm (BOA, Givi & Hubálovská 2023) and the Dynamic Butterfly Optimisation Algorithm (DBOA,
Tubishat 2020).

**The targets the P2 report set** (§4.1.6): block latency **≤ 250 ms**, federated detection
accuracy **≥ 97 %**, block validation rate **≥ 95 %** under Proof of Authority, and **≥ 30 %** lower
latency than BFT-style baselines through DB-BOA leader scheduling.

> **Worth noticing, in hindsight.** Three of the numbers our June draft later reported, and that
> our own audit then found no code had produced, sit on these targets: 97.38 % accuracy (target
> ≥ 97 %), 180 ms latency (target ≤ 250 ms) and a 28.4 % latency cut (target ≥ 30 %). The
> repository does not record how those numbers entered the draft. It does record that none of
> them came from a run, and that we deleted them (Part III).

**What the repository does not record.** The discussions in which the team chose the base paper,
the financial-fraud domain and Fabric are not in the repository. The P2 report is the earliest
artefact, and this journal starts from it.

---

# PART II — THE FIRST BUILD (20 MAY – 2 JUNE 2026)

### 📅 20 May, 19:44 — The first commit: the P2 design as running code (`b903bf2`)

**What existed.** A complete pipeline that ran end to end:

- `db_boa_framework/main.py` orchestrated six phases: DB-BOA leader selection → DB-BOA
  hyperparameter search → ADTCN training → evaluation → multi-round consensus simulation → plots.
- `algorithms/boa.py`, `dboa.py`, `db_boa.py`: the three optimisers.
- `models/adtcn.py`: "ADTCN", which was in fact a scikit-learn **`MLPClassifier`**.
- `models/federated_adtcn.py`, `models/federation_manager.py`: three banks whose aggregation
  weights DB-BOA "Job 3" optimised.
- `blockchain/leader_block.py`: a 10-node consortium; the "consensus round" was `time.sleep`
  plus arithmetic on resource scores.
- `data/data_loader.py`: a **synthetic** 20,000-row generator (30 features, 5 % fraud) with PTC/NTC
  rolling-window feature engineering.
- `utils/metrics.py`: the base paper's objective, Obf2 = Acc + Pre + NPV + MCC + **1/FPR**, and
  base-paper baseline numbers typed in by hand. `utils/visualizer.py` had 12 plots.
- `db_boa_fabric/`: real Fabric chaincode (`chaincode/lib/db_boa_chaincode.js`, the
  `DBBOAContract` with functions such as `updateNodeMetrics`, `updateIncentive`,
  `recordFraudResult`, `recordConsensusRound`, `recordFederationRound`, `getNodeStatus`), an
  Express REST server (`api-server/server.js`), a web dashboard (`api-server/index.html`), wallet
  enrolment scripts and a launcher.
- `fabric/install-fabric.sh` (Fabric installer) and `hello-world/docker-compose.yml` (a Docker
  smoke test).

**What we believed.** That this was the thesis system: FL-ADTCN with DB-BOA in three roles.

**What we did not yet know.** Almost every component was a placeholder for what its name
claimed. The rest of Part II is the list of those placeholders being found and replaced.

### 📅 20 May, 20:14 — The report sources, the reference papers, and the first review (`fe39499`)

**Files made.** The LaTeX thesis (`FINAL YEAR THESIS REPORT/main.tex`, Chapters 1, 2, 3, 5, 6 and 9,
the `core/` front matter, `bibliography/references.bib` and about 30 images); the first nine
reference PDFs in `all papers/`; and **`NOVELTY_TIPS.md`**.

**The first review, `NOVELTY_TIPS.md` version 1**, was blunt about what an examiner would see:

- *Tier 1:* replace synthetic data with a real benchmark ("99 % accuracy on self-generated data
  proves nothing"); add real Byzantine-robust aggregation (**Krum**, Blanchard 2017), because a
  token penalty fires only *after* a poisoned model has already corrupted the global weights;
  add **differential privacy** to weight sharing (Dwork 2006).
- *Tier 2:* replace the MLP with a real temporal model, since "an MLP treats the 10-timestep
  sequence as a flat vector"; replace DB-BOA Job 3 with **Shapley values** ("a population
  metaheuristic to optimise 3 scalars that sum to 1 is overkill").
- *Tier 3:* transaction-graph features.
- *What not to do:* no more metaheuristics, no Ethereum claims on Fabric, no extra chaincode
  features, and no claim that DB-BOA beats Optuna or random search without evidence.

### 📅 20 May, 21:27 — Tips 1–6 implemented in one evening (`c7d3012`)

The detail is in `build_log.md` (Tips 1–6). In short:

- **Tip 1 — real data.** The loader now reads the **ULB Credit Card Fraud** dataset
  (`creditcard.csv`: 284,807 transactions, 492 frauds, 0.17 %, 28 PCA components). *Why ULB:* it
  is the standard public benchmark and the one the literature compares against. *Its catch,
  which would take three months to bite:* **it has no customer identifiers.**
- **Tip 2 — Krum.** Each bank's update is scored by its distance to its nearest neighbours, and
  the most central one becomes the global model.
- **Tip 3 — DP.** Weights are clipped to L2 norm 1 and Gaussian noise is added, with σ ≈ 4.84 at
  ε = 1, δ = 1e-5.
- **Tip 4 — a real 1-D CNN.** `Conv1d → ReLU → Conv1d → GlobalMaxPool → Linear` over windows of
  10 consecutive transactions, with a class-weighted loss for the 0.17 % imbalance.
- **Tip 5 — Shapley.** Exact values over the 7 coalitions of 3 banks decide the aggregation and
  reward weights.
- **Tip 6 — "graph features".** Three features from amount-bucket co-occurrence (ULB has no
  account IDs, so there is no real graph to build).
- `build_log.md` was created that evening to record each change.

### 📅 20 May, late evening — The review turns on its own fixes (`2b38904`, `69aadb0`, `6a2b4a3`)

**What changed in our thinking.** `NOVELTY_TIPS.md` stopped being a list of tips and became
**"Open Issues — Must Fix Before Submission"**, ordered by how quickly each would end a defense.
Its first line: *"The 1D-CNN replacement is the only change that is solid. Everything else has
bugs, false claims in docstrings, or broken mathematical guarantees."* Among its findings:

- Krum's guarantee needs **n ≥ 2f + 3**. With 3 banks and f = 1 it does not hold (Fix 1: f = 0,
  and the "Byzantine fault tolerance" wording removed in Fix 7).
- The Shapley docstring said "no shared labels needed", but it needs a labelled validation set
  at the aggregator (Fix 2).
- The "graph features" are a sliding-window frequency counter, and the "PageRank" is a degree
  ratio (Fix 3, features renamed honestly).
- The activation-comparison plot was **invented**, drawn as offsets from one measured accuracy
  (Fix 9, replaced by a stub).
- DB-BOA's epoch dimension was flat because the surrogate capped training at 5 epochs (Fix 10:
  the search became 2-D).
- The surrogate trained at the wrong fraud rate (Fix 11).
- "MTTA attention" was a global max-pool (Fix 12, relabelled).
- The comparison table was empty (Fix 17: `run_baselines.py`, which runs FedAvg / +Krum / +DP /
  proposed).

**Files made.** `db_boa_framework/run_baselines.py`, and **`defense_questions.md`**: 150 likely
viva questions (Q1–Q150), phrased against the current code. From here on most fixes cite the
question they answer (Q30, Q50, Q144 …).

### 📅 21 May — Twelve more fixes, mostly disclosures (`50492f4`)

Fixes 18–29. Some were code (surrogate minimum 30 fraud rows; Shapley on `X_val` rather than
`X_test`; FedAvg weighted by bank size, as McMahan 2017 defines it). Most were **disclosures
written into the code itself**: ReLU is used and no activation ablation was run; latency is
simulated; the window length 10 is an empirical choice; padding affects 0.003 % of rows; the
PTC/NTC features the loader builds are discarded by the CNN; and the most consequential one:

> **Fix 26 — at ε = 1, σ ≈ 4.84 is 370–800× larger than each weight.** *"The DP-shared global model
> is near-random weights."* We wrote this down in May as a disclosure. It became the thesis in June.

### 📅 1 June — A second loop finds fabricated plots we had missed (`e1bb51a`)

Recorded in `build_log.md` as Fixes 30–36. The most important:

- `visualizer.py` still drew **invented** convergence curves for four optimisers and **hard-coded
  AUCs** for EfficientNet, ResNet, DenseNet and DTCN. None had ever been run. Both were removed.
- `main.py --attack` crashed on a key only the dead DB-BOA Job-3 path returned.
- The eval subset had been rebalanced to 50/50 fraud. Once it kept the real rate it held about 5
  fraud rows, so the surrogate's `rng.choice(fraud_idx, 30, replace=False)` crashed, and we made
  it sample **with replacement**. ⚠ *Three months later (4 Sep) this line turned out to be how the
  search came to validate on memorised rows.*

### 📅 2 June — Ten bugs in one session (`37b5d4a`)

BF-1 to BF-10 in `build_log.md`. Two were serious outside machine learning. BF-7: a
**path-traversal hole** in `server.js` (`/api/plots/../../etc/passwd`). BF-8: `recordFraudResult`
was called with its arguments **in the wrong order**, which silently broke all on-chain fraud
recording. Also: an accuracy-delta assignment in the wrong loop, a falsy `0` overriding
`n_pop`, a crash in `get_training_info`, a CouchDB iterator never closed, the attack
simulation's `predict` override invisible to Shapley, and `get_eval_subset` made genuinely
stratified.

**What Part II taught.** Each round of review found claims the code did not support, and the
pattern was always the same: *a name promising more than its implementation.* "Temporal network"
was an MLP. "Graph features" were a counter. "Attention" was a max-pool. "Byzantine-tolerant"
held at f = 0. "Measured" latency was `time.sleep`. By 2 June the code was far more honest than
the thesis draft describing it, and that gap is Part III.

---

# PART III — THE AUDIT AND THE TURN (2 – 13 JUNE 2026)

*This work happened on a Linux/WSL2 machine and was committed to the separate `-v2` repository
(6, 7, 8 and 11 June). It reached the canonical repository only on 30 August.*

### 📅 ~2 – 6 June — The report-vs-code audit: D1–D17

**What we were thinking.** The code had been fixed; the LaTeX report had not. The rule we adopted
(written in `final_report_data/README.md`): *"If something has not been used in the actual project
work, it must not appear in the final report."*

**What we did.** We read the draft chapter by chapter against the code and wrote
**`final_report_data/00_report_vs_code_divergences.md`**. It lists 17 divergences:

- 🔴 **D1:** the report's "primary novel contribution", DB-BOA Job 3 for aggregation weights, was
  **dead code**. The live path was Shapley + Krum + DP, and Job 3's saved fitness was the
  degenerate constant −1.0000000397e8. It had never optimised anything.
- 🔴 **D2:** the architecture was described as a dilated TCN with a 64-d embedding, BatchNorm and
  softmax attention. The code is a 2-layer 1-D CNN with global max-pool.
- 🔴 **D3:** FedProx with μ = 0.05 and a sensitivity table. Not implemented anywhere.
- 🔴 **D4:** an 8-model comparison table **copied from the base paper**.
- 🔴 **D5:** "97.38 % accuracy, MCC 0.966", produced by no run. The saved run said 99.45 % / 0.941,
  and the report's own Table 4.8 said 0.7812, contradicting it.
- 🔴 **D6–D9:** a 28/18/4 leader-selection split over 50 rounds (the code ran 5 rounds and elected
  Node 7 every time); **85 TPS / 180 ms / "28.4 % latency cut"** (simulated arithmetic); paired
  t-tests over 3 seeds (no multi-seed experiment existed); token balances 458/312/178 over 50
  rounds (3 rounds had been saved).
- 🟠 D10–D14: the dataset described as synthetic in two chapters, a 3-D search that was 2-D, a
  stale results JSON, an activation comparison never run, an attack table never computed.
- 🟢 D15–D17: "Reinforcement Learning" in the title with no RL in the code; wrong SDK versions;
  DP described as future work when it was implemented.

**Files made.** `00_report_vs_code_divergences.md`, `00_ground_truth_implementation.md` (what the
system actually is), `README.md`, per-chapter notes `01_introduction.md` to `06_conclusion.md`,
`07_actions_checklist.md`, and paste-ready rewrites `REWRITE_00_title_abstract.md`,
`REWRITE_01_introduction.md`, `REWRITE_02_literature_and_bib.md`, `REWRITE_03_requirements.md`,
`REWRITE_05_methodology.md`, `REWRITE_06_results.md` and `REWRITE_09_conclusion.md`. Working rule:
**drafts first, `.tex` only after review.**

**Decision — "Option A".** We re-framed the novelty around what was actually built rather than
switching the code back to a Job 3 that did not work.

### 📅 ~2 – 6 June — Deciding what kind of novelty this is (`novel_plan.md`)

**What we were thinking.** Once the invented numbers were gone, what remained? Every algorithm in
the stack was already published: DP, Krum, Shapley, FedAvg, Fabric. We could not honestly claim
to have invented anything.

**The answer, written into `novel_plan.md`:** **"characterisation novelty, not new-algorithm
novelty."** Take a system assembled from published parts and find an *interaction* the literature
only studies in isolation. The research question became: ***When does a Shapley-weighted,
blockchain-enforced incentive mechanism stay honest?*** Two failure modes to characterise: (A)
privacy noise, (B) strategic adversaries. Two rules came with it: *convert the project's biggest
weakness (DP broken at ε = 1, Krum at f = 0) into the contribution*, and **never claim "first to
bind Shapley to blockchain"**, because FedCoin (2020) did that first.

### 📅 ~2 – 6 June — Task A: does privacy noise break the rewards?

**Hypothesis.** The Gaussian noise that buys privacy also scrambles the contribution signal
Shapley reads. Shapley weights drive token rewards on an immutable ledger, so DP noise would
produce *measurably wrong money that cannot be undone*.

**What we built.** `db_boa_framework/experiments/privacy_incentive_sweep.py`. It sweeps ε; at each
budget it compares the noisy Shapley split with the noise-free one (L1, cosine, Spearman ρ) and
measures token error and the rate at which the reward *ranking* inverts. To have an honest
ranking to invert, it injects label noise `{A: 0, B: 0.10, C: 0.25}`. That is a constructed
ground truth, and we disclose it as one.

**What happened.** Rewards were mis-ordered at **every budget up to ε\* = 3000**, which is no privacy
at all in practice. The global model sat near 50 % balanced accuracy until ε ≈ 300 (results:
`results/privacy_incentive_sweep.json`, figures `privacy_incentive_tradeoff.png`,
`privacy_incentive_reward_bars.png`, draft `TASKA_privacy_incentive_results.md`).

### 📅 ~2 – 6 June — Tasks B, C, D: the other title words, tested

- **Task B — economic Byzantine tolerance** (`experiments/economic_byzantine_sweep.py`). Could the
  reward layer itself act as a defence, isolating an attacker whose Shapley share keeps falling?
  *Result:* a colluding **2-of-3 majority is caught** (+40.78 pp always-fraud, +79.52 pp label-flip),
  but a **lone attacker and a free-rider are not**. Kept as a negative result.
- **Task C — scalable attribution** (`experiments/scalability_sweep.py`, plus exact and Monte-Carlo
  Shapley in `federation_manager.py`, and `make_org_splits` in `config.py`). *Why:* the title says
  "Scalable", and exact Shapley is O(2ⁿ). *Result:* exact takes 0.14 s at n = 3 and 113 s at n = 12;
  Monte-Carlo reaches n = 20 in 95 s but is barely faster below n ≈ 10. We decided that "Scalable"
  can only honestly mean **scalable contribution attribution**, not throughput.
- **Task D — Krum where its theorem holds** (`experiments/byzantine_robustness_sweep.py`). At n = 3
  and f = 0, Krum is only outlier rejection. We ran it at **n = 5 / f = 1 and n = 7 / f = 2** against
  four weight-poisoning attacks. *Result:* the attacker was rejected **8/8**, and unprotected
  FedAvg collapsed only under the scaled attack (87.46 % against Krum's 99.95 %). "Secure" became
  a claim backed in code, not just in wording.
- Alongside: `utils/metrics.py` gained `coalition_score` (balanced accuracy), because Obf2's 1/FPR
  term exploded to ~1e8 inside Shapley and made the weights meaningless.

### 📅 ~2 – 6 June — The title word we had no code for: Reinforcement Learning

**The problem.** D15: the title says "Reinforcement Learning" and there was no RL anywhere. The
title had been approved and could not change. The options were to keep a word the code did not
support, or to find where RL is genuinely the right tool.

**Where it fit.** Leader selection. DB-BOA picks a leader each round by minimising a static cost,
so it keeps electing the same node and cannot notice one going bad. That is a sequential
decision problem.

**What we built.** `db_boa_framework/blockchain/rl_leader.py`: linear function-approximation
Q-learning. The state is each node's metrics, reputation, tokens, load and failure rate; the
action is which node leads; the reward is the node's on-chain token payout. It is wired into
`leader_block.py` (`attach_rl_agent`, `select_leader_rl`, `run_rl_round`,
`compare_leader_methods`), with `experiments/rl_leader_sweep.py` and a write-up in
`db_boa_framework/final_report_data/08_rl_leader_selection.md`.

**Result, stated with its caveat.** Stationary: RL *ties* DB-BOA on reward and spreads leadership
more fairly (Gini 0.90 → 0.78). Non-stationary: when a node's reliability collapses, DB-BOA keeps
electing it 100 % of the time and RL about 15 %. **That win depends on a fault we injected**
(`reliability = 0.15`). It is a simulation, and it is recorded as a secondary contribution.

### 📅 6 June — The brutal critique (`new_issues.md`, `title_issue.md`)

**What we were thinking.** Before writing the report we asked for the harshest reading of our own
work. `new_issues.md` ("Brutal Research Analysis", 2026-06-06) said:

- **N1:** the configuration `main.py` ships with (DP at ε = 1 + Krum + Shapley) is the **worst row in
  its own ablation**, below a coin flip. *"What is the contribution if the system you propose is
  the worst row in your own table?"*
- **N2:** Task A actually proves DP is unusable here, and ε\* is unstable: it had already moved from
  1000 to 3000.
- **N3:** two of the four novelty results rest on conditions we inserted.
- **N4:** half the title is simulation.
- **N5:** there is no ML novelty.

Its verdict: the work holds together *"as an honest characterisation study, not as the framework
the title sells."*

`title_issue.md` scored the six title words: Blockchain-Integrated ✅, Incentivized ✅ (not novel),
RL ✅ (qualified), Secure ✅ (qualified), **Consensus Mechanisms ⚠ overstated** (we use Fabric's
stock Raft), **Scalable ML ⚠ overstated** (only attribution scales).

### 📅 6 – 7 June — The insight: move the noise (`0f02ce8`, "now title sticks word to word")

**What we were thinking.** Task A showed DP destroying the rewards. N1 and N2 said the default
system was broken. The obvious move was to turn DP off and stop talking about it, and that is
exactly the move `novel_plan.md` forbade. So we asked a different question: *is the problem
privacy itself, or where we put the noise?*

**The insight.** We were adding noise to a **~111,874-dimensional weight vector** and then trying to
read a **3-number** contribution split through the fog. Noise per coordinate grows with dimension.
Protect the *contribution score itself* instead: apply **output perturbation** (Chaudhuri 2011)
directly to the 3-dimensional Shapley vector φ before it is published and paid out.

**What we built.**

- `models/federation_manager.py::_privatise_incentive`: clip φ at C = ‖φ‖₂ and add Gaussian noise.
- `config.py`: weight-channel `use_dp` became **off by default, a swept knob**; `use_private_incentive` added.
- `experiments/private_incentive_sweep.py`: both channels, the same seeded noise draws
  (`np.random.seed(3000 + rep)`), 100 draws per budget.
- Results: `results/private_incentive_sweep.json`, `private_incentive_channel.png`, draft `TASKB1_private_incentive_results.md`.

**What happened.** ε\* fell from **3000 (weight channel) to 50 (output channel)**. At ε = 50 the
reward ranking's Spearman ρ went from **−0.288 (rewards backwards) to +0.950**. **This became the
thesis's central contribution:** *the privacy ↔ incentive failure is a property of the channel, not
of DP-plus-Shapley in principle.* (The "60×" ratio was later retired as a headline; see 4 Sep.)

**Also that day: two more results.**

- **B2, real Fabric.** `db_boa_fabric/api-server/measure_consensus.js` timed `recordConsensusRound`
  on the live test network: **mean 2115.9 ms** (the 2 s Raft batch timeout is the floor), and
  **peak 40.3 TPS** at 10 concurrent clients, collapsing past that (50 % failed at 20 clients).
  These replaced the invented 85 TPS and 180 ms. Files: `results/fabric_consensus_measured.json`,
  `experiments/write_b2_draft.py`, `final_report_data/B2_fabric_consensus_measured.md`, and a
  `load_measured_consensus` hook in `leader_block.py`.
- **B3, does the temporal model pay?** We built the dilated-conv + attention ADTCN the report had
  described (`_DilatedBlock`, `_DilatedAttnClassifier`, `make_temporal_model` in `adtcn.py`) and
  compared it on shuffled and time-ordered windows (`experiments/temporal_pipeline_ablation.py`,
  `experiments/architecture_ablation.py`). The plain CNN on shuffled windows won (0.770); the
  dilated + attention model on real time order scored 0.459. We concluded that "temporal
  complexity does not pay on ULB" and kept the CNN. *Three months later BankSim showed why: ULB's
  windows had no one's history in them.*

### 📅 8 June — The ablation, and the report rebuilt (`cbe5035`, `7d84bf4`, `81981f0`)

- `run_baselines.py` wrote **`results/baselines.json`**: FedAvg MCC 0.569 → **+Krum 0.776** → +DP (ε = 1)
  0.000 → full proposed pipeline ≈ 0. We reported the collapse as evidence for the trade-off, not
  as something to hide.
- The REWRITE drafts were merged into the `.tex` chapters; the bibliography grew to 71 verified
  entries; `REWRITE_08_limitations_disclosures.md` (threat model and limitations) and
  `02_literature_review_DRAFT.md` were written; **`finalreport_checklist.md`** tracked the merge.
- `new_issues.md` was deleted once its fixes had been folded in. Build artefacts were cleaned out.

### 📅 10 – 11 June — The detector's own claim, the brief, and a last audit (`94c071c`)

- **DB-BOA against a hand-set default** (`experiments/dbboa_vs_default.py`). A paired retraining
  gave default MCC 0.785 against tuned 0.313, so we reported that *DB-BOA does not beat a hand-set
  default*. ⚠ *On 31 August we found 0.313 was produced by a different CPU thread count, and
  withdrew it.*
- `experiments/plot_federated_confusion.py` (Figure 5.4 from saved counts, no re-run).
- **`SUPERVISOR_BRIEF.md` / `.tex` / `.pdf`**: the one-read walkthrough (problem → approach → the
  turn → results → honest claims → limitations).
- Report polish against an approved example thesis (`example_thesis/`): nomenclature
  (`core/nomenclature.tex`, `.latexmkrc`), a methodology figure (`images/methodology_overview.svg`
  and `.png`), a live capture of the Fabric network (`images/fabric_live_capture.png`), a class
  distribution figure, a Comparison with Prior Work section, and Limitations and Future Work.
- **Final audit, 11 June** (`final_report_data/REWRITE_FIX_2026-06-11.md`): ten more errors found
  and fixed that day. The "3-organisation Fabric deployment" was really 2 organisations on one
  host. The feature count is 33, not 29 or 30. Training uses plain Adam, not AdamW with a
  scheduler. A "threshold optimisation" section described code that did not exist. The round
  counts were wrong. Statistics were re-derived from the raw CSV.

### 📅 13 June — The thesis is accepted

The submitted thesis stands on ULB alone. Its central claim is the privacy ↔ incentive coupling,
reported at the weight-DP-off operating point. Its detector headline was Acc 99.85 % / MCC 0.677
(later replaced by a five-seed distribution). Its honest negatives: DB-BOA does not beat the
default; MC-Shapley's top contributor is unreliable; the economic layer does not stop a lone
attacker; Krum has no guarantee at n = 3.

---

# PART IV — THE QUIET MONTHS (13 JUNE – 29 AUGUST)

No commits exist for this period. Two facts about the repositories matter for what came next:

1. The **canonical** repository (`Undergraduate-Thesis`) was still at the 2 June state. It was
   missing every experiment the accepted report describes. Those lived only in `-v2`.
2. The two repositories had **independent histories** (no shared commit), so they could not
   simply be merged.

---

# PART V — THE PAPER EXTENSION (30 AUGUST – 13 SEPTEMBER 2026)

*The aim was to turn the accepted thesis into a publishable paper by 19 September.*

### 📅 30 August — Sync, the mission board, and the dataset question

**The repository.** A local commit from that morning (`69ef481`, "refreshing") had tracked
`creditcard.csv` (143.8 MB, above GitHub's 100 MB per-file limit) and 17 `.pyc` files. A push would
have failed, and deleting the file in a later commit would not help, because the blob stays in
history. The commit was dropped with `git reset --soft`, and a branch `backup-before-v2-sync`
keeps it. The tree was then synced to v2's tip and committed as **`ccdd19f`**, "Sync repository to
the state described by the final thesis report". That commit brought all of Part III into the
canonical repository, plus 21 more reference PDFs, the report figures and the Fabric samples.
`.gitignore` now keeps datasets and bytecode out.

**`TASK.md` — "Operation: Honest Federation".** A mission board with standing rules. Two remarks
of Shadman's that day, recorded in the session notes, set its tone: *"we are doing an
undergraduate thesis and it has to be novel and completely honest"* and *"my works are the latest
works … always prioritize my work here."* The rules:

1. Novel **and** completely honest; honesty wins when they conflict.
2. No number ships that a script did not produce.
3. Negative results stay negative.
4. No result-shopping; write predictions down first.
5. The repo is the truth; handbooks and chat exports are claims to check.
6. Scope every claim.
7. No invention claims.
8. A new dataset must be able to overturn a conclusion, and breadth across claims beats breadth
   across datasets.

(Rules 9–12 were added later, each after a specific mistake.)

**The dataset question.** A supervisor-facing "research status handbook" and a ChatGPT consultation
both pushed toward more datasets. We checked them against the repository first. Per the session
notes, the handbook listed "212 s" as an open problem; it was a units typo for the measured
2.12 s. What survived that
check was a real scientific question. **ULB has no customer column, so every 10-transaction
"sequence" our detector reads belongs to 10 unrelated strangers.** We had been testing a model
built to learn a person's habits on data containing no one's habits. Was ADTCN genuinely worse,
or had it never had a fair test?

**Plan agreed:** keep ULB, and add entity-linked datasets, **BankSim first**. BankSim is one CSV file,
594,643 transactions, 4,112 customers with about a hundred transactions each, and small enough to
iterate on in minutes. The Fraud Detection Handbook, PaySim, AMLSim and IEEE-CIS were noted for
later.

### 📅 31 August — Closing the loose ends a reviewer would find first

**Loose end 1: two files disagreed about the detector (0.677 against 0.313).** Thirteen full
training runs (`experiments/detector_multiseed.py`, `check_thread_sensitivity.py`). The answer
was not what we guessed. Training is bitwise deterministic for a fixed **seed, torch version and
CPU thread count**, and our two scripts used 4 and 8 threads. The tuned configuration scores
0.708 / 0.746 / 0.800 at 4 / 2 / 8 threads, while the default is identical at any thread count.
**0.313 was withdrawn**: it does not reproduce under any condition we could construct. The honest
headline became a distribution: **tuned 0.753 ± 0.055, default 0.706 ± 0.076, difference +0.047,
p = 0.34.** DB-BOA neither wins nor loses. torch was pinned at 2.12.0 in `requirements.txt`, and
from then on no detector number is quoted without its torch version and thread count. Files:
`results/detector_multiseed.json`, `results/_multiseed_runs/*`,
`final_report_data/OBJ2_detector_multiseed.md`.

**Loose end 2: the measured Fabric numbers were missing from the report.** They went into Chapter 6
(`sec:fabric-measured`). Measuring them exposed a design flaw the simulation had hidden: **the
reward scheme pays a 15-token bonus for commits under 300 ms, and the real network's floor is
2.1 s.** No one can ever earn that bonus.

**Loose end 3: two definitions of the objective.** `utils/metrics.py` still held the unbounded
Obf2 while the surrogate used the bounded one. There is now one definition, checked
bit-identical over 2,000 random inputs.

**The second dataset: BankSim, and a trap caught before it fired.** We wrote
`db_boa_framework/data/banksim_loader.py`. Before windowing anything we checked the file order,
and BankSim's raw CSV **places its frauds next to each other within each day**: 3,635 adjacent
fraud pairs, against 84 after a shuffle within the day. The day is the finest time unit, so the
order within a day comes from how the simulator wrote the file, not from the world. Windows built
in file order would have made the *control* arm look strongly predictive (P(fraud | previous
fraud) = 0.50) and quietly destroyed the experiment. The loader breaks same-day ties with a
**seeded permutation**. Lesson, now a mandatory check: *if a time column is coarser than the row
order, the row order carries information from the generator.*

**The window grid (OBJ-1)** (`experiments/banksim_temporal_grid.py`, `experiments/_seqtrain.py`,
and `build_sequences(groups=…)` in `adtcn.py`): 7 architectures × {global, customer-linked}
windows × 3 seeds, 5.3 h. **Neither pre-registered branch was right.** Customer-linked windows
help **every** architecture (+0.10 to +0.17 MCC), so "temporal complexity does not pay" had been a
statement about entity-less windows, not about time. But ADTCN ranks **7th of 7** on linked
windows, behind its own no-attention ablation (DTCN). Verdict: *the window was the problem, not
the sequence model, and ADTCN is still the wrong sequence model.*

**Re-implementing the base paper's comparison (OBJ-1b).** The deleted 8-model table (D4) could
only be replaced honestly by running the models ourselves. So:
`db_boa_framework/models/basepaper_models.py` (EfficientNet-, ResNet- and DenseNet-1D, DTCN, LSTM),
`algorithms/mbo.py` and `algorithms/wsa.py` (the two optimisers we did not have), and
`experiments/basepaper_comparison.py`. On ULB, ADTCN ranks **5th of 7**. On ULB, **all five
optimisers reached Obf2 = exactly 5.0000**, the theoretical maximum. Five different algorithms
agreeing to four decimals says nothing about the algorithms; it says the *scoring function* is
broken. That opened **OBJ-13**. A Windows trap cost two 35-minute runs: with stdout redirected,
`─` separators crash the cp1252 codec, so every redirected run now sets `PYTHONIOENCODING=utf-8`.

**The federated ablation on BankSim, both partitions** (`run_baselines.py --dataset banksim
--partition {stratified,customer}`, `experiments/federated_cross_dataset.py`). **The DP collapse
replicated**, which is the most important result of the day, because the whole privacy
contribution rests on it. **And its direction flipped:** on ULB the DP model flags nothing and
accuracy *rises*; on BankSim it flags 119,784 transactions and accuracy falls 97.55 pp. Both
have MCC ≈ 0. An accuracy-reported DP pipeline hides the failure completely, which is an argument
for our choice of metric.

**OBJ-13 opened** (`experiments/objective_noise_audit.py`): one fixed configuration, re-scored 25
times, spans Obf2 3.45 – 5.00. The fitness function redrew its data split and its torch seed on
every call, so the search could not tell good candidates from lucky ones.

### 📅 1 September — The wrong question, the right one, and a control

**Morning: a direction review that reached the wrong conclusion.** Rev. A of `TASK.md`'s direction
review looked at ADTCN losing on both datasets and proposed reframing the thesis around two
claims. Shadman corrected it: **the unit is FL-ADTCN**, a system of six components making six title
claims, and privacy and incentives matter as much as detector accuracy. ADTCN's rank among
classifiers is a component ablation that backs *none* of the six. Rev. B reversed the conclusion
and found the real gap: **five of the six title claims rested on ULB alone**, because the four
system sweeps hard-coded the ULB loader and none of their result files recorded which dataset
produced them. Rule 9 was added (*judge the system, not a component*). Rule 10 (*Krum is
security, Shapley is fairness*) and Rule 11 (*name the metric you quote*, first enforced in
OBJ-14 that same day) belong to the same days. A standing instruction
followed: **ask before concluding**, because verifying numbers is not enough when the yardstick
is wrong.

**OBJ-14: retracted numbers were back in live drafts.** A draft generated after the June purge
quoted the withdrawn 0.313 and "loses to the default (0.785)". The bug was in the *process*: the
only way to regenerate that draft was a multi-hour re-run, so a stale draft was cheaper to leave
broken. `objective_noise_audit.py --render-only` rebuilds it from JSON in seconds. Eight
superseded drafts were stamped `SUPERSEDED` (their numbers left untouched, as a historical
record). A repo-wide sweep found three more breaches, including the supervisor brief itself.

**OBJ-15: take the system claims off ULB.** We wrote `experiments/_dataset.py`, one place that
resolves `--dataset` / `--partition`, forwards the real feature count (BankSim has 79 features,
and the model would otherwise silently keep 33), stamps provenance into every JSON and suffixes
every filename. All four sweeps gained the flags. The smoke test found two latent bugs, fixed
before the overnight run: file writes crashed on `→` under cp1252 even with `PYTHONIOENCODING`
set (explicit `encoding="utf-8"` was added to all 15 scripts), and fixed draft names would have
overwritten the ULB drafts. Four expectations were **pre-registered at 13:05, before any output**.
Run with `experiments/run_obj15_banksim.ps1`, 3 h 15 m.

**Reading sweep 1's output instead of trusting its exit code** found two more holes. Figures were
never suffixed, so the BankSim run **overwrote the ULB figures**: they were recovered
byte-identical from git, and the rest were backed up to `results/_ulb_figures_backup/`. And the
generated drafts printed *static* ULB conclusions ("on this 0.17 %-fraud data", "not isolated")
directly under tables showing BankSim results that contradicted them. All four generators now
compute their prose from the data. Lesson: *"dataset-aware" had been scoped to filenames and
stopped at the prose.*

**Scored at 16:20.** Krum 8/8 held, *but it had predicted the right answer to the wrong
question*. Krum's **utility** flipped negative on BankSim (−12.57 to −1.48 pp), and the prediction
had said nothing about utility. "Lone attacker not caught" became "caught and not helped".
"MC top-1 fails from n ≈ 6" was withdrawn: it is intermittent, not a threshold. The lesson
written down: **pre-register the property, not just the number.**

**The control, launched at 20:20.** Every ULB-vs-BankSim comparison had changed *dataset and
partition together*: ULB ran stratified, BankSim entity-disjoint. Any difference could belong to
either. The four sweeps were re-run on **BankSim with the stratified partition**, so that ULB vs
BankSim/stratified isolates the dataset and BankSim/stratified vs BankSim/customer isolates the
partition. `experiments/sweeps_cross_condition.py` was written to print both effects as separate
columns. Six expectations were pre-registered. The most important read: *"if Krum is still
negative under stratified, that mechanism is wrong and the finding is about BankSim."* Logs split
per run (`results/_obj15_logs/banksim_<partition>/`); `db_boa_framework/scratchpad/resume_new.md`
was the handoff note. Landed 23:41.

### 📅 4 September — The day three things we believed stopped being true

**Morning: the control is scored, and our best new claim dies.** Krum is negative in **7 of 8** cells
under the *stratified* partition too. The dataset effect is negative in all 8 rows; at n = 7
the partition effect is *positive* in all four, which is the opposite of the mechanism we had
argued. **"Krum pays 1.5–12.6 pp of utility when banks hold different customers" was withdrawn.**
No mechanism was put in its place, and saying so is the honest position. Two other
"federation-structure" readings fell the same way: lone-attacker isolation and MC rank fidelity
both track the *dataset*. Final score 4 held, 2 wrong, both wrong identically. The pre-registration
had named the falsifier in advance, so the dead claim arrived as a finding rather than as a
reviewer's objection. **Rule 12 was added:** *a control that can overturn a finding you already
have outranks a dataset that can add a new one.* Draft: `final_report_data/OBJ15_two_factor_decomposition.md`.

**The locked plan** (with Shadman): consolidate first, then expand; the Handbook's go/no-go on
12 September; the **16–19 September writing block protected**, so that *if the third dataset
overruns, the dataset gets cut, not the writing*.

**OBJ-18: propagating a withdrawal is its own job.** The dead mechanism was still being printed by
two *generators* (`byzantine_robustness_sweep.py`, `federated_cross_dataset.py`). One of them
emitted it into the stratified draft, where it is false by construction. Lesson: *a generator may
only assert what its own input can falsify.* The supervisor brief was corrected ("≥60×" with a
censoring box; Krum's rejection separated from Krum's utility). The PDF build had been silently
broken since 30 August (MiKTeX lacked scalable fonts, so `microtype` aborted), which meant **the
brief the supervisor reads had been two revisions stale**. Fixed by installing `cm-super`. The
collator now marks **contaminated cells**: an attacked FedAvg scoring *above* its own no-attack
reference, which an attack cannot genuinely cause. The quoted "−12.57" was one of them; the
clean range is −4.73 to −1.48 pp, and the withdrawal survives on the clean cells too.

**OBJ-17: we measured the privacy factor, and then retired it.** Every ε\* so far had been a
*lower bound*: the weight channel was still inverting at ε = 3000, the top of the grid. We
snapshotted the old results (`results/_obj17_pre_extension/` + `README.md`), extended the grid to
**11 points up to ε = 30000**, added a fragility table showing how many draws decided each
threshold (exact Clopper–Pearson intervals), and ran it (`experiments/run_obj17_epsgrid.ps1`,
~78 min). All three conditions de-censored: **60× (ULB), 30× (BankSim/stratified), 33×
(BankSim/customer)**. Two problems remained.

- ⛔ **ULB did not reproduce its own 30 August result.** At identical seeds 13 of 18 cells moved,
  and even the noise-free ground truth moved ([0.619, 0.250, 0.131] → [0.554, 0.336, 0.110]).
  Both BankSim conditions reproduced bitwise, and ULB reproduced *itself* the same day. Every code
  change on the path was audited, along with all six configuration dicts and the thread count. The
  library version at the time is unrecoverable. **Recorded as unexplained. We did not guess a
  cause.** Every sweep JSON now stamps its full software environment (`environment()`).
  Tool: `experiments/check_obj17_nesting.py`.
- **The `--repeats 1000` pass** (pre-registered before launch). The single-draw thresholds were
  real (7, 11, 16 and 10 per 1000). But BankSim/customer re-censored to **≥ 100×** on a single
  draw in a thousand. The deeper finding: ε\* = the largest budget at which *any* inversion is seen,
  so it **can only grow** with more draws or a wider grid. Re-derived at slightly different cut-offs
  it swings **20× to ≥ 100×**, and the ranking of the three conditions changes. **We retired the
  factor as a headline.** The claim now rests on two statements with no arbitrary threshold: the
  paired **ordering** (the output channel inverts less often at essentially every budget), and the
  **rank-correlation gap at a stated budget**, which at ε = 50 is ULB −0.199 → +0.995,
  BankSim/stratified +0.017 → +0.945, BankSim/customer −0.070 → +0.805.

**Afternoon: OBJ-13, the surrogate repair applied.** The staged patch was sitting in a session
temp directory; it was rescued into the tracked `db_boa_framework/scratchpad/`
(`obj13_new_objective.py`, `apply_obj13.py`, `verify_obj13_legacy.py`). Backups
`models/adtcn.py.pre_obj13` and `config.py.pre_obj13` were taken; they are the *only* copy of the
pre-patch state, because `adtcn.py` had uncommitted edits. The repair adds `eval_mode ∈ {legacy,
deterministic, averaged}`: `deterministic` freezes the random draw, and `averaged` scores every
candidate on the same 3 draws (**common random numbers**). It also revives a dead search axis: the
old batch formula `max(32, n // spe)` pinned every candidate to 43 steps per epoch. `legacy` was
verified **bit-for-bit** against the backup, to 10 decimals on 3 configurations. Applying it caught
**three defects that would have shipped**:

1. The patch silently re-pointed `objective_noise_audit.py`, the script that *produced the
   evidence of the problem*, at the new default, where it would have reported zero noise and read
   as "the objective was always fine". It now pins `legacy` on purpose.
2. The same silent switch in `basepaper_comparison.py`. It now stamps the mode into its JSON and
   refuses to overwrite a file recorded under another mode.
3. Two drafts overwrote each other because they were named by dataset instead of by results file.
   This was the third instance of that class of bug.

**Evening: the shipped path is checked, and it is worse.** `experiments/obj13_shipped_path_check.py`
(no training) asked what the *deployed* search actually sees. The audit had used a 6,000-row
pool; `main.py` uses a 3,000-row pool. At ULB's 0.17 % that holds **5 unique frauds**, and the
surrogate needs 30, so June's sample-with-replacement line repeated them ~6 times. **9 of 9
validation positives were copies of training rows.** The perfect 5.0000 had been memorisation,
not luck. The repair did **not** fix it (`deterministic` leaked 9/9 identically). **The fix:**
`eval_subset` 3,000 → 36,000, which costs no training time, plus a `RuntimeWarning` whenever
sampling with replacement fires. Re-verified: **0 of 11** leaked.

**The knee sweep** (`objective_noise_audit.py --knee` → `results/objective_size_knee.json`) landed
at 20:38, while the board said nothing was running. Pre-registration 6 was **wrong on both
halves**. ULB's noise improved sharply while its fraud count stayed pinned at 30, and BankSim's
did not improve with 8× more fraud rows. The "n ≈ 9 positives" explanation was withdrawn. No knee
is locatable at one seed, so no chosen surrogate size is quoted.

**20:54: the side-by-side is launched** (`experiments/obj13_surrogate_repair.py`,
`experiments/run_obj13_sidebyside.ps1`, detached, with a preflight that asserts both the patch and
the pool fix). The written recommendation was a cheap ~1.1 h version. **Shadman chose the full
production budget (~16 h)** because only that produces configurations comparable to what was
shipped.

### 📅 5 September — The search the framework is named after, measured

**13:13, 16.3 h, 9 of 9 runs clean.** Three surrogate modes × three seeds on BankSim, each chosen
configuration re-scored on held-out draws from a **row-disjoint** pool and compared **paired**,
because the yardstick shares its draws.

- **Not one of the nine beats a hand-set 128 filters / 150 steps per epoch.** Mean paired
  Δ −0.1982. The shipped 142/76 loses too (−0.1900). The negative result now rests on a direct
  measurement, not on a p = 0.34 tie.
- **Determinism does not make a noisy search reproducible; shared draws do.** The spread of chosen
  filter counts across seeds: legacy 66, deterministic **81 (wider)**, averaged **12**. This was
  pre-registered the other way round, and it is the most transferable finding of the project.
- **The converged mode generalises worst** (averaged 3.4237 against 3.5857 and 3.5552 held-out). It
  converges on high-filter configurations that overfit its three shared draws. **The noise the
  repair removed had been acting as accidental regularisation.**
- The revived axis is **alive but coarse**: 23 distinct batch sizes across 201 settings. Quote the
  batch size, never the step count.
- ⛔ `ADTCN.fit` (line 681) still uses the old floored formula. At ~400 k rows the floor never
  binds there, so **only the surrogate's axis was ever dead**. The search had been blind to a
  dimension the deployed model responds to. Whether to change it is still undecided, because
  changing it invalidates every trained model.

**Everything committed.** About 20 hours of results had existed only as untracked files, one
`git clean` from gone. Commit `889d623` landed OBJ-1b, 15, 17 and 18; `ac606de` landed OBJ-13 and
`TASK.md`. A LaTeX `*.log` rule in `.gitignore` had been silently swallowing **experiment run
logs**, so an exception was added. That evening (`33bb97f`, "new datasets") the first
**`WORK_REPORT.tex` / `.pdf`** was committed: a work report for the supervisor covering 30 August
– 5 September.

### 📅 8 September — The third dataset's loader

**Why the Fraud Detection Handbook.** On 4 September the third dataset had been re-gated. The
control had already answered the question AMLSim was promoted for, so AMLSim was demoted: *"right
dataset, wrong fortnight."* The live question was now
*is Krum's utility cost a BankSim artefact?*, and **any** third dataset could answer that. The
Handbook was the cheapest one that could run all four sweeps, and the only one with
**sub-daily time** (timestamps to the second) plus **two** entity links (customer and terminal),
so it could also finish the "does any architecture exploit time?" question BankSim had only
half-answered.

**What we built.** `db_boa_framework/data/handbook_loader.py` (reads 183 daily pickles cloned into
`datasets/handbook_raw/` and consolidates them into `datasets/handbook_transactions.csv`), plus
`HANDBOOK_CONFIG`, `DATASETS["handbook"]` and `ENTITY_PARTITIONS` in `config.py`,
`experiments/check_handbook_loader.py` (**42 acceptance checks, all passing, no training**) and
`scratchpad/verify_handbook_no_regression.py` (ULB and BankSim unchanged).

**What we measured.** 1,754,155 transactions, 4,990 customers, 10,000 terminals, 183 days, 0.837 %
fraud. The BankSim file-order trap does **not** fire here (lift 1.02×). It was checked, not
assumed. A surprise: **terminal linkage (71.65×) is five times stronger than customer linkage
(13.35×)**, because the generator's largest fraud scenario compromises a terminal for 28 days. We
deliberately built a *thinner* feature matrix than the Handbook's own baseline, whose features are
entity aggregates that would contaminate the global-vs-linked comparison, and we wrote down that
our MCC will be lower than theirs by design. The real cost was about 11 h of CPU, not the budgeted
6. One housekeeping check: `datasets/bsNET140513_032310.csv` looked like a spare dataset but is a
byte-identical column projection of BankSim. Counting a file twice is not breadth.

### 📅 10 September — Figures from stored results, and the report rebuilt (`6d8e41e`)

`experiments/make_report_figures.py` draws every WORK_REPORT figure from result JSONs only (no
training, no sampling), printing each panel's provenance. It writes to `results/figures/`: the
architecture scoreboard, cross-dataset comparison, memorisation before/after, process
optimisation, scalability, score index, statistical significance, system performance,
`score_index.csv` and `score_index_table.tex`. The WORK_REPORT was rebuilt as a story with a
decision log.

### 📅 11 September — Rule 8 suspended; three new datasets taken to measurement

**The decision.** Shadman: *"we will break this rule."* Rule 8, which gated datasets on what they
could overturn, was suspended, and **all five assessed datasets would be run and compared**. It
was marked SUSPENDED in `TASK.md` with its original text kept. Rules 1–4 still bind: a dataset
without the structure to test a claim is reported as *not testing it*, never skipped silently.
Placement: **ULB and the Handbook on the laptop**, sharing BankSim's environment so their columns
stay comparable; **PaySim and AMLSim on Kaggle**, because they could not fit before 16 September.
For AMLSim, **the real IBM simulator built with Java**, not the pre-generated substitute. And:
**every pre-registration is drafted by the assistant and approved by the team lead before any
run.**

**PaySim: reconnaissance before any loader decision** (`experiments/paysim_recon.py`). 6.36 M
transactions, 0.129 % fraud, fraud only in TRANSFER and CASH_OUT, and 99.7 % of senders appear
exactly once, so there are no sender histories. ⛔ **The time order itself carries the label:** the
simulator writes each fraud's two legs together, so P(fraud | previous fraud) is 0.72 in raw
order and still **151.6×** the base rate after a within-hour shuffle. On PaySim the "global
window" arm is *not* a no-signal control. We caught this before any model saw it. Design choices
(`data/paysim_loader.py`, `experiments/check_paysim_loader.py`): TRANSFER + CASH_OUT only (the
other types hold zero fraud across 3.59 M rows, and keeping them inflates accuracy); row fields
plus raw balances, nothing engineered; stratified partition only (no banks). A pre-registered
**all-rows sensitivity arm** tests whether that scope choice changes any conclusion.

**AMLSim: generated, discarded, regenerated.** `data/make_amlsim_config.py` takes the shipped
10K configuration and splits accounts across three banks at 50/30/20, plus three forced key
repairs. `data/generate_amlsim.py` pins IBM/AMLSim `7338a4bc`, builds MASON 20 from source
(SHA-256 recorded), and uses Maven 3.9.9, JBR 21 and Python 3.8 with networkx 1.11. It also
repairs a Windows line-ending bug upstream (`\r\r\n`, which the Java half reads as empty
records) and asserts that nothing else changed. ⛔ **The first derivation was wrong.** Contiguous
bank blocks tied the bank to account activity: volumes of **94.6 / 5.1 / 0.2 %**, with 43 % of one
bank's transactions being laundering. **The bank had become a proxy for the label.** The
reconnaissance (`experiments/amlsim_recon.py`) caught it before any model trained. We discarded
the data and regenerated with the banks interleaved: **50.1 / 30.0 / 19.9 %**, and the bank columns
no longer predict anything. Final dataset: 198,015 transactions, 685 laundering (0.346 %), three
**native** banks, the label is *laundering* not fraud, rows carry nothing (logistic MCC 0.0001),
and links carry a lot (sender 26×, receiver 41.5×). Checked by
`experiments/check_amlsim_loader.py`; loader `data/amlsim_loader.py`.

**Kaggle, and the traps it set** (`experiments/kaggle_jobs.py`: bundle the code with a SHA-256
and pinned library versions, push one private job per experiment, keep five running, and import
results only if the job exited 0). A smoke test found two failures by running, not reading:
Kaggle **drops empty folders**, and **torch ignored `OMP_NUM_THREADS`** (it ran 2 threads, not 4).
Overnight two more appeared:

- The queue substring-matched "error" in the CLI output and wrote off a job Kaggle still reported
  RUNNING. It now parses the status token.
- Kaggle **refused a sixth session while the CLI exited 0**, so the queue waited all night on a
  job that never existed. `push()` now requires an explicit success line.

The API token lives only in the gitignored `kaggle.md`.

**The windowing correction** (`experiments/check_partition_windowing.py`, no training). On
1 September we had verified that the partition was "the only factor moving" in the control. **That
was wrong.** When a sweep passes no entity groups, each bank's model windows its rows *in the order
the partition leaves them*. The stratified split shuffles them; an entity split leaves them
grouped by customer. So **91.22 %** of BankSim/customer's training windows are single-customer,
against 0 % under stratified (Handbook 96.37 %, AMLSim 88.76 %). The Krum withdrawal survives,
because it rested on the *dataset* effect, where both sides use random windows. Every "partition
effect" is now quoted as *"entity-disjoint split (which also changes the training windows)"*. We
rescoped it in a dated box rather than quietly correcting the table.

**The detector track is safe.** `experiments/check_detector_repro.py` re-ran both 31 August
detector records: **bit for bit identical**. The unexplained ULB reproduction failure is confined to
the private-incentive sweep.

**First scorecards, same evening.** AMLSim (f): under a shuffled split the model catches **0 of
106** laundering cases (MCC −0.011, at the floor, as predicted). (g): under native banks, MCC
**+0.158**. Linkage lifts it off the floor. AMLSim (c) scored 11/11 on the established reading and
8/11 on the strict wording of the pre-registration, and the strict wording was the assistant's
drafting error. The proposal was to label it "held" under the established reading. Shadman
declined (quoted in the session notes as *"don't show as holding, show both scores"*; recorded in
`TASK.md` as the operator's ruling). That became a standing rule: **where two readings disagree,
give both counts and no verdict.** PaySim (a) was **wrong**: Krum *protects* accuracy on PaySim's clean
cells (+3.8 to +19.2 pp), which is ULB's pattern, not BankSim's.

**A deviation, written down before its result existed.** AMLSim's training pool holds 17 of the
20 × 8,000-row shards the scalability sweep needs, and the job crashed at 23:35.
`scalability_sweep.py --fit-pool` stops the Monte-Carlo range at 16 and leaves the exact range
(3–12, the one the pre-registration names) untouched. The first 16 shards are verified identical
to the full protocol.

**The ULB regeneration runs overnight** (`experiments/run_obj13_ulb_regen.ps1`,
`run_main_ulb_rerun.ps1`). The two files the surrogate leak had invalidated are regenerated on
the fixed pool. Phase 8 of `main.py` had crashed on an old bug (the attacker's stand-in `predict`
lacked a `groups` argument); fixed and re-run.

### 📅 12 September — The day the Handbook broke two defences

**Overnight, Kaggle finished** (29 jobs, 28.8 h of sessions). PaySim: (a) wrong; (b), (c) and (e)
held. (c) held at **11 of 11 strictly**, the strongest privacy result in the project. (d): receiver
linking helps **6 of 7** architectures despite the time artefact, pre-registered the other way with
that exact alternative named. The row-scope arm changed no conclusion, but it swung the collapsed
DP model's *accuracy* from 98.70 % to 28.22 % with MCC near zero both times, the clearest
demonstration in the project that accuracy can hide a total collapse. AMLSim (d): linked windows
lift every model off the floor, and ADTCN takes its **only first place anywhere**, on *global*
windows where every model is at the floor. We report that and do not lean on it.

**01:48: ULB's ablation regenerated, and it moved.** FedAvg 0.569 → **0.425**; Krum's advantage
+0.207 → **+0.326**. The DP collapse **changed direction**: the August model flagged nothing, the new
one flags 43,089 transactions. Direction is not a property of a dataset. WORK_REPORT had printed
the August accuracy *rise* of 0.11 pp as a *fall*; the sign error was corrected in a dated box, not
quietly. The detector arms from the repaired search tie the default on the test set too (OBJ-13
pre-registration 5 held; p = 0.199 and 0.239). `db_boa_results.json`'s best surrogate score is now
3.93, not the memorised 5.0000. **04:03:** the `main.py` rerun reproduces phases 1–7 bitwise.

**04:03: the Handbook suite starts** (`experiments/run_obj16_handbook.ps1`, gated on the
pre-registration reading APPROVED). At 10:55 its ~16 h window grid moved to Kaggle, split 7 ways
by architecture (`handbook_temporal_grid.py --archs / --skip-reference / --merge`). Every
comparison the grid is scored on is within the Handbook, so a second environment cannot touch the
verdict. It was merged at 15:09, an hour before the runner reached it; the runner's grid step
returned in 22 seconds, and the laptop reproduced Kaggle's windowing statistics exactly.

**The scorecards. Six of the Handbook's eight predictions were wrong.**

- ⛔ **(a) Krum selects a label-flipping attacker**, 6/8 on stratified and 7/8 on customer. The
  poisoned model has the **lowest** Krum score, more central than every honest one, and Krum's
  accuracy falls 18.6, 11.8 and 13.4 pp. "Attacker caught 8/8" had been called *"held by
  construction"*. That wording is falsified.
- ⛔ **(b) The incentive layer never isolates anyone** (12 of 12 scenarios, every gap exactly 0). So
  "isolation does not help a lone attacker" holds only **vacuously**, and the collusion catch that
  worked five times fails. **Worse: on the customer split the reputation ordering inverts.** An
  honest bank ends at 0.5068 against the 0.5 expulsion floor while the attacker's reputation
  *rises*. One more round would have expelled the honest bank and kept the attacker.
- ⛔ **(c) The core privacy contribution fails on the customer split**: 9 of 11 under *both*
  readings, below the pre-registered 10. The ranking gap at ε = 50 still points the right way, so
  what fails is the inversion-rate criterion, not the direction of the effect.
- ⛔ **(d) Fraud adjacency does not predict learnable signal; it inverts.** The 72× terminal ordering
  trains the **worst** models for all seven architectures, and even unlinked windows beat it. This
  contradicts BankSim, where linked windows helped every architecture.
- **We asserted no cause** for any of the three failures. What is measured is that each defence
  needs a signal that separates organisations, and on the Handbook none separates.

**16:01, `baselines_customer` lands, and it corrects one of our own sentences.** That morning we
had written that grouping by entity *"does not lift the Handbook's models at all"*. With only the
partition changed, the ablation goes from MCC **0.0439 to 0.1500**. Entity **partitioning** lifts
the Handbook; entity **windowing** does not. The two factors point in opposite directions, which
is exactly the separation our own control had forced us to make. The same file showed our
detector **degenerating to constant-positive** on the customer split (recall 100 %, 351,280 false
positives, accuracy 0.95 %), and DP collapsing while accuracy *rose* 0.64 pp.

**Decision 21.** The Handbook's stratified ablation landed at 0.0439, just under the 0.05 floor we
had borrowed from *AMLSim's* pre-registration, after three of its items were already scored. We
**did not relabel them untestable**. Applying another dataset's threshold after seeing a
prediction fail would launder a failure into a non-result. Both defence failures concern
*selection*, not accuracy, so they stand either way.

**Committed at 18:02** (`f92b6de`, 395 files). **23:53: the suite completes**, 11 of 11 steps, 19 h
50 m. The last step forced a correction (`2efa8e7`). Shapley-fidelity metrics mean something only
where the 8,000-row shard models learn, and on PaySim, AMLSim and Handbook/stratified they sit at
50 %. There, "3 of 10 top-1 matches" is what guessing gives (expected 1.60). So it is **seven
conditions, four able to test fidelity**, and we stopped averaging the other three into the
quoted series.

### 📅 13 September — The last audits, and the story written by dataset

**00:17: a coverage audit finds one more hollow pass** (`13157f8`). Each Byzantine sweep stores a
*separation margin*: the lowest attacker score minus the highest honest score. Ordered by margin:
BankSim +67 to +149, PaySim +48 to +134, ULB +7 to +8, **AMLSim −20 to −60**, Handbook −36 to −224.
On AMLSim the attacker sits *inside* the honest range. It merely was not the single lowest score,
so its "8/8" is a pass **by ordering, not by detection**, on models at 50 % accuracy. The three
non-separating conditions are the three weakest-modelled. We record the association and assert
no cause.

**00:23: AMLSim cannot test Krum at all** (`7ef1ca8`). The Byzantine sweep runs at 5 and 7 banks so
that Krum's n ≥ 2f + 3 guarantee holds; AMLSim has **three** native banks. The partition whose
models work is too small to host the test, and the partition large enough is inert. We recorded
why and **did not fill the cell** with an invalid result. Three alternatives are written down,
none run.

**00:44: "The five datasets, one at a time"** (`4a8052e`). WORK_REPORT is organised by claim, which
makes it hard to answer *what did this one dataset buy us?* A new section answers five questions
per dataset (why it is here, what it can and cannot test, predicted vs found, what broke, what it
contributed), with a coverage matrix and a cross-dataset synthesis. Its conclusion: **most of what
we measured is a property of a dataset, not of fraud detection**. It closes on the lesson three
separate findings converged on: ***a defence that cannot be observed failing is not a defence that
passed.*** The long-standing overfull page in the report was fixed by converting the scoreboard
to `xltabular`, leaving 65 pages with zero overfull vboxes and zero undefined references.

**00:48: the last commit** (`c66d1d1`). The WORK_REPORT cover gains a clickable repository link, the
branch name, and a note that the work is **not yet merged to `main`**; the cover dates are
corrected to 30 August – 13 September.

---

# PART VI — WHERE THINGS STAND AT THE LAST COMMIT (13 SEPTEMBER 2026)

**The repository record ends here.** No commit and no file in the working tree is newer than
`c66d1d1` (checked 26 September). The 19 September deadline and the protected writing block
(16–19 September) fall after the last commit. Whatever happened then is not in the repository,
and this journal does not guess.

### The six title claims

| Claim | Where it stands |
|---|---|
| **Secure (privacy)** | Adding DP to the weights drops MCC below 0.05 in **all 7 testable conditions**. *How* the model fails is not stable, and twice accuracy **rose** while it failed (PaySim +1.16 pp, Handbook/customer +0.64 pp). |
| **Secure (attacks)** | Krum rejects the attacker 8/8 in ULB, both BankSim splits and PaySim. It **selects a label-flipper on both Handbook splits**, **fails to separate** one on AMLSim, and AMLSim **cannot host the test**. Krum's accuracy cost tracks the dataset: it helps on ULB (+0.326 MCC) and PaySim, and costs on BankSim and Handbook/customer. **No mechanism is on record.** |
| **Incentivized (catching cheats)** | Collusion caught in 5 conditions and **not at all on the Handbook**, where reputation inverts against an honest bank. Lone attacker: isolation fires 0–3 of 3 depending on the condition and **helps in 0 of 3 everywhere** (vacuously on the Handbook). |
| **Incentivized (privacy vs rewards), the core contribution** | Holds on ULB, both BankSim splits and PaySim (11/11 strictly). Two scores, no verdict on AMLSim/native and Handbook/stratified. **Fails on Handbook/customer** (9/11 under both readings). The improvement *factor* is retired; the ordering and the ρ gap at a stated ε carry the claim. |
| **Scalable (attribution)** | Exact-Shapley cost is structural (same 4,095 coalitions everywhere). "Fails from ~6 banks" is withdrawn. 4 of 7 conditions can test fidelity at all. |
| **RL / Consensus / Blockchain** | Dataset-independent. Leadership Gini 0.90 → 0.782; degraded-node election 100 % → 15.2 % (with an injected fault); Fabric 2115.9 ms mean, 40.3 TPS peak. |

**Tally.** 49 pre-registered predictions scored, **20 wrong**. Three headline claims withdrawn by our
own tests (Krum's cost mechanism, "MC fails from n ≈ 6", the privacy factor), plus one rescoping
(windowing) and three counts reclassified as untestable.

### What was never finished (all of it recorded as open in `TASK.md`)

- **OBJ-7:** formalise the DP guarantee (adjacency, clipping, σ, δ, composition, an accountant).
  Flagged must-fix before publication.
- **OBJ-8:** freeze the on-chain / off-chain execution map (chaincode must be deterministic; stochastic
  search cannot live inside it).
- **OBJ-9:** scope the two overstated title words ("Scalable" → contribution attribution;
  "Consensus" → stock Raft plus a simulated round).
- **OBJ-6:** Shapley under a BankSim entity-disjoint split. (Partly answered: AMLSim's native banks
  give [0.550, 0.316, 0.135], the first non-uniform split we have.)
- The **`ADTCN.fit:681`** batch-formula decision.
- The unexplained **30 August ULB reproduction failure** in the private-incentive sweep.
- Merging `obj13-surrogate-repair` into `main`, and rotating the Kaggle token.
- A small reporting inconsistency: `detector_multiseed.json` stores tuned MCC **0.7531 ± 0.0553**
  (sample SD), while WORK_REPORT's per-dataset section prints **± 0.0494 / ± 0.0676** (population SD).
  Pick one convention before the thesis quotes it.

---

# PART VII — WHAT WE LEARNED

These are the practices the project paid for. Each is tied to the moment it was learned.

1. **A name is a claim.** "Temporal network", "attention", "graph features", "Byzantine-tolerant",
   "measured latency": in May every one of them promised more than its code delivered. (Part II)
2. **If no script produced it, it does not exist.** Nine invented numbers were deleted in June. The
   thesis survived because the real contribution was found in what remained. (June audit)
3. **Turn the weakness into the question.** DP broke the rewards. Asking *where* the noise goes,
   instead of turning DP off, produced the contribution. (6–7 June)
4. **Pin the environment, or a number is not a result.** Seed, torch version and thread count; an
   MCC of 0.313 was a thread count. (31 August)
5. **Check the file order before building sequences.** BankSim and PaySim both hid the label in row
   order; AMLSim's first derivation hid it in the bank column. (31 August, 11 September)
6. **Judge the system, not a component, and ask before concluding.** A review with correct numbers
   reached a wrong conclusion because it used the wrong yardstick. (1 September)
7. **Pre-register the property, not just the number, and name the falsifier.** A correctly predicted
   security result hid an unpredicted utility reversal. (1 September)
8. **Run the control that can kill your best claim before building on it.** Three hours cost us
   the claim in private instead of in public. (4 September)
9. **A statistic that can only grow with effort describes the effort.** ε\* = "largest budget where
   any inversion is seen" is monotone in draws and grid extent. Report thresholds that do not move
   with how hard you look. (4 September)
10. **Diagnostics must run on the path that ships.** The audit's pool was never the deployed pool;
    the leak lived in the gap. (4 September)
11. **Common random numbers, not determinism, make a noisy search reproducible**, and a search that
    can finally rank reliably will overfit what it ranks on. (5 September)
12. **Generated prose must read its conclusions from its data**, and a generator may only assert
    what its own input can falsify. (1 and 4 September)
13. **Where two readings disagree, show both scores and no verdict.** (11 September)
14. **Do not apply a threshold after seeing a result fail.** (Decision 21, 12 September)
15. **A defence that cannot be observed failing is not a defence that passed.** Vacuous isolation,
    fidelity measured at chance, and Krum's 8/8 on an inert model were all reclassified as
    untestable. (12–13 September)

---

# APPENDIX A — EVERY FILE IN THE PROJECT, AND WHY IT EXISTS

*Dates are when the file first appeared. "v2" means it was first committed to the `-v2`
repository in June and arrived here in `ccdd19f` on 30 August. Vendored third-party trees
(`fabric/fabric-samples/`, `db_boa_fabric/api-server/node_modules/`) are described as a whole.
Untracked files that were made are listed at the end.*

## A.1 Repository root

| File | Since | What it is and why it exists |
|---|---|---|
| `.gitignore` | 20 May | Keeps datasets (`creditcard.csv` is 143.8 MB, over GitHub's limit), bytecode, LaTeX build files, the Kaggle token and generated simulator output out of git. The `results/**/*.log` exception was added 5 Sep because the LaTeX rule was swallowing experiment logs. |
| `build_log.md` | 20 May | Changelog of every change set (Phase 0 → X-41). |
| `research_journey.md` | 26 Sep | This journal. |
| `NOVELTY_TIPS.md` | 20 May | Began as the six novelty tips; became the "Open Issues — Must Fix Before Submission" list re-audited after every fix batch. Last edited 1 June. |
| `defense_questions.md` | 20 May | 150 viva questions (Q1–Q150) phrased against the current code; fixes cite them. |
| `novel_plan.md` | v2 (Jun) | The novelty decision: characterisation, not new algorithms. Tasks A and B. No priority claims. |
| `title_issue.md` | v2 (6 Jun) | Claim-by-claim audit of the six title words. |
| `finalreport_checklist.md` | v2 (8 Jun) | Draft → `.tex` merge tracker, June audit record. Stamped SUPERSEDED 1 Sep (its DB-BOA entry rests on the withdrawn 0.313). |
| `DEMO_GUIDE.md` | v2 (Jun) | How to demo the Python pipeline and the Fabric layer on the June WSL2 machine. |
| `SUPERVISOR_BRIEF.md` / `.tex` / `.pdf` | v2 (11 Jun) | The one-read walkthrough for the supervisor. Corrected 1 Sep (0.785 claim) and 4 Sep (≥60×, Krum split); the PDF was rebuilt after the `cm-super` fix. |
| `TASK.md` | 30–31 Aug (committed 5 Sep) | "Operation: Honest Federation", the live board: rules, objectives OBJ-1 … OBJ-18, pre-registrations and scorecards, confirmed kills, verified numbers, field map. |
| `WORK_REPORT.tex` / `.pdf` | 5 Sep | Supervisor-facing account of 30 Aug – 13 Sep, told as a story with a decision log, the six-claims scoreboard, "what may be quoted", and the per-dataset journey (65 pp). |
| `kaggle.md` | 11 Sep (untracked) | Holds the Kaggle API token. Never printed, never committed. |
| `creditcard.csv` | (untracked) | The ULB dataset. Deliberately not in git. |

## A.2 `db_boa_framework/` — the Python system

**Entry points**

| File | Since | Purpose |
|---|---|---|
| `main.py` | 20 May | The full pipeline: leader selection, DB-BOA search, training, evaluation plus a default comparison, consensus simulation, federation rounds (DP → Krum → Shapley), `--attack` simulation, plots. `--dataset` / `--partition` since 31 Aug; Phase 8 `groups` fix on 11 Sep. |
| `run_baselines.py` | 20 May (Fix 17) | The federated ablation (FedAvg / +Krum / +DP / proposed) → `results/baselines*.json`. McMahan-weighted since 21 May; `--dataset` / `--partition` / `--filters` since 31 Aug. |
| `config.py` | 20 May | Every parameter. Later additions: `DATASETS`, `get_loader`, `BANKSIM_CONFIG`, `HANDBOOK_CONFIG`, `ENTITY_PARTITIONS`, PaySim/AMLSim configs, `RL_LEADER_CONFIG`, `make_org_splits`, `eval_subset` = 36,000. |
| `config.py.pre_obj13` | 4 Sep | ⛔ The only copy of `config.py` before the OBJ-13 patch. Never delete. |
| `requirements.txt` | 20 May | Dependencies; torch pinned 2.12.0 since 31 Aug. |

**`algorithms/`**

| File | Since | Purpose |
|---|---|---|
| `boa.py` | 20 May | Billiards Optimisation Algorithm, BOA (Givi & Hubálovská 2023). |
| `dboa.py` | 20 May | Dynamic Butterfly Optimisation Algorithm, DBOA (Tubishat 2020); NaN fragrance fix 1 Jun. |
| `db_boa.py` | 20 May | The hybrid DB-BOA (Dynamic Butterfly–Billiards) from the base paper; range-normalised switching rule. |
| `mbo.py` | 31 Aug | Mine Blast Optimiser (Sadollah 2013), for the base-paper optimiser comparison. |
| `wsa.py` | 31 Aug | Water Strider Algorithm (Kaveh 2020), same purpose. |

**`blockchain/`**

| File | Since | Purpose |
|---|---|---|
| `leader_block.py` | 20 May | Consortium simulation, leader selection, incentive updates; RL hooks and `compare_leader_methods` (Jun); `load_measured_consensus` (Jun). |
| `rl_leader.py` | v2 (Jun) | `RLLeaderSelector`: linear-FA Q-learning leader election, added to honour the title's "RL". |

**`data/`**

| File | Since | Purpose |
|---|---|---|
| `data_loader.py` | 20 May | ULB loader (synthetic generator until Tip 1); stratified 70/10/20 split; `split_for_orgs`; `get_eval_subset`; `raw_feature_count`. |
| `graph_features.py` | 20 May (Tip 6) | Three amount-recurrence features, honestly renamed from "graph features" (Fix 3). |
| `banksim_loader.py` | 31 Aug | BankSim with customer IDs, seeded within-step tie-break, customer/stratified partitions. |
| `handbook_loader.py` | 8 Sep | Fraud Detection Handbook: 35 row features, global/customer/terminal orderings and partitions. |
| `paysim_loader.py` | 11 Sep | PaySim, TRANSFER + CASH_OUT scope (plus the `paysim_all` sensitivity dataset), stratified only. |
| `make_amlsim_config.py` | 11 Sep | Derives our AMLSim configuration (3 interleaved banks at 50/30/20) from upstream's 10K files. |
| `generate_amlsim.py` | 11 Sep | End-to-end AMLSim generation with pinned toolchain, line-ending repair and SHA-256 per output. |
| `amlsim_loader.py` | 11 Sep | AMLSim, `is_sar` label, native `bank` and `stratified` partitions, sender/receiver orderings. |

**`models/`**

| File | Since | Purpose |
|---|---|---|
| `adtcn.py` | 20 May | The detector: `_Conv1dClassifier` (deployed CNN), `_DilatedAttnClassifier` (report-faithful ADTCN), `build_sequences(groups=)`, and `_ADTCNObjective`, the DB-BOA surrogate with `eval_mode` and the leak guard. |
| `adtcn.py.pre_obj13` | 4 Sep | ⛔ The only copy of `adtcn.py` before OBJ-13; `verify_obj13_legacy.py` loads it. |
| `federated_adtcn.py` | 20 May | Per-bank detector wrapper; weight extraction, with or without DP. |
| `federation_manager.py` | 20 May | DP → Krum → Shapley (exact and MC) → the output-channel private incentive. |
| `basepaper_models.py` | 31 Aug | EfficientNet-, ResNet- and DenseNet-1D, DTCN, LSTM, for the base-paper and window-grid comparisons. |

**`utils/`**

| File | Since | Purpose |
|---|---|---|
| `metrics.py` | 20 May | All metrics; `obf2_value` (bounded since 31 Aug); `coalition_score` (balanced accuracy, June). |
| `visualizer.py` | 20 May | Pipeline plots; fabricated series removed (Fix 9, 1 Jun fixes). |

**`experiments/`** (one script per result)

| File | Since | Produces / purpose |
|---|---|---|
| `__init__.py` | 31 Aug | Package marker. |
| `_dataset.py` | 1 Sep | Shared `--dataset` / `--partition` resolution, provenance, suffixes, `--redraft`. |
| `_seqtrain.py` | 31 Aug | Shared sequence-training helper for the grids. |
| `privacy_incentive_sweep.py` | v2 (Jun) | Task A: weight-channel DP vs Shapley fidelity and rewards. |
| `private_incentive_sweep.py` | v2 (7 Jun) | B1: weight vs output channel; `--eps-grid`, `--repeats`, fragility table (4 Sep). |
| `economic_byzantine_sweep.py` | v2 (Jun) | Task B: collusion, lone attackers, free-riders, isolation. |
| `byzantine_robustness_sweep.py` | v2 (Jun) | Task D: Krum at n = 5 / f = 1 and n = 7 / f = 2, four attacks, separation margins. |
| `scalability_sweep.py` | v2 (Jun) | Task C: exact vs MC Shapley cost and fidelity; `--fit-pool` (11 Sep). |
| `rl_leader_sweep.py` | v2 (Jun) | RL vs DB-BOA leader selection over 5 seeds. |
| `temporal_pipeline_ablation.py` | v2 (7 Jun) | B3: random vs time-ordered windows × CNN vs dilated + attention. |
| `architecture_ablation.py` | v2 (7 Jun) | Architecture ablation on ULB. |
| `write_b2_draft.py` | v2 (7 Jun) | Writes the Fabric-measurement draft from its JSON. |
| `dbboa_vs_default.py` | v2 (11 Jun) | The paired default-vs-tuned retraining (source of the withdrawn 0.313). |
| `plot_federated_confusion.py` | v2 (11 Jun) | Federated confusion-matrix figure from saved counts. |
| `detector_multiseed.py` | 31 Aug | The 5-seed detector headline and the thread-count probe; picks up repaired arms from `extra_configs.json`. |
| `check_thread_sensitivity.py` | 31 Aug | Thread-count determinism check. |
| `banksim_temporal_grid.py` | 31 Aug | OBJ-1: 7 architectures × {global, customer} windows on BankSim. |
| `basepaper_comparison.py` | 31 Aug | OBJ-1b: base-paper classifiers and optimisers, both datasets; `--eval-mode` with overwrite refusal. |
| `federated_cross_dataset.py` | 31 Aug | Collates every federated ablation across conditions. |
| `objective_noise_audit.py` | 31 Aug | OBJ-13 diagnosis (pins `legacy` on purpose); `--render-only`, `--knee`, `--knee-redraft`, `--knee-seeds`, `--components`. |
| `sweeps_cross_condition.py` | 1 Sep | **The only correct way to compare conditions**: dataset vs partition effects, contaminated cells, clean-subset table. |
| `check_obj17_nesting.py` | 4 Sep | Checks the nested seed structure of the privacy sweep and the ULB reproduction. |
| `obj13_shipped_path_check.py` | 4 Sep | Found the memorised-rows leak, with zero training. |
| `obj13_surrogate_repair.py` | 4 Sep | The three-mode side-by-side with the held-out paired yardstick. |
| `check_handbook_loader.py` | 8 Sep | 42 acceptance checks on the Handbook loader. |
| `make_report_figures.py` | 10 Sep | Every WORK_REPORT figure from stored results; prints provenance. |
| `paysim_recon.py` | 11 Sep | PaySim reconnaissance: repeat structure, the file-order trap, link strength. |
| `amlsim_recon.py` | 11 Sep | AMLSim reconnaissance (caught the first derivation's bank/label tie). |
| `check_paysim_loader.py` | 11 Sep | Re-derives every number the PaySim configuration quotes. |
| `check_amlsim_loader.py` | 11 Sep | The same for AMLSim. |
| `check_partition_windowing.py` | 11 Sep | How entity-linked each split's training windows are (the windowing correction). |
| `check_detector_repro.py` | 11 Sep | Bit-for-bit re-run check of stored detector records. |
| `entity_temporal_grid.py` | 11 Sep | PaySim/AMLSim window grids, split per architecture for Kaggle, with a strict merge. |
| `handbook_temporal_grid.py` | 11 Sep | The Handbook's 7 × 3 window grid; resumable, `--archs` / `--merge`. |
| `kaggle_jobs.py` | 11 Sep | Bundle / push / status / queue / track / fetch for Kaggle jobs. |
| `run_obj15_banksim.ps1` | 1 Sep | Detached runner for the four BankSim sweeps (`-Partition`). |
| `run_obj15_banksim.sh` | 1 Sep | Bash version; does not work on this machine (no `python` on Git Bash's PATH). Kept as a record. |
| `check_obj15.ps1` | 1 Sep | Read-only progress checker for the OBJ-15 runs. |
| `run_obj17_epsgrid.ps1` | 4 Sep | Runner for the extended privacy grid. |
| `run_obj13_sidebyside.ps1` | 4 Sep | Detached 16 h side-by-side runner with a preflight. |
| `run_obj13_ulb_regen.ps1` | 11 Sep | ULB regeneration and repaired detector arms. |
| `run_main_ulb_rerun.ps1` | 11 Sep | `main.py` re-run that doubles as a reproducibility check. |
| `run_obj16_handbook.ps1` | 11 Sep | The Handbook suite, gated on the APPROVED pre-registration. |

**`scratchpad/`** (tracked deliberately; each file is the record of a decision)

| File | Since | Purpose |
|---|---|---|
| `resume_new.md` | 1 Sep | Session handoff while the confound control ran. |
| `obj13_new_objective.py` | 4 Sep | The staged OBJ-13 objective, rescued from a temp directory. |
| `apply_obj13.py` | 4 Sep | Idempotent patch applier (`--check` writes nothing). |
| `verify_obj13_legacy.py` | 4 Sep | Proves `legacy` = pre-patch, bit for bit. |
| `verify_handbook_no_regression.py` | 8 Sep | Proves the Handbook plumbing left ULB and BankSim unchanged. |

**`final_report_data/` (inside `db_boa_framework/`)**

| File | Since | Purpose |
|---|---|---|
| `08_rl_leader_selection.md` | v2 (Jun) | The honest RL write-up for the report. |

**`results/`** (the only source of truth for numbers)

*Pipeline outputs (20 May, regenerated since):* `db_boa_results.json` (last regenerated 11 Sep on
the fixed pool), `activation_accuracy.png` (from before Fix 9; the plot was retired),
`classifier_comparison.png`, `confusion_matrix.png`, `cost_function_convergence.png`,
`federation_weights.png`, `incentive_tokens.png`, `leader_selection.png`,
`org_accuracy_progression.png`, `roc_curve.png`, `summary_comparison.png`,
`throughput_latency.png`, `token_balance_history.png`.

*June experiments:* `baselines.json` (regenerated 12 Sep), `dbboa_vs_default.json`,
`fabric_consensus_measured.json`, `temporal_pipeline_ablation.json`, `rl_leader_sweep.json`,
`rl_leader_adaptivity.png`, `rl_leader_reward_fairness.png`, `privacy_incentive_sweep.json`,
`privacy_incentive_tradeoff.png`, `privacy_incentive_reward_bars.png`, `confusion_matrix_federated.png`.

*The four system sweeps, one JSON (+ figures) per condition.* Suffix = condition; no suffix = ULB.
Conditions: `_banksim_stratified`, `_banksim_customer`, `_handbook_stratified`,
`_handbook_customer`, `_paysim_stratified`, `_amlsim_stratified`, `_amlsim_bank`.

- `byzantine_robustness_sweep{,_banksim_stratified,_banksim_customer,_handbook_stratified,_handbook_customer,_paysim_stratified,_amlsim_stratified}.json` + `byzantine_robustness_krum_vs_fedavg*.png`
- `economic_byzantine_sweep{,_banksim_stratified,_banksim_customer,_handbook_stratified,_handbook_customer,_paysim_stratified,_amlsim_stratified,_amlsim_bank}.json` + `economic_isolation_trajectory*.png` + `economic_accuracy_protection*.png`
- `private_incentive_sweep{,_banksim_stratified,_banksim_customer,_handbook_stratified,_handbook_customer,_paysim_stratified,_amlsim_stratified,_amlsim_bank}.json` + `private_incentive_channel*.png`; also `private_incentive_sweep_rep2.json` (same-day ULB re-run) and the 1000-draw passes `private_incentive_sweep{,_banksim_stratified,_banksim_customer}_r1000.json` with their `private_incentive_channel*_r1000.png` / `_rep2.png`
- `scalability_sweep{,_banksim_stratified,_banksim_customer,_handbook_stratified,_handbook_customer,_paysim_stratified,_amlsim_stratified}.json` + `scalability_shapley_runtime*.png` + `scalability_fidelity_accuracy*.png`

*Federated ablations:* `baselines_banksim_{stratified,customer}.json`,
`baselines_handbook_{stratified,customer}.json`, `baselines_paysim_stratified.json`,
`baselines_paysim_all_stratified.json`, `baselines_amlsim_{stratified,bank}.json`,
collated in `federated_cross_dataset.json`.

*Detector and search:* `detector_multiseed.json`, `detector_repro_check.json`,
`basepaper_comparison_ulb.json`, `basepaper_optimisers_ulb.json`, `basepaper_optimisers_banksim.json`,
`objective_noise_audit.json`, `objective_size_knee.json`, `obj13_shipped_path_check.json`,
`obj13_surrogate_repair_banksim.json`.

*Window grids:* `banksim_temporal_grid.json`; `handbook_temporal_grid.json`, `paysim_temporal_grid.json` and
`amlsim_temporal_grid.json` (merged), each with seven per-architecture parts
`*_temporal_grid_{cnn,lstm,dtcn,dilated_attn,resnet,densenet,efficientnet}.json`.

*Reconnaissance and checks:* `paysim_recon.json`, `amlsim_recon.json`, `partition_windowing.json`.

*Figures for WORK_REPORT:* `figures/fig_architecture_scoreboard.png`, `fig_cross_dataset_comparison.png`,
`fig_memorisation_before_after.png`, `fig_process_optimisation.png`, `fig_scalability.png`,
`fig_score_index.png`, `fig_statistical_significance.png`, `fig_system_performance.png`,
`score_index.csv`, `score_index_table.tex`.

*Archives and backups:* `_ulb_figures_backup/` (8 ULB figures saved before the BankSim run could
overwrite them); `_obj17_pre_extension/` (`README.md` plus the 9-point-grid JSONs and figures, the
only evidence for the factors as first published); `_smoke/` (`paysim_temporal_grid_quick.json`,
`handbook_temporal_grid_quick.json`, which are never results).

*Per-run detector records:* `_multiseed_runs/` (25 files): `_determinism.json`,
`dbboa_tuned_seed{42..46}_t2.json`, `dbboa_tuned_seed42_t{4,8}.json`,
`hand_set_default_seed{42..46}_t2.json`, `hand_set_default_seed42_t8.json`,
`dbboa_repaired_{deterministic,averaged}_seed{42..46}_t2.json`, `extra_configs.json`.

*Run logs (kept since the 5 Sep `.gitignore` fix):* `_obj13_logs/` (knee, scout, shipped-path
before and after the fix, side-by-side, and `ulb_regen/` with 45 files including
`db_boa_results_phase1to7_2250.json` and `main_rerun_compare.txt`); `_obj15_logs/banksim_{customer,stratified}/`;
`_obj16_logs/handbook/` (27 files); `_obj17_logs/{grid,r1000,rep2}/`. The per-sweep logs are UTF-16LE.

*Kaggle:* `_kaggle/` (184 files). Queue state (`_queue.json`, `_queue*.log`, `_queue3.pid`,
`_track.*`) and one folder per job, each holding the kernel log, `out/job.log`,
`out/kaggle_job.json` (with the code bundle's SHA-256) and the job's outputs. Jobs:
`paysim-smoke`; `paysim-{baselines,all-baselines,byzantine-robustness,economic-byzantine,private-incentive,scalability}-stratified`;
`paysim-grid-*` ×7; `amlsim-baselines-{stratified,bank}`; `amlsim-byzantine-robustness-stratified`;
`amlsim-economic-byzantine-{stratified,bank}`; `amlsim-private-incentive-{stratified,bank}`;
`amlsim-scalability-stratified` (and `__failed_2335`, the crash that led to `--fit-pool`);
`amlsim-grid-*` ×7; `handbook-grid-*` ×7.

## A.3 `final_report_data/` — drafts and the audit trail

| File(s) | Since | Purpose |
|---|---|---|
| `README.md` | v2 (Jun) | Index; the rule "if it was not used, it is not in the report". |
| `00_report_vs_code_divergences.md` | v2 (Jun) | **The D1–D17 audit.** |
| `00_ground_truth_implementation.md` | v2 (Jun) | What the system actually is. |
| `01_introduction.md`, `02_literature_review.md`, `03_requirements_impacts_constraints.md`, `04_methodology.md`, `05_results.md`, `06_conclusion.md` | v2 (Jun) | Per-chapter correction notes (several stamped SUPERSEDED 1 Sep). |
| `02_literature_review_DRAFT.md` | v2 (8 Jun) | Literature-review draft. |
| `07_actions_checklist.md` | v2 (Jun) | The June to-do list (superseded by `TASK.md`). |
| `REWRITE_00_title_abstract.md`, `REWRITE_01_introduction.md`, `REWRITE_02_literature_and_bib.md`, `REWRITE_03_requirements.md`, `REWRITE_05_methodology.md`, `REWRITE_06_results.md`, `REWRITE_08_limitations_disclosures.md`, `REWRITE_09_conclusion.md` | v2 (Jun) | Paste-ready LaTeX for each chapter; the record of what was merged into the thesis. |
| `REWRITE_FIX_2026-06-11.md` | v2 (11 Jun) | The final audit fix batch (stamped SUPERSEDED for its 0.313 entry). |
| `TASKA_privacy_incentive_results.md` | v2 (Jun) | Task A draft. |
| `TASKB_economic_byzantine_results*.md` (ULB + 7 conditions) | Jun / Sep | Task B drafts per condition. |
| `TASKB1_private_incentive_results*.md` (ULB, `_rep2`, `_r1000` + conditions) | Jun / Sep | B1 drafts per condition, with the fragility tables. |
| `TASKB3_temporal_pipeline.md` | v2 (7 Jun) | B3 draft. |
| `TASKC_scalability_results*.md` (ULB + 6 conditions) | Jun / Sep | Task C drafts. |
| `TASKD_byzantine_robustness_results*.md` (ULB + 6 conditions) | Jun / Sep | Task D drafts. |
| `B2_fabric_consensus_measured.md` | v2 (7 Jun) | Fabric measurement draft (CONSUMED into Chapter 6, 31 Aug). |
| `OBJ1_banksim_temporal_grid.md` | 31 Aug | OBJ-1 verdict. |
| `OBJ2_detector_multiseed.md` | 31 Aug | OBJ-2 / OBJ-4 verdict. |
| `BASEPAPER_comparison_ulb.md`, `BASEPAPER_optimisers_ulb.md`, `BASEPAPER_optimisers_banksim.md` | 31 Aug | OBJ-1b drafts. |
| `FEDERATED_cross_dataset.md` | 31 Aug | DP replication across datasets. |
| `OBJECTIVE_noise_audit.md` | 31 Aug | OBJ-13 diagnosis (its "n ≈ 9" mechanism was withdrawn; the numbers stand). |
| `OBJECTIVE_size_knee.md` | 4 Sep | Knee sweep: no knee locatable at one seed. |
| `OBJ13_surrogate_repair_banksim.md` | 5 Sep | The 0-of-9 headline. |
| `OBJ15_two_factor_decomposition.md` | 4 Sep | Dataset vs partition effects (generated; never hand-edit). |
| `OBJ16_paysim_temporal_grid.md`, `OBJ16_amlsim_temporal_grid.md`, `OBJ16_handbook_temporal_grid.md` | 12 Sep | Grid drafts for the three new datasets. |

## A.4 `db_boa_fabric/` — the blockchain layer

| File | Since | Purpose |
|---|---|---|
| `chaincode/lib/db_boa_chaincode.js` | 20 May | `DBBOAContract`: node metrics, incentives (reputation clamped to [0.5, 2.0]), fraud results, leader selections, consensus rounds, hyperparameters, federation rounds (20-token pool split by Shapley weight), and queries. Iterator leak fixed (BF-10). |
| `chaincode/index.js`, `chaincode/package.json` | 20 May | Chaincode entry point and dependencies (`fabric-contract-api` 2.5.x). |
| `api-server/server.js` | 20 May | Express REST API + server-sent-event logs + dashboard routes; path-traversal and argument-order fixes (BF-7–9). |
| `api-server/index.html` | 20 May | The web dashboard. |
| `api-server/measure_consensus.js` | v2 (7 Jun) | The real latency and throughput measurement (B2). |
| `api-server/enrollAdmin.js`, `registerUser.js`, `wallet/admin.id`, `wallet/appUser.id` | 20 May | Fabric CA enrolment and the test-network identities. |
| `api-server/start_db_boa.sh` | 20 May | Launcher. |
| `api-server/package.json`, `package-lock.json`; `api-server/node_modules/` | 20 May | Node dependencies (`fabric-network` / `fabric-ca-client` 2.2.20); `node_modules/` is vendored. |
| `README.md`, `package.json`, `package (1).json`, `package (2).json` | 20 May | Layer README; the two bracketed files are duplicate downloads of `package.json`. |

## A.5 Thesis, papers, reports, infrastructure

| Path | Since | Purpose |
|---|---|---|
| `FINAL YEAR THESIS REPORT/main.tex`, `.latexmkrc` | 20 May / Jun | Thesis root and build configuration (the nomenclature index rule). |
| `FINAL YEAR THESIS REPORT/chapters/chapter_1.tex`, `chapter_2.tex`, `chapter_3.tex`, `chapter_5.tex`, `chapter_6.tex`, `chapter_9.tex` | 20 May | The six chapters as numbered in the template (Introduction; Literature Review; Requirements; Methodology; Result Analysis; Conclusion). Rewritten in June and re-based on 31 Aug. `chapter_7.tex` was deleted 11 Jun. |
| `FINAL YEAR THESIS REPORT/chapters/Blank diagram (6).png` | 20 May | A diagram image kept beside the chapters; no chapter references it now. |
| `FINAL YEAR THESIS REPORT/core/abstract.tex`, `acknowledgement.tex`, `approval.tex`, `declaration.tex`, `titlepage.tex`, `nomenclature.tex`, `dedication.tex`, `ethics_statement.tex` | 20 May / Jun | Front matter (the last two are empty template files). |
| `FINAL YEAR THESIS REPORT/appendix/appendix_1.tex`, `appendix_2.tex` | v2 | Empty template appendices (removed from `main.tex` 10 Jun). |
| `FINAL YEAR THESIS REPORT/bibliography/references.bib` | 20 May | 71+ verified entries. |
| `FINAL YEAR THESIS REPORT/main.pdf`, `main.ilg`, `main.nls` | Jun | The compiled thesis and its nomenclature index. |
| `FINAL YEAR THESIS REPORT/images/` | 20 May / Jun | Figures. The originals from May (`Blank diagram (8).png`, `Fig_ure_1.png`, `Figure3.png`, `Figure4.png`, `Figure_5_Precision_Recall.png`, `Figure_8_Privacy.png`, `Figure_11_Comparative_Performance.png`, `Figure_13_Radar_Chart.png`, `GRA sig.png`, `fig-5.jpg`, `fig-17.jpg`, `fig-18.jpg`, `fig2.png`, and the `DB-BOA Metrics/` folder of pre-audit plots) and the June result figures (`byzantine_robustness_krum_vs_fedavg.png`, `classifier_comparison.png`, `confusion_matrix.png`, `confusion_matrix_federated.png`, `cost_function_convergence.png`, `economic_accuracy_protection.png`, `economic_isolation_trajectory.png`, `fabric_live_capture.png`, `federation_weights.png`, `incentive_tokens.png`, `leader_selection.png`, `methodology_overview.png` / `.svg`, `org_accuracy_progression.png`, `privacy_incentive_reward_bars.png`, `privacy_incentive_tradeoff.png`, `private_incentive_channel.png`, `rl_leader_adaptivity.png`, `roc_curve.png`, `scalability_fidelity_accuracy.png`, `scalability_shapley_runtime.png`, `summary_comparison.png`, `throughput_latency.png`, `token_balance_history.png`, `ulb_class_distribution.png`). |
| `all papers/` (30 PDFs) | 20 May / Jun | The reference set, one file per cited method or related work: Abdallah-AlShaibani 2020, Andrew 2021, Bai 2018, Blanchard 2017, Chaudhuri 2011, Commey 2025, Dwork 2006, Fraboni 2020, Ghorbani & Zou 2019, Givi & Hubálovská 2023, Hsieh 2020, Jaramillo-Velez 2026, Li 2020 (FedProx), Li 2023 (Fabric-SCF), Li 2025, Liu 2020 (FedCoin), Liu 2021, Lopez-Rojas 2016, McMahan 2017, McMahan 2018, Nourmohammadi 2022, **Prabanand & Thanabal 2025**, Saveetha 2024, Truong 2024, Tsoulias 2020, Tubishat 2020, Wang 2020 (FedSV), Watkins & Dayan 1992, Yang 2024, Zhao 2026 (SI-ChainFL). |
| `reports/P2_REPORT_T2430460.pdf`, `P2_POSTER_T2430460.pdf`, `report.txt` | v2 | The P2 proposal (Part I) and its text. |
| `example_thesis/T2430427_P3 (1) (1).pdf`, `T2430427_P3_Slides (1).pptx` | v2 (11 Jun) | An approved BRAC thesis used as the rubric benchmark. |
| `fabric/install-fabric.sh` | 20 May | Fabric installer. |
| `fabric/fabric-samples/` | 30 Aug (`ccdd19f`) | Vendored upstream Hyperledger Fabric samples, including the test network the chaincode runs on. |
| `hello-world/docker-compose.yml` | 20 May | Two-line Docker smoke test. Removed from v2 on 8 Jun; kept here. |

## A.6 Made but not in git (by design)

| Path | Why untracked |
|---|---|
| `creditcard.csv`, `datasets/creditcard.csv` | ULB, 143.8 MB, above GitHub's limit. From Kaggle. |
| `datasets/bs140513_032310.csv` | BankSim (Kaggle). |
| `datasets/bsNET140513_032310.csv` | A projection of BankSim; verified **not** a separate dataset. |
| `datasets/handbook_raw/`, `datasets/handbook_transactions.csv` | Handbook clone and its consolidated CSV (105 MB); also uploaded as the private Kaggle dataset `fl-adtcn-handbook`. |
| `datasets/paysim/` | PaySim (Kaggle `ealaxi/paysim1`). |
| `datasets/amlsim/` | Generated AMLSim output, the upstream checkout and the Python 3.8 environment; uploaded as the private Kaggle dataset `fl-adtcn-amlsim-10k-3banks`. |
| `kaggle.md` | The Kaggle API token. |
| `SUPERVISOR_BRIEF.aux/.log/.out`, `WORK_REPORT.aux/.log/.out/.toc` | LaTeX build files. |
| Outside the repo | `%LOCALAPPDATA%\Programs\fl-adtcn-kaggle-venv` (Kaggle CLI 2.2.4), Maven 3.9.9, the MASON 20 jar, and the private Kaggle dataset `fl-adtcn-code` (the code bundles). |

---

# APPENDIX B — NUMBERS THAT WERE WITHDRAWN, AND WHY

| Number | Where it appeared | Why withdrawn | When |
|---|---|---|---|
| Synthetic-data metrics, activation plot, base-paper baselines, optimiser curves, AUCs 0.94–0.98 | May code and plots | Not measured on our data, or invented outright | 20 May – 1 Jun |
| 97.38 % accuracy, MCC 0.966 | Thesis draft | No run produced them (D5) | Jun |
| 85 TPS, 180 ms, "28.4 % latency cut" | Thesis draft | Simulated arithmetic (D7); replaced by the measured 2115.9 ms / 40.3 TPS | Jun |
| 8-model comparison table | Thesis draft | Copied from the base paper (D4) | Jun |
| Paired t-tests, 28/18/4 leader split, 458/312/178 tokens, weights converging to [0.52, 0.32, 0.16] | Thesis draft | Never computed (D6, D8, D9) | Jun |
| MCC 0.313 and the "0.47 paired gap"; "DB-BOA loses to the default (0.785)" | June report, brief, drafts | 0.313 was a thread-count artefact and does not reproduce; the real difference is +0.047, p = 0.34 | 31 Aug |
| "Krum pays 1.5–12.6 pp under entity-disjointness" | OBJ-15 | Falsified by our own control; part of the range came from contaminated cells | 4 Sep |
| "MC top-1 fails from n ≈ 6" | June report, OBJ-15 | Intermittent, not a threshold; later shown untestable in 3 of 7 conditions | 1 Sep, 12 Sep |
| "≈60×" / 30× / 33× / ≥100× privacy improvement factor | Report, brief | ε\* can only grow with effort; it swings 20× to ≥100× with the cut-off | 4 Sep |
| "The fraud rate is not what separates ULB from BankSim"; "Obf2 = 5.0000 is reachable by luck"; "n ≈ 9 positives" | OBJ-13 | The shipped surrogate validated on memorised rows; the knee sweep falsified the mechanism | 4 Sep |
| 30 August ULB `db_boa_results.json` / `baselines.json` numbers (FedAvg 0.569, Krum +0.207, MCC 0.677 as tuned) | Report | Chosen by a broken search on a leaking surrogate, plus the unexplained reproduction failure; regenerated 11–12 Sep | 4–12 Sep |
| Any "partition effect" quoted as the partition alone | OBJ-15 control | The entity split also changed the training windows (0 % → ~91 % single-entity) | 11 Sep |
| The first AMLSim dataset | — | Bank had become a proxy for the label | 11 Sep |
| "The DP collapse lowers accuracy 0.11 pp" (ULB) | WORK_REPORT | Sign error: it was a rise | 12 Sep |
| "Entity grouping does not lift the Handbook's models at all" | TASK.md, WORK_REPORT | Entity *partitioning* lifts them (0.0439 → 0.1500); only *windowing* does not | 12 Sep |
| "Attacker caught 8/8" as universal; "held by construction" | Everywhere until 12 Sep | Krum selects a label-flipper on the Handbook; AMLSim's 8/8 is ordering, not detection | 12–13 Sep |
| "5/10, 5/10, 3/10, 3/10, 3/10" as one fidelity series | WORK_REPORT | Three of those conditions measure noise (models at chance) | 12 Sep |

---

# APPENDIX C — THE DECISION REGISTER

| # | Date | Question | Decision, and who | What it produced |
|---|---|---|---|---|
| — | pre-May | What system? | Blockchain-integrated FL for financial fraud on Fabric, DB-BOA + ADTCN from the base paper; title fixed (team) | The P2 design |
| — | 20 May | Keep synthetic data? | No: ULB (team) | Real data, and later the missing-customer problem |
| — | 20 May | Krum at n = 3? | f = 0 and drop the BFT wording; later test at n ≥ 2f + 3 (team) | Honest "Secure" (Task D) |
| — | Jun | Report says DB-BOA Job 3; code says Shapley. Which story? | **Option A**: tell the story of what is built (team) | D1 closed |
| — | Jun | What kind of novelty? | Characterisation, not new algorithms; no priority claims (team) | `novel_plan.md` |
| — | Jun | The title says RL and there is none | Keep the title and add RL where it fits: leader selection (team) | `rl_leader.py` |
| — | 6–7 Jun | DP breaks the rewards. Drop DP? | No: move the noise to the contribution channel (team) | The thesis's contribution (B1) |
| — | 30 Aug | Which repository is current? | The user's own work; sync to v2 (Shadman) | `ccdd19f` |
| 1 | 30–31 Aug | Replace ULB or keep it? | Keep ULB, add BankSim (team) | Linkage priced; DP direction flip; two ULB-only defects found |
| — | 1 Sep | Judge the thesis by ADTCN's rank? | No: judge the system (Shadman corrected rev. A) | Rule 9; the real gap found |
| 2 | 1 Sep | Third dataset, or get the claims off ULB? | Claims first (Rule 8) | Five of six claims off ULB |
| 3 | 1 Sep | Build on the Krum finding, or try to break it? | Break it: run the control | Our best new claim died, found by us |
| 4 | 4 Sep | Extend the grid, raise the draws, or both? | Both, on one shared grid | De-censored, and the estimator shown unstable |
| 5 | 4 Sep | Publish the measured factor? | No: retire it | The claim rests on threshold-free statements |
| 6 | 4 Sep | Fix the leaking pool how? | Enlarge 3,000 → 36,000 + a runtime guard (Shadman approved) | 0 of 11 leaked |
| 7 | 4 Sep | Cheap 1.1 h comparison or the 16.7 h deployment budget? | The expensive one, against the written recommendation (Shadman) | 0 of 9 beat the default; two predictions inverted |
| 8 | 4 Sep | Repair the dead axis or drop it? | Repair | Alive but coarse; the deployed axis was never dead |
| 9 | 4 Sep | Name a cause for the ULB non-reproduction? | No: record it as unexplained | Environment stamping in every JSON |
| 10 | 4 Sep | Consolidate or expand? | Consolidate first; protect the writing block (Shadman) | The repair preceded any new dataset |
| 11 | 5 Sep | Commit, or keep iterating? | Commit | ~20 h of results safe; hidden logs recovered |
| 12 | 11 Sep | Keep Rule 8? | **Suspended** (Shadman) | All five datasets run |
| 13 | 11 Sep | Where to run them? | Laptop for ULB + Handbook, Kaggle for PaySim + AMLSim (Shadman) | Each dataset in one environment |
| 14 | 11 Sep | Real AMLSim or the substitute? | The real simulator (Shadman) | A reproducible native-bank dataset |
| 15 | 11 Sep | PaySim: all rows or the fraud-bearing two? | Two types + a sensitivity arm (delegated: "do what's best for the research") | No accuracy padding; scope shown not to matter |
| 16 | 11 Sep | AMLSim v1 ties bank to label | Discard and regenerate | Bank columns predict nothing |
| 17 | 11 Sep | The control's partition effect also moved the windows | Rescope in writing, not a silent fix | The withdrawal survives; wording changed everywhere |
| 18 | 11 Sep | AMLSim is too small for one sweep | Cut the MC range at 16, written down before the run | Pre-registered statements untouched |
| 19 | 11 Sep | 11/11 or 8/11 by reading | Both scores, no verdict (Shadman) | A rule binding every document |
| 20 | 12 Sep | ULB accuracy sign error | Correct it where it stood, in a dated box | Honest record of the error |
| 21 | 12 Sep | Apply a borrowed floor after scoring? | No: items stay scored | Failures stay failures |
| — | 12 Sep | Handbook grid: laptop (~16 h) or Kaggle? | Kaggle, split 7 ways (Shadman) | ~16 h saved; verdict unaffected (all within-dataset) |
| — | 13 Sep | Fill the AMLSim × Krum cell? | No: record why it cannot be run | Krum "untestable on AMLSim" |

*Numbered rows 1–21 are WORK_REPORT's decision log. Unnumbered rows come from the earlier
documents.*

---

# APPENDIX D — GLOSSARY

- **FL-ADTCN:** the whole system: federated ADTCN detectors + DP + Krum + Shapley + DB-BOA + RL leader
  election, on Hyperledger Fabric. Six components, six title claims.
- **ADTCN:** the fraud detector. It reads a window of 10 consecutive transactions. Deployed as a
  1-D CNN; the dilated + attention version exists and is ablated.
- **DB-BOA:** the hybrid Butterfly / Billiards optimiser from the base paper, used to pick detector
  settings and (at cold start) the leader. The **surrogate** is the small model it trains to score
  each candidate.
- **Obf2:** the surrogate's score. Since June it is 2·MCC + Specificity + Precision + NPV (maximum 5).
- **MCC:** Matthews correlation coefficient, our headline metric, because at 0.17 % fraud a model
  that flags nothing already scores 99.83 % accuracy.
- **FedAvg / Krum:** averaging every bank's weights vs keeping the single most central submission.
  Krum is security, not fairness.
- **Shapley value:** each bank's average marginal contribution over all coalitions; decides rewards.
  Fairness, not security.
- **DP, ε:** calibrated noise for privacy; smaller ε means more noise. **Weight channel vs output
  channel:** noising the shared weights vs noising the published contribution score.
- **Rank inversion; ε\*:** rewards paid in the wrong order; ε\* is the largest budget at which an
  inversion was *seen* (retired as a headline because it only grows with effort).
- **Stratified vs entity-disjoint split:** deal rows to banks keeping the fraud rate, vs deal whole
  customers so no one's history is split across banks.
- **Global vs entity-linked window:** the next 10 rows in the file vs the next 10 transactions of the
  same customer, terminal, sender or receiver.
- **Pre-registration:** the prediction and its falsifier, written before a run and never edited;
  the score goes underneath.
- **Confound control:** a run that changes one factor only, so an effect can be attributed.
- **Contaminated cell:** an attacked baseline scoring above its own no-attack reference, which is
  a metric artefact rather than robustness.
- **Floor:** federated FedAvg MCC < 0.05. A condition at the floor cannot test the system claims.
- **Common random numbers:** scoring every candidate on the same random draws, so comparisons are
  not decided by luck.

---

*End of the journal. The next entry belongs after commit `c66d1d1`.*
