# Fix batch — final audit corrections (2026-06-11)

Applied directly to the `.tex` with user approval ("do the full"). Substantive rewrites summarised here; all numbers below verified against code/JSON/CSV.

## 1. DB-BOA vs default (ch5 §DB-BOA HPO, tab:dbboa_vs_default) — disclosure route
- Caption no longer says "single full run"; table now has **three rows**: default 128/150 → 99.92%/0.785 (paired run), tuned 142/76 → 99.85%/0.677 (headline run), tuned 142/76 → 98.77%/**0.313** (paired run, from `dbboa_vs_default.json`, `mcc_gap_default_minus_tuned: 0.4716`).
- "Honest finding" paragraph now reports gap ≈0.11 (vs headline) AND ≈0.47 (paired), plus the 0.677-vs-0.313 run-to-run variance of the tuned config (low steps/epoch ⇒ very large batches ⇒ unstable training) as part of the finding.
- ch6 (conclusion) parenthetical updated to point at the variance disclosure.
- Ch5 intro sentence now mentions the dedicated paired retraining run.

## 2. Two-org deployment honesty (ch4 "Hyperledger Fabric Network Deployment")
- Removed the false "extended with a third organisation (BankC/Org3)" narrative, peer0.org3/ca.org3 rows, and "2-of-3" deployment claims.
- Rewritten to match reality (and Figure fabric_live): two peer orgs on a single host, CCaaS chaincode, majority endorsement; three-bank consortium runs in the off-chain layer with node state in chaincode world state. Docker table now lists the real containers/ports from the live capture (peer0.org1 7051, peer0.org2 9051, ca_org1 7054, ca_org2 8054, ca_orderer 9054, orderer 7050, couchdb0/1, 2× db-boa CCaaS 9999).
- "Consortium Network Topology" section + figure caption now explicitly framed as design, with pointer to Limitations for the 2-org measured deployment. Ch1 Phase-5 "with three organisations" dropped.

## 3. Code-truth fixes
- Features: 29/30 → **33** (30 base + 3 temporal-amount recurrence features, `use_graph_features: True`) everywhere: ch5 data-prep + MJE + math (x_t∈R^33, Conv1d 33→F) + arch table + TikZ diagram + (N,10,33); recurrence features now described.
- Training config: AdamW/weight-decay/grad-clip/LR-scheduler claims → plain **Adam lr 1e-3**, fixed 30 epochs, batch from steps/epoch; dev-env table row fixed. (CNN path has no dropout — none claimed.)
- "Threshold Optimisation" subsection + workflow step deleted (code predicts by argmax; no threshold tuning exists).
- "weighted sampling" → class-weighted loss only.
- Rounds: "50 consensus / 10 federation" → 5 simulated consensus rounds, 3 federation rounds, 15-round attack (ch4 Phase 6 + ch5 workflow step).
- DB-BOA math: switch rule now the implemented range-normalised threshold τ_t (base paper's best/worst ratio noted as the origin); LSAM now Gaussian mutation μ=0.1, 5 iterations (Lévy noted as the base paper's description).
- "re-runs if degradation detected" softened to manual re-run; leader re-selection per round kept.

## 4. Data corrections (vs raw creditcard.csv)
- tab:stat_summary: Time mean/std/25/75 → 94,814 / 47,488 / 54,202 / 139,321; Amount 25/75/max → 5.60 / 77.16 / 25,691.16; V1 quartiles → −0.92/0.02/1.32; V2 → −0.60/0.07/0.80 (and min −72.72).
- z-score table: max 25,691.16 → 102.36; min −0.3532. Cleaning text max fixed.
- Class weight "(578.0)" → "≈578".

## 5. Minor
- ε* definition unified: "largest budget at which the reward ordering still inverts" (ch5 §private-incentive wording + nomenclature z02).
- Approval: "B.Sc. in Computer Science and Engineering **on** June 13, 2026". (Semester "Spring, 2026" left — user to confirm vs June date.)
- ch9 repo-URL sentence **deleted** (option 1; both URLs 404).
- ch2 "receptive field and capacity ... tuned" → "capacity ... tuned".
- Empty unused `chapters/chapter_7.tex` deleted.
