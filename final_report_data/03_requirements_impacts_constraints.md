# Chapter 3 — Requirements, Impacts and Constraints (corrections)

The current `chapters/chapter_3.tex` is solid (societal/environmental/ethical/standards/
risk/economic sections are reasonable and can largely stay). Fix the technical-spec
details so they match the code.

## Fixes

### 1. Dataset (🔴 D10)
Change "synthetically generated … 20,000 samples … 5% fraud" to the **real ULB dataset**
(284,807 transactions, V1–V28 + Amount + Time, 0.17% fraud), stratified 50/30/20 across
BankA/BankB/BankC. (Same fix as Ch.1.)

### 2. DB-BOA roles (🔴 D1)
Currently lists DB-BOA across "three distinct roles" including per-org aggregation
weights. The third role is **Shapley**, not DB-BOA. State:
- DB-BOA → (1) ADTCN hyperparameter search, (2) leader selection.
- Federation contribution weights → **exact Shapley values**; robust model selection →
  **Krum**; privacy → **differential privacy (Gaussian mechanism)**.

### 3. Baseline list (🔴 D4)
Remove "evaluated against MBO-ADTCN, WSA-ADTCN, DBOA-ADTCN, BOA-ADTCN, EfficientNet,
ResNet, DenseNet, DTCN." The real baselines are **FedAvg, FedAvg+Krum, FedAvg+DP, and the
proposed (Krum+DP+Shapley)** (`run_baselines.py`).

### 4. Software/resource requirements (🟢 D16)
- Blockchain SDK: **`fabric-network` 2.2.20 + `fabric-ca-client` 2.2.20** (legacy
  wallet-based SDK), **not** `@hyperledger/fabric-gateway`.
- Chaincode: `fabric-contract-api` / `fabric-shim` 2.5.x (Fabric 2.5.x). ✓
- ML stack: Python 3, NumPy, pandas, scikit-learn, **PyTorch** (used by ADTCN — pin it in
  `requirements.txt`, where it is currently commented out as optional), Matplotlib.
- Only claim GPU/CUDA if a GPU run was actually performed.

### Standards section nuance
The "IEEE 2986-2023 federated ML privacy" and Fabric determinism claims are fine. The
determinism point is genuinely true of the chaincode (uses `ctx.stub.getTxTimestamp()`,
no `Math.random()`/`new Date()` in state-mutating functions — verify and keep).

## Risk-management table (referenced but currently missing a table body)
`chapter_3.tex` references `tab:risk_management` but the table itself isn't in the file.
Add a small real table, e.g.:

| Risk | Impact | Mitigation (as implemented) |
|------|--------|------------------------------|
| Model poisoning by an org | Corrupted global model | Krum consensus-aligned selection + Shapley down-weighting + on-chain token/reputation penalty |
| Privacy leakage via shared weights | Re-identification | DP Gaussian mechanism (ε=1.0) before sharing |
| Non-deterministic chaincode → endorsement mismatch | Consensus failure | All randomness/optimisation kept off-chain; only finalised results written on-chain |
| Class imbalance (0.17% fraud) | Trivial "all-normal" classifier | Weighted cross-entropy; MCC/ROC-AUC as primary metrics |
| Single-bank dataset | Limited ecological validity | Stated as a limitation; future work = real multi-bank streams |
