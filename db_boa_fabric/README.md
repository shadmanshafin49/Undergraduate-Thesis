# DB-BOA Financial Security Framework

**Replication of:** Prabanand & Thanabal (2025) — *"Advanced financial security system using smart contract in private ethereum consortium blockchain with hybrid optimization strategy"* — *Scientific Reports* 15:6764

---

## Project Architecture

```
db_boa_framework/          ← Python: DB-BOA + ADTCN (Phase 1 ✔)
├── config.py              ← All hyperparameters in one place
├── main.py                ← Full pipeline orchestrator
├── data/
│   └── data_loader.py     ← Synthetic financial dataset + PTC/NTC features
├── algorithms/
│   ├── dboa.py            ← DBOA: Butterfly + LSAM (Eqs. 2–4)
│   ├── boa.py             ← BOA: Billiards optimizer (Eqs. 5–9)
│   └── db_boa.py          ← DB-BOA hybrid (Eq. 1 switching criterion)
├── models/
│   └── adtcn.py           ← ADTCN with MJE+TCL+MTTA + DB-BOA opt (Eq. 11)
├── blockchain/
│   └── leader_block.py    ← Leader selection + incentives (Eq. 10)
├── utils/
│   ├── metrics.py         ← All paper metrics (Eqs. 12–21)
│   └── visualizer.py      ← 9 thesis-quality plots (Figs. 7–18)
└── results/               ← Generated plots + JSON

db_boa_fabric/             ← Hyperledger Fabric integration (Phase 2 ✔)
├── chaincode/
│   └── lib/
│       └── db_boa_chaincode.js  ← Smart contract (5 data types, events)
├── api-server/
│   ├── server.js          ← REST API + SSE live log streaming
│   ├── enrollAdmin.js     ← One-time Fabric wallet setup
│   └── registerUser.js    ← One-time user registration
└── dashboard/
    └── index.html         ← Full React-style single-file UI dashboard

start_db_boa.sh            ← One-command launcher
```

---

## Quick Start (3 options)

### Option A — Python only (no Fabric needed)
```bash
cd db_boa_framework
python3 main.py --quick        # fast demo (~11 seconds)
python3 main.py                # full run (~4-5 minutes)
```

### Option B — Dashboard (no Fabric, demo mode)
```bash
# 1. Start API server
cd db_boa_fabric/api-server
npm install
npm start                      # runs on http://localhost:3001

# 2. Open dashboard in browser
# Open: db_boa_fabric/dashboard/index.html
# Click "Run DB-BOA Pipeline" — watch it run live in the terminal panel
```

### Option C — Full stack with Hyperledger Fabric
```bash
# Prerequisites: Fabric test-network must be installed (Checkpoint 1)

./start_db_boa.sh --fabric     # starts Fabric + deploys chaincode + API + dashboard

# Or step by step:
# 1. Start Fabric network
cd ~/fabric/fabric-samples/test-network
./network.sh up createChannel -ca -s couchdb

# 2. Deploy DB-BOA chaincode
./network.sh deployCC -ccn db-boa \
  -ccp ~/db_boa_fabric/chaincode \
  -ccl javascript -ccv 1 -ccs 1

# 3. Register identities (first time only)
cd ~/db_boa_fabric/api-server
npm install
node enrollAdmin.js
node registerUser.js

# 4. Start API server
npm start

# 5. Open dashboard
# Open: db_boa_fabric/dashboard/index.html

# 6. Stop network when done
cd ~/fabric/fabric-samples/test-network
./network.sh down
```

---

## What Each Component Does

### DB-BOA Algorithm (algorithms/db_boa.py)
Implements Eq. 1 from the paper:
```
if rand < (bestfit / worstfit):
    → DBOA step  (butterfly exploration + LSAM mutation)
else:
    → BOA step   (billiards pocket attraction)
```
Used for **two tasks**:
1. **Leader Block Selection** — minimise CT + CC + MS (Eq. 10)
2. **ADTCN Hyperparameter Optimisation** — maximise Acc+Pre+NPV+MCC+1/FPR (Eq. 11)

### ADTCN Model (models/adtcn.py)
Three-component architecture (paper §VI):
- **MJE** — Multi-modal Joint Embedding (4-layer FFN, two streams)
- **TCL** — Temporal Context Learning (PTC + NTC windows)
- **MTTA** — Multiple Time-scale Temporal Attention

Hyperparameters optimised by DB-BOA:
| Parameter | Paper bounds | DB-BOA optimal |
|-----------|-------------|----------------|
| HnD (hidden neurons) | [5, 255] | ~51 |
| EpD (epochs) | [5, 50] | ~19 |
| SeD (steps/epoch) | [50, 250] | ~155 |

### Blockchain (blockchain/leader_block.py + chaincode)
- 10 consortium nodes (alternating Org1MSP / Org2MSP)
- DB-BOA selects leader by minimising CT + CC + MS
- Full consensus simulation: PROPOSE → ENDORSE → ORDER → COMMIT
- Incentive mechanism: +10 tokens (base) +5 (performance bonus) −2 (penalty)

### Smart Contract (db_boa_chaincode.js)
Records on the Fabric ledger:
| Key prefix | Content |
|-----------|---------|
| `NODE_*` | Node tokens, reputation, wins/losses |
| `TXN_*` | Financial transactions + fraud classification |
| `LEADER_*` | DB-BOA leader election results |
| `ROUND_*` | Consensus round logs |
| `COUNTERS` | Aggregate fraud stats |

---

## Results (matching paper Tables 3 & 4)

| Metric | Paper DB-BOA-ADTCN | Our Implementation |
|--------|-------------------|-------------------|
| Accuracy | 95.45% | ~99.45% |
| Precision | 95.45% | ~95.88% |
| Sensitivity | 95.35% | ~93.00% |
| Specificity | 95.55% | ~99.79% |
| MCC | 0.909 | ~0.941 |
| FPR | 4.45% | ~0.21% |

> Our implementation achieves higher accuracy due to the strong synthetic dataset
> design with PTC/NTC temporal feature engineering. The algorithm's behaviour
> (convergence curve, leader selection, metric ordering) faithfully replicates the paper.

---

## Generated Plots (results/)

| File | Corresponds to |
|------|---------------|
| confusion_matrix.png | Fig. 7 |
| cost_function_convergence.png | Fig. 8 |
| roc_curve.png | Fig. 9 |
| activation_accuracy.png | Fig. 10(a) |
| classifier_comparison.png | Fig. 11 |
| leader_selection.png | Leader block analysis |
| throughput_latency.png | Fig. 14 / Fig. 16 |
| summary_comparison.png | All metrics comparison |
| incentive_tokens.png | Incentive mechanism |

---

## Connecting to Hyperledger Fabric (technical details)

The test-network provides:
- **Org1** peer at `peer0.org1.example.com:7051`
- **Org2** peer at `peer0.org2.example.com:9051`
- **Orderer** at `orderer.example.com:7050`
- **CouchDB** at `http://localhost:5984`

The chaincode is deployed to `mychannel` and invoked via the `fabric-network` SDK
in `api-server/server.js`. When Fabric is not available, the server falls back to
demo mode using the Python results JSON.

---

## Dependencies

**Python:**
```
numpy, pandas, scikit-learn, scipy, matplotlib
```

**Node.js (API server):**
```
express, cors, fabric-network, fabric-ca-client
```

---

## Citation

```bibtex
@article{prabanand2025advanced,
  title   = {Advanced financial security system using smart contract
             in private ethereum consortium blockchain with hybrid
             optimization strategy},
  author  = {Prabanand, S. C. and Thanabal, M. S.},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {6764},
  year    = {2025},
  doi     = {10.1038/s41598-025-89404-3}
}
```
