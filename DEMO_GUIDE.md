# DB-BOA / FL-ADTCN — Supervisor Demo Guide

This guide is tailored to **this machine**, where the full stack has already been set up
and verified end-to-end on the **real ULB Credit Card Fraud dataset** (284,807
transactions). It covers both layers of the project:

1. **Python ML pipeline** — DB-BOA + ADTCN + federated learning (DP + Krum + Shapley) + Byzantine attack.
2. **Hyperledger Fabric blockchain** — `DBBOAContract` chaincode + REST API + web dashboard.

---

## 0. One-time environment notes (already done on this machine)

These were needed to make it run here; you do **not** need to repeat them unless you start fresh:

- Dataset symlinked into place: `ln -sf ../creditcard.csv datasets/creditcard.csv`
- Fabric binaries+samples installed: `./fabric/install-fabric.sh -f 2.5.10 -c 1.5.13 binary samples`
- **Docker 29 fix:** `/etc/systemd/system/docker.service.d/min-api.conf` sets
  `DOCKER_MIN_API_VERSION=1.24` (Fabric 2.5's peer needs the old Docker API).
- Chaincode runs as **CCaaS** (Chaincode-as-a-Service), because Docker 29 can't do the
  peer-side chaincode image build. This is transparent to the demo.

If the machine was rebooted, see **§4 Restart from scratch**.

---

## 1. Quick "is everything up?" check

```bash
# Fabric containers (expect 10: CAs, couchdb x2, orderer, 2 peers, 2 ccaas chaincode)
docker ps --format '{{.Names}}' | sort

# API server + Fabric connection
curl -s http://localhost:3001/health
# → {"status":"OK","fabricConnected":true,...}

# Live ledger query (the 10 consortium nodes)
curl -s http://localhost:3001/api/nodes | head -c 200
```

If `fabricConnected:true`, you are ready to demo. Open the dashboard:
**http://localhost:3001/**

---

## 2. The demo flow (recommended order, ~10–15 min)

### Part A — The blockchain dashboard (live)
1. Open **http://localhost:3001/** in a browser → "DB-BOA Financial Security Dashboard".
2. Show the **consortium nodes** (10 nodes, tokens=100, reputation) — these are read live
   from the Hyperledger Fabric ledger via the `DBBOAContract` chaincode.
3. **Submit a transaction** from the dashboard (or via API):
   ```bash
   curl -s -X POST http://localhost:3001/api/submit-transaction \
     -H 'Content-Type: application/json' \
     -d '{"txnId":"DEMO-TX-001","amount":4200.50,"isFraud":true,"fraudScore":0.97}'
   ```
   Then show it persisted on-chain:
   ```bash
   curl -s http://localhost:3001/api/transactions | head -c 300
   ```
4. Talking point: *raw data never leaves a bank; only verdicts/metrics/weights are written
   to the immutable ledger; incentive rules are enforced by chaincode no single org controls.*

### Part B — The ML pipeline (live, fast mode)
Run the full research pipeline live in **quick mode** (~3–5 min on CPU). This trains the
ADTCN on the real dataset, runs DB-BOA, the DP+Krum+Shapley federation, and the Byzantine
attack:
```bash
cd db_boa_framework
python3 main.py --quick --attack
```
Narrate the phases as they print:
- **Phase 1** DB-BOA leader selection (DBOA/BOA adaptive switching visible in the log).
- **Phase 2** DB-BOA hyperparameter search (2-D: filters, steps/epoch).
- **Phase 3–4** ADTCN training + evaluation on the held-out test set (Accuracy / MCC / FPR…).
- **Phase 7** Federation: `DP weight sharing` → `Krum selected` → `Shapley values` →
  `Aggregation weights`. **This is the Option-A novelty** — Shapley contribution weights
  drive the on-chain token split.
- **Phase 8** Byzantine attack: BankC always reports fraud → Shapley assigns it a near-zero
  weight.

> For the **strong numbers** (full 30-epoch training), use `python3 main.py --attack`
> *without* `--quick`. It takes ~45–60 min on CPU, so run it **before** the meeting and
> show the saved results, then run `--quick` live for speed. Outputs land in
> `db_boa_framework/results/` (JSON + 12 PNG plots).

### Part C — Tie them together
- Show the generated plots in `db_boa_framework/results/` (confusion matrix, ROC,
  federation weights, token balances, leader selection, convergence).
- Point out that the dashboard + ledger is where these results are recorded and incentives
  enforced.

---

## 3. Handy commands during the demo

```bash
# Query the chaincode directly (proves it's a real ledger, not a mock)
cd fabric/fabric-samples/test-network
export PATH=$PATH:$PWD/../bin FABRIC_CFG_PATH=$PWD/../config
source scripts/envVar.sh; setGlobals 1
peer chaincode query -C mychannel -n db-boa -c '{"function":"getNodeStatus","Args":[]}' | head -c 300

# API endpoints (all live from the ledger)
curl -s http://localhost:3001/api/leader-history
curl -s http://localhost:3001/api/consensus-history
curl -s http://localhost:3001/api/federation-history
curl -s http://localhost:3001/api/plots          # list generated plot files
```

---

## 4. Restart from scratch (if rebooted / containers gone)

```bash
# 0. Docker must be running with the API-version fix already in place (it persists via systemd).
sudo systemctl restart docker        # only if needed; the min-api.conf drop-in persists

# 1. Bring up the network + CAs + CouchDB + create channel
cd fabric/fabric-samples/test-network
export PATH=$PATH:$PWD/../bin
./network.sh down
./network.sh up createChannel -ca -s couchdb

# 2. Deploy the chaincode AS A SERVICE (this is the key step on Docker 29)
./network.sh deployCCAAS -ccn db-boa -ccp ../db-boa-ccaas -cci initLedger -ccv 1 -ccs 1

# 3. Enroll API identities + start the API server
cd ../../../db_boa_fabric/api-server
rm -rf wallet
node enrollAdmin.js && node registerUser.js
node server.js          # http://localhost:3001   (Ctrl+C to stop)
```

> Note: do **not** use `./network.sh deployCC` (the normal path) — it tries to build the
> chaincode image inside the peer, which fails on Docker 29. Always use **deployCCAAS**.

If `initLedger` reports an init error during deployCCAAS, seed it manually afterward:
```bash
cd fabric/fabric-samples/test-network && source scripts/envVar.sh && setGlobals 1
peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls \
  --cafile "$PWD/organizations/ordererOrganizations/example.com/tlsca/tlsca.example.com-cert.pem" \
  -C mychannel -n db-boa \
  --peerAddresses localhost:7051 --tlsRootCertFiles "$PWD/organizations/peerOrganizations/org1.example.com/tlsca/tlsca.org1.example.com-cert.pem" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "$PWD/organizations/peerOrganizations/org2.example.com/tlsca/tlsca.org2.example.com-cert.pem" \
  -c '{"function":"initLedger","Args":[]}'
```

---

## 5. Honest talking points (for supervisor Q&A)

Be upfront about scope — examiners reward this:
- The **consensus latency/throughput numbers are simulated** (the Python `leader_block.py`
  models them); the real Fabric network demonstrates the architecture and on-chain logic.
- The **three "banks" are volume-splits of one bank's ULB dataset** (controlled simulation,
  not true cross-institution data).
- **Differential privacy at ε=1.0** is a deliberately tight budget that degrades the shared
  weights — a real privacy/utility trade-off to discuss (DP-SGD / larger ε is future work).
- The deployed Fabric network is **2 orgs** (Org1/Org2); the chaincode/data model supports 3.
- Federation contribution weights use **exact Shapley values + Krum + DP** (the real
  Option-A novelty), with DB-BOA driving hyperparameters and leader selection.

---

## 6. Known limitation discovered during setup

The Byzantine-attack token side can **reward** the attacker when the honest models are
under-trained (quick mode), because consensus over mostly-normal transactions plus low
precision lets an always-fraud verdict blend in. Use a **full (non-quick) run** for the
attack demo, where the Shapley side clearly isolates the attacker (near-zero weight). See
`final_report_data/00_report_vs_code_divergences.md` for the full list of report-vs-code
items to reconcile.
