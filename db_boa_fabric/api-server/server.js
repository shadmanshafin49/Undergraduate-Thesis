'use strict';

/**
 * api-server/server.js
 * ====================
 * REST API Server — bridges the React dashboard to:
 *   1. Hyperledger Fabric ledger (via fabric-network SDK)
 *   2. Python DB-BOA engine      (via child_process / python3)
 *
 * Endpoints:
 *   GET  /health                    — server health check
 *   GET  /api/nodes                 — all node statuses from ledger
 *   GET  /api/transactions          — all transactions from ledger
 *   GET  /api/fraud-stats           — aggregated fraud statistics
 *   GET  /api/leader-history        — past leader elections
 *   GET  /api/consensus-history     — past consensus rounds
 *   POST /api/run-db-boa            — trigger full Python DB-BOA pipeline
 *   POST /api/submit-transaction    — submit & classify a financial transaction
 *   GET  /api/db-boa-results        — latest results from Python engine
 *   GET  /api/plots/:filename       — serve generated plot images
 *
 * The server uses Server-Sent Events (SSE) to push real-time
 * terminal log lines to the dashboard as DB-BOA runs.
 *   GET  /api/sse/logs              — SSE stream of live log lines
 *
 * Usage:
 *   npm install
 *   node enrollAdmin.js       (first time only)
 *   node registerUser.js      (first time only)
 *   npm start
 */

const express    = require('express');
const cors       = require('cors');
const path       = require('path');
const fs         = require('fs');
const { spawn }  = require('child_process');

// ── Fabric SDK imports ────────────────────────────────────────────────────────
// These are dynamically required so the server can still start even if
// fabric-network isn't installed (pure Python / demo mode).
let Gateway, Wallets;
try {
    ({ Gateway, Wallets } = require('fabric-network'));
} catch (_) {
    console.warn('[WARN]  fabric-network not installed — running in DEMO mode');
}

// ── Config ────────────────────────────────────────────────────────────────────
const PORT         = process.env.PORT || 3001;
const CHANNEL      = process.env.CHANNEL      || 'mychannel';
const CHAINCODE    = process.env.CHAINCODE     || 'db-boa';
const WALLET_PATH  = process.env.WALLET_PATH  || path.join(__dirname, 'wallet');
const CC_PATH      = process.env.CC_PATH       ||
    path.join(__dirname, '..', '..', 'fabric', 'fabric-samples',
              'test-network', 'organizations', 'peerOrganizations',
              'org1.example.com', 'connection-org1.json');
const PYTHON_SCRIPT = path.join(__dirname, '..', '..', 'db_boa_framework', 'main.py');
const RESULTS_DIR   = path.join(__dirname, '..', '..', 'db_boa_framework', 'results');

const app = express();
app.use(cors());
app.use(express.json());

// ── In-memory state (populated after Python run / Fabric queries) ─────────────
let latestResults   = null;
let sseClients      = [];    // SSE subscriber list
let fabricConnected = false;
let contract        = null;

// ─── SSE helpers ──────────────────────────────────────────────────────────────

function broadcastLog(line) {
    const data = `data: ${JSON.stringify({ type: 'log', line })}\n\n`;
    sseClients.forEach(c => c.write(data));
}

function broadcastEvent(type, payload) {
    const data = `data: ${JSON.stringify({ type, payload })}\n\n`;
    sseClients.forEach(c => c.write(data));
}

// ─── Fabric connection ────────────────────────────────────────────────────────

async function connectFabric() {
    if (!Gateway) return false;
    try {
        const ccpPath = CC_PATH;
        if (!fs.existsSync(ccpPath)) {
            console.warn(`[WARN]  Connection profile not found: ${ccpPath}`);
            return false;
        }
        const ccp    = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));
        const wallet = await Wallets.newFileSystemWallet(WALLET_PATH);
        const identity = await wallet.get('appUser');
        if (!identity) {
            console.warn('[WARN]  Identity "appUser" not found in wallet. Run registerUser.js');
            return false;
        }
        const gateway = new Gateway();
        await gateway.connect(ccp, {
            wallet,
            identity      : 'appUser',
            discovery     : { enabled: true, asLocalhost: true },
        });
        const network = await gateway.getNetwork(CHANNEL);
        contract      = network.getContract(CHAINCODE);
        fabricConnected = true;
        console.log(`[INFO]  Connected to Fabric: channel=${CHANNEL}  chaincode=${CHAINCODE}`);
        return true;
    } catch (err) {
        console.warn(`[WARN]  Fabric connection failed: ${err.message}`);
        return false;
    }
}

// ─── Helper: Fabric query with demo fallback ──────────────────────────────────

async function fabricQuery(fnName, args = []) {
    if (contract) {
        try {
            const result = await contract.evaluateTransaction(fnName, ...args);
            return JSON.parse(result.toString());
        } catch (err) {
            console.error(`[Fabric] query ${fnName} failed:`, err.message);
        }
    }
    // Demo fallback — return from in-memory results
    return demoFallback(fnName);
}

async function fabricSubmit(fnName, args = []) {
    if (contract) {
        try {
            const result = await contract.submitTransaction(fnName, ...args);
            return JSON.parse(result.toString());
        } catch (err) {
            console.error(`[Fabric] submit ${fnName} failed:`, err.message);
        }
    }
    return demoFallback(fnName);
}

function demoFallback(fnName) {
    if (!latestResults) return { demo: true, message: 'Run DB-BOA first' };
    const r = latestResults;
    switch (fnName) {
        case 'getNodeStatus':
            return r.nodes || [];
        case 'getFraudStats':
            return r.fraudStats || {};
        case 'getLeaderHistory':
            return r.leaderHistory || [];
        case 'getConsensusHistory':
            return r.consensusRounds || [];
        case 'getAllTransactions':
            return r.transactions || [];
        case 'getFederationHistory':
            return r.fed_rounds || [];
        case 'getOrgModels':
            return r.org_models || [];
        default:
            return { demo: true };
    }
}

// ─── Routes ───────────────────────────────────────────────────────────────────

// Health
app.get('/health', (req, res) => res.json({
    status         : 'OK',
    fabricConnected,
    latestResults  : !!latestResults,
    timestamp      : new Date().toISOString(),
}));

// SSE — live log stream
app.get('/api/sse/logs', (req, res) => {
    res.setHeader('Content-Type',  'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection',    'keep-alive');
    res.flushHeaders();
    sseClients.push(res);
    req.on('close', () => {
        sseClients = sseClients.filter(c => c !== res);
    });
});

// Node status
app.get('/api/nodes', async (req, res) => {
    try {
        const data = await fabricQuery('getNodeStatus');
        res.json(data);
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// Transactions
app.get('/api/transactions', async (req, res) => {
    try {
        const data = await fabricQuery('getAllTransactions');
        res.json(data);
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// Fraud statistics
app.get('/api/fraud-stats', async (req, res) => {
    try {
        const data = await fabricQuery('getFraudStats');
        res.json(data);
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// Leader history
app.get('/api/leader-history', async (req, res) => {
    try {
        const data = await fabricQuery('getLeaderHistory');
        res.json(data);
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// Consensus history
app.get('/api/consensus-history', async (req, res) => {
    try {
        const data = await fabricQuery('getConsensusHistory');
        res.json(data);
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// Federation history
app.get('/api/federation-history', async (req, res) => {
    try {
        const data = await fabricQuery('getFederationHistory');
        // Fallback: also try in-memory results
        if (!data || (Array.isArray(data) && data.length === 0)) {
            return res.json(latestResults?.fed_rounds || []);
        }
        res.json(data);
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// Org model metrics
app.get('/api/org-models', async (req, res) => {
    try {
        const data = await fabricQuery('getOrgModels');
        res.json(data);
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// Trigger a federation round only
app.post('/api/run-federation-round', async (req, res) => {
    const roundNum = req.body.round_num || 1;
    broadcastEvent('federation_start', {
        round_num: roundNum,
        orgs: ['BankA', 'BankB', 'BankC'],
    });
    res.json({ status: 'started', round_num: roundNum });

    const args = ['python3', PYTHON_SCRIPT, '--fed-only', '--round', String(roundNum)];
    const proc = spawn(args[0], args.slice(1), {
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    proc.stdout.on('data', chunk => {
        chunk.toString().split('\n').forEach(line => {
            if (line.trim()) broadcastLog(line);
        });
    });
    proc.stderr.on('data', chunk => {
        chunk.toString().split('\n').forEach(line => {
            if (line.trim()) broadcastLog(`[STDERR] ${line}`);
        });
    });

    proc.on('close', async code => {
        if (code === 0) {
            const jsonPath = path.join(RESULTS_DIR, 'db_boa_results.json');
            if (fs.existsSync(jsonPath)) {
                try {
                    const r = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
                    latestResults = enrichResults(r);
                    const fedRounds = latestResults.fed_rounds || [];
                    const lastRound = fedRounds[fedRounds.length - 1];
                    if (lastRound) {
                        broadcastEvent('federation_weights', {
                            round_num   : lastRound.round_num,
                            weights     : lastRound.org_contributions,
                            best_fitness: lastRound.best_fitness,
                        });
                        broadcastEvent('federation_done', {
                            round_num      : lastRound.round_num,
                            accuracy_deltas: lastRound.accuracy_deltas || {},
                        });
                        await writeFederationToFabric(lastRound);
                    }
                } catch (e) {
                    broadcastLog(`[ERROR] ${e.message}`);
                }
            }
        }
    });
});

// Latest DB-BOA results (from JSON file written by Python)
app.get('/api/db-boa-results', (req, res) => {
    if (!latestResults) return res.json({ ready: false });
    res.json({ ready: true, ...latestResults });
});

// Serve plot images
app.get('/api/plots/:filename', (req, res) => {
    const resolved   = path.resolve(RESULTS_DIR, req.params.filename);
    const resultsAbs = path.resolve(RESULTS_DIR);
    if (!resolved.startsWith(resultsAbs + path.sep) && resolved !== resultsAbs) {
        return res.status(400).json({ error: 'Invalid filename' });
    }
    if (!fs.existsSync(resolved)) return res.status(404).json({ error: 'Plot not found' });
    res.sendFile(resolved);
});

// List available plots
app.get('/api/plots', (req, res) => {
    if (!fs.existsSync(RESULTS_DIR)) return res.json([]);
    const files = fs.readdirSync(RESULTS_DIR).filter(f => f.endsWith('.png'));
    res.json(files);
});

// ── POST /api/run-db-boa ─────────────────────────────────────────────────────
/**
 * Trigger the Python DB-BOA pipeline.
 * Streams stdout line-by-line to SSE clients in real time.
 * When done, reads results JSON and writes key records to Fabric ledger.
 */
app.post('/api/run-db-boa', async (req, res) => {
    const quick = req.body.quick !== false;   // default: quick mode for web demo

    broadcastEvent('pipeline_start', { quick, timestamp: new Date().toISOString() });
    res.json({ status: 'started', message: 'DB-BOA pipeline started. Watch SSE stream.' });

    const args = ['python3', PYTHON_SCRIPT];
    if (quick) args.push('--quick');

    const proc = spawn(args[0], args.slice(1), {
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    proc.stdout.on('data', chunk => {
        chunk.toString().split('\n').forEach(line => {
            if (line.trim()) broadcastLog(line);
        });
    });

    proc.stderr.on('data', chunk => {
        chunk.toString().split('\n').forEach(line => {
            if (line.trim()) broadcastLog(`[STDERR] ${line}`);
        });
    });

    proc.on('close', async code => {
        if (code === 0) {
            broadcastLog('✔  DB-BOA pipeline finished successfully.');

            // Read results JSON written by Python
            const jsonPath = path.join(RESULTS_DIR, 'db_boa_results.json');
            if (fs.existsSync(jsonPath)) {
                try {
                    const raw = fs.readFileSync(jsonPath, 'utf8');
                    const r   = JSON.parse(raw);
                    latestResults = enrichResults(r);
                    broadcastEvent('results_ready', latestResults);

                    // Write key results to Fabric ledger
                    await writeTofabric(latestResults);
                } catch (e) {
                    broadcastLog(`[ERROR] Could not parse results: ${e.message}`);
                }
            }
        } else {
            broadcastLog(`✗  DB-BOA pipeline exited with code ${code}`);
            broadcastEvent('pipeline_error', { code });
        }
    });
});

// ── POST /api/submit-transaction ──────────────────────────────────────────────
/**
 * Submit a single transaction, classify it with the trained ADTCN model,
 * and write both the transaction and the fraud result to the ledger.
 */
app.post('/api/submit-transaction', async (req, res) => {
    try {
        const { txnId, amount, senderId, receiverId, features } = req.body;
        if (!txnId || !amount) return res.status(400).json({ error: 'txnId and amount required' });

        const ts = new Date().toISOString();

        // Submit raw transaction
        await fabricSubmit('submitTransaction', [
            txnId, String(amount), senderId || 'anon',
            receiverId || 'anon', ts,
            JSON.stringify(features || []),
        ]);

        // Simple rule-based demo classification (real version calls Python predict)
        const isFraud    = parseFloat(amount) > 5000 || Math.random() < 0.05;
        const fraudScore = isFraud ? (0.7 + Math.random() * 0.3).toFixed(4)
                                   : (Math.random() * 0.15).toFixed(4);

        // Chaincode signature: recordFraudResult(ctx, txnId, orgName, isFraud, fraudScore)
        await fabricSubmit('recordFraudResult', [
            txnId, 'DemoNode', String(isFraud), fraudScore,
        ]);

        broadcastEvent('transaction_classified', { txnId, isFraud, fraudScore });

        res.json({ txnId, isFraud, fraudScore, timestamp: ts });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// ─── Write results to Fabric ──────────────────────────────────────────────────

async function writeTofabric(r) {
    if (!contract) return;

    const writeDone = (fnName, success, err) => broadcastEvent('fabric_write_done', {
        function_name: fnName, success, error: err || undefined,
    });

    try {
        // Write 1: Hyperparameters
        if (r.optimal_hyperparams) {
            const hp = r.optimal_hyperparams;
            const stats = r.db_boa_stats || {};
            await fabricSubmit('recordHyperparams', [
                String(hp.hidden_neurons  || 0),
                String(hp.epoch_count     || 0),
                String(hp.steps_per_epoch || 0),
                String(Math.abs(stats.Min || 0)),
                JSON.stringify(stats),
            ]);
            writeDone('recordHyperparams', true);
        }

        // Write 2: Leader selection
        if (r.leader_node !== undefined) {
            const leaderStats = r.db_boa_stats || {};
            await fabricSubmit('recordLeaderSelection', [
                '1',
                `Node-${String(r.leader_node).padStart(2, '0')}`,
                String(r.leader_ct   || 0),
                String(r.leader_cc   || 0),
                String(r.leader_ms   || 0),
                String(r.leader_cost || 0),
                JSON.stringify(r.db_boa_history || []),
                JSON.stringify(leaderStats),
            ]);
            writeDone('recordLeaderSelection', true);
        }

        // Write 3: Org model metrics (per-org accuracy before federation)
        const orgs = ['BankA', 'BankB', 'BankC'];
        for (let i = 0; i < orgs.length; i++) {
            const org = orgs[i];
            const acc = r.metrics?.Accuracy || 0;
            const mcc = r.metrics?.MCC      || 0;
            await fabricSubmit('submitModelMetrics', [
                org, '1',
                String(acc), String(mcc),
                '1000',                      // placeholder sample count
                `hash_${org}_round1`,        // placeholder weight hash
                JSON.stringify({}),
            ]);
        }
        writeDone('submitModelMetrics', true);

        // Write 4: Consensus rounds
        if (Array.isArray(r.consensus_rounds)) {
            for (const rnd of r.consensus_rounds) {
                await fabricSubmit('recordConsensusRound', [
                    String(rnd.round),
                    `Node-${String(rnd.leader).padStart(2, '0')}`,
                    String(rnd.n_txns),
                    String(rnd.n_endorsed),
                    String(rnd.consensus_ok),
                    String(rnd.latency_ms),
                    String(rnd.throughput),
                ]);
            }
            writeDone('recordConsensusRound', true);
        }

        // Write 5: Federation rounds
        if (Array.isArray(r.fed_rounds)) {
            for (const fr of r.fed_rounds) {
                await writeFederationToFabric(fr);
            }
        }

        broadcastLog('[FABRIC] All results written to ledger ✔');
    } catch (e) {
        broadcastLog(`[FABRIC] Write error: ${e.message}`);
        broadcastEvent('fabric_write_done', { function_name: 'batch', success: false, error: e.message });
    }
}

async function writeFederationToFabric(fr) {
    if (!contract) return;
    try {
        // aggregationWeightsJson needs {org: weight} dict form for on-chain reward distribution
        const aggregationWeights = fr.org_contributions || {};
        const orgContributions   = fr.org_contributions || {};
        const hash               = fr.global_weights_hash || `fed_hash_round_${fr.round_num}`;
        await fabricSubmit('recordFederationRound', [
            String(fr.round_num),
            JSON.stringify(aggregationWeights),
            JSON.stringify(orgContributions),
            JSON.stringify(fr.db_boa_history || []),
            String(fr.best_fitness || 0),
            hash,
        ]);
        broadcastEvent('fabric_write_done', { function_name: 'recordFederationRound', success: true });
    } catch (e) {
        broadcastEvent('fabric_write_done', {
            function_name: 'recordFederationRound', success: false, error: e.message,
        });
    }
}

// ─── Enrich Python results for frontend ───────────────────────────────────────

function enrichResults(r) {
    const metrics = r.metrics || {};

    // Build per-node token balances: leader earns more
    const fedRounds   = r.fed_rounds || [];
    const orgTokenMap = { BankA: 100, BankB: 100, BankC: 100 };
    for (const fr of fedRounds) {
        for (const [org, w] of Object.entries(fr.org_contributions || {})) {
            orgTokenMap[org] = (orgTokenMap[org] || 100) + Math.floor(20 * w);
        }
    }

    const nodes = Array.from({ length: 10 }, (_, i) => {
        const nRounds  = r.consensus_rounds?.length || 1;
        const isLeader = i === r.leader_node;
        const baseTokens = isLeader ? 100 + 15 * nRounds : 100;
        // Map node 0,2,4 to BankA,B,C federation rewards
        const orgMap  = { 0: 'BankA', 2: 'BankB', 4: 'BankC' };
        const orgName = orgMap[i];
        const tokens  = orgName ? orgTokenMap[orgName] : baseTokens;
        return {
            nodeId     : `Node-${String(i).padStart(2, '0')}`,
            org        : i % 2 === 0 ? 'Org1MSP' : 'Org2MSP',
            tokens,
            reputation : (1.0 + i * 0.05).toFixed(4),
            isLeader,
        };
    });

    return {
        ...r,
        nodes,
        fraudStats: {
            totalTransactions : 4000,
            totalFraud        : Math.round(4000 * 0.05),
            totalNormal       : Math.round(4000 * 0.95),
            accuracy          : metrics.Accuracy,
            mcc               : metrics.MCC,
            fpr               : metrics.FPR,
        },
        metrics,
        leaderHistory: [{
            roundId      : 1,
            winnerNodeId : `Node-${String(r.leader_node).padStart(2, '0')}`,
            totalCost    : r.leader_cost,
        }],
        ready: true,
    };
}

// ─── Dashboard — single-file HTML at api-server/index.html ───────────────────
// The manual specifies db_boa_fabric/dashboard/index.html; the file is kept
// alongside the server for convenience. Both paths are served below.
const DASHBOARD_FILE = path.join(__dirname, 'index.html');
app.get('/',          (req, res) => res.sendFile(DASHBOARD_FILE));
app.get('/dashboard', (req, res) => res.sendFile(DASHBOARD_FILE));

// ─── Start ────────────────────────────────────────────────────────────────────
async function start() {
    await connectFabric();
    app.listen(PORT, () => {
        console.log(`\n[API]   DB-BOA REST API running on http://localhost:${PORT}`);
        console.log(`[API]   Fabric connected: ${fabricConnected}`);
        console.log(`[API]   Python script   : ${PYTHON_SCRIPT}`);
        console.log(`[API]   Results dir     : ${RESULTS_DIR}\n`);
    });
}

start().catch(console.error);
