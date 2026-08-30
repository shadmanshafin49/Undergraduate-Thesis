/*
 * measure_consensus.js  —  B2: REAL Fabric consensus measurement
 * ===============================================================
 * Replaces the SIMULATED latency/throughput in
 * db_boa_framework/blockchain/leader_block.py:simulate_consensus_round()
 * (time.sleep + resource arithmetic) with WALL-CLOCK measurements taken
 * against the live Hyperledger Fabric test-network (Raft orderer, 2 orgs,
 * CouchDB), submitted through the fabric-network SDK gateway so there is no
 * per-call `peer` CLI process-spawn overhead — the timed interval is the real
 * propose -> endorse -> order(Raft) -> validate -> commit path.
 *
 * Two honest measurements:
 *   (1) CONSENSUS-ROUND LATENCY — sequential `recordConsensusRound` invokes
 *       (the actual on-chain consensus-round object). submit->commit ms.
 *   (2) SUSTAINED THROUGHPUT — concurrent `updateNodeMetrics` spread across the
 *       10 distinct NODE_* keys (NO shared counter → no MVCC hotspot), so the
 *       number reflects Fabric's ordering/commit throughput, not the chaincode's
 *       counter-serialisation. Concurrency is swept; only VALID (committed)
 *       transactions are counted as goodput.
 *
 * Output: results/fabric_consensus_measured.json  (consumed by the Python side).
 *
 * Usage:  node measure_consensus.js [--quick]
 */

const path = require('path');
const fs   = require('fs');
const { Gateway, Wallets } = require('fabric-network');

const QUICK     = process.argv.includes('--quick');
const CHANNEL   = process.env.CHANNEL   || 'mychannel';
const CHAINCODE = process.env.CHAINCODE || 'db-boa';
const WALLET    = path.join(__dirname, 'wallet');
const CCP_PATH  = process.env.CC_PATH || path.join(
    __dirname, '..', '..', 'fabric', 'fabric-samples', 'test-network',
    'organizations', 'peerOrganizations', 'org1.example.com',
    'connection-org1.json');
const OUT = path.join(__dirname, '..', '..', 'db_boa_framework', 'results',
                      'fabric_consensus_measured.json');

const N_LAT      = QUICK ? 20  : 50;          // sequential latency samples
const TPUT_TOTAL = QUICK ? 60  : 100;         // txns per throughput level
const CONC_SWEEP = QUICK ? [1, 5, 10] : [1, 2, 5, 10, 20, 40];
const N_NODES    = 10;                         // NODE_1 .. NODE_10 (distinct keys)

const pct = (arr, p) => {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(p / 100 * s.length))];
};
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;

async function connect() {
    const ccp    = JSON.parse(fs.readFileSync(CCP_PATH, 'utf8'));
    const wallet = await Wallets.newFileSystemWallet(WALLET);
    if (!(await wallet.get('appUser')))
        throw new Error('appUser identity missing — run registerUser.js');
    const gateway = new Gateway();
    await gateway.connect(ccp, {
        wallet, identity: 'appUser',
        discovery: { enabled: true, asLocalhost: true },
        eventHandlerOptions: { commitTimeout: 300 },
    });
    const network  = await gateway.getNetwork(CHANNEL);
    const contract = network.getContract(CHAINCODE);
    return { gateway, contract };
}

// bounded-concurrency pool: run `tasks` with at most `conc` in flight
async function pool(tasks, conc) {
    let i = 0, ok = 0, fail = 0;
    async function worker() {
        while (i < tasks.length) {
            const t = tasks[i++];
            try { await t(); ok++; } catch (_) { fail++; }
        }
    }
    await Promise.all(Array.from({ length: conc }, worker));
    return { ok, fail };
}

async function main() {
    console.log(`[B2] connecting to Fabric (channel=${CHANNEL}, cc=${CHAINCODE}) …`);
    const { gateway, contract } = await connect();
    console.log('[B2] connected. warming up …');

    const nodeKey = i => `Node-${String(i).padStart(2, '0')}`;   // Node-00 .. Node-09

    // ── warmup (JIT + endorsement discovery + couchdb caches) ────────────────
    for (let w = 0; w < 3; w++)
        await contract.submitTransaction('updateNodeMetrics', nodeKey(0),
            '0.5', '0.5', '0.5');

    // ── (1) CONSENSUS-ROUND LATENCY (sequential, real consensus path) ────────
    console.log(`[B2] measuring consensus-round latency (${N_LAT} sequential rounds) …`);
    const lat = [];
    const stamp = Date.now();
    for (let r = 0; r < N_LAT; r++) {
        const roundNum = `${stamp}${r}`;            // unique → no key collision
        const t0 = process.hrtime.bigint();
        await contract.submitTransaction('recordConsensusRound',
            roundNum, nodeKey(r % N_NODES), '50', '2', 'true', '0', '0');
        const ms = Number(process.hrtime.bigint() - t0) / 1e6;
        lat.push(ms);
        if ((r + 1) % 10 === 0) console.log(`     round ${r + 1}/${N_LAT}  ${ms.toFixed(1)} ms`);
    }
    const latency = {
        n: N_LAT, mean_ms: mean(lat), p50_ms: pct(lat, 50),
        p95_ms: pct(lat, 95), min_ms: Math.min(...lat), max_ms: Math.max(...lat),
    };
    console.log(`[B2] LATENCY  mean=${latency.mean_ms.toFixed(1)}ms  `
        + `p50=${latency.p50_ms.toFixed(1)}  p95=${latency.p95_ms.toFixed(1)}ms`);

    // ── (2) SUSTAINED THROUGHPUT (concurrency sweep, contention-free keys) ───
    console.log('[B2] measuring throughput (concurrency sweep, distinct NODE_* keys) …');
    const throughput = [];
    for (const conc of CONC_SWEEP) {
        const tasks = [];
        for (let k = 0; k < TPUT_TOTAL; k++) {
            const nodeId = nodeKey(k % N_NODES);
            const v = (0.3 + (k % 7) * 0.1).toFixed(3);
            tasks.push(() => contract.submitTransaction(
                'updateNodeMetrics', nodeId, v, v, v));
        }
        const t0 = process.hrtime.bigint();
        const { ok, fail } = await pool(tasks, conc);
        const elapsed = Number(process.hrtime.bigint() - t0) / 1e9;
        const tps = ok / elapsed;
        throughput.push({ concurrency: conc, total: TPUT_TOTAL, committed: ok,
                          failed: fail, elapsed_s: elapsed, tps });
        console.log(`     conc=${String(conc).padStart(2)}  committed=${ok}/${TPUT_TOTAL}`
            + `  failed=${fail}  ${elapsed.toFixed(2)}s  → ${tps.toFixed(1)} tps`);
    }
    const peak = throughput.reduce((a, b) => b.tps > a.tps ? b : a);

    const result = {
        measured_on: new Date().toISOString(),
        network: { orderer: 'etcdraft (Raft)', orgs: 2, peers: 2,
                   state_db: 'couchdb', sdk: 'fabric-network gateway',
                   note: 'real submit->endorse->order->commit; CLI spawn excluded' },
        latency_ms: latency,
        throughput: throughput,
        peak_tps: peak.tps,
        peak_tps_concurrency: peak.concurrency,
        quick: QUICK,
    };
    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    fs.writeFileSync(OUT, JSON.stringify(result, null, 2));
    console.log(`[B2] PEAK throughput ${peak.tps.toFixed(1)} tps @ conc=${peak.concurrency}`);
    console.log(`[B2] wrote ${OUT}`);

    await gateway.disconnect();
}

main().catch(e => { console.error('[B2] ERROR:', e); process.exit(1); });
