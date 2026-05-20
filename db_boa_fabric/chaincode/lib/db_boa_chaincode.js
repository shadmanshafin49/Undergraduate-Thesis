'use strict';

/**
 * db_boa_chaincode.js
 * ===================
 * Hyperledger Fabric Chaincode for DB-BOA Financial Security System
 * (Complete rebuild — includes federation functions)
 *
 * Data types stored on the ledger (all have a docType field):
 *
 *   node            NODE_{id}              — peer node metrics, tokens, reputation
 *   transaction     TXN_{id}               — anonymised transaction + verdicts
 *   leaderElection  LEADER_{roundId}       — DB-BOA Job 2 output per round
 *   consensusRound  ROUND_{roundNum}       — block consensus metrics
 *   federationRound FED_{roundNum}         — Shapley-weighted aggregation output + reward record
 *   orgModel        MODEL_{org}_{round}    — per-org accuracy & weight hash
 *   hyperparams     HYPERPARAMS            — DB-BOA Job 1 optimal hyperparameters
 *   counters        COUNTERS               — running totals
 *
 * Incentive rules (hardcoded, part of consortium agreement):
 *   Fraud confirmed by consensus            +10 tokens
 *   Fraud confirmed AND latency < 300 ms    +15 tokens (bonus replaces base)
 *   Selected as leader, round succeeds      +10 tokens
 *   Verdict disputed by majority            –2 tokens
 *   Consensus round fails (applied to leader) –2 tokens
 *   Federation participation                +20 tokens shared by Shapley contribution weight
 *   Reputation on success                   +0.02
 *   Reputation on failure                   –0.05
 *
 * IMPORTANT — determinism rules:
 *   • Never use Math.random() inside state-mutating functions
 *   • Never use new Date() inside state-mutating functions
 *   • Use _txTime(ctx) which reads ctx.stub.getTxTimestamp() — agreed by all
 *     peers for the same transaction, therefore deterministic.
 */

const { Contract } = require('fabric-contract-api');

const BASE_FEDERATION_REWARD = 20;   // tokens split by aggregation weights

/** _txTime — deterministic timestamp from the transaction envelope.
 *  ctx.stub.getTxTimestamp() is set by the orderer and is identical on
 *  every endorsing peer, satisfying Fabric's determinism requirement.
 */
function _txTime(ctx) {
    const ts = ctx.stub.getTxTimestamp();
    return new Date(ts.seconds.low * 1000).toISOString();
}

class DBBOAContract extends Contract {

    // ═══════════════════════════════════════════════════════════════════════
    // INITIALISATION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * initLedger — seed 10 nodes and create COUNTERS document.
     * Called once on chaincode deployment.
     * Uses fixed (deterministic) resource values so the ledger is reproducible.
     */
    async initLedger(ctx) {
        console.log('=== initLedger: DB-BOA Financial Security ===');

        // Fixed node profiles (deterministic; API server will update via updateNodeMetrics)
        const nodeProfiles = [
            { ct: 0.32, cc: 0.18, ms: 0.41 },
            { ct: 0.51, cc: 0.29, ms: 0.63 },
            { ct: 0.44, cc: 0.12, ms: 0.55 },
            { ct: 0.27, cc: 0.35, ms: 0.38 },
            { ct: 0.61, cc: 0.22, ms: 0.71 },
            { ct: 0.38, cc: 0.41, ms: 0.29 },
            { ct: 0.19, cc: 0.14, ms: 0.47 },
            { ct: 0.55, cc: 0.33, ms: 0.58 },
            { ct: 0.42, cc: 0.25, ms: 0.34 },
            { ct: 0.67, cc: 0.38, ms: 0.72 },
        ];

        for (let i = 0; i < nodeProfiles.length; i++) {
            const p    = nodeProfiles[i];
            const node = {
                docType         : 'node',
                nodeId          : `Node-${String(i).padStart(2, '0')}`,
                org             : i % 2 === 0 ? 'Org1MSP' : 'Org2MSP',
                tokens          : 100,
                reputation      : (1.0 + i * 0.05).toFixed(4),
                ct              : p.ct,
                cc              : p.cc,
                ms              : p.ms,
                roundsAsLeader  : 0,
                consensusWins   : 0,
                consensusFails  : 0,
                updatedAt       : '2026-01-01T00:00:00.000Z',
            };
            await ctx.stub.putState(
                `NODE_${node.nodeId}`,
                Buffer.from(JSON.stringify(node))
            );
            console.log(`  Registered ${node.nodeId} (${node.org})`);
        }

        const counters = {
            docType           : 'counters',
            totalTransactions : 0,
            totalFraud        : 0,
            totalNormal       : 0,
            consensusRounds   : 0,
            leaderElections   : 0,
            federationRounds  : 0,
        };
        await ctx.stub.putState('COUNTERS', Buffer.from(JSON.stringify(counters)));

        console.log('initLedger complete — 10 nodes registered.');
        return JSON.stringify({ status: 'OK', nodes: nodeProfiles.length });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NODE MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * updateNodeMetrics — update CT, CC, MS for a node.
     * Called by API server at the start of each round with real measurements.
     */
    async updateNodeMetrics(ctx, nodeId, ct, cc, ms) {
        const buf = await ctx.stub.getState(`NODE_${nodeId}`);
        if (!buf || buf.length === 0) throw new Error(`Node ${nodeId} not found.`);
        const node      = JSON.parse(buf.toString());
        node.ct         = parseFloat(ct);
        node.cc         = parseFloat(cc);
        node.ms         = parseFloat(ms);
        node.updatedAt  = _txTime(ctx);
        await ctx.stub.putState(`NODE_${nodeId}`, Buffer.from(JSON.stringify(node)));
        return JSON.stringify(node);
    }

    /**
     * updateIncentive — credit / debit tokens and reputation to a node.
     * reason: 'CONSENSUS_SUCCESS' | 'PERFORMANCE_BONUS' | 'PENALTY' | 'FEDERATION_REWARD'
     */
    async updateIncentive(ctx, nodeId, deltaTokens, deltaRep, reason) {
        const buf = await ctx.stub.getState(`NODE_${nodeId}`);
        if (!buf || buf.length === 0) throw new Error(`Node ${nodeId} not found.`);

        const node = JSON.parse(buf.toString());
        const dt   = parseInt(deltaTokens, 10);
        const dr   = parseFloat(deltaRep);

        node.tokens     = Math.max(0, node.tokens + dt);
        node.reputation = Math.min(2.0, Math.max(0.5,
                            parseFloat(node.reputation) + dr)).toFixed(4);

        if (reason === 'CONSENSUS_SUCCESS' || reason === 'PERFORMANCE_BONUS') {
            node.consensusWins += 1;
        } else if (reason === 'PENALTY') {
            node.consensusFails += 1;
        }

        node.updatedAt = _txTime(ctx);
        await ctx.stub.putState(`NODE_${nodeId}`, Buffer.from(JSON.stringify(node)));

        ctx.stub.setEvent('IncentiveApplied', Buffer.from(JSON.stringify({
            nodeId, deltaTokens: dt, deltaRep: dr, reason,
            newTokens: node.tokens, newReputation: node.reputation,
        })));

        return JSON.stringify(node);
    }

    /** getNodeStatus — return all docType='node' documents. */
    async getNodeStatus(ctx) {
        return await this._queryByDocType(ctx, 'node');
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TRANSACTION AND FRAUD DETECTION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * submitTransaction — record an anonymised transaction on the ledger.
     * featuresJson: JSON string of float array (no PII — only feature vector).
     */
    async submitTransaction(ctx, txnId, amount, senderId, receiverId,
                             timestamp, featuresJson) {
        const existing = await ctx.stub.getState(`TXN_${txnId}`);
        if (existing && existing.length > 0) {
            throw new Error(`Transaction ${txnId} already exists.`);
        }

        const txn = {
            docType    : 'transaction',
            txnId,
            amount     : parseFloat(amount),
            senderId,
            receiverId,
            timestamp,
            features   : JSON.parse(featuresJson),
            status     : 'PENDING',
            verdicts   : {},
            consensus  : null,
            blockNumber: null,
            detectedAt : null,
        };

        await ctx.stub.putState(`TXN_${txnId}`, Buffer.from(JSON.stringify(txn)));
        await this._incrementCounter(ctx, 'totalTransactions');

        ctx.stub.setEvent('TransactionSubmitted', Buffer.from(JSON.stringify({
            txnId, amount: txn.amount, timestamp,
        })));

        return JSON.stringify(txn);
    }

    /**
     * recordFraudResult — submit one org's verdict for a transaction.
     * After all 3 orgs have submitted, majority vote determines consensus.
     * isFraud: "true" | "false"
     */
    async recordFraudResult(ctx, txnId, orgName, isFraud, fraudScore) {
        const buf = await ctx.stub.getState(`TXN_${txnId}`);
        if (!buf || buf.length === 0) throw new Error(`Transaction ${txnId} not found.`);

        const txn = JSON.parse(buf.toString());
        txn.verdicts[orgName] = {
            isFraud   : isFraud === 'true',
            fraudScore: parseFloat(fraudScore),
        };

        // Check if all 3 orgs have submitted
        const nVerdicts = Object.keys(txn.verdicts).length;
        if (nVerdicts >= 3) {
            const fraudVotes = Object.values(txn.verdicts)
                .filter(v => v.isFraud).length;
            const legVotes   = nVerdicts - fraudVotes;

            if (fraudVotes > legVotes) {
                txn.consensus = 'CONFIRMED_FRAUD';
                await this._incrementCounter(ctx, 'totalFraud');
            } else if (legVotes > fraudVotes) {
                txn.consensus = 'CONFIRMED_LEGITIMATE';
                await this._incrementCounter(ctx, 'totalNormal');
            } else {
                txn.consensus = 'DISPUTED';
            }
            txn.status     = txn.consensus;
            txn.detectedAt = _txTime(ctx);

            // Apply incentives — +10 for each org whose verdict matched consensus
            const isConsensusFraud = txn.consensus === 'CONFIRMED_FRAUD';
            for (const [org, verdict] of Object.entries(txn.verdicts)) {
                const matched = verdict.isFraud === isConsensusFraud;
                if (matched && txn.consensus !== 'DISPUTED') {
                    await this.updateIncentive(ctx, `Node-${this._orgToNode(org)}`,
                        '10', '0.02', 'CONSENSUS_SUCCESS');
                } else if (!matched && txn.consensus !== 'DISPUTED') {
                    await this.updateIncentive(ctx, `Node-${this._orgToNode(org)}`,
                        '-2', '-0.05', 'PENALTY');
                }
            }

            ctx.stub.setEvent('ConsensusReached', Buffer.from(JSON.stringify({
                txnId, consensus: txn.consensus, fraudVotes, legVotes,
            })));
        }

        await ctx.stub.putState(`TXN_${txnId}`, Buffer.from(JSON.stringify(txn)));
        return JSON.stringify(txn);
    }

    async getAllTransactions(ctx) {
        return await this._queryByDocType(ctx, 'transaction');
    }

    async getTransaction(ctx, txnId) {
        const buf = await ctx.stub.getState(`TXN_${txnId}`);
        if (!buf || buf.length === 0) throw new Error(`Transaction ${txnId} not found.`);
        return buf.toString();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LEADER BLOCK SELECTION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * recordLeaderSelection — store DB-BOA Job 2 output for one round.
     * dbBoaHistoryJson: JSON array of best-fitness per iteration.
     * dbBoaStatsJson:   JSON object {Min, Max, Mean, Std, Median}.
     */
    async recordLeaderSelection(ctx, roundId, winnerNodeId, ct, cc, ms,
                                 totalCost, dbBoaHistoryJson, dbBoaStatsJson) {
        const record = {
            docType      : 'leaderElection',
            roundId      : parseInt(roundId, 10),
            winnerNodeId,
            ct           : parseFloat(ct),
            cc           : parseFloat(cc),
            ms           : parseFloat(ms),
            totalCost    : parseFloat(totalCost),
            dbBoaHistory : JSON.parse(dbBoaHistoryJson),
            dbBoaStats   : JSON.parse(dbBoaStatsJson),
            electedAt    : _txTime(ctx),
        };

        await ctx.stub.putState(
            `LEADER_${String(roundId).padStart(6, '0')}`,
            Buffer.from(JSON.stringify(record))
        );

        // Update node's leadership count
        const nodeBuf = await ctx.stub.getState(`NODE_${winnerNodeId}`);
        if (nodeBuf && nodeBuf.length > 0) {
            const node = JSON.parse(nodeBuf.toString());
            node.roundsAsLeader += 1;
            await ctx.stub.putState(`NODE_${winnerNodeId}`, Buffer.from(JSON.stringify(node)));
        }

        await this._incrementCounter(ctx, 'leaderElections');

        ctx.stub.setEvent('LeaderSelected', Buffer.from(JSON.stringify({
            roundId, winnerNodeId, totalCost: record.totalCost,
        })));

        return JSON.stringify(record);
    }

    async getLeaderHistory(ctx) {
        return await this._queryByDocType(ctx, 'leaderElection');
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONSENSUS ROUNDS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * recordConsensusRound — log a full consensus round event.
     * consensusOk: "true" | "false"
     */
    async recordConsensusRound(ctx, roundNum, leaderNodeId, nTransactions,
                                nEndorsed, consensusOk, latencyMs, throughput) {
        const ok = consensusOk === 'true';
        const lat = parseFloat(latencyMs);

        const round = {
            docType       : 'consensusRound',
            roundNum      : parseInt(roundNum, 10),
            leaderNodeId,
            nTransactions : parseInt(nTransactions, 10),
            nEndorsed     : parseInt(nEndorsed, 10),
            consensusOk   : ok,
            latencyMs     : lat,
            throughput    : parseFloat(throughput),
            committedAt   : _txTime(ctx),
        };

        await ctx.stub.putState(
            `ROUND_${String(roundNum).padStart(6, '0')}`,
            Buffer.from(JSON.stringify(round))
        );

        await this._incrementCounter(ctx, 'consensusRounds');

        // Leader incentive
        if (ok) {
            const reward  = lat < 300 ? 15 : 10;
            const reason  = lat < 300 ? 'PERFORMANCE_BONUS' : 'CONSENSUS_SUCCESS';
            await this.updateIncentive(ctx, leaderNodeId, String(reward), '0.02', reason);
        } else {
            await this.updateIncentive(ctx, leaderNodeId, '-2', '-0.05', 'PENALTY');
        }

        ctx.stub.setEvent('ConsensusCompleted', Buffer.from(JSON.stringify(round)));
        return JSON.stringify(round);
    }

    async getConsensusHistory(ctx) {
        return await this._queryByDocType(ctx, 'consensusRound');
    }

    // ═══════════════════════════════════════════════════════════════════════
    // HYPERPARAMETERS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * recordHyperparams — store DB-BOA Job 1 optimal hyperparameters.
     * Overwrites if exists (only one HYPERPARAMS record).
     */
    async recordHyperparams(ctx, hiddenNeurons, epochCount, stepsPerEpoch,
                             bestObf2, dbBoaStatsJson) {
        const record = {
            docType       : 'hyperparams',
            hiddenNeurons : parseInt(hiddenNeurons, 10),
            epochCount    : parseInt(epochCount, 10),
            stepsPerEpoch : parseInt(stepsPerEpoch, 10),
            bestObf2      : parseFloat(bestObf2),
            dbBoaStats    : JSON.parse(dbBoaStatsJson),
            optimisedAt   : _txTime(ctx),
        };
        await ctx.stub.putState('HYPERPARAMS', Buffer.from(JSON.stringify(record)));
        return JSON.stringify(record);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FEDERATION — NEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * submitModelMetrics — record one org's model metrics before a fed round.
     * weightHash: SHA-256 of the weight array (proves integrity; weights stay off-chain).
     */
    async submitModelMetrics(ctx, orgName, roundNum, accuracy, mcc,
                              nSamples, weightHash, trainingInfoJson) {
        const record = {
            docType      : 'orgModel',
            orgName,
            roundNum     : parseInt(roundNum, 10),
            accuracy     : parseFloat(accuracy),
            mcc          : parseFloat(mcc),
            nSamples     : parseInt(nSamples, 10),
            weightHash,
            trainingInfo : JSON.parse(trainingInfoJson),
            submittedAt  : _txTime(ctx),
        };

        await ctx.stub.putState(
            `MODEL_${orgName}_${String(roundNum).padStart(6, '0')}`,
            Buffer.from(JSON.stringify(record))
        );

        return JSON.stringify(record);
    }

    /**
     * recordFederationRound — record Shapley-weighted aggregation and distribute rewards.
     * (DB-BOA Job 3 was replaced by exact Shapley values for fairness attribution.)
     *
     * aggregationWeightsJson: '{"BankA":0.61,"BankB":0.24,"BankC":0.15}'  Shapley weights
     * orgContributionsJson:   same format
     * dbBoaHistoryJson:       kept for schema compatibility (pass '[]' if unused)
     * bestFitness:            Obf2 of the Krum-selected global model
     * globalWeightsHash:      SHA-256 of the Krum-selected weight array
     */
    async recordFederationRound(ctx, roundNum, aggregationWeightsJson,
                                 orgContributionsJson, dbBoaHistoryJson,
                                 bestFitness, globalWeightsHash) {
        const weights       = JSON.parse(aggregationWeightsJson);
        const contributions = JSON.parse(orgContributionsJson);

        const record = {
            docType            : 'federationRound',
            roundNum           : parseInt(roundNum, 10),
            aggregationWeights : weights,
            orgContributions   : contributions,
            dbBoaHistory       : JSON.parse(dbBoaHistoryJson),
            bestFitness        : parseFloat(bestFitness),
            globalWeightsHash,
            timestamp          : _txTime(ctx),
        };

        await ctx.stub.putState(
            `FED_${String(roundNum).padStart(6, '0')}`,
            Buffer.from(JSON.stringify(record))
        );

        await this._incrementCounter(ctx, 'federationRounds');

        // Distribute federation rewards proportional to aggregation weights
        // Map org names to node IDs (BankA→Node-00, BankB→Node-02, BankC→Node-04)
        const orgNodeMap = { BankA: 'Node-00', BankB: 'Node-02', BankC: 'Node-04' };
        for (const [org, w] of Object.entries(weights)) {
            const nodeId = orgNodeMap[org];
            if (!nodeId) continue;
            const earned = Math.floor(BASE_FEDERATION_REWARD * w);
            if (earned > 0) {
                await this.updateIncentive(ctx, nodeId,
                    String(earned), '0.02', 'FEDERATION_REWARD');
            }
        }

        ctx.stub.setEvent('FederationCompleted', Buffer.from(JSON.stringify({
            roundNum: record.roundNum, aggregationWeights: weights, globalWeightsHash,
        })));

        return JSON.stringify(record);
    }

    async getFederationHistory(ctx) {
        return await this._queryByDocType(ctx, 'federationRound');
    }

    /** getOrgModels — return all docType='orgModel' documents. */
    async getOrgModels(ctx) {
        return await this._queryByDocType(ctx, 'orgModel');
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATISTICS
    // ═══════════════════════════════════════════════════════════════════════

    async getFraudStats(ctx) {
        const buf = await ctx.stub.getState('COUNTERS');
        if (!buf || buf.length === 0) return JSON.stringify({});
        const c     = JSON.parse(buf.toString());
        const total = c.totalTransactions || 1;
        return JSON.stringify({
            ...c,
            fraudRate        : ((c.totalFraud / total) * 100).toFixed(2) + '%',
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INTERNAL HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    async _incrementCounter(ctx, field) {
        const buf = await ctx.stub.getState('COUNTERS');
        const c   = buf && buf.length > 0 ? JSON.parse(buf.toString()) : {};
        c[field]  = (c[field] || 0) + 1;
        await ctx.stub.putState('COUNTERS', Buffer.from(JSON.stringify(c)));
    }

    async _queryByDocType(ctx, docType) {
        const query   = JSON.stringify({ selector: { docType } });
        const iter    = await ctx.stub.getQueryResult(query);
        const results = [];
        let res = await iter.next();
        while (!res.done) {
            results.push(JSON.parse(res.value.value.toString()));
            res = await iter.next();
        }
        return JSON.stringify(results);
    }

    /**
     * Map org name to a node ID suffix (two-digit string).
     * BankA → '00', BankB → '02', BankC → '04', others → '06'
     */
    _orgToNode(orgName) {
        const map = { BankA: '00', BankB: '02', BankC: '04' };
        return map[orgName] || '06';
    }
}

module.exports = DBBOAContract;
