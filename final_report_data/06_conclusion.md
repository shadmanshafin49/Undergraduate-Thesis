# Chapter — Conclusion (corrections)

The current `chapters/chapter_9.tex` is one paragraph short and repeats the fabricated
headline numbers. Rewrite around the real contribution.

## Remove from the conclusion
- "97.38% accuracy and MCC of 0.966" (🔴 D5 — no run produces this).
- "85 TPS with average 180 ms latency … 28.4% reduction" (🔴 D7 — simulated, not measured).
- "isolates a malicious organisation within 12 rounds" — only state if the `--attack` run
  actually shows it; report the real disputed-round count and depletion instead.
- "primary novel contribution — DB-BOA for federated aggregation weight determination"
  (🔴 D1 — the code uses Shapley, not DB-BOA, for this).

## Real contribution to claim (defensible)
> This thesis implements, on a **real Hyperledger Fabric consortium**, a federated fraud-
> detection system that combines a DB-BOA-tuned 1-D temporal CNN detector, DB-BOA-driven
> consensus leader selection, and a **privacy-preserving, fairness-aware federation layer**
> — differentially-private weight sharing, Krum robust model selection, and **exact
> Shapley-value contribution attribution** — whose Shapley weights are bound to an on-chain
> token-incentive mechanism enforced by the `DBBOAContract` chaincode. The contribution is to
> **characterise** the privacy↔incentive coupling (noise→Shapley fidelity→reward error) of
> **mathematically-derived (Shapley) contribution weights bound to blockchain-enforced economic
> incentives** on a production-grade permissioned ledger, **extending** prior Shapley-on-
> blockchain work (FedCoin, 2020) with a deployed Fabric implementation.

> **🔴 Do not claim "first."** FedCoin (2020) already does Shapley reward allocation on a
> blockchain; the lit review cites it, so a priority claim is self-contradicting. Frame as
> characterisation/extension, never priority (applied across REWRITE_01/05/09).

## Honest results summary (VERIFIED — settled numbers)
- Centralised ADTCN: **Acc 99.85% / MCC 0.677** (DB-BOA-tuned, single headline; selected without
  manual search but does not beat — slightly trails — a hand-set default at MCC 0.785). Do not reuse
  97.38 / 0.966 / 0.941.
- Federated ablation: FedAvg (MCC 0.569) → +Krum (**0.776**, best) → both DP ε=1.0 rows collapse to
  a degenerate single-class predictor (MCC≈0; failure direction varies by noise draw). The DP collapse
  at ε=1.0 is the expected privacy/utility cost, consistent with the ε\*=3000 incentive-fidelity threshold (§6.6).
- Incentives: token/reputation dynamics reward contribution and penalise the always-fraud
  attacker; the Shapley layer zeroes the attacker's aggregation weight (real numbers from
  `--attack`, REWRITE_06 §6.4).

## Limitations to state plainly (these strengthen the thesis)
- Three orgs are **volume-splits of one bank's ULB data**, not true cross-institution data.
- Consensus latency/throughput are **simulated**, not measured on a live multi-host network.
- DP at **ε=1.0** heavily degrades shared weights (deliberate tight budget).
- Krum with f=0 provides **consensus alignment, not full Byzantine tolerance**.
- ADTCN is a **1-D CNN with max-pooling**, not an attention/dilated architecture.

## Future work (real next steps)
Larger consortia (K>3) on real multi-bank streams; relaxed/adaptive DP (DP-SGD, larger ε);
true Byzantine-tolerant Krum (f≥1, n≥2f+3); a deployed multi-host Fabric network with
measured throughput/latency; activation and architecture ablations; optionally re-enabling
DB-BOA Job 3 as a comparison against Shapley.

## Keep
The public code-availability statement (GitHub link) is good — keep it, it supports
reproducibility.
