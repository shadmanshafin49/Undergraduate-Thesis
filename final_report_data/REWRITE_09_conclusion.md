# REWRITE — Conclusion (chapter_9.tex), Option A, paste-ready

Replace the entire conclusion with the version below. The headline numbers are settled at
**99.85\% accuracy / MCC 0.677** (DB-BOA-tuned detector, fresh single fixed-objective run, see
`REWRITE_06_results.md` §6.1/§6.5) and already filled in below. Do not reuse 97.38\% / 0.966 (no run
produces those), and do not quote 0.785 as the headline (it is the hand-set default, reported only
in the §6.5 comparison).

```latex
This thesis presented FL-ADTCN, a blockchain-integrated federated learning framework for
collaborative financial fraud detection, implemented end-to-end on a real Hyperledger Fabric
consortium. The framework resolves the three fundamental barriers to cross-institutional
fraud detection. Privacy is protected by federated learning with differentially private
weight sharing: raw transaction data never leaves an institution, and shared parameters carry
a formal $(\epsilon,\delta)$ guarantee. Trust is provided by Hyperledger Fabric: consensus
decisions and incentive rules are recorded and enforced by the \texttt{DBBOAContract}
chaincode, which no single organisation controls and the immutable ledger cannot retroactively
alter. Incentive alignment is achieved by coupling exact Shapley-value contribution weights to
an on-chain token mechanism, so that economic reward tracks measured contribution
automatically.

The primary contribution is a \emph{characterisation of the privacy--incentive coupling} of a
Shapley-driven, chaincode-enforced reward mechanism on a real permissioned network, together with
a concrete fix: differential privacy and contribution-based incentives are in direct tension, and
moving the privacy noise from the shared model weights onto the released contribution channel
restores rank-faithful, DP-protected on-chain rewards at roughly two orders of magnitude lower
privacy budget ($\epsilon^\star\approx 3000\to 50$). This central result rests on, but is
distinct from, the engineering substrate that makes it possible: the federated aggregation layer
(differentially private weight sharing, Krum robust selection, exact/Monte-Carlo Shapley
attribution) bound to blockchain-enforced incentives; a hybrid DB-BOA metaheuristic that tunes the
ADTCN detector's hyperparameters and selects the consensus leader; and a deliberately minimal
linear-Q-learning policy that handles \emph{sequential} leader rotation and, in simulation, adapts
to a node whose reliability degrades at runtime -- a secondary, consensus-layer component, not a
fraud-detection improvement. Around the central result we contribute a characterisation suite --
the decoupling of accuracy from reward fidelity under DP, the economic isolation of a colluding
majority, statistical Byzantine fault tolerance, and the cost--fidelity limits of Shapley
attribution as the federation grows -- that bounds where the mechanism can and cannot be trusted.
Rather than claiming algorithmic priority, this work extends prior Shapley-on-blockchain incentive
work (FedCoin~\cite{fedcoin2020}) with a deployed Hyperledger Fabric implementation and a
noise$\to$fidelity$\to$reward-error analysis.

Empirically, the DB-BOA-tuned centralised ADTCN reached 99.85\% accuracy and an MCC of 0.677
on the ULB Credit Card Fraud dataset -- a working detector selected automatically without manual
search, though it did not exceed a hand-tuned default (MCC 0.785); DB-BOA's value here is automation
and leader selection, not a detection-accuracy gain -- and the federated configuration preserved
detection quality while keeping each
institution's data local. These detection figures are reported at the deployment operating point,
in which weight-channel differential privacy is disabled (its $\epsilon=1.0$ noise near-randomises
the global model, as the ablation shows) and privacy is instead enforced on the incentive channel
-- the very configuration the central contribution shows to be both private and useful. An
ablation over FedAvg, FedAvg+Krum, and FedAvg+DP quantified the privacy/robustness trade-offs and
makes this cost explicit rather than hiding it. Byzantine resilience was demonstrated on two
fronts: at the parameter level, Krum -- run in the regime where its theorem holds ($n=5,f=1$
and $n=7,f=2$) -- rejected every one of four weight-poisoning attacks while plain averaging fell
to 87.5\% balanced accuracy under norm-boosting; at the behavioural level, the Shapley mechanism
assigns an always-fraud adversary a near-zero contribution weight, isolating a coordinated
majority that out-numbers any statistical defence.

The work has clear limitations, stated honestly: the three institutions are volume-splits of a
single bank's dataset rather than genuinely distinct sources; consensus latency and throughput
are obtained from a simulation rather than a multi-host deployment; the differential-privacy
budget $\epsilon=1.0$ is deliberately tight and degrades the shared weights; the demonstrated
Byzantine tolerance holds for up to $f$ colluding organisations and a sufficiently subtle
poisoning attack narrows Krum's separation margin; and the default three-organisation deployment
runs at $f=0$, so statistical robustness is shown in the larger $n\ge 2f+3$ configurations rather
than the headline pipeline. Future work will extend the consortium beyond three organisations on
real multi-bank streams, relax the privacy budget with DP-SGD, harden the aggregator against
subtle (close-to-honest) poisoning, deploy a measured multi-host Fabric network, and investigate
quantum-resistant ledger cryptography. The codebase
-- Fabric network configuration, \texttt{DBBOAContract} chaincode, DB-BOA implementation, and
the FL-ADTCN model -- is publicly available at
\url{https://github.com/shadmanshafin49/DB-BOA-FEL-ADTCN-Hyperledger-Fabric}.
```

### Removed from the old conclusion
- "97.38\% accuracy and MCC of 0.966" (no run produces these).
- "85 TPS with average 180 ms latency ... 28.4\% reduction" (simulated, not measured).
- "DB-BOA for federated aggregation weight determination" as the novelty (it is Shapley).
- "within 12 rounds" isolation claim unless the full attack run shows it.
