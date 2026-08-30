# Thesis Brief for Supervisor

**Privacy-Preserving, Fairness-Aware Federated Fraud Detection on Hyperledger Fabric**

*A one-read walkthrough: the problem, where it came from, what we built, what we found,
what we can honestly claim, and where we fall short. Every section points to where it lives
in the report. All numbers below are taken from the re-run code, not from earlier drafts.*

---

## 1. The problem — and where it came from

Cross-institutional fraud detection has a structural deadlock. A single bank only sees its
own transactions, so its fraud model is blind to patterns that show up across institutions.
The obvious fix — pool everyone's data — is blocked by privacy law, competitive secrecy, and
the absence of any party everyone trusts to hold the data.

**Federated learning (FL)** is the textbook escape: train locally, share only model updates,
never move raw data. But once you try to deploy FL across mutually-distrusting banks, three
new problems appear that the textbook quietly assumes away:

1. **Trust / coordination** — who aggregates the updates, and why would a rival trust them?
2. **Incentives / fairness** — why would a bank contribute good data if a free-rider gets the
   same model for nothing? Contribution must be *measured* and *rewarded*.
3. **Privacy of the updates themselves** — model gradients leak information, so updates need
   differential privacy (DP).
4. **Security** — a malicious participant can poison the shared model.

The seductive pitch is that you can bolt all four fixes together — blockchain for trust, DP
for privacy, Shapley values for fair reward, Krum for security — and get a system that is
private, fair, secure, and decentralised *at the same time*. **Our research question is
whether that promise survives an honest implementation.** We build directly on Prabanand &
Thanabal (2025)'s DB-BOA-ADTCN detector and ask what happens when it is moved into a real
federated, on-chain setting.

> **In the report:** Chapter 1 (Introduction) — background, the cross-institutional fraud gap,
> the FL/blockchain/incentive barriers, and the objectives list. Chapter 2 (Literature Review)
> — where each component comes from and the gap we occupy (esp. FedCoin's prior Shapley-on-
> blockchain work, which we *extend*, not claim priority over).

---

## 2. Our solution approach — what we set out to build

We implemented the full stack end-to-end rather than simulating it on paper:

- **Detector:** an ADTCN fraud classifier whose hyperparameters are tuned automatically by a
  hybrid **DB-BOA** metaheuristic (no manual search), which *also* drives consensus
  leader-selection.
- **Federation:** three simulated banks (BankA/B/C) training locally, combined through a
  pipeline of **DP weight sharing + Krum robust selection + exact Shapley contribution
  attribution**.
- **Incentives:** Shapley contribution weights drive an **on-chain token reward pool**, so
  payment tracks measured contribution automatically.
- **Blockchain:** a real **Hyperledger Fabric** consortium running the `DBBOAContract`
  chaincode for consensus logging and token accounting — not a simulated ledger.

The intellectual core, though, was not "assemble the stack." It was to **test whether privacy,
fairness, and security actually compose** — and to characterise the trade-offs where they
don't.

> **In the report:** Chapter 4 (Proposed Methodology) — the six-phase pipeline, the DB-BOA
> tuning, the Krum/DP/Shapley federation layer, and the chaincode design. The threat model
> (the precise, bounded definition of what "Secure" means) also belongs here / Chapter 3.

---

## 3. What we tried — and the turn the research took

When we wired privacy and fairness together and ran it **honestly at a tight privacy budget
(ε = 1.0)**, it did not merely underperform — it **broke**:

- The full proposed pipeline (Krum + DP + Shapley at ε=1.0) **collapsed to a degenerate
  single-class predictor (MCC ≈ 0)**. The DP noise on the ~111,874-dimensional weight vector
  overwhelmed the signal, and Krum could not help (selecting one noised model forgoes the
  noise-cancellation that averaging gives).

This is the pivot of the whole thesis. A weaker project would have quietly tuned the numbers
until it "worked." Instead we made the failure the subject: **where, exactly, is the boundary
at which privacy destroys fair incentives — and can it be fixed by design rather than by
fudging?**

The fix came from a single insight: the problem was not privacy itself, but **where the noise
was injected**. We were perturbing the high-dimensional model weights and then trying to read
contribution through that fog. So we moved the DP noise off the weight channel and applied
**output-perturbation DP directly to the 3-dimensional Shapley contribution vector φ**.

> **In the report:** Chapter 5 (Result Analysis) §6.2 (the DP collapse / ablation) and §6.6
> (the privacy↔incentive characterisation and the output-channel mechanism). This is also the
> reframed novelty statement carried from Chapter 1.

---

## 4. What we found — the verified results

All figures below are from the re-run code and the `final_report_data/` ground-truth drafts.

**(a) Centralised detector works, and we report it honestly.**
DB-BOA-tuned ADTCN on the ULB test set (n=56,962, 98 fraud): **Acc 99.85%, MCC 0.677**,
Precision 54.25%, Sensitivity 84.69%. We emphasise **MCC**, not accuracy, because under 0.17%
fraud a "predict-all-normal" model already scores 99.83%. Honest caveat: the *tuned* detector
(MCC 0.677) slightly **trails** a hand-set default (MCC 0.785) — so DB-BOA's real win is
*automation without manual search*, not beating a human.

**(b) The privacy↔incentive trade-off, characterised and then resolved (the centrepiece).**
Moving the incentive signal from the weight channel to an output-perturbation channel on φ
improves the privacy budget at which on-chain rewards stay rank-faithful from
**ε\* = 3000 (weight) → ε\* = 50 (output)** — a **~60× improvement**, landing honest incentives
in the practical DP regime (ε ≈ 10–50) instead of an unusable one. The Spearman rank-fidelity
of rewards rises from ≈ −0.29 to **+0.95** at ε=50. The negative result is therefore a property
of the *channel*, not of DP-plus-Shapley in principle.

**(c) Security holds where the theorem holds.**
Run in the regime where Krum's precondition (n ≥ 2f+3) is satisfied — n=5/f=1 and n=7/f=2 —
Krum **rejected the Byzantine org in 8/8 (regime × attack) cases**, keeping the global model at
**≈99.9%** balanced accuracy. The damage it prevents is largest under norm-boosting attacks,
where unprotected FedAvg collapses to **≈87.5%**. This is genuine *statistical* BFT, not the
weaker outlier-rejection of the default 3-org setup.

**(d) The federation ablation is honest, not cherry-picked.**
FedAvg MCC 0.569 → FedAvg+Krum **0.776** (best single config) → FedAvg+DP(ε=1.0) **0.000** →
full proposed pipeline at ε=1.0 **≈0** (collapse). We present the collapse as evidence for the
trade-off, not as something to hide.

**(e) The blockchain layer is measured, not invented.**
Against the live Fabric test-network: consensus-round latency **mean 2116 ms** (the real 2 s
Raft `BatchTimeout` floor), peak sustained throughput **40.3 tps** at concurrency 10, with a
documented `MVCC_READ_CONFLICT` knee past that. These replace earlier fabricated 85-tps / 180-ms
figures.

> **In the report:** Chapter 5 (Result Analysis) — §6.1 (centralised detector), §6.2
> (federation ablation), §6.6 (privacy↔incentive mechanism), the Byzantine-robustness section
> (Krum 8/8), and the Fabric consensus-measurement section. Plots regenerated into
> `db_boa_framework/results/`.

---

## 5. What we can honestly claim — the contributions

In order of strength, and in language that survives scrutiny:

1. **A privacy-preserving incentive mechanism** that applies DP to the *contribution score*
   rather than the *model weights*, buying ~60× privacy budget and keeping on-chain rewards
   rank-faithful. *(This is the one genuinely novel mechanism.)*
2. **A characterisation of the privacy↔incentive trade-off** — empirical evidence, with real
   numbers, that naïvely stacking DP and Shapley incentives produces rewards *worse than
   random*, and a clear map of the failure boundary. *(Legitimate negative-result novelty.)*
3. **A working, integrated systems contribution** — DB-BOA + DP + Krum + Shapley + on-chain
   incentives, running on a real Hyperledger Fabric consortium with *measured* consensus
   behaviour. *(Integration of established parts, honestly labelled as such.)*

We **do not** claim to have invented DP, Krum, Shapley, or Fabric, and we **do not** claim
"first to bind Shapley to blockchain incentives" — FedCoin (2020) did that; we extend it.

> **In the report:** Chapter 1 (contribution list / novelty statement) and Chapter 6
> (Conclusion).

---

## 6. Where we fall short — limitations (stated plainly)

These are disclosed in the report, not buried — examiners reward this far more than inflated
numbers:

- **Single-source data.** The three "banks" are a stratified volume-split (50/30/20) of *one*
  institution's ULB dataset, not genuinely distinct sources. Cross-institution distribution
  shift is therefore *not* exercised — the federation is a controlled simulation of heterogeneity.
- **Incentive defence is conditional.** Reputation-driven economic isolation works against a
  *colluding majority* of label-corrupters (restoring +41 to +80 pts of balanced accuracy) but
  does **not** deter a lone attacker or a passive free-rider — a free-rider can even be
  mis-attributed almost the whole token pool. Deterring free-riding needs a contribution
  *floor*, left as future work.
- **"Secure" and "Scalable" are bounded claims.** Krum's BFT guarantee holds only for ≤ f
  colluding orgs at n ≥ 2f+3 (the default n=3/f=0 pipeline is outlier rejection, not BFT); and
  a subtle retrained label-flip attacker sits only just outside the honest cluster
  (margin ≈ 10⁰). "Scalable" covers *contribution attribution*, not blockchain throughput
  (Fabric goodput collapses past concurrency 10).
- **No inter-round local training**, so the 3-round convergence is not a general FL convergence
  result; and Shapley values carry a small, disclosed **validation circularity** (computed on an
  in-distribution held-out slice, not an independent institution's data).
- **Simulation scope.** Consensus latency/throughput and the RL leader-selection result come
  from a single-host test-network; the transferable findings are the *shapes* (batching
  amortises the orderer timeout), not the absolute constants.

> **In the report:** Chapter 5 / Chapter 6 — the "Threat model and security scope" paragraph
> and the consolidated "Limitations and disclosures" block (drafted in
> `final_report_data/REWRITE_08_limitations_disclosures.md`).

---

## 7. The one-paragraph version (if he only reads this)

> We built a privacy-preserving, fairness-aware federated fraud-detection system on a real
> Hyperledger Fabric consortium, extending the DB-BOA-ADTCN detector. The core finding is that
> privacy, fairness, and security **do not compose for free**: at a meaningful privacy budget,
> naïvely combining differential privacy with Shapley-based incentives makes contribution-based
> rewards *worse than random* and collapses the model. We characterise exactly where this
> breaks, and we resolve it with a mechanism that applies DP to the contribution score rather
> than the model weights — improving the usable privacy budget ~60× (ε\* 3000 → 50). Krum gives
> genuine Byzantine fault tolerance where its theorem holds (attacker rejected 8/8), and the
> Fabric layer's consensus behaviour is measured, not assumed. The contribution is one novel
> mechanism, one honest characterisation of a real trade-off, and a working integrated system —
> with its limitations (single-source data, conditional incentive defence, simulated consensus)
> stated openly.

---

*Report chapter map: Ch1 Introduction · Ch2 Literature Review · Ch3 Requirements, Impacts &
Constraints · Ch4 Proposed Methodology · Ch5 Result Analysis · Ch6 Conclusion. Ground-truth
numbers: `final_report_data/`. Defense Q&A: `defense_questions.md`.*
