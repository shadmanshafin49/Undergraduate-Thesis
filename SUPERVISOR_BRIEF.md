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
fraud a "predict-all-normal" model already scores 99.83%. Honest caveat: the *tuned* detector does
**not** beat a hand-set default — and does not trail one either. Over 5 seeds: tuned
**0.753 ± 0.055** vs default **0.706 ± 0.076**, a gap of **+0.047 at p=0.34**, statistically
indistinguishable. (An earlier draft reported the tuned detector trailing a default at MCC 0.785;
OBJ-2 withdrew that — 13 runs showed it was CPU thread count, not the optimiser.) So DB-BOA's real
win is *automation without manual search*, not beating a human.

**(b) The privacy↔incentive trade-off, characterised and then resolved (the centrepiece).**
Moving the incentive signal from the weight channel to an output-perturbation channel on φ
improves the privacy budget at which on-chain rewards stay rank-faithful by **one to two orders
of magnitude**, landing honest incentives in the practical DP regime instead of an unusable one.
The negative result is therefore a property of the *channel*, not of DP-plus-Shapley in
principle.

**What the mechanism buys, stated in the two forms that survive scrutiny.** Both are measured
in all three run conditions and neither depends on an arbitrary cut-off.

1. **The ordering.** The contribution channel keeps rewards better-ordered than the weight
   channel at **essentially every privacy budget**: 11 of 11 on rank-correlation in all three
   conditions, and 11/11 · 11/11 · 10/11 on the stricter mis-ranking-rate metric. The single
   counter-example is at ε=1 on BankSim/entity-disjoint (weight mis-ranks 85/100, contribution
   88/100), where *both* channels are failing near-totally. This is a **paired** comparison —
   the same noise draw drives both channels in the same iteration — so more data changes its
   precision, not its direction. It has now held across three conditions, two sample sizes
   (100 and 1000 draws) and two metrics.
2. **The reward-ordering quality at a fixed, stated budget.** At ε=50, the Spearman correlation
   between paid tokens and true contribution:

| Condition | weight channel | contribution channel |
|---|---|---|
| ULB | **−0.199** (rewards backwards) | **+0.995** |
| BankSim, stratified | +0.017 (no signal) | **+0.945** |
| BankSim, entity-disjoint | −0.070 (no signal) | **+0.805** |

That table is the centrepiece. At a genuinely private setting the old channel pays contributors
in an order that is uncorrelated with — or on ULB actively opposite to — what they contributed,
and the new channel gets it nearly right. No threshold, no grid, no extrapolation.

> ⚠ **On the "60×"-style budget-improvement factor: we are retiring it as a headline, and the
> reason is a result in itself.** The factor is ε\*(weight)/ε\*(output), where ε\* is the
> largest budget at which rewards are still mis-ranked *at all*. We extended the grid to
> ε=30000 and then re-measured the deciding budgets at 1000 draws instead of 100. Three things
> came out of that:
>
> (i) **ε\* can only ever move up.** More draws, or more budgets, can only *find* additional
> mis-rankings — never remove one. Extending the grid raised BankSim/entity-disjoint from
> "≥10×" to 33×; the 1000-draw pass then raised it again to **≥100×** and pushed it back
> outside our grid, so it is a lower bound once more — on a single mis-ranked draw in a
> thousand. Two successive increases from looking harder, in one day.
>
> (ii) **The factor is not stable under its own definition.** Re-deriving it with the threshold
> set at a small non-zero mis-ranking rate rather than "any at all":
>
> | Condition | rate > 0 | rate > 0.01 | rate > 0.05 |
> |---|---|---|---|
> | ULB | 60× | 20× | 33× |
> | BankSim, stratified | 30× | 60× | 20× |
> | BankSim, entity-disjoint | ≥100× | 33× | 33× |
>
> The value swings between **20× and ≥100×**, and the *ranking of the three conditions* changes
> with the choice. A number that mobile should not carry a claim. **Our honest summary is "one
> to two orders of magnitude, definition-sensitive"**, with the ordering and the ρ table above
> doing the actual work.
>
> (iii) **What we did confirm:** the thresholds that rested on a single draw in 100 were **real,
> not flukes** — at 1000 draws they came back as 7, 11, 16 and 10 per 1000, all consistent with
> a true rate near 0.01. So the earlier fragility was about *precision*, not about whether the
> effect was there.

> ⛔ **One reproducibility failure, disclosed.** The ULB result stored on 2026-08-30 does **not**
> reproduce under the current environment (13 of 18 shared cells moved). Both BankSim conditions
> reproduce **bitwise**, and ULB reproduces *itself* on same-day re-run, so the pipeline is
> deterministic — but the cause of the drift is **unexplained**: we eliminated every code change
> on the path, all six configuration dicts, and the CPU thread count, and the library version at
> the time is unrecoverable because the version pin was added afterwards. All sweep results now
> record their full software environment so this cannot recur. **The ULB figures above are from
> re-verified 2026-09-04 runs and stand on their own evidence; they cannot be reconciled with
> the older stored number.**

**(c) Security holds where the theorem holds — and security is not the same as utility.**
Run in the regime where Krum's precondition (n ≥ 2f+3) is satisfied — n=5/f=1 and n=7/f=2 —
Krum **rejected the Byzantine org in 8/8 (regime × attack) cases**, keeping the global model at
**≈99.9%** balanced accuracy on ULB. The damage it prevents is largest under norm-boosting
attacks, where unprotected FedAvg collapses to **≈87.5%**. This is genuine *statistical* BFT,
not the weaker outlier-rejection of the default 3-org setup. **The rejection result is the
strongest thing we have: 8/8 in all three run conditions** (ULB, and BankSim under both a
stratified and an entity-disjoint bank split).

> ⚠ **What does *not* replicate is the accuracy Krum buys.** The +12.49 pp advantage over
> unprotected FedAvg is a **ULB** figure. On BankSim the same experiment gives a *negative*
> advantage — Krum rejects the attacker and the selected model is still worse than the plain
> average — in 7 of 8 cells under the stratified split and 8 of 8 under the entity-disjoint one.
> **Why it costs utility there is unexplained; we report the measurement and do not supply a
> mechanism.** An earlier reading that blamed entity-disjoint banks was withdrawn on 2026-09-04
> by our own confound control, which found the same cost on a partition where the banks share
> customers. Two consequences for how this is written: **report rejection and accuracy as
> separate claims**, and treat "Krum's value shows up under attack" as retired wording.
> (A further caution for the tables: in some BankSim cells the *attacked* FedAvg baseline scores
> above its own no-attack reference, which an attack cannot genuinely do — those rows are a
> metric artefact and are flagged individually in
> `final_report_data/TASKD_byzantine_robustness_results_banksim_*.md`.)

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
   rather than the *model weights*, keeping on-chain rewards rank-faithful at essentially every
   budget we swept, in all three conditions — Spearman **+0.995 / +0.945 / +0.805** against
   **−0.199 / +0.017 / −0.070** for the weight channel at ε=50. The budget improvement is
   **one to two orders of magnitude but definition-sensitive** (20× to ≥100×; see §4(b)), so we
   lead with the ordering rather than a factor. *(This is the one genuinely novel mechanism.)*
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

- **Simulated, not genuinely multi-institution.** The three "banks" of the headline pipeline are
  a stratified volume-split (50/30/20) of *one* institution's ULB dataset, not distinct sources.
  A second dataset (BankSim, 594,643 tx with 4,112 real customer IDs) has since been added and
  the system experiments re-run on it under **both** a stratified and a genuinely
  entity-disjoint bank split — so the results are no longer single-source, and five of the six
  title claims now stand on two datasets. But even the entity-disjoint split deals customers of
  *one* generator out to orgs: cross-**institution** distribution shift is still not exercised.
  The federation remains a controlled simulation of heterogeneity.
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
> than the model weights — restoring rank-faithful rewards at a genuinely private budget
> (Spearman +0.995 vs −0.199 at ε=50 on ULB, same direction in all three conditions). The
> corresponding budget-improvement factor spans **one to two orders of magnitude and is
> sensitive to how the threshold is defined** (20× to ≥100×), so we report the ordering, which
> holds at 11 of 11 budgets on rank-correlation and 10–11 of 11 on mis-ranking rate, rather
> than a single multiplier. Krum gives
> genuine Byzantine fault tolerance where its theorem holds — the attacker is rejected **8/8 in
> all three conditions** — though on our second dataset that rejection *costs* accuracy rather
> than buying it, which we report as measured and do not explain away. The Fabric layer's
> consensus behaviour is measured, not assumed. The contribution is one novel mechanism, one
> honest characterisation of a real trade-off, and a working integrated system — with its
> limitations (simulated rather than genuinely multi-institution federation, conditional
> incentive defence, simulated consensus) stated openly.

---

*Report chapter map: Ch1 Introduction · Ch2 Literature Review · Ch3 Requirements, Impacts &
Constraints · Ch4 Proposed Methodology · Ch5 Result Analysis · Ch6 Conclusion. Ground-truth
numbers: `final_report_data/`. Defense Q&A: `defense_questions.md`.*
