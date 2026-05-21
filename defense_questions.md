# Thesis Defense Questions — DB-BOA Financial Security Framework (Updated)

**Framework**: DB-BOA + ADTCN + Federated Learning + Hyperledger Fabric for Credit Card Fraud Detection  
**Based on**: Prabanand & Thanabal (2025), Scientific Reports 15, 6764  
**Codebase state**: All 25 known issues from NOVELTY_TIPS.md have been addressed.  
Questions reflect the *current* implementation — wrong assumptions from old versions are corrected.

---

## 1. Overview & Motivation

**Q1.** In one sentence, what is the core problem your thesis solves, and why does it need a blockchain?

**Q2.** Why is credit card fraud detection specifically suited to a federated learning setup across banks, rather than a centralised model trained on pooled data? Name at least two practical barriers to data pooling.

**Q3.** The ULB dataset has 0.17% fraud. How does such extreme class imbalance affect your model, and how did you address it?

**Q4.** What would happen if you simply used a standard centralised XGBoost or LightGBM trained on all data? Why is that approach unacceptable in practice, and how does federated learning address the gap?

**Q5.** Your system has five major components: DB-BOA, ADTCN, Federated Learning, Krum, Shapley, DP, and Hyperledger Fabric. Which single component contributes the most to detection accuracy, and how do you know? Your Phase 4 DB-BOA-vs-defaults comparison is evidence here — what did it show?

**Q6.** What is the threat model of your system? Who is the adversary, what can they do, and what are the limits of your security guarantees?

---

## 2. DB-BOA Algorithm

**Q7.** Explain the DB-BOA switching criterion. When does it choose DBOA over BOA, and why? Your implementation uses `1 − |f_max − f_min| / max(|f_min|, |f_max|)` — what does this approach when the population converges, and what does it approach when spread out?

**Q8.** Prove that the switching threshold `1 − |f_max − f_min| / max(|f_min|, |f_max|)` is always in [0, 1]. Why does this matter for the switching criterion to function correctly?

**Q9.** DBOA has two movement equations — global search (Eq.3) and local search (Eq.4). What determines which fires, and what does ρ=0.8 mean for the exploration/exploitation balance?

**Q10.** What is LSAM (Local Search with Adaptive Mutation)? How many fitness evaluations does it add per iteration? For population=20 and iterations=30, what is the total evaluation budget including LSAM?

**Q11.** BOA uses billiards-table pockets. Explain in plain language what a "pocket" represents and how the billiards movement equation (Eq.8) shifts a solution toward a pocket.

**Q12.** DB-BOA is applied to three different jobs in your system. List each job's objective function and its dimensionality.

**Q13.** What is Obf2? Write the full formula, and explain why each component is included.

**Q14.** Phase 4 of your pipeline now trains a default-hyperparameter model alongside the DB-BOA-optimal model and prints a side-by-side comparison. What does this comparison prove, and what would it mean if the default model performs equally well?

**Q15.** Your DB-BOA statistical summary reports Min, Max, Mean, Std, Median. What exactly is being summarised — the per-iteration best-so-far fitness or the mean fitness?

---

## 3. ADTCN Architecture

**Q16.** Your paper calls the model "ADTCN — Adaptive Deep Temporal Context Networks." Describe the exact PyTorch layer sequence in your implementation and what each layer does.

**Q17.** You use Conv1d with a sliding window of SEQ_LEN=10 transactions. How do you construct these sequences from the flat transaction feature matrix? What happens for the first 9 rows, and what bias does this introduce?

**Q18.** Why 1D-CNN instead of LSTM or Transformer for temporal modelling? What is the trade-off in terms of expressiveness, training speed, and interpretability?

**Q19.** Your architecture labels its layers MJE, TCL, MTTA, OUT. What does MTTA stand for in the paper, and what does your code actually implement in that position? Your docstring was updated to be honest about this — what does it say?

**Q20.** GlobalMaxPool is used instead of GlobalAveragePool. Justify this choice for fraud detection. What signal does max-pooling preserve that average-pooling would dilute?

**Q21.** How does the weighted cross-entropy loss work for 0.17% fraud? Compute the class weight assigned to the fraud class if the training set has 400 fraud and 226,700 normal samples.

**Q22.** Config.py previously had `"activation": "tanh"` but the CNN hardcodes `nn.ReLU()`. How was this resolved in the updated code? What does the current config say about activation choice?

---

## 4. DB-BOA Hyperparameter Search — 2D Search

**Q23.** The search space for ADTCN hyperparameter optimisation is now 2-dimensional: (n_filters, steps_per_epoch). Why was epoch count removed from the search space in the latest version?

**Q24.** What was the specific problem with including epoch count as a DB-BOA dimension? Why did the surrogate's hard cap at 5 epochs make this dimension "flat"?

**Q25.** With epoch count fixed at the config default (30), is this value ever optimised anywhere in the pipeline? If not, how do you justify the choice of 30 epochs?

**Q26.** The `steps_per_epoch` parameter controls `batch_size = max(32, n_train // spe)`. Walk through how DB-BOA varying steps_per_epoch affects the training dynamics.

---

## 5. Dataset & Feature Engineering

**Q27.** Describe the ULB Credit Card Fraud dataset. How many samples, features, and what is the fraud rate? What are the V1–V28 features, and why were they PCA-transformed?

**Q28.** Your feature pipeline produces 301-dimensional engineered features (30 base + 3 recurrence features + PTC + NTC). Describe what PTC (Periodic Temporal Context) and NTC (Non-Periodic Temporal Context) windows contribute.

**Q29.** Your "graph features" are actually three sliding-window recurrence features: `amount_recurrence_before`, `amount_recurrence_after`, and `degree_ratio`. Why were they originally called graph features? What real graph feature would these proxy if the ULB dataset had account IDs?

**Q30.** The 1D-CNN only uses the first 33 columns (30 base + 3 recurrence features) of the 301-dimensional engineered matrix. Why doesn't it use all 301 features?

**Q31.** You use an 80/20 train-test split and 10% of training for validation. Does your test set leak into any part of the optimisation? Specifically, does DB-BOA's surrogate ever see test-set samples?

**Q32.** How do you split data across the three federated organisations (BankA 50%, BankB 30%, BankC 20%)? Is this i.i.d. or non-i.i.d.? Does the fraud rate remain 0.17% in each slice?

**Q33.** The ULB dataset comes from a single bank's transactions. Your config now explicitly acknowledges this as a "controlled simulation, not a real cross-bank federated deployment." How does this limit the generalisability of your results?

---

## 6. Surrogate Evaluation & Class Distribution

**Q34.** The DB-BOA surrogate evaluates ADTCN fitness on a 2,000-row subsample. Originally this used ~50% fraud; now it matches the real 0.17% rate. Why does this matter for the validity of the hyperparameters found?

**Q35.** With `_SURROGATE_ROWS=2,000` and 0.17% fraud, the raw count would be `int(2000 × 0.0017) = 3` fraud samples. Your code sets `_MIN_FRAUD_ROWS=30`. Why is 30 chosen as the minimum? What happens to the class weight calculation with only 3 fraud samples?

**Q36.** With 30 fraud samples in ~2,000 total surrogate rows, the effective fraud rate in the surrogate is ~1.5% rather than 0.17%. Does this still create a mismatch between surrogate and deployment conditions, and how significant is it?

**Q37.** The surrogate trains for `_SURROGATE_EPOCHS=5` regardless of what DB-BOA proposes. Is 5 epochs sufficient for the CNN to converge enough to give a meaningful fitness signal?

---

## 7. Federated Learning

**Q38.** What is Federated Averaging (FedAvg)? Why is it your primary baseline and not standard centralised training?

**Q39.** In your federated simulation, organisations share model weights, not raw data. What information can an adversary potentially infer from the shared weights alone (model inversion / membership inference)?

**Q40.** Your code includes a "FL Validity note" (added in the latest update) commenting that orgs do not perform local gradient updates between federation rounds. Why does this matter? What does a real federated learning deployment do between rounds that your simulation skips?

**Q41.** Your federation triggers every 5 consensus rounds. What is the architectural reason for this interval, and how does it relate to the Hyperledger Fabric consensus cycle?

**Q42.** You simulate 3 federation rounds. Your code acknowledges that "the absence of inter-round drift means the 3-round convergence result does not generalise to real FL deployments." How would you redesign the simulation to add drift?

**Q43.** The data split (50/30/20) means BankA contributes much more data. Your config notes this violates the i.i.d. assumption of McMahan et al. (AISTATS 2017). What is the i.i.d. assumption in FedAvg, and why does your split violate it?

**Q44.** FedProx (Li et al., MLSys 2020) is cited in your config as more appropriate for heterogeneous distributions. What does FedProx add over FedAvg, and would switching to it strengthen your thesis?

---

## 8. Differential Privacy

**Q45.** State the Gaussian mechanism formula for (ε, δ)-DP. What is σ as a function of sensitivity C, ε, and δ?

**Q46.** Your code uses C=1.0, ε=1.0, δ=1e-5. Compute σ numerically (your build log states σ ≈ 4.84 — verify this).

**Q47.** What does it mean for your weight sharing to be (1.0, 1e-5)-DP? What adversarial capability does it protect against?

**Q48.** Your updated code now logs DP composition after each federation round: "after k round(s) ε_total=k·ε, δ_total=k·δ (basic composition)." With ε=1.0 and 3 rounds, what is ε_total? Is basic composition tight, or can it be improved?

**Q49.** `run_baselines.py` now prints a DP accuracy cost comparison between FedAvg (no DP) and FedAvg+DP. What does this measurement prove, and why is it important for your defense?

**Q50.** DP noise with σ ≈ 4.84 is added to every weight tensor before sharing. The Conv1d weight tensors have values typically in [-1, 1]. Is σ=4.84 small relative to weight magnitudes? What does a very large σ relative to weight magnitudes imply for the federated model?

**Q51.** Clipping each weight tensor to L2 norm ≤ 1.0 bounds the sensitivity C. What happens if a weight tensor has L2 norm of 300? Does the clipping introduce bias?

**Q52.** ε=1.0 is considered a moderate privacy budget. Would ε=0.1 be stronger or weaker privacy? What would it cost in terms of accuracy, and is there any bound on this?

---

## 9. Krum — Consensus Alignment (not Byzantine Robustness)

**Q53.** Your module docstring was updated to remove "Byzantine fault tolerance" and replace it with "outlier-weight rejection for consensus alignment." Explain this distinction clearly. What does Krum actually do when f=0?

**Q54.** Krum requires n ≥ 2f+3. With n=3 organisations, what is the maximum f (number of tolerated adversaries)? Why is f=0 the correct setting, and what does this mean operationally?

**Q55.** Krum scores each org as the sum of squared L2 distances to its k=max(1, n−f−2) nearest neighbours. With n=3 and f=0, what is k? Walk through the score computation for three orgs A, B, C.

**Q56.** If f=0 means no Byzantine adversary is assumed, why keep Krum at all? What value does it add beyond a plain weighted average?

**Q57.** How would you redesign the system to support genuine Byzantine fault tolerance? How many organisations would you need, and what would change in the federation round?

**Q58.** The build log says an early smoke test showed "BankC score ~113 vs BankA/BankB ~0.03 — correct rejection." What scenario caused such a large BankC score, and is this still valid with f=0?

---

## 10. Shapley Values

**Q59.** State the exact Shapley value formula for player i. What does each term represent in the context of your federated learning system?

**Q60.** For n=3 organisations, list all 7 non-empty coalitions. Write down the Shapley formula for BankA (i=0) with each coalition S explicitly.

**Q61.** How do you compute the coalition value v(S) in your implementation? What does it measure, and what dataset does it require?

**Q62.** Shapley computation requires a trusted aggregator with a shared labelled validation set. What privacy concern does this introduce? What paper discusses this trade-off, and how does your docstring address it?

**Q63.** Your system uses the shared test set slice `X_test[:500]` as the Shapley validation set. Is this a data leakage concern? What is the correct practice?

**Q64.** What happens to an organisation with a negative Shapley value? How is the aggregation weight handled in your code?

**Q65.** If BankC is a malicious attacker that always predicts fraud=1, what Shapley value would you expect it to receive on a dataset with 0.17% fraud, and why?

**Q66.** What is the computational complexity of exact Shapley for n=3? At what n would you need to switch to approximate methods (e.g., Monte Carlo Shapley), and why?

---

## 11. Krum + Shapley Independence

**Q67.** Krum (security) and Shapley (fairness) operate independently in your system. Your updated architecture note states this is "intentional by design." Explain: Krum selects one model, Shapley computes weights — which feeds into which decision?

**Q68.** An org whose weights are Krum-rejected can still earn full federation tokens based on its Shapley value. Is this fair? Under what circumstance would Krum reject a high-Shapley-value org?

**Q69.** Can Krum and Shapley ever conflict in a way that gives a malicious organisation an advantage? Walk through a specific attack scenario.

---

## 12. Blockchain & Hyperledger Fabric

**Q70.** What is the difference between a public blockchain (e.g., Ethereum) and a consortium blockchain (Hyperledger Fabric)? Why is a consortium blockchain appropriate for inter-bank fraud detection?

**Q71.** Describe the Hyperledger Fabric transaction flow: Propose → Endorse → Order → Commit. Which phase is the latency bottleneck?

**Q72.** What is a "leader block" in your system? What objective does DB-BOA minimise for leader selection (Eq.10), and which three node properties are included?

**Q73.** The leader selection objective includes `−0.05 × reputation`. What is the purpose of the reputation term, and how does reputation accumulate?

**Q74.** Your simulated latency now has an explicit disclosure comment: "all latency values are derived from normalised resource scores and time.sleep, not from a live Hyperledger Fabric network." What would a real latency measurement require?

**Q75.** What are the three chaincode functions in your system (`recordFraudResult`, `recordConsensusRound`, `recordFederationRound`)? What data does each write to the ledger?

**Q76.** After Shapley replaced DB-BOA Job 3, both `config.py` and the chaincode were updated. What did the old comments say, and what do they say now?

---

## 13. Incentive Mechanism

**Q77.** Explain the token economy in your system. What are the four events that affect a node's token balance, and what are the reward/penalty values?

**Q78.** A node starts with 100 tokens and earns +10 for leader success and +15 for latency < 300ms. After 5 rounds as leader with all successes and low latency, what is its balance?

**Q79.** Your updated code includes a limitation comment in Phase 8: "A bank that always reports fraud earns tokens for every real fraud event... the mechanism provides no formal deterrent against a patient attacker at higher fraud base rates." Explain this vulnerability with a concrete numerical example at a 5% fraud base rate.

**Q80.** If BankC is a malicious attacker who always votes fraud=1, under what condition does it gain tokens? Under what condition does it lose tokens? Does the current mechanism adequately discourage this behaviour?

**Q81.** Reputation is bounded at [0.5, 2.0]. Why are these specific bounds chosen? Is a node with reputation=0.5 ejected from the network, or merely disadvantaged?

**Q82.** What stronger mechanism would provide formal deterrence against malicious over-reporting? Name one approach from the literature.

---

## 14. Evaluation & Metrics

**Q83.** Why is accuracy alone a poor metric for fraud detection on a 0.17% fraud dataset? What metric would you prioritise, and why?

**Q84.** Explain MCC (Matthews Correlation Coefficient). What does MCC=0.9 mean? What does MCC=0 mean?

**Q85.** Write the full Obf2 formula. The term `1/FPR` creates a numerical singularity when FPR→0. How does your code handle this, and is this a meaningful risk in practice?

**Q86.** NPV is included in Obf2. Why is NPV important for fraud detection beyond Precision?

**Q87.** Sensitivity (Recall) vs. Precision — in fraud detection, why is Sensitivity arguably more important? What is the real-world cost of a False Negative vs. a False Positive?

**Q88.** Your evaluation computes metrics on the test set. Did you check for data leakage? Specifically: could any test samples have influenced the DB-BOA hyperparameter search through the surrogate fitness function?

---

## 15. Baselines & Comparison (run_baselines.py)

**Q89.** Describe the four configurations run by `run_baselines.py` and what each tests.

**Q90.** All four baseline runs share the same DB-BOA hyperparameter search. Why is this important for a fair comparison?

**Q91.** If you ran `run_baselines.py` and FedAvg+Krum performed better than DB-BOA-ADTCN on accuracy, what would that imply about the contribution of Shapley and DP to the system?

**Q92.** `run_baselines.py` uses only one federation round per baseline. The full pipeline in `main.py` uses three rounds. Does this difference invalidate the comparison?

**Q93.** After running `run_baselines.py`, you paste the results into `baseline_metrics()` in `metrics.py`. The visualiser then uses these values. If you have not yet run this script and pasted the results, what does your comparison plot show?

---

## 16. Known Remaining Limitations — Be Ready for These

**Q94.** (Architecture) The Shapley computation uses `X_test[:500]` as the validation set. This is a slice of the same test set used for final evaluation. Does this create any circularity in your results?

**Q95.** (Simulation fidelity) Orgs do not run local training epochs between federation rounds. Your code documents this as a limitation. How would the model accuracy trajectory differ if each org ran 5 local epochs between rounds?

**Q96.** (Generalisability) The ULB dataset comes from a single bank. Your config acknowledges this is a "controlled simulation, not a real cross-bank deployment." What evidence would you need to demonstrate real cross-bank applicability?

**Q97.** (Krum at f=0) With f=0, Krum selects the most consensus-aligned org even when all three orgs are honest. Is the selected org's model actually better than a weighted average of all three, or could Krum sometimes discard useful information?

**Q98.** (Surrogate mismatch) Even with `_MIN_FRAUD_ROWS=30`, the surrogate's effective fraud rate (~1.5%) is 9× higher than deployment (0.17%). How confident are you that the 2D search still finds better hyperparameters than a grid search or random search on the real distribution?

**Q99.** (Sequence padding) The first 9 transactions in any batch use a padded context. Your docstring says this "affects ~0.003% of the dataset and does not meaningfully bias aggregate metrics." What would you do to verify this claim?

**Q100.** (SEQ_LEN=10) Your code now states this was "chosen empirically" and an ablation over {5, 10, 20} is future work. If an examiner asks why 10 is optimal, what is your honest answer?

---

## 17. Novelty & Contributions

**Q101.** List your five claimed novel contributions. For each one, name the closest prior work and explain precisely what is new in your approach.

**Q102.** DB-BOA hybridises DBOA and BOA. Both already exist. What is the specific novelty of the hybrid beyond the sum of its parts — particularly the switching criterion?

**Q103.** Using Shapley values for federated learning contribution attribution is already in Wang et al. (FedSV, IEEE BigData 2020). What does your system add on top of FedSV?

**Q104.** Krum for federated learning is from Blanchard et al. (NeurIPS 2017). DP in FL is from McMahan et al. (2018). ADTCN is from the 2025 paper. What is the specific novelty of the combination on Hyperledger Fabric for financial fraud detection?

**Q105.** Your system runs on the ULB dataset from one bank. A reviewer argues: "Federated learning adds no value here because there is only one real data source." How do you respond?

---

## 18. Design Choices & Alternatives

**Q106.** You chose Krum over coordinate-wise median (Yin et al., ICML 2018). What is the theoretical difference, and when would the median outperform Krum?

**Q107.** You chose the Gaussian mechanism over the Laplace mechanism for DP. Why is Gaussian preferred for vector-valued outputs like model weights?

**Q108.** You chose exact Shapley over approximate methods. At n=3 this requires only 7 coalition evaluations. Would you need to change this for n=10 organisations?

**Q109.** The federation pool of 20 tokens is distributed by Shapley weights. Is this a linear or proportional distribution? Does an org with Shapley weight 0.5 receive exactly 10 tokens?

**Q110.** Why did you choose Hyperledger Fabric over a simpler shared ledger or IPFS for storing federation records? What specific Fabric features justify the added complexity?

**Q111.** The sequence length SEQ_LEN=10 is fixed and unjustified beyond "empirical choice." If you had time, how would you properly ablate this? What range of values would you test?

---

## 19. Scalability & Production Readiness

**Q112.** Your federation runs with 3 organisations and 10 blockchain nodes. How would the system scale to 50 banks and 1,000 nodes? What components are the bottlenecks?

**Q113.** Exact Shapley has O(2^n) complexity. At n=10 organisations you need 1,023 coalition evaluations per round. At n=20 you need 1,048,575. What approximation methods exist, and which would you use?

**Q114.** Basic DP composition after 3 rounds gives ε_total=3.0. In a production system with 1,000 rounds, ε_total=1000 — essentially no privacy. What composition theorem would you use to tighten this bound (Rényi DP, zero-Concentrated DP)?

**Q115.** In a real deployment, would the 500-sample validation set used for Shapley evaluation be available to the aggregator? Who controls this data and how is it kept private?

**Q116.** The latency simulation uses arithmetic formulas. What infrastructure would be required to measure real Hyperledger Fabric throughput across banks, and what latency would you realistically expect for a cross-bank deployment?

---

## 20. Related Work & Positioning

**Q117.** Name three other federated learning papers specifically targeting financial fraud detection. How does your approach differ from each?

**Q118.** What is the difference between horizontal federated learning (your setup) and vertical federated learning? Which is more appropriate for banks sharing fraud data across different customer bases?

**Q119.** FedProx (Li et al., 2020) adds a proximal regularisation term to handle heterogeneous data. Would using FedProx over FedAvg as your baseline have strengthened or weakened your novelty claim?

**Q120.** How does your system compare to GAN-based synthetic data generation approaches for handling class imbalance in fraud detection?

**Q121.** Your paper is a replication and extension of Prabanand & Thanabal (2025). What are the three most significant ways your implementation differs from the original paper?

---

## 21. General Thesis Defense Questions

**Q122.** What is the single most important result in your thesis, and what does it prove?

**Q123.** If you had to redo this thesis from scratch, what would you do differently?

**Q124.** What is the most significant limitation of your work, and how did you document it?

**Q125.** What are the top three follow-up research questions your work raises?

**Q126.** A reviewer says: "Your blockchain adds complexity without a clear security advantage over a trusted aggregator server." How do you respond?

**Q127.** A reviewer says: "Your federated learning simulation uses data from a single dataset split across artificial organisations — this is not real federated learning." How do you respond?

**Q128.** A reviewer says: "With f=0, Krum provides no Byzantine fault tolerance, yet you still call it a security mechanism." How do you respond? (Your updated docstrings directly address this — cite them.)

**Q129.** How would a practitioner at a real bank deploy your system? What infrastructure would they need, and what would the onboarding process look like?

**Q130.** What ethical considerations arise from automating fraud detection decisions using a model whose MTTA layer is GlobalMaxPool, not true attention? How does the lack of interpretability affect accountability?

---

## 22. Mathematical Depth Questions

**Q131.** The Butterfly Optimisation fragrance update is `g_j = d × |f_j|^b`. With d=0.01, b=0.1, and f_j=−5.0, compute g_j to 4 significant figures.

**Q132.** Write the DBOA global search equation (Eq.3) and explain every symbol.

**Q133.** The Shapley formula is `φ_i = Σ [|S|!(n−|S|−1)!/n!] × [v(S∪{i}) − v(S)]`. For n=3 and i=0 (BankA), list all coalitions S ⊆ {1,2} and compute the coefficient for each term. Verify the coefficients sum to 1.

**Q134.** The Gaussian mechanism: σ = C√(2ln(1.25/δ))/ε. With C=1, ε=1, δ=1e-5, compute σ. Your build log states σ ≈ 4.84 — verify this to 2 decimal places.

**Q135.** MCC formula: `(TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))`. If TP=450, TN=56,200, FP=80, FN=30, compute MCC to 3 decimal places.

**Q136.** The DB-BOA switching threshold: `1 − |f_max − f_min| / max(|f_min|, |f_max|)`. What happens when all population members have the same fitness (f_max = f_min = f)? Compute the threshold and explain the behaviour.

**Q137.** For n=3 organisations and f=0, what is k in Krum? With flat weight vectors A=[1,0,0], B=[0,1,0], C=[10,10,10], compute the Krum scores for each and identify which org is selected.

---

## 23. Implementation-Level Questions

**Q138.** In `_ADTCNObjective.__call__`, epoch count is now fixed at `_SURROGATE_EPOCHS=5` and the variable `n_ep` is no longer passed as `params[1]`. Confirm: if DB-BOA tries to propose a configuration with epochs=40 and filters=64, what actually executes?

**Q139.** Your `_make_sequences` function pads the first 9 rows by repeating row 0. If the first transaction happens to be a fraud case, does this mean the first 9 model evaluations see a fraudulent context window? What effect could this have?

**Q140.** Why does your federation manager use `copy.deepcopy(org_models[template_key])` inside `coalition_value()`? What would go wrong if you loaded weights into the original model object directly?

**Q141.** In the attack simulation, `attack_models[attacker_name].predict = lambda X: np.ones(len(X), dtype=int)`. This overrides predict at the instance level. Does this also affect `evaluate_on_validation()`, which calls `self.predict(X_val)` internally?

**Q142.** The results JSON includes a SHA-256 hash of the global weights. What attack does this hash guard against, and what does it not guard against?

**Q143.** `run_baselines.py` evaluates the global model by loading it into `list(org_models.values())[0]` — always BankA. Does this introduce a bias in the comparison between configurations? Why or why not?

**Q144.** In `run_baselines.py`, the FedAvg baseline uses `_avg_weights()` which is a plain equal-weight average. The full pipeline uses Shapley-weighted aggregation. Is this comparison fair — is FedAvg in the baselines the same algorithm as the FedAvg described in McMahan et al. (AISTATS 2017)?

---

*Prepared for thesis defense. Questions are ordered from broad to technical. Section 16 (Q94–Q100) targets the remaining honest limitations disclosed in the updated code.*
