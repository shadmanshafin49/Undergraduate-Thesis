# Thesis Defense Questions — DB-BOA Financial Security Framework

**Framework**: DB-BOA + ADTCN + Federated Learning + Hyperledger Fabric for Credit Card Fraud Detection  
**Based on**: Prabanand & Thanabal (2025), Scientific Reports 15, 6764

---

## 1. Overview & Motivation

**Q1.** In one sentence, what is the core problem your thesis solves, and why does it need a blockchain?

**Q2.** Why is credit card fraud detection specifically suited to a federated learning setup across banks, rather than a centralised model trained on pooled data?

**Q3.** The ULB dataset has only 0.17% fraud. How does such extreme class imbalance affect your model, and how did you address it?

**Q4.** What would happen if you simply used a standard centralised LSTM or transformer trained on pooled bank data? Why is that approach unacceptable in practice?

**Q5.** Your system has five major components: DB-BOA, ADTCN, Federated Learning, Krum, Shapley, DP, and Hyperledger Fabric. Which single component contributes the most to detection accuracy, and how do you know?

**Q6.** What is the threat model of your system? Who is the adversary, what can they do, and what are the limits of your security guarantees?

---

## 2. DB-BOA Algorithm

**Q7.** Explain the DB-BOA switching criterion (Equation 1). When does it choose DBOA over BOA, and why?

**Q8.** Your switching threshold is `1 − |f_max − f_min| / max(|f_min|, |f_max|)`. What does this ratio approach when the population has converged? What does it approach when spread out?

**Q9.** DBOA has two movement equations (Eq.3 and Eq.4 — global and local search). What determines which one fires, and what does the parameter ρ=0.8 mean for exploration vs. exploitation balance?

**Q10.** What is LSAM (Local Search with Adaptive Mutation)? How many evaluations does it add per iteration, and what is the total evaluation count for your default settings (population=20, iterations=30)?

**Q11.** BOA uses billiards-table pockets. Explain in plain language what a "pocket" represents and how the billiards movement equation (Eq.8) shifts a solution toward a pocket.

**Q12.** DB-BOA is applied to three different jobs in your system (Job 1: hyperparameter tuning, Job 2: leader selection, Job 3: federated weights). Do all three use the same fitness function? If not, what changes?

**Q13.** What is Obf2, and why does DB-BOA minimise `−Obf2` rather than maximise Obf2 directly?

**Q14.** How do you verify that DB-BOA actually improves over default hyperparameters? What is the baseline comparison for Job 1?

**Q15.** Your summary statistics (Min, Max, Mean, Std, Median) over convergence history — what exactly is being summarised? Is it the best-so-far fitness or the per-iteration best?

---

## 3. ADTCN Architecture

**Q16.** Your paper calls the model "ADTCN — Adaptive Deep Temporal Context Networks." Describe the exact layer sequence of your implementation and what each layer does.

**Q17.** You use Conv1d with a sliding window of SEQ_LEN=10 transactions. How do you construct these sequences from the flat transaction feature matrix? What happens for the first 9 rows?

**Q18.** Why 1D-CNN instead of LSTM or Transformer for temporal modelling of transaction sequences? What is the trade-off?

**Q19.** Your architecture is `Conv1d(30→F, k=3) → ReLU → Conv1d(F→2F, k=3) → GlobalMaxPool → Linear(2F→2)`. The paper labels these as MJE, TCL, MTTA, OUT layers. Map each label to the corresponding PyTorch layer.

**Q20.** GlobalMaxPool is used instead of GlobalAveragePool. What is the theoretical justification for this choice in the context of anomaly/fraud detection?

**Q21.** How does the weighted cross-entropy loss work for a 0.17% fraud rate? Compute the class weight ratio your code would assign to the fraud class if the training set has 300 fraud and 227,000 normal samples.

**Q22.** The ADTCN uses `activation=tanh` in config but the 1D-CNN layers use ReLU. Which is actually used, and does this match the paper?

---

## 4. Dataset & Feature Engineering

**Q23.** Describe the ULB Credit Card Fraud dataset. How many samples, features, and what is the fraud rate?

**Q24.** The dataset has 28 PCA-transformed features V1–V28 plus Amount and Time. Why does your code treat these as 30 base features? What are the 3 "graph features" appended on top?

**Q25.** Your "graph features" are actually sliding-window frequency features (`amount_recurrence_before`, `amount_recurrence_after`, `degree_ratio`). Why do you call them graph features? What real graph-based feature would require account-level IDs?

**Q26.** You use an 80/20 train-test split and 10% of training for validation. Does your test set leak into any part of the optimisation or training process?

**Q27.** In the DB-BOA surrogate evaluation, you use a 2,000-row stratified subsample with a ~50% fraud rate, but the deployment data has 0.17% fraud. How does this class distribution mismatch affect the hyperparameters found by DB-BOA?

**Q28.** How do you split data across three federated organisations (BankA 50%, BankB 30%, BankC 20%)? Is this an i.i.d. or non-i.i.d. split? Does the fraud rate remain 0.17% in each organisation's slice?

---

## 5. Federated Learning

**Q29.** What is Federated Averaging (FedAvg)? Why is it your baseline and not standard centralised training?

**Q30.** In your federated simulation, organisations share model weights, not raw data. What information can an adversary potentially infer from the shared weights alone (model inversion / gradient leakage)?

**Q31.** Your federation triggers every 5 consensus rounds. What is the architectural reason for this interval, and how does it relate to the Hyperledger Fabric consensus cycle?

**Q32.** After a federation round, all organisations load the global weights. Does each organisation's model then continue local training, or does it freeze at the global model? What are the implications for model drift?

**Q33.** You simulate 3 federation rounds. Is this sufficient to draw conclusions about convergence? How many rounds would a production system require?

**Q34.** The data split (50/30/20) means BankA contributes much more data. How does this affect the fairness of the aggregation without Shapley weighting?

---

## 6. Differential Privacy

**Q35.** State the Gaussian mechanism formula for (ε, δ)-DP: what is σ as a function of sensitivity C, ε, and δ?

**Q36.** Your code uses C=1.0 (L2 clipping norm), ε=1.0, δ=1e-5. Compute σ numerically.

**Q37.** What does it mean for your weight sharing to be (1.0, 1e-5)-DP? What adversarial capability does it protect against?

**Q38.** DP noise is added to weights before sharing. How does this affect model accuracy? Did you measure the accuracy drop from DP at ε=1.0?

**Q39.** Clipping each weight tensor to L2 norm ≤ 1.0 is a prerequisite for the DP guarantee. What happens if a weight tensor has norm 500? Does the clipping introduce bias?

**Q40.** ε=1.0 is considered a moderate privacy budget. Would ε=0.1 be stronger or weaker privacy? What would it cost in terms of accuracy?

**Q41.** Does your DP guarantee compose across multiple federation rounds? If you run 3 rounds, what is the total privacy budget consumed (using basic or advanced composition)?

---

## 7. Krum & Byzantine Fault Tolerance

**Q42.** State the Krum Byzantine-robust aggregation algorithm. What does the Krum score represent, and which organisation is selected?

**Q43.** Krum requires n ≥ 2f+3. With n=3 organisations, what is the maximum f (number of Byzantine adversaries) it can tolerate?

**Q44.** Your configuration sets `byzantine_f=0` with n=3. The code comment says "f=0 means no Byzantine adversary is assumed." If f=0, what does Krum actually do in your system?

**Q45.** The module docstring says "Krum → security (Byzantine fault tolerance)" but f=0 means no tolerance. Is this claim accurate, and how would you correct it?

**Q46.** If you wanted genuine Byzantine fault tolerance with Krum, how many organisations would you need? What would change in the system?

**Q47.** In the attack simulation, BankC always reports isFraud=True. Does your Krum configuration (f=0) prevent BankC's malicious weights from corrupting the global model?

**Q48.** What is the difference between Krum (which selects one model) and coordinate-wise median (Yin et al., ICML 2018)? When would you prefer one over the other?

---

## 8. Shapley Values

**Q49.** State the Shapley value formula for player i. What does each term represent?

**Q50.** For n=3 organisations, how many coalitions are there? List all of them.

**Q51.** How do you compute the coalition value v(S) in your implementation? What model does it produce, and what dataset does it evaluate on?

**Q52.** Shapley computation requires a trusted aggregator with a shared labelled validation set. What privacy concern does this introduce? Which paper discusses this trade-off?

**Q53.** Your Shapley weights become the on-chain incentive record (replacing DB-BOA Job 3). What happens to an organisation with a negative Shapley value? How is the aggregation weight handled?

**Q54.** What is the computational complexity of exact Shapley for n=3 orgs? At what n would you need approximation methods (e.g., Monte Carlo Shapley)?

**Q55.** If BankC is a malicious attacker that always predicts fraud=1, what Shapley value would you expect it to receive, and why?

---

## 9. Blockchain & Hyperledger Fabric

**Q56.** What is the difference between a public blockchain (e.g., Ethereum) and a consortium blockchain (Hyperledger Fabric)? Why is a consortium blockchain appropriate for inter-bank fraud detection?

**Q57.** Describe the Hyperledger Fabric transaction flow: Propose → Endorse → Order → Commit. Which phase is the bottleneck for latency?

**Q58.** What is a "leader block" in your system? Who selects the leader, and what objective does DB-BOA optimise for leader selection (Eq.10)?

**Q59.** The leader selection objective is `CT + CC + MS − 0.05 × reputation`. Why is a reputation bonus included? How does this affect fairness across nodes?

**Q60.** Your simulation uses 10 nodes. What is the quorum (minimum endorsements required)? Is this realistic for a production consortium?

**Q61.** What are the three chaincode functions in your system (`recordFraudResult`, `recordConsensusRound`, `recordFederationRound`)? What data does each write to the ledger?

**Q62.** The `INCENTIVE_CONFIG` comment still says "shared by DB-BOA weight across orgs" after Shapley replaced DB-BOA Job 3. Which mechanism actually distributes the federation pool — Shapley or DB-BOA?

**Q63.** How does the on-chain incentive mechanism discourage a malicious bank from consistently misreporting fraud verdicts?

---

## 10. Incentive Mechanism

**Q64.** Explain the token economy in your system. What are the four events that affect a node's token balance, and what are the reward/penalty values?

**Q65.** A node starts with 100 tokens and earns +10 for leader success and +15 for latency < 300ms. After 5 rounds as leader with all successes and low latency, what is its balance?

**Q66.** If BankC is an attacker who always reports fraud, under what condition does it gain tokens rather than lose them? Does this mean the incentive mechanism can reward malicious behaviour?

**Q67.** Reputation is bounded at [0.5, 2.0]. Why are these specific bounds chosen? What happens to a node whose reputation falls to 0.5 — is it ejected from the network?

**Q68.** The latency threshold is 300ms. Is this realistic for a cross-bank Hyperledger Fabric deployment? What would you expect real-world latency to be?

---

## 11. Evaluation & Metrics

**Q69.** Why is accuracy alone a poor metric for fraud detection on a 0.17% fraud dataset? What metric would you prioritise, and why?

**Q70.** Explain MCC (Matthews Correlation Coefficient). What does MCC=0.9 mean, and what does MCC=0 mean?

**Q71.** What is Obf2 (your composite fitness function)? Write the formula and explain why each component is included.

**Q72.** Obf2 includes `1/FPR`. What happens to this term if FPR approaches zero? Does this create a numerical instability in your fitness evaluation?

**Q73.** NPV (Negative Predictive Value) is included in Obf2. Why is NPV important for fraud detection beyond Precision?

**Q74.** You report Sensitivity (Recall). In fraud detection, why is Sensitivity arguably more important than Precision? What is the cost of a False Negative vs. a False Positive?

**Q75.** Your evaluation computes metrics on the test set. Did you check for data leakage — specifically, could any test samples have influenced the DB-BOA hyperparameter search?

---

## 12. Baselines & Comparison

**Q76.** Your `baseline_metrics()` function returns empty dictionaries. How do you compare your model against baselines if no baseline results exist?

**Q77.** The original paper compares against MBO-ADTCN, WSA-ADTCN, DBOA-ADTCN, BOA-ADTCN. Why were these synthetic baselines removed from your replication?

**Q78.** What would be a fair ablation study for your system? List the four configurations you would need to run.

**Q79.** FedAvg is your primary baseline. How would you implement a pure FedAvg baseline using your existing code? What flags would you set?

**Q80.** How would you compare your system against a strong centralised baseline (e.g., XGBoost or LightGBM trained on all data)?

---

## 13. Known Weaknesses — Be Ready for These

**Q81.** (Critical) With n=3 and f=0, Krum satisfies the formula n ≥ 2f+3 but tolerates zero Byzantine adversaries. How does this affect the validity of citing Blanchard et al. (NeurIPS 2017) in the context of your system's security claim?

**Q82.** (Critical) Your DB-BOA surrogate uses a 50/50 class split for hyperparameter optimisation, but the real deployment data has 0.17% fraud. How confident are you that the optimal hyperparameters found under the surrogate distribution are also optimal for the real distribution?

**Q83.** (Significant) The baseline comparison table in your thesis is empty. How do you substantiate the claim that DB-BOA-ADTCN outperforms FedAvg?

**Q84.** (Significant) What is the runtime of your full DB-BOA hyperparameter search with population=20 and iterations=30? Is this acceptable for a production banking system?

**Q85.** (Minor) `BASELINE_NAMES` in config.py still lists MBO-ADTCN, WSA-ADTCN, etc. These were produced on synthetic data. If these names appear on your evaluation plots, how do you explain the mismatch?

**Q86.** (Minor) The `federation_pool` comment in INCENTIVE_CONFIG says "shared by DB-BOA weight" but Shapley now distributes this pool. Which mechanism actually runs in your code?

---

## 14. Novelty & Contributions

**Q87.** List your five claimed novel contributions. For each one, name the closest prior work and explain precisely what is new in your approach.

**Q88.** DB-BOA hybridises DBOA and BOA. Both already exist. What is the specific novelty of the hybrid beyond the sum of its parts?

**Q89.** Using Shapley values for federated learning contribution attribution is already in Wang et al. (FedSV, IEEE BigData 2020). What does your system add on top of FedSV?

**Q90.** Krum for Byzantine-robust federated learning is from Blanchard et al. (NeurIPS 2017). Differential privacy in FL is from McMahan et al. (2018). ADTCN fraud detection is from the 2025 paper. What is the specific novelty of combining these on Hyperledger Fabric?

**Q91.** Your system uses both Krum (security) and Shapley (fairness). These optimise different objectives. Can they conflict? For example, if Krum rejects BankC's model but Shapley assigns BankC a high contribution, what happens?

---

## 15. Design Choices & Alternatives

**Q92.** You chose Krum over coordinate-wise median (Yin et al., ICML 2018) and FLTrust (Cao et al., 2022) for Byzantine robustness. What is the reason for this choice?

**Q93.** You chose Gaussian mechanism for DP. What is the alternative (Laplace mechanism), and why is Gaussian preferred for vector-valued outputs like model weights?

**Q94.** You chose exact Shapley over approximate methods. At n=3, exact computation is cheap (7 coalitions). Would you need to change this for 10 organisations?

**Q95.** Why did you choose Hyperledger Fabric over a simpler shared database or IPFS for storing federation records?

**Q96.** The sequence length SEQ_LEN=10 is fixed. How did you choose this value? Did you ablate over different sequence lengths?

**Q97.** You use GlobalMaxPool over the temporal dimension. Would attention over time steps improve performance? Why or why not?

**Q98.** The ULB dataset has no account IDs — transactions are anonymous. Does this mean your "federated" setup is actually simulating independent institutions, not truly federated ones with user overlap?

---

## 16. Scalability & Production Readiness

**Q99.** Your federation runs with 3 organisations and 10 blockchain nodes. How would the system scale to 50 banks and 1,000 nodes? What components are the bottlenecks?

**Q100.** Exact Shapley has O(2^n) complexity. At n=10 organisations, you need 1,023 coalition evaluations per round. Is this feasible?

**Q101.** What happens to the privacy guarantee if the number of federation rounds increases from 3 to 1,000? How does DP composition affect ε over many rounds?

**Q102.** In a real deployment, would the validation set (used for Shapley coalition evaluation) be available to the aggregator? Who controls this data and how is it kept private?

**Q103.** The latency simulation in your blockchain module uses `time.sleep` and arithmetic formulas. Is this a valid measure of Hyperledger Fabric throughput? How would you measure real throughput?

---

## 17. Related Work & Positioning

**Q104.** Name three other federated learning papers specifically targeting financial fraud detection. How does your approach differ from each?

**Q105.** What is the difference between horizontal federated learning (your setup) and vertical federated learning? Which is more appropriate for banks sharing fraud data?

**Q106.** FedProx (Li et al., 2020) adds a proximal term to handle heterogeneous data. Does your non-i.i.d. data split (50/30/20) motivate using FedProx over FedAvg?

**Q107.** Your paper cites McMahan et al. (AISTATS 2017) for FedAvg. What is the core assumption of FedAvg that your non-i.i.d. data split violates?

**Q108.** How does your system compare to GAN-based synthetic data generation approaches for handling class imbalance in fraud detection?

---

## 18. General Thesis Defense Questions

**Q109.** What is the single most important result in your thesis, and what does it prove?

**Q110.** If you had to redo this thesis from scratch, what would you do differently?

**Q111.** What is the most significant limitation of your work?

**Q112.** What are the top three follow-up research questions that your work raises?

**Q113.** A reviewer says: "Your blockchain adds complexity without a clear security advantage over a trusted aggregator." How do you respond?

**Q114.** A reviewer says: "Your federated learning simulation uses data from a single dataset split across artificial organisations — this is not real federated learning." How do you respond?

**Q115.** A reviewer says: "Without a real baseline comparison on ULB, you cannot claim state-of-the-art performance." How do you respond?

**Q116.** How would a practitioner at a real bank deploy your system? What infrastructure would they need, and what would the onboarding process look like?

**Q117.** What ethical considerations arise from automating fraud detection decisions using a black-box model? How does your system address explainability?

**Q118.** If your system incorrectly flags a legitimate transaction as fraud (False Positive) and the bank blocks the customer's card, who is accountable — the algorithm, the bank, or the model developer?

---

## 19. Mathematical Depth Questions

**Q119.** The Butterfly Optimisation Algorithm fragrance update is `g_j = d × |f_j|^b`. With d=0.01, b=0.1, and f_j=−5.0, compute g_j.

**Q120.** Write the DBOA global search equation (Eq.3) and explain every symbol.

**Q121.** The Shapley formula is `φ_i = Σ [|S|!(n−|S|−1)!/n!] × [v(S∪{i}) − v(S)]`. For n=3, i=0 (BankA), list every coalition S and compute the coefficient for each term.

**Q122.** The Gaussian mechanism adds noise σ = C√(2ln(1.25/δ))/ε. With C=1, ε=1, δ=1e-5, compute σ to 4 decimal places.

**Q123.** MCC formula: `(TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))`. If TP=400, TN=56,000, FP=100, FN=50, compute MCC.

**Q124.** The DB-BOA convergence threshold is `1 − |f_max − f_min| / max(|f_min|, |f_max|)`. Why is this bounded to [0,1]? Prove it cannot be negative.

---

## 20. Implementation-Level Questions

**Q125.** In `_ADTCNObjective.__call__`, the surrogate training hard-caps epochs at `_SURROGATE_EPOCHS=5` regardless of what DB-BOA proposes. Why? Does this invalidate the fitness signal?

**Q126.** Your `_make_sequences` function pads the first SEQ_LEN−1 rows by repeating row 0. Does this boundary padding create a bias in how the first few transactions are classified?

**Q127.** Why does your federation manager use `copy.deepcopy(org_models[template_key])` to evaluate coalition values? What would go wrong if you loaded weights into the original model object?

**Q128.** In the attack simulation, `attack_models[attacker_name].predict = lambda X: np.ones(len(X), dtype=int)`. Is this a valid override of the model's predict method? What limitation does this have?

**Q129.** The `baseline_metrics()` function returns empty dicts but `main.py` iterates over them to print accuracy deltas. What output does the user see when baselines are empty?

**Q130.** Your results are saved as JSON with a SHA-256 hash of the global weights. What is the purpose of this hash, and what does it prove about the integrity of the federation round?

---

*Prepared for thesis defense — questions are ordered from broad to detailed, with known weak points in Section 13.*
