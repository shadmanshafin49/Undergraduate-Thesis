# Novelty Tips for Thesis Improvement

## Tier 1 — High impact, low effort (do these first)

### ✅ 1. Replace synthetic data with a real benchmark — DONE

**Why it matters**: 99% accuracy on self-generated data proves nothing. Real data has class
imbalance, noise, and concept drift — problems your system claims to solve.

**Concretely**: Use one of these free datasets:
- **PaySim** (Kaggle) — based on real M-Pesa transactions, 6.3M rows, 0.13% fraud
- **ULB Credit Card Fraud** (Kaggle) — 284k rows, 0.17% fraud, 28 PCA features

Drop-in replacement for `db_boa_framework/data/data_loader.py`. Accuracy will fall to
~92–96% — that is a stronger result because it is honest.

---

### ✅ 2. Add real Byzantine-robust aggregation (Krum) — DONE

**Why it matters**: The current "attack resilience" is theatrical. A token penalty does not
prevent a malicious model from corrupting global weights before the penalty fires.

**Concretely**: Implement the **Krum rule** (Blanchard et al., NeurIPS 2017). ~20 lines:

```python
# For each org, compute sum of squared distances to its n-f-2 nearest neighbours.
# Select the org with the minimum score as the global model (no averaging).
```

Replaces the weighted average in `federation_manager.py`. Makes the attack resilience
claim technically defensible with a peer-reviewed citation.

---

### ✅ 3. Add differential privacy to weight sharing — DONE

**Why it matters**: The current federated setup requires a **shared labelled validation
set at the aggregator** — this breaks the privacy premise entirely. DP fixes it.

**Concretely**: Clip weights to norm 1, then add Gaussian noise before sharing:

```python
def extract_weights_with_dp(self, epsilon=1.0, delta=1e-5):
    weights = self.extract_weights()
    sensitivity = 1.0
    sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
    return [w + np.random.normal(0, sigma, w.shape) for w in weights]
```

Gives a formal ε-DP guarantee. Removes the need for shared validation data in the
aggregation objective. Cite Dwork et al. (2006). FL + DP is one of the most active
areas in 2025–2026 research.

---

## Tier 2 — Medium effort, strong thesis differentiator

### ✅ 4. Replace MLPClassifier with a real temporal model — DONE

**Why it matters**: Calling something "Temporal Context Learning" while using
`sklearn.MLPClassifier` is indefensible. An MLP treats the 10-timestep sequence as a
flat feature vector — it ignores temporal order entirely.

**Concretely**: A PyTorch 1D-CNN over the last 10 transactions (sequence_length=10 is
already configured):

```python
# Input shape: (batch, 10, 30) — 10 timesteps, 30 features
# Conv1d(30, 64, kernel_size=3) → ReLU → Conv1d(64, 128, 3) → GlobalMaxPool → Linear(2)
```

~30 lines of PyTorch. The model actually exploits temporal order. The "TCL" claim
becomes true.

---

### ✅ 5. Replace DB-BOA Job 3 with Shapley-value contribution weights — DONE

**Why it matters**: Using a population metaheuristic to optimise 3 scalars that sum to 1
is overkill AND requires shared data. Shapley values solve the same problem — how much
did each org contribute? — with game-theoretic fairness guarantees and no shared dataset.

**Concretely**: For 3 orgs there are only 7 non-empty coalitions. Compute exact Shapley
values in 7 forward passes. Each org's aggregation weight becomes proportional to its
marginal contribution. Replaces `_run_db_boa_job3()` entirely.

**2026 relevance**: Shapley-based FL fairness (FedSV, CGSV) is one of the most cited FL
research directions right now.

---

## Tier 3 — High effort, only if time allows

### ✅ 6. Add transaction graph features — DONE

Build a bipartite sender → receiver graph with NetworkX. Extract node-level features
(in-degree, out-degree, PageRank) and append them to the existing 30 features. Graph-
based fraud detection (GraphSAGE, GCN) dominates the 2024–2026 literature. Even non-GNN
graph features consistently beat tabular-only baselines.

---

## What NOT to do

- Do not add more metaheuristics — the space is saturated and reviewers are fatigued by it.
- This system runs exclusively on Hyperledger Fabric — do not introduce references to other blockchain platforms.
- Do not add more chaincode features. The chain is already doing little that is critical;
  more ledger entries will not strengthen the contribution.
- Do not justify DB-BOA over Optuna or random search without a convergence speed
  experiment. If you cannot show DB-BOA is faster or more sample-efficient, the
  optimisation claim is weak.

---

## Recommended thesis narrative for 2026

> *"We propose a privacy-preserving federated fraud detection framework with formally-
> bounded differential privacy guarantees (ε-DP), Byzantine-robust aggregation (Krum),
> and blockchain-based audit trails on Hyperledger Fabric. We evaluate on the PaySim
> benchmark using a temporal 1D-CNN classifier. Compared to vanilla FedAvg, we show X%
> accuracy retention under Y% Byzantine nodes at ε = 1.0."*

That is a concrete, falsifiable claim using 2026-relevant techniques — fully buildable
by one undergraduate in 2–3 months.

---

## Key citations to add

| Contribution | Citation |
|---|---|
| Krum (Byzantine-robust FL) | Blanchard et al., NeurIPS 2017 |
| Differential privacy | Dwork et al., 2006 |
| FedAvg baseline | McMahan et al., AISTATS 2017 |
| Shapley FL fairness | Wang et al. (FedSV), 2020 |
| PaySim dataset | Lopez-Rojas et al., 2016 |
| Graph fraud detection | Liu et al. (Pick & Choose GNN), 2021 |
