## ▮ RESUME HERE — handoff 2026-09-01 (updated 20:20)

> First thing a new session reads. Delete or rewrite this block once the run below has landed.

### ⏳ IN FLIGHT — BankSim/`stratified` confound control, launched 2026-09-01 20:20

The four sweeps are running again on BankSim with the **stratified** partition. ETA ~3 h
(the entity-disjoint run took 3 h 15 m). Nothing else should touch the CPU until it lands.

```powershell
cd "d:\THESIS\FINAL PROJECT\DB-BOA-FEL-ADTCN-Hyperledger-Fabric-main\db_boa_framework"
.\experiments\check_obj15.ps1 -Partition stratified      # progress, read-only, run any time
```

**Why this run exists.** Every BankSim↔ULB comparison in OBJ-15 changed **dataset and partition
together**, so not one of its divergences could be attributed to either alone. This run separates
them:

| Comparison | What moves | What it isolates |
|---|---|---|
| ULB vs **BankSim/stratified** | dataset only | **the dataset** |
| **BankSim/stratified** vs BankSim/customer | partition only | **entity-disjointness** |

**✅ The single-factor claim was verified before launching, not assumed.** None of the four sweeps
passes `groups=` to `ADTCN.fit()` (grepped: `economic_byzantine_sweep:194`,
`byzantine_robustness_sweep:119`, `private_incentive_sweep:107`, `scalability_sweep:234,309` all
call `split_for_orgs` and then `fit()` with no groups), and `fit(groups=None)` builds **global**
windows (`adtcn.py:593`). So **both partitions window identically** and the partition really is the
only factor moving. The windowing confound recorded for the federated ablation in CONFIRMED KILLS
(`baselines_banksim_*`) does **not** apply to these four sweeps — those two runs differ in
partition *and* windowing; these do not.

> ⚠ **The other side of that same fact, and it must be reported:** the OBJ-15 entity-disjoint
> sweeps therefore measured a federation whose orgs hold **disjoint customers but global windows**.
> "Krum pays 1.5–12.6 pp of utility under entity-disjointness" is measured *without* entity-linked
> windows inside each org. That is honest scoping, not a bug — but do not describe those sweeps as
> using customer-linked sequences, because they do not.

**What each sweep is expected to say (pre-registered 2026-09-01 20:20, before any output — rule 4).**
Do not edit these after results land; score them underneath.

- **Krum's utility cost tracks the *partition*, not the dataset.** Predicted: BankSim/stratified
  puts Krum back at or near parity with unprotected FedAvg (ULB stratified was +12.49 pp,
  BankSim/customer was −12.57 to −1.48 pp). The mechanism argued in OBJ-15 — Krum selects one
  org's weights, which generalise worse only when orgs hold *different* customers — predicts the
  cost is a property of entity-disjointness. **If Krum is still negative under stratified, that
  mechanism is wrong and the finding is about BankSim, not about federation structure.** That is
  the most valuable outcome available here.
- **Krum still rejects 8/8.** It is geometric; nothing in the partition should touch it.
- **The lone-attacker isolation result follows the partition too.** Predicted: back toward ULB's
  0/3 firing. If it still fires 3/3 under stratified, the cause is the dataset (or the even-voter
  quorum arithmetic already flagged), not org heterogeneity.
- **ε\* ordering holds a third time** (output channel ≫ weight channel at every ε). The constants
  may move again; the ordering is the claim. **ε\* will still be right-censored** — the grid still
  stops at 3000 and the weight channel was still inverting there on both previous runs.
- **Exact-Shapley cost is unchanged.** O(2ⁿ) over the same 4,095 coalitions; genuinely nothing to
  learn, stated so it is not later reported as a finding.
- **MC top-1 stays intermittent.** Withdrawn as a threshold in OBJ-15; predicted to stay
  unreliable at every n, with no clean partition story.

**Operational notes worth keeping:**
- `.\experiments\check_obj15.ps1 [-Partition customer|stratified]` reports liveness, elapsed,
  per-sweep state and landed JSONs. Both runner and checker now take `-Partition`.
- **Logs are now per-run**: `results/_obj15_logs/banksim_<partition>/`. The 2026-09-01 09-01
  entity-disjoint logs were moved into `banksim_customer/` — a second run would otherwise have
  overwritten the first one's evidence, which is the same class of mistake as the unsuffixed
  figures caught during OBJ-15.
- Judge a running sweep by **CPU time climbing**, not log freshness — prints are flushed but
  sparse (the first line after the banner waits for a whole org to train).
- The per-sweep logs are **UTF-16LE** (PowerShell 5.1 `*>`), so `grep` from Git Bash reads
  nothing from them; `iconv -f UTF-16LE` first. `_runner.out.log` is ASCII.
- ULB figures for the four sweeps are backed up in `results/_ulb_figures_backup/`.
- ⚠ **Do not edit `private_incentive_sweep.py` (the ε grid) while this run is queued** — the
  sweeps launch as separate processes in sequence, so an edit lands mid-run and the stratified
  arm would no longer share a grid with the other two.

**Regenerating any draft is free** — no re-run needed:

```powershell
python experiments\<sweep>.py --dataset banksim --partition stratified --redraft
python experiments\objective_noise_audit.py --render-only
```

### Then, in order
