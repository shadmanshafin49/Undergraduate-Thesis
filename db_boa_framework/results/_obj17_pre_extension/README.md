# Pre-OBJ-17 snapshot — the 9-point ε grid

Archived 2026-09-04, immediately before OBJ-17 re-ran all three conditions on the
extended 11-point grid `[1,5,10,30,50,100,300,1000,3000,10000,30000]`.

**Why this directory exists.** The two BankSim JSONs were untracked, so the
re-run would have destroyed them. These six files are the *only* evidence for
the budget factors that were published while the grid stopped at 3000:

| Condition | ε\*(weight) | ε\*(output) | factor as published | weight inversions @ ε=3000 |
|---|---|---|---|---|
| ULB / stratified          | 3000 | 50  | ≥60× | 20 / 100 |
| BankSim / stratified      | 3000 | 100 | ≥30× | **1 / 100** |
| BankSim / entity-disjoint | 3000 | 300 | ≥10× | 38 / 100 |

Every one of those factors is a right-censored **lower bound**: ε\* sat on the
top of the grid in all three conditions, so the true ε\*(weight) lies beyond it.
The BankSim/stratified row is additionally decided by a single inverted draw at
*both* ends of its ratio (output ε\*=100 also sits at 1/100), which is why
OBJ-17 pairs the grid extension with a `--repeats` probe rather than treating
the extension as sufficient on its own.

These files predate the `eps_grid` / `n_repeats` / `*_censored` fields, so a
reader must take the grid from the `sweep` row list. That is exactly the defect
OBJ-17 fixed; do not use these as a template for a new run.
