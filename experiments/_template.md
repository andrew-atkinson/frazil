# <Experiment name>

**Date:** YYYY-MM-DD · **Branch:** `exp/<name>` · **Status:** planned / in progress / done · **Issue:** ISS-nnn

## Question
One or two sentences: what are we trying to find out, and why now?

## Setup
- Recipe: `configs/experiment/<name>.yaml` (if it trains a model), seeded from `<checkpoint>`.
- Data: which datasets, years, and any new preprocessing.
- What is compared against what (the A/B, the baseline, the control).

## Commands
```bash
python train.py +experiment=<name>
python scripts/project/freerun_monthly.py ... --note "<name>: ..."
```

## Runs
| what | ledger run_id | output |
| --- | --- | --- |

## Results
Tables and figures (figures go in `plots/`; link or describe them).

## Conclusion
What it means, what it rules in or out, and what to do next (new issues in `docs/ISSUES.md`).
