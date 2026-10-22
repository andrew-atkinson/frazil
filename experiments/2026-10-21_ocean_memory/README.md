# Ocean memory: information test

**Date:** 2026-10-21 · **Branch:** `exp/daily-degree-days` · **Status:** done (no training) · **Issue:** ISS-002

## Question
The emulator sees only the current month. The real ocean stores the summer's absorbed sunlight and melts ice later and nearby, which could explain too little summer melt on the Pacific shelf. Before building an ocean-memory input (sunlight absorbed by open water since spring) and training on it, does it carry information the emulator lacks? The degree-day experiment trained on inputs that turned out to add none; this test checks first.

## Setup
`info_test.py` (this folder). On the QM-target datacube the model trains on, every 4th cell of the seasonal ice zone (ice in March, not full in September), months t = June–September predicting July–October:
- **Baseline inputs:** everything the emulator sees at a step: its 6 states at t, the 4 forcings at t and t+1, the 4 degree-days, the month and its position (24 inputs).
- **+ memory:** the sum over April..t of (1 − sic) × top-of-atmosphere insolation (exact from latitude and date).
- **+ last two months of sic:** a generic "any history" comparison.
- A small neural network (2 × 128) per variant, fit on 1995–2006, early-stopped on 2007–10, scored on 2011–18, three seeds each. 451,392 rows.

## Results
Test RMSE of next-month concentration, 2011–2018, mean of 3 seeds (seed-to-seed spread of the baseline in brackets):

| | all | July | August | September | October | East Siberian | Chukchi | Beaufort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.1218 (0.003) | 0.139 | 0.127 | 0.105 | 0.113 | 0.148 (0.009) | 0.119 (0.009) | 0.138 |
| + memory | −1.1% | −1.3% | −1.4% | +0.5% | −1.8% | −0.3% | +0.5% | −0.7% |
| + last two months of sic | −0.9% | −0.6% | −2.1% | +1.1% | −1.8% | −0.9% | +0.3% | −0.3% |

The memory term improves prediction by about 1%, which is within the seed-to-seed spread (2.5% overall, 6% on the Pacific shelf), and not at all where the summer gap is (East Siberian −0.3%, Chukchi +0.5%). Any two months of history do about as well. For the teacher, next month's ice is about as predictable from the current state and weather as with its history: the current thickness, snow and concentration already carry what the past summer did.

## Conclusion
Fails the test: no ocean-memory input, no training run. The emulator isn't missing information about the next month. That fits what else is known: one-month skill is good (51–61% better than persistence), while the September gap is small in the first years of a free-run (+0.12 in 2019–22) and larger later (+0.34 in 2023–25). The gap is the emulator's own error accumulating as it runs on its own predictions, not a missing input. Next experiments should target that drift (training on longer stretches of its own predictions), not new inputs.
