# Loose ice that opens too slowly

**Date:** 2026-11-05 · **Branch:** `exp/loose-ice` · **Status:** in progress (both hypotheses rejected; the gap is partly-open ice, also on training years; sampler clamp ruled out; suspect the censored loss at 0)

## Question
From the target's own state, v0.4.0's August and September shelf ice cover is too high: loose ice opens too slowly and compact ice melts too fast, including on years it trained on ([summer_melt](../2026-10-30_summer_melt/)). Ocean heat isn't the cause. Is this how the model was trained, or how it's sampled?

## Hypotheses
- **Training from its own states.** 80% of v0.4.0's fine-tune starts from a state the model generated. That input carries the model's own errors, so what it learns should depend less on the starting state than the truth does: too little melt from low concentrations, too much from high ones, when started from a clean state. That is the pattern seen, and the one-step error (+0.22 M km²) is larger than the free runs' first summer (+0.15), which start from generated states.
- **Sampler.** 20 steps may be too few to resolve a fast melt-out.

## Setup
No training. August and September 2015–18, 8 members, `step_test.py`, `run_tests.sh`, `compare.py` (`compare_output.txt`), `members.py` (`members_output.txt`). Each run's outputs: `data/models/loose_ice_tests/`.

| run | model | start | sampler steps | tests |
| --- | --- | --- | --- | --- |
| `v04_truth` | v0.4.0 | target | 20 | baseline (August 2015 matches the saved prediction exactly, largest difference 0.0) |
| `v04_generated` | v0.4.0 | its own prediction | 20 | is the model calibrated from the states it trained on? |
| `v04_truth_n40` | v0.4.0 | target | 40 | sampler |
| `qm_nopf_truth` | `monthly_qm` (no pushforward in its history) | target | 20 | control, against its own targets `sicqm` |
| `qm_pf_truth` | `monthly_pf_qm` (pushforward 0.8) | target | 20 | partner for the control, same targets |

**Limits, known before running.** `monthly_qm`'s averaged weights still hold about 61% of its seed (ISS-005), and `monthly_pf_qm`'s about 37% of its own, so that pair compares lineages, not clean recipes. Inference sampling is second order with 20 steps and EMA weights, while pushforward generation used first order, 10 steps and the raw weights, so `v04_generated` is close to the training condition but not identical.

## Decision, written before the runs
- **Training from its own states is supported** if, on the Pacific shelves, `v04_generated` cuts by at least half both the starting-class errors (August 50–80% about +0.07, August >95% about −0.12) and the extent errors (+0.22, +0.23) compared with `v04_truth`, and `qm_nopf_truth` shows a clearly weaker pattern than `qm_pf_truth`. Then the next step is a training run: the v0.4.0 recipe with less or no pushforward.
- **Supported only in part** if just one of the two holds. Then look further before any training.
- **Rejected** if the same pattern stays from generated starts. Then the problem is the model's step whatever it starts from: the loss or capacity.
- **Sampler:** ruled out if 40 steps moves the shelf extent error by under 0.02 M km² and the class errors by under 0.01.

## Results
**Sampler: ruled out.** 40 steps moves the shelf extent error by 0.003 M km² (August +0.223 → +0.220, September +0.228 → +0.225) and every class error by 0.001 or less.

**Training from its own states: rejected.** Pacific shelves, error in the predicted concentration by the member's starting concentration (positive = too much ice):

| run | Aug 15–50% | Aug 50–80% | Aug 80–95% | Aug >95% | Sep 15–50% | Sep 50–80% | extent Aug / Sep (M km²) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.4.0, from the target | +0.020 | +0.069 | −0.052 | −0.119 | +0.098 | +0.081 | +0.223 / +0.228 |
| v0.4.0, from its own prediction | −0.017 | +0.075 | +0.047 | +0.067 | +0.094 | +0.115 | +0.287 / +0.328 |
| `monthly_qm` (no pushforward), from the target | +0.014 | +0.028 | −0.120 | −0.143 | +0.170 | +0.209 | +0.176 / +0.263 |
| `monthly_pf_qm` (pushforward), from the target | +0.012 | +0.027 | −0.117 | −0.147 | +0.163 | +0.202 | +0.132 / +0.264 |

- **From its own prediction, the loose-ice excess stays and the extent error grows** (+0.29 / +0.33). Only the compact classes change, and they flip sign. Neither condition of the rule holds.
- **The pushforward pair is indistinguishable**: every class error within 0.007. Both have compact ice melting too fast in August (−0.12 to −0.15). The pair's lineage caveat stands, but the pattern isn't something pushforward adds.
- **Compact ice melting too fast in August is pan-Arctic.** From the target, >95% ice in August: Barents/Kara −0.116, Laptev −0.096, Canadian −0.060, Greenland −0.044, Pacific −0.119. It covers under 1% of the shelf area in August, so it hardly moves extent. Loose ice opening too slowly is mainly Pacific (and Laptev in September).

**What the error is (`members.py`): too few cells end partly open.** v0.4.0 from the target, Pacific shelves, by starting concentration. "Partly open" means it ends between 0 and 15%. "Open water" means it ends at 0 (≤ 1e-4, see the erratum):

| | target partly open | members | target open water | members | spread / error of the mean |
| --- | --- | --- | --- | --- | --- |
| August, 15–50% | 36% | 22% | 54% | 61% | 0.084 / 0.101 |
| August, 50–80% | 23% | **12%** | 6.1% | 8.6% | 0.137 / 0.176 |
| September, 15–50% | 34% | 18% | 23% | 25% | 0.143 / 0.191 |
| September, 50–80% | 4.4% | 3.9% | 1.2% | 2.4% | 0.119 / 0.156 |

- **Erratum (first version of this table):** it said no member ever reaches open water. That was wrong. The sampler clamps each member to the bound, but rounding leaves it about 1e-7 above 0, so the `== 0` test missed every member. With a 1e-4 tolerance, members reach open water a little *more* often than the target.
- **The gap is the partly-open band.** Members put about half as much area as the target between 0 and 15%. The cells either go to exactly 0 or stay above 15%. Barents/Kara is close to the target in August (`members_output.txt`).
- **The ensemble is too narrow where the error is.** Its spread is below the error of the mean in every loose class.

## Training years and the 0 bound (`train_years.py`, `train_years_output.txt`)

**The same gap appears on years the model trained on.** v0.4.0, 2011–14, one month from the target (`one_month_fields_2011`), Pacific shelves:

| | target partly open | members | target open water | members |
| --- | --- | --- | --- | --- |
| August, 15–50% | 41% | 23% | 49% | 64% |
| August, 50–80% | 29% | 17% | 5.5% | 9.9% |
| September, 15–50% | 31% | 19% | 22% | 29% |
| September, 50–80% | 2.5% | 3.8% | 1.7% | 2.0% |

- **Not rarity in the data.** From 50–80% starting ice in August, 20% of the training targets (1995–2014) end partly open on the Pacific shelves and 28% in Barents/Kara. That is common, not a tail.
- **Not a generalisation failure.** On training years the members miss the band the same way as in 2015–18. So the model doesn't fit it at all. Barents/Kara fits on training years too.
- **What the code does at 0:**
  - **Loss (`gensim/train_module.py:198`).** Where the target is exactly 0, the Gaussian likelihood is replaced by a censored one, `neglogcdf(error)`. That term only penalises a prediction that ends *above* 0. Ending anywhere below 0 costs nothing.
  - **Sampler (`gensim/sampler.py:72`).** At every step, the predicted endpoint is clamped to the bounds. The decoder clamps again at the end (`gensim/encoder_decoder.py:104`).
  - **Together:** for cells that might open, the model is free to aim well below 0, and the clamps turn anything below 0 into exactly 0. That would explain both too much open water and too little partly-open ice: mass aimed below 0 drags nearby outcomes down with it. This fits the numbers but is not yet tested.
- **Planned next test, no training needed:** rerun `step_test.py` with the sampler's clamp off, and record how far below 0 the unclamped members aim. If partly-open ice comes back, the clamp is the mechanism. If not, the loss is left as the suspect, and testing that needs training.

## Sampler clamp off (`step_test.py --no-clamp`, `members_noclamp_output.txt`, `aim.py`, `aim_output.txt`)

**The sampler's clamp is not the cause.** Same members and seeds, but the sampler doesn't clamp at each step. The decoder output is clamped to [0, 1] only at the end. Pacific shelves, 2015–18:

| | target partly open | clamped | clamp off | target open water | clamped | clamp off |
| --- | --- | --- | --- | --- | --- | --- |
| August, 15–50% | 36% | 22% | 17% | 54% | 61% | 66% |
| August, 50–80% | 23% | 12% | 10% | 6.1% | 8.6% | 9.3% |
| September, 15–50% | 34% | 18% | 13% | 23% | 25% | 29% |

- **Partly open ice doesn't come back.** It falls slightly further, and slightly more ends at 0. Bias and error of the mean move by 0.006 at most.
- **The members aim below 0, not just at it.** Unclamped, the members that go below 0 have a median of −0.07 to −0.14 sic. A tenth of them go below −0.12 to −0.21, and the lowest reaches −0.40 (`aim_output.txt`). On the Pacific shelves in August, starting from 50–80%, 9.4% of members aim below 0, against 6.1% of targets at 0. Barents/Kara behaves the same way.
- **Reading:** this fits the censored loss. Where a training target is exactly 0, ending anywhere below 0 costs nothing, so the model has learned to put outcomes there that should be partly open. It is still not proven. Training without censoring at 0 is the test that would show it.
