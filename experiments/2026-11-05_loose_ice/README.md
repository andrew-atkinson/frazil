# Loose ice that opens too slowly

**Date:** 2026-11-05 · **Branch:** `exp/loose-ice` · **Status:** in progress (no-training tests done; both hypotheses rejected)

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

**What the error is (`members.py`): the members almost never open loose ice right up.** v0.4.0 from the target, Pacific shelves, by starting concentration:

| | target opens below 15% | members do | target goes to 0 | members do | spread / error of the mean |
| --- | --- | --- | --- | --- | --- |
| August, 15–50% | 90% | 83% | 53% | 40% | 0.084 / 0.101 |
| August, 50–80% | 29% | 20% | 6.1% | **0.0%** | 0.137 / 0.176 |
| September, 15–50% | 57% | 44% | 23% | 14% | 0.143 / 0.191 |
| September, 50–80% | 5.6% | 6.3% | 1.2% | 0.0% | 0.119 / 0.156 |

- **From 50–80% ice, no member ever reaches open water in one month**, while the target does in 6% of the area in August (Barents/Kara: 7.7% against 0.0%). Below 15%, the members open a third too rarely in August on the Pacific shelves. In Barents/Kara they open as often as the target (31% against 29%) but still never to 0.
- **The ensemble is too narrow where the error is** (spread under the error of the mean in every loose class), so the missing melt-outs are in the tail the members don't reach, not a shift of the whole distribution.
- **Reading:** a large one-month drop, loose ice to open water, is something the model doesn't produce. It now looks like how the model represents the step (its loss, and how it handles the 0 bound), not its inputs, sampler or pushforward. This is a description from four summers, not yet a cause.
