# Drift, with the weather averaged out

**Date:** 2026-10-22 · **Branch:** `exp/multistep-pushforward` · **Status:** done · **Issue:** ISS-025

## Question
The hindcast's drift growth (September bias of 2021–22 minus 2019–20) is dominated by year-to-year weather and can't separate models. Measured properly, does the model drift, how much, and how much do two runs of the same recipe differ?

## Setup
`scripts/evaluate/drift.py`: free-runs (2 members, fast sampler) from the model's own training-target state each December 2008–2014, each to December 2018 under real ERA5 weather, scored month by month against that target, which saw the same weather. Drift rate = trend of the annual-mean extent error over lead years, mean of the 7 starts, with its standard error. 588 model months per model.

## Runs
| model | drift run_id |
| --- | --- |
| `monthly_pf_qm` | `aba97a721598` |
| `monthly_pf_qm_ema` | `324a877dedf7` |
| `monthly_pf_qm_ema_s1` | `b95831629980` |

## Results
Extent error against the training target (QM targets), M km²:

| | drift rate per year | September drift rate per year | error, lead years 1 / 3 / 5 / 7 / 9 |
| --- | --- | --- | --- |
| `monthly_pf_qm` | −0.009 ± 0.006 | −0.040 ± 0.021 | −0.07 / −0.13 / −0.06 / −0.14 / −0.18 |
| `monthly_pf_qm_ema` | −0.007 ± 0.003 | −0.020 ± 0.007 | −0.06 / −0.10 / −0.04 / −0.12 / −0.11 |
| `monthly_pf_qm_ema_s1` | −0.005 ± 0.002 | −0.011 ± 0.007 | −0.09 / −0.13 / −0.06 / −0.14 / −0.13 |

- **No drift.** Against the target it was trained on, none of the three gets icier as it runs: the error sits at about −0.1 M km² (slightly *less* ice than the target) from the first year to the ninth, with a trend of −0.005 to −0.009 per year, close to zero.
- **Noise floor:** two runs of the same recipe differ by 0.002 per year in drift rate (0.009 in September), well inside the standard errors.

## Conclusion
In 2009–2018 the model doesn't drift: it tracks its training target with a small, steady offset. So the growing gap against observations in the 2019–25 hindcast (September +0.12 in 2019–22, +0.34 in 2023–25) isn't compounding error. The remaining explanations are outside the model's own dynamics: year-to-year weather (which the old measure couldn't separate), the training target itself drifting away from reality after 2018 (the QM correction was fitted on 1995–2018 and doesn't follow the trend), or the model under-responding to recent summers warmer than any it trained on. Caveat: the starts are training years; drift under unfamiliar weather could be larger, which the 2019–25 hindcast can't separate from the other explanations.

For the roadmap, "reduce drift" stops being a target: there is none to reduce. A drift guardrail (no model should start to drift) is the useful form.
