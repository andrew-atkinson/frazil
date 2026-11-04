# Thickness-corrected targets

**Date:** 2026-10-29 · **Branch:** `exp/sit-target-correction` · **Status:** growth correction tested and rejected; static-ratio sensitivity test trained: spring thickness fixed, September much worse (the targets melt less) · **Issues:** ISS-002, ISS-028

## Question
[ocean_input_and_teacher_checks](../2026-10-29_ocean_input_and_teacher_checks/) found the teacher's (neXtSIM-OPA's) winter ice on the East Siberian and Chukchi shelves 0.26–0.29 m too thick by April against the CS2SMOS-constrained neXtSIM reanalysis, about twice the Laptev's, and Frazil v0.4.0 reproduces it (+0.29/+0.33 m). The concentration correction (QM) never touched thickness. Does training on corrected thickness targets reduce Frazil's Pacific-shelf September excess?

## Setup
- **Targets:** `scripts/data/correct_sit_targets.py` scales the v0.4.0 targets' `sit` (`monthly_datacube_sicqmw9`) by one ratio per cell and calendar month: the reanalysis mean over 2011–2018 divided by the targets' mean (`monthly_datacube_sicqmw9_sitra`; concentration and everything else unchanged).
  - **October–April** (CS2SMOS months): the ratio as fitted. East Siberian + Chukchi: 0.92 in January, 0.87 in April.
  - **May–September:** interpolated per cell from April's ratio to October's. The two models differ in summer melt (the teacher grows too much in winter and melts more, so by September they agree on the shelves: 0.43 vs 0.40 m), and the reanalysis's weaker melt is unconstrained by satellites (its September is too icy there), so the teacher's summer physics is kept, starting from corrected spring ice. Shelves: 0.91 in June, 0.97 in September.
  - Identity outside the reanalysis's domain and where either mean is under 5 cm; ratios clipped to [0.5, 2].
  - Effect: April volume 11–13% lower in every year (1995: 27,190 → 24,030 km³).
  - **Ceiling:** one ratio for all years assumes the excess is proportional to thickness before 2011 too; there is no observed thickness to check it there.
- **Recipe:** `configs/experiment/pf_qm_ema_qmw_sit.yaml` = v0.4.0's recipe (`pf_qm_ema_qmw`: seeded from `monthly_pf_qm`, 20k steps, EMA 0.999, seed 1) with suffix `_sicqmw9_sitra`. A/B partner: `monthly_pf_qm_ema_qmw` (v0.4.0).
- **Hindcast** starts from the corrected December 2018 state (v0.4.0's started from the uncorrected one).

**Reading it** (development years 2019–22 only; run-to-run noise about 0.01 on September bias and ice-edge error):

| | v0.4.0 | what would count as working |
| --- | --- | --- |
| September bias, East Siberian | +0.161 | clearly lower |
| September bias, Chukchi | +0.083 | clearly lower |
| April thickness minus reanalysis, East Siberian / Chukchi | +0.29 / +0.33 m | near zero |
| September bias / ice-edge error, pan-Arctic | +0.082 / 1.133 | no worse than noise |
| June / July bias | +0.655 / +0.291 | no worse than noise |

If the shelves' spring ice gets thinner but the September excess stays, the thickness is not what keeps the shelves icy in Frazil, and the remaining candidates are its own regional thickening (drift) and the teacher's ocean.

## Commands
```bash
python scripts/data/correct_sit_targets.py
python scripts/data/build_zarr_monthly.py --datacube-dir data/train_data/monthly_datacube_sicqmw9_sitra --suffix _sicqmw9_sitra
python train.py +experiment=pf_qm_ema_qmw_sit
python scripts/project/freerun_monthly.py --ckpt data/models/monthly_pf_qm_ema_qmw_sit/last.ckpt --forcing era5 --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members --datacube data/train_data/monthly_datacube_sicqmw9_sitra --era5-ext data/train_data/era5_forcing_datacube_era5x --note "pf_qm_ema_qmw_sit hindcast"
python scripts/evaluate/drift.py --model monthly_pf_qm_ema_qmw_sit --run --fields
python scripts/evaluate/scorecard.py --model monthly_pf_qm_ema_qmw_sit
python experiments/2026-10-29_ocean_input_and_teacher_checks/analysis.py --model monthly_pf_qm_ema_qmw_sit
```

## Runs
| what | ledger run_id | output |
| --- | --- | --- |
| ERA5 hindcast 2019–25, 8 members | `3120e7ff38f4` | `plots/freerun/snapshots/monthly_pf_qm_ema_qmw_sit_era5_2018` |
| drift runs with fields (starts 2008–14) | `c338608fd874` | `plots/freerun/drift` |
| scorecard (development years only) | `4fadec0766d7` | `results/scorecards/monthly_pf_qm_ema_qmw_sit.json` |
| shelf comparison (`analysis.py --model monthly_pf_qm_ema_qmw_sit`) | | `analysis_output_sit.txt` |
| summer thickness loss, targets and hindcasts | | `melt.py` (output below) |

## Results
### Does the correction need to change over time? (`growth_test.py`, `growth_test_output.txt`)
A fixed ratio per cell and month (above) carries the 2011–18 ice regime back to 1995, against the project's purpose of following a changing Arctic. Two questions: does the error change over time, and can a rule tied to the ice itself, not the calendar, capture it?

**The error does vary, but the data can't pin down a trend.** The teacher's April thickness minus the reanalysis swings widely from year to year:

| April, m | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| East Siberian | 0.26 | 0.43 | 0.38 | 0.28 | 0.26 | 0.40 | 0.08 | 0.01 |
| Chukchi | 0.19 | 0.45 | 0.44 | 0.25 | 0.35 | 0.44 | 0.05 | 0.13 |
| Beaufort | 0.26 | 0.40 | 0.27 | 0.02 | 0.22 | 0.18 | −0.26 | 0.03 |

The rising reanalysis/teacher ratio (East Siberian + Chukchi +0.16 ± 0.09 per decade, Beaufort +0.28 ± 0.12) rests mostly on 2017–18. Before 2011 the reanalysis has no satellite thickness, so it can't give the trend there; extending eight years back to 1995 isn't defensible.

**A rule tied to the ice regime: rejected.** Winters 2011–18 suggested the teacher grows first-year ice too fast (reanalysis growth ÷ teacher growth 0.75–0.76 for ice under 1 m in November, steady from year to year). The test rebuilt thickness from 1995 with winter growth scaled by α for first-year ice (1 for ice over 2 m, linear between, carried month to month so the correction follows the ice), fitted α on winters 2011–14 and tested 2015–18. A flat α for all ice is the control.

| method (fitted 2011–14) | α | cell RMSE, Nov–Apr 2015–18 (m) | April minus reanalysis, 2015–18: E. Siberian + Chukchi | Beaufort | central (>82° N) | ratio to teacher, trend per decade 2011–18 | April volume trend 1995–2018 (km³ per decade) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| teacher | — | 0.44 | +0.21 | +0.04 | +0.05 | — | −2,320 |
| static ratio | — | **0.34** | −0.09 | −0.17 | −0.04 | 0.00 | −2,130 |
| growth, flat α | 0.75 | 0.41 | −0.15 | −0.32 | −0.31 | −0.01 | −2,680 |
| growth, first-year ice | 0.70 | 0.40 | −0.20 | −0.30 | −0.20 | −0.03 (shelves), −0.12 (Beaufort) | −2,890 |
| reanalysis (target to match) | | | | | | +0.17 (shelves), +0.28 (Beaufort) | |

The growth rule over-corrects out of sample, and its change over time goes the wrong way: as first-year ice spreads it corrects more, while the data shows the gap closing. It also steepens the targets' volume trend by a quarter, which is exactly the kind of imposed trend this test was meant to avoid. The static ratio is better out of sample on average, keeps relative trends (it is multiplicative), but cannot follow any change and over-corrects the Beaufort in the latest years.

### The sensitivity test: thinner spring ice, icier September
Training on the static-ratio targets (`pf_qm_ema_qmw_sit`) fixed the spring thickness and made September worse. Development years 2019–22 against v0.4.0 (run-to-run noise about 0.01 on biases and ice-edge error):

| | v0.4.0 | `pf_qm_ema_qmw_sit` |
| --- | --- | --- |
| April thickness minus reanalysis, East Siberian / Chukchi / Beaufort / Laptev (m) | +0.29 / +0.33 / +0.22 / +0.14 | **+0.08 / +0.11 / +0.01 / −0.05** |
| September bias, pan-Arctic (M km²) | +0.082 | **+0.219** |
| August / July / June bias | +0.077 / +0.291 / +0.655 | +0.266 / +0.375 / +0.700 |
| September bias, East Siberian / Chukchi / Beaufort / Laptev | +0.161 / +0.083 / +0.012 / −0.071 | +0.191 / +0.093 / +0.017 / −0.037 |
| ice-edge error, September / December–March | 1.133 / 0.798 | 1.138 / 0.796 |
| September r | 0.949 | 0.958 |
| drift rate against target (starts 2008–14) | 0.0034 ± 0.0043 | 0.0066 ± 0.0035 |

**Why: the corrected targets melt less, and Frazil learned it.** The ratio is 0.87 in April but ramps towards October's in summer (0.97 in September on the shelves), so April loses about 13% and September about 3%, which takes about 15% out of the summer thickness loss (`melt.py`, area means, m):

| | April | September | April − September |
| --- | --- | --- | --- |
| targets, 1995–2014: teacher / corrected | 0.92 / 0.81 | 0.30 / 0.30 | 0.62 / **0.51** |
| same, East Siberian + Chukchi | 2.18 / 1.88 | 0.60 / 0.55 | 1.58 / **1.34** |
| hindcast 2019–22: v0.4.0 / `_sit` | 0.83 / 0.74 | 0.18 / 0.20 | 0.64 / **0.54** |
| same, East Siberian + Chukchi | 1.90 / 1.70 | 0.36 / 0.38 | 1.54 / **1.32** |

Frazil's spring ice came out as the reanalysis has it, but its September ice is slightly thicker than v0.4.0's and its September extent 0.14 M km² larger: about 0.08 of it in the four sectors above (most in the East Siberian and Laptev), the rest elsewhere.

## Conclusion
- **Frazil's September follows the summer melt it learned, not its spring thickness.** Bringing April shelf ice to within 0.1 m of the reanalysis did nothing for the September shelves; 15% less melt in the targets made September clearly icier. Any per-month thickness ratio that differs between April and September changes the melt Frazil learns, so this recipe can't isolate spring thickness; a clean test would have to keep the teacher's April-to-September loss.
- **Lesson: the linear summer interpolation was the mistake.** May–September had no satellite thickness, and I filled the gap with a straight line between April's ratio and October's. That looked harmless month by month, but it made up a new summer melt, about 15% weaker than the teacher's, and Frazil learned it faithfully ([summer_melt](../2026-10-30_summer_melt/)). Where data are missing, leave the teacher as it is (identity) or carry its own physics forward; never fill the gap with a line. And check the month-to-month changes a correction implies, not only the months it was fitted on.
- **Neither correction is justified for production targets.** The data supports neither a fixed ratio nor a time-varying one: the error is large in some years and near zero in others, and the one physically motivated rule (first-year growth) is contradicted out of sample. The thickness targets stay as the teacher made them.
- **Thickness is not the lever.** The v0.4.0 targets have thick spring ice on the shelves *and* the observed September extent there (QM: 0.00 bias), so in the targets thick spring ice does not produce a September excess. Frazil's excess is above its own target ([regional_model_vs_target](../2026-10-25_regional_model_vs_target/)), and its April shelf ice in 2020–22 (+0.19 to +0.40 m over the reanalysis) is at least as thick as the teacher's worst years, after the teacher's gap had closed in 2017–18. That points back at Frazil's own regional thickening in free runs.
- **The static-ratio targets stay a sensitivity result, not a correction.** `pf_qm_ema_qmw_sit` is not a candidate; v0.4.0 stands.
- **The reanalysis stays as a monitor:** April shelf thickness against it is a useful diagnostic for any model.
