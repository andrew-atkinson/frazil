# Where the 2019–25 gap to observations comes from

**Date:** 2026-10-23 · **Branch:** `main` (analysis only) · **Status:** done (no training) · **Issues:** ISS-002, ISS-026

## Question
The model doesn't drift against its training target ([drift_protocol](../2026-10-22_drift_protocol/)), yet its September gap to NSIDC is larger in 2023–25 (+0.34 M km²) than in 2019–22 (+0.12). Two candidate explanations, testable without training: (2) the training target itself drifts away from the observations over time; (3) the model under-responds to summers warmer than any it trained on.

## Setup
`analysis.py` (this folder). September extent on the model grid over the ocean, against NSIDC-0051. Test 2: the QM training target and the original neXtSIM, 1995–2018. Test 3: June–August mean 2 m temperature north of 70° N (the model's own `tus` input) against September extent, for `monthly_pf_qm_ema` (2009–18 from the drift free-runs, mean over starts; 2019–25 from the hindcast) and NSIDC, same years.

## Results
**Test 2: the QM target drifts away from the observations; neXtSIM itself doesn't.**

| September extent trend, 1995–2018 | per decade |
| --- | --- |
| NSIDC | −1.14 ± 0.18 |
| neXtSIM (original targets) | −1.13 ± 0.19 |
| QM targets (what the model trains on) | **−0.93 ± 0.17** |
| neXtSIM − NSIDC | +0.01 ± 0.10 |
| QM target − NSIDC | **+0.21 ± 0.07** |

QM target minus NSIDC averages −0.16 in 1995–2006 and +0.14 in 2007–2018. neXtSIM has the right trend and a steady excess (about +0.36); the quantile mapping removes that excess on average but also flattens the trend by about a fifth, because it is one fixed mapping per cell and month fitted on all of 1995–2018: too strong a correction early, too weak late. Extended at +0.21 per decade, the target would sit about +0.3 above NSIDC around 2022 and +0.36 around 2024. With the model's steady −0.1 offset to its target, that predicts a model gap of about +0.2 in 2019–22 and +0.27 in 2023–25, against the +0.12 and +0.34 found: consistent within the year-to-year weather noise (about ±0.1 on a 3–4-year mean).

**Test 3: the model responds to summer warmth like the observations.**

| September extent per +1 °C of June–August temperature, 2009–2025 | all | year-to-year (trends removed) |
| --- | --- | --- |
| model | −1.68 ± 0.44 (r −0.70) | −2.12 ± 0.49 |
| NSIDC | −1.63 ± 0.37 (r −0.75) | −1.98 ± 0.41 |

The model-minus-NSIDC gap doesn't depend on the temperature (r −0.05), and in the 3 summers warmer than any training year (2016, 2020, 2024) it is smaller (+0.09) than in the other 14 (+0.21). No under-response.

## Conclusion
Explanation 2 holds, 3 doesn't. The growing gap comes from the training targets: the QM correction that de-iced the model also flattened its trend by about a fifth, so the target gets icier relative to reality by about +0.2 M km² per decade, and the model reproduces its target faithfully. The fix is in the data, not the model: a trend-preserving correction (for example the quantile mapping fitted separately in shorter periods, or quantile delta mapping), checked with this same test (QM target − NSIDC trend ≈ 0) before any retraining. ISS-026.

## The fix, checked (2026-10-23)
`correct_sic_targets_qm.py --window N` fits each year's maps on the N years centred on it (shifted inward at the ends), so the correction follows the observed trend. Checked with `check_targets.py` (this folder) against NSIDC-0051, 1995–2018:

| September: target − NSIDC | trend per decade | mean 1995–2006 | mean 2007–2018 | RMS of yearly difference |
| --- | --- | --- | --- | --- |
| neXtSIM | +0.01 ± 0.10 | +0.37 | +0.37 | 0.49 |
| QM, one map (current targets) | +0.21 ± 0.07 | −0.16 | +0.14 | 0.27 |
| QM, 13-year windows | +0.06 ± 0.07 | −0.05 | +0.04 | 0.23 |
| **QM, 9-year windows** | **+0.03 ± 0.07** | **−0.00** | **+0.04** | 0.23 |

Both windowed versions remove the trend (within its standard error); 9-year windows most completely. They also lower the ice-edge error against NSIDC in every month (September 1.01 → 0.79 for 9-year windows). The 9-year targets are built (`monthly_datacube_sicqmw9`, `train_sicqmw9.zarr`) with the recipe `pf_qm_ema_qmw` (same as `pf_qm_ema_s1` except the targets).

**A separate observation, not changed by the windowing:** every QM version sits about +0.2 M km² above NSIDC from October to June (neXtSIM itself about 0 to +0.3). Part of it is how it's scored here: NSIDC has no data on about 0.45 M km² of the model's ocean (coastal cells), counted as open water. Part is real: the mapping adds winter ice in the Greenland and Baffin seas (+0.1 to +0.2 in 2016). Logged as ISS-027; it doesn't affect the summer gap.

## Retrained on the windowed targets (2026-10-25)
`monthly_pf_qm_ema_qmw` (recipe `pf_qm_ema_qmw`): identical to `monthly_pf_qm_ema_s1` (same seed) except the 9-year-window targets. Runs: hindcast `9f2cbe0d8b4d`, drift `71093e5f2af0`, scorecard `47a9df01b371`. Development years 2019–22, per member, M km² (`ema` vs `ema_s1` shows run-to-run noise):

| | `ema` | `ema_s1` | `ema_qmw` |
| --- | --- | --- | --- |
| September bias | +0.12 | +0.11 | **+0.08** |
| July / August bias | +0.23 / +0.10 | +0.31 / +0.15 | +0.29 / **+0.08** |
| June bias | +0.64 | +0.63 | +0.66 |
| ice-edge error Sep / Jun / Jul–Sep / Dec–Mar | 1.20 / 1.36 / 1.42 / 0.80 | 1.19 / 1.35 / 1.42 / 0.81 | **1.13** / 1.33 / **1.38** / 0.80 |
| September correlation | 0.92 | 0.94 | 0.95 |
| East Siberian / Chukchi September | +0.19 / +0.11 | +0.19 / +0.10 | +0.16 / +0.08 |
| drift rate against its own target, per year | −0.007 ± 0.003 | −0.005 ± 0.002 | +0.003 ± 0.004 |

Better than its same-seed partner on every September and ice-edge measure, by more than the run-to-run noise: September bias −0.03, August −0.08, September ice-edge error −0.06 (5%), summer ice-edge error −0.03, East Siberian −0.02. Still no drift. The improvement is smaller than the targets' change would suggest if it carried over one for one (the old target's gap extended to 2020 is about +0.3, the new one's about +0.06); the decisive test of the trend is the 2023–25 holdout, where the old gap was largest.

**One-month skill guardrail (2026-10-25, run `7ed18eccd467`):** skill against persistence, 2015–18, 8 members, against its own targets: concentration 58.0%, thickness 51.4%, damage 50.2%, drift 55.5% / 55.3%, snow 61.3%. Every variable is at or above its floor (the v0.2 baseline minus 2 points; closest: thickness 51.4 against 50.4, snow 61.3 against 60.4), and within 0.1 of `monthly_pf_qm_ema`. Passes.

**CMIP stability and projection (2026-10-25, run `f12f5d95e695`):** 2015–2100 under MPI-ESM1-2-LR ssp245, 2 members: no non-finite values in 1,020 months, so the last guardrail holds. September extent, mean of the members' own extents, M km²:

| | 2015–24 | 2035–44 | 2055–64 | 2075–84 | 2095–99 |
| --- | --- | --- | --- | --- | --- |
| `monthly_pf_qm_ema` | 5.39 | 4.36 | 3.05 | 2.64 | 1.36 |
| `monthly_pf_qm_ema_qmw` | 5.37 | 4.24 | 2.87 | 2.39 | 1.21 |

The same start, then a steeper decline: 0.12 lower by 2035–44 and 0.25 by 2075–84, which is what correcting the targets' too-shallow trend should do. First September below 1 M km² still 2091. The 2019–24 CMIP level (5.33) still sits +0.6 above the ERA5 hindcast (4.74) and +0.8 above NSIDC (4.54): the CMIP forcing's own cold bias (ISS-009, v0.7.0).

`monthly_pf_qm_ema_qmw` now meets every v0.4.0 guardrail (September correlation, December–March ice-edge error, one-month skill, finite CMIP run, no drift) and is the v0.4.0 release candidate.

## Release-candidate holdout, 2023–25 (2026-10-25)
Scored once, as the roadmap allows for a release candidate (`scorecard.py --holdout`). Per member, M km²:

| | `ema` 2019–22 / 2023–25 | `ema_qmw` 2019–22 / 2023–25 | v0.4.0 target |
| --- | --- | --- | --- |
| September bias | +0.12 / +0.34 | +0.08 / **+0.34** | ≤ +0.20 |
| August bias | +0.10 / +0.53 | +0.08 / **+0.38** | |
| ice-edge error Jun / Jul–Sep / Sep | 1.36 / 1.42 / 1.20 ; 1.37 / 1.52 / 1.26 | 1.33 / 1.38 / 1.13 ; 1.39 / 1.51 / 1.28 | Jun ≤ 1.41, Jul–Sep ≤ 1.65 |
| East Siberian / Beaufort September | +0.19 / +0.01 ; +0.10 / +0.30 | +0.16 / +0.01 ; +0.11 / +0.31 | ≤ +0.10 each |

September bias by year and sector shows where the 2023–25 jump comes from: **the Beaufort Sea, about +0.3 in each of 2023, 2024 and 2025, against about 0 in 2019–22**, in both models alike. East Siberian is +0.1 to +0.25 in every year; the other sectors stay near zero.

So the trend fix improved the development years and August in the holdout, but not the 2023–25 September excess, which is a regional shift in the Beaufort Sea from 2023 that neither model reproduces, not the targets' general trend. Earlier, the extrapolated target trend appeared to account for the whole jump; with the sectors in view, it doesn't (ISS-028).

**Against the v0.4.0 targets on the holdout:** June and July–September ice-edge errors met; September bias (+0.34), East Siberian (+0.11) and Beaufort (+0.31) not. With three years, the holdout correlation (0.89) isn't meaningful.
