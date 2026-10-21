# Degree-days that count thaw days

**Date:** 2026-10-20 · **Branch:** `exp/daily-degree-days` · **Status:** done · **Issue:** ISS-007

## Question
The model's melt inputs are degree-days computed from monthly-mean temperature, so a month that averages below freezing counts no melt even when it had thaw days. Summer melt is the model's weakest point: too little of it on the Pacific shelf, and a melt rate that extra training moves freely ([more_finetune](../2026-10-18_more_finetune/)). Do degree-days from daily temperature, which see those thaw days, give the model the information it lacks?

## Setup
- **Inputs:** the same four degree-day channels (`pdd_month`, `fdd_month`, `pdd_year`, `fdd_year`), same names, units, threshold (271.35 K) and encoder normalization, rebuilt from each month's mean temperature plus the typical day-to-day spread for that calendar month (`scripts/data/spread_degree_days.py`; see *Results* for why not daily data). Replaced rather than added: added channels were barely learned before (radiation 6–10% weight).
- **Recipe** `configs/experiment/pf_qm_ema_dd.yaml`: identical to `pf_qm_ema_s1` (from `monthly_pf_qm`, 20k steps, EMA 0.999, seed 1) except the data. A/B partner: `monthly_pf_qm_ema_s1`; noise between training runs is ~0.01 M km² on September bias and ice-edge errors, 0.05–0.08 on July–August bias.
- **Scored** with the scorecard on 2019–22, focusing on the two missed v0.4.0 targets (East Siberian September bias, drift growth) and July–August bias.
- **CMIP:** the same spread applies to CMIP monthly means, so a model trained this way can run projections once the CMIP cube's degree-days are rebuilt the same way.

## Commands
```bash
python scripts/data/degree_day_spread.py --check     # accuracy against real daily degree-days (Results)
python scripts/data/degree_day_spread.py             # data/auxiliary/t2m_daily_spread.nc
python scripts/data/spread_degree_days.py --datacube-dir data/train_data/monthly_datacube_sicqm
python scripts/data/spread_degree_days.py --datacube-dir data/train_data/monthly_datacube
python scripts/data/spread_degree_days.py --datacube-dir data/train_data/era5_forcing_datacube
python scripts/data/build_zarr_monthly.py --datacube-dir data/train_data/monthly_datacube_sicqm_dd --suffix _sicqm_dd
python train.py +experiment=pf_qm_ema_dd
python scripts/project/freerun_monthly.py --ckpt data/models/monthly_pf_qm_ema_dd/last.ckpt --forcing era5 --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members --datacube data/train_data/monthly_datacube_dd --era5-ext data/train_data/era5_forcing_datacube_dd --note "pf_qm_ema_dd hindcast"
python scripts/evaluate/scorecard.py --model monthly_pf_qm_ema_dd
```

## Runs
| what | ledger run_id |
| --- | --- |
| hindcast (on the `_dd` cubes) | `63faf41cf99a` |
| scorecard | `5da5de37abdc` |

## Results
**Change of method (2026-10-20).** Daily ERA5 from the CDS would take days to weeks: daily statistics are computed on request, and a year counts as a large request (40 running at once across all users, ~1,700 queued). Instead, the degree-days come from the monthly mean plus the typical day-to-day spread of daily temperature for that calendar month, assuming daily temperatures in a month are normally distributed (`frazil.preprocess.expected_degrees`, the positive-degree-day method of Braithwaite 1985 and Calov & Greve 2005). The spread comes from WeatherBench2's public ERA5 (6-hourly, 1.5°, 1995–2014; `scripts/data/degree_day_spread.py`). This also works for CMIP monthly means, which daily data wouldn't without daily CMIP downloads.

**Accuracy check** against degree-days from the real daily temperatures, 2015–2022 (outside the spread's 1995–2014 climatology), over the cells that have sea ice in March. Mean daily positive degrees (K) and errors:

| month | true | monthly-mean method: bias / RMSE | spread formula: bias / RMSE | formula with each month's own spread: RMSE |
| --- | --- | --- | --- | --- |
| May | 0.57 | −0.28 / 0.42 | +0.13 / 0.24 | 0.11 |
| June | 3.05 | −0.03 / 0.08 | −0.00 / 0.05 | 0.04 |
| July | 4.84 | −0.00 / 0.01 | +0.00 / 0.01 | 0.00 |
| August | 4.83 | −0.02 / 0.05 | +0.00 / 0.04 | 0.02 |
| September | 2.89 | −0.13 / 0.24 | +0.09 / 0.18 | 0.06 |
| October | 1.08 | −0.20 / 0.36 | +0.16 / 0.32 | 0.12 |
| winter months | 0.03–0.25 | −0.02 to −0.09 / 0.12–0.24 | +0.02 to +0.08 / 0.10–0.20 | 0.05–0.08 |

- The monthly-mean method misses 5% of the year's melt over the ice, almost all of it at **melt onset (May: half its melt) and freeze-up (September–October)**. July and August are essentially exact: over melting ice the daily temperatures sit above the threshold, so averaging first loses nothing.
- The formula cuts the error by 20–45% in the months that matter but overshoots (+3.4% of annual melt; +0.13 in May, +0.16 in October). With each month's own spread the error is less than half, so most of the remaining error is the fixed 1995–2014 spread, not the normal shape.
- Over the whole model ocean (including open water far south) the monthly-mean method misses only 2%.

**Training result (2026-10-21).** `monthly_pf_qm_ema_dd` against its same-seed partner `monthly_pf_qm_ema_s1`, development years 2019–22, per member, M km² (`monthly_pf_qm_ema`, another run of the same recipe, shows the run-to-run noise):

| | `ema` | `ema_s1` | `ema_dd` |
| --- | --- | --- | --- |
| September bias / ice-edge error | +0.12 / 1.20 | +0.11 / 1.19 | +0.14 / 1.20 |
| June bias / ice-edge error | +0.64 / 1.36 | +0.63 / 1.35 | +0.62 / 1.34 |
| May bias / ice-edge error | +0.37 / 0.85 | +0.34 / 0.84 | +0.35 / 0.84 |
| October bias / ice-edge error | −0.12 / 1.00 | −0.17 / 1.03 | −0.15 / 1.03 |
| July–September ice-edge error | 1.42 | 1.42 | 1.42 |
| drift growth | +0.12 | +0.09 | +0.08 |
| East Siberian September | +0.19 | +0.19 | +0.20 |

Every difference from `ema_s1` is within the run-to-run noise (the `ema`–`ema_s1` gap), including May and October, where the inputs changed most.

## Conclusion
No effect, and in hindsight none was possible: with a fixed climatological spread, the new degree-days are a fixed function of the month's mean temperature (`tus`, already an input) and the calendar month. They add no information the network couldn't compute itself; they only change how it is presented. This should have been caught before training. Degree-days that carry real information would need each year's own day-to-day spread (the accuracy check found it halves the error), which needs daily data for every year: WeatherBench2 covers 1959 to January 2023, so the 2023–25 holdout and CMIP projections would lack it. Not pursued. ISS-007 closed.

Together with radiation (no measurable effect) and longer training (overfits the summer), this points away from monthly atmospheric inputs and toward the model's own memory: the ocean-memory input (sunlight absorbed by the model's own open water over the summer) is information the current inputs don't contain.
