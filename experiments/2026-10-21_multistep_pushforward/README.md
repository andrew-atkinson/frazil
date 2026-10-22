# Multi-step pushforward

**Date:** 2026-10-21 · **Branch:** `exp/daily-degree-days` · **Status:** done · **Issue:** ISS-002

## Question
Radiation, thaw-day degree-days and an ocean-memory information test all came out null: the emulator isn't missing information about the next month, and its one-month skill is good. But the September gap grows over a free-run (+0.12 M km² in 2019–22, +0.34 in 2023–25): it is error compounding as the model runs on its own predictions. Pushforward teaches correction of one generated month. Does training on corrections after several generated months reduce that compounding?

## Setup
- **Code:** `gensim/train_module.py`, new `surrogate.pushforward_steps` (K). Each pushforward sample chains k ~ uniform{1..K} of the model's own generated months from a true state, then trains the step from the k-th generated month to the true month after it, with the true forcing and degree-days for every month. K = 1 is the original behaviour, with identical random draws (`tests/test_pushforward.py`).
- **Recipe** `configs/experiment/pf_qm_ema_ms.yaml`: identical to `pf_qm_ema_s1` (from `monthly_pf_qm`, 20k steps, EMA 0.999, pushforward probability 0.8, 10 sampler steps, seed 1) except K = 3 and 5-month samples (`data.n_rollout_steps: 4`, 236 samples instead of 238). A/B partner: `monthly_pf_qm_ema_s1`.
- **Cost:** a short test run measured ~2.5 s per batch against ~1.6 s single-step: ~7 h for the full run.
- **Watch:** drift growth and the 2023–25 holdout difference (the compounding), September and July–August bias, East Siberian September; guardrails as in the roadmap. Run-to-run noise is ~0.01 on September bias and ice-edge errors, 0.05–0.08 on July–August, 0.03 on drift growth.

## Commands
```bash
python train.py +experiment=pf_qm_ema_ms
python scripts/project/freerun_monthly.py --ckpt data/models/monthly_pf_qm_ema_ms/last.ckpt --forcing era5 --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members --datacube data/train_data/monthly_datacube_era5x --era5-ext data/train_data/era5_forcing_datacube_era5x --note "pf_qm_ema_ms hindcast"
python scripts/evaluate/scorecard.py --model monthly_pf_qm_ema_ms
```

## Runs
| what | ledger run_id |
| --- | --- |
| hindcast | `c419a5f2bc3a` |
| scorecard | `b152e4047427` |

## Results
Development years 2019–22, per member, M km² (`ema` and `ema_s1` are two runs of the single-step recipe, so their gap is the run-to-run noise):

| | `ema` | `ema_s1` | `ema_ms` (K = 3) |
| --- | --- | --- | --- |
| September bias | +0.12 | +0.11 | **+0.19** |
| June / July / August bias | +0.64 / +0.23 / +0.10 | +0.63 / +0.31 / +0.15 | +0.70 / +0.32 / +0.23 |
| ice-edge error Sep / Jun / Jul–Sep / Dec–Mar | 1.20 / 1.36 / 1.42 / 0.80 | 1.19 / 1.35 / 1.42 / 0.81 | 1.19 / 1.35 / 1.41 / 0.81 |
| September correlation | 0.92 | 0.94 | 0.92 |
| drift growth (2021–22 minus 2019–20) | +0.12 | +0.09 | **+0.04** |

By year:

| | 2019 | 2020 | 2021 | 2022 |
| --- | --- | --- | --- | --- |
| September bias, `ema_s1` / `ema_ms` | +0.16 / +0.27 | −0.03 / +0.06 | −0.06 / −0.03 | +0.37 / +0.45 |
| annual-mean bias, `ema_s1` / `ema_ms` | +0.24 / +0.30 | +0.19 / +0.25 | +0.18 / +0.21 | +0.22 / +0.27 |

- **Icier, by a roughly constant amount from the first year** (annual mean +0.03 to +0.06 every year, September +0.03 to +0.11), not by more as the run goes on. The ice edge is no more and no less accurate (ice-edge errors within 0.015).
- **The lower drift growth is not less drift.** It comes from 2019–20 getting icier, not from 2021–22 getting less icy (2022: +0.45 against +0.37).
- **The drift-growth measure can't see drift.** September bias swings from −0.06 to +0.37 between years in every model, driven by that year's weather; a difference of two years against two years has an uncertainty around ±0.15, far above the target (≤ +0.09) and the differences between models (ISS-025).

## Conclusion
No improvement: multi-step pushforward shifts the model icier by about 0.05 throughout and leaves the ice edge unchanged. Not adopted; `monthly_pf_qm_ema` stays the candidate. The more important finding is about measurement: a 4-year hindcast can't measure drift, so the drift-growth target can't separate models. Judging drift experiments needs a different protocol (ISS-025) before trying more.
