# More of the EMA fine-tune

**Date:** 2026-10-18 · **Branch:** `exp/pf-qm-ema2` · **Status:** done · **Issue:** —

## Question
`monthly_pf_qm` → `monthly_pf_qm_ema` (20k steps of the same fine-tune at EMA 0.999) was the largest gain so far. Does another 20k steps keep improving, or is the cheapest lever exhausted?

## Setup
- Recipe `configs/experiment/pf_qm_ema2.yaml`: exactly `pf_qm_ema`, seeded from `monthly_pf_qm_ema` (weights and EMA copy); optimizer and learning-rate schedule restart. 20k steps ≈ 10k batches ≈ 170 more epochs of the 238 training samples.
- Compared with `monthly_pf_qm_ema` on the scorecard (version 2), development years 2019–22 only.

## Commands
```bash
python train.py +experiment=pf_qm_ema2
python scripts/project/freerun_monthly.py --ckpt data/models/monthly_pf_qm_ema2/last.ckpt --forcing era5 --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members --datacube data/train_data/monthly_datacube_era5x --era5-ext data/train_data/era5_forcing_datacube_era5x
python scripts/evaluate/scorecard.py --model monthly_pf_qm_ema2
```

## Runs
| what | ledger run_id |
| --- | --- |
| hindcast | `36addb25aa8a` |
| scorecard | `e0d5acb45f04` |

## Results
Per member against NSIDC-0051, 2019–22, M km² except r ([scorecard](../../results/scorecards/monthly_pf_qm_ema2.json)):

| measure | `monthly_pf_qm` | `monthly_pf_qm_ema` | `monthly_pf_qm_ema2` | v0.4.0 target |
| --- | --- | --- | --- | --- |
| September bias | +0.27 | **+0.12** | +0.22 | ≤ +0.20 |
| June / July / August bias | +0.54 / +0.34 / +0.08 | +0.64 / +0.23 / +0.10 | +0.69 / +0.46 / +0.32 | |
| ice-edge error: Sep / Jun / Jul–Sep / Dec–Mar | 1.41 / 1.56 / 1.76 / 0.91 | 1.20 / 1.36 / 1.42 / 0.80 | 1.19 / 1.34 / 1.40 / 0.80 | Jun ≤ 1.41, Jul–Sep ≤ 1.65 |
| September r | 0.95 | 0.92 | 0.95 | ≥ 0.85 |
| drift growth (2021–22 minus 2019–20) | +0.21 | +0.12 | +0.09 | ≤ +0.09 |
| Sep bias E. Siberian / Beaufort / Chukchi / Laptev | +0.26 / +0.06 / +0.15 / −0.09 | +0.19 / +0.01 / +0.11 / −0.08 | +0.20 / +0.01 / +0.10 / −0.05 | E. Siberian, Beaufort ≤ +0.10 |

The ice-edge errors improve by about 1%; the extent biases get worse in summer (September +0.10, July +0.23, August +0.22), so the ice it adds is mostly misplaced edge, not missing ice found. Drift growth reaches +0.09, on the target, but from two years against two years with one training run that is within noise.

## Conclusion
The lever is exhausted: a second 20k steps leaves the ice edge where it was and makes the summer icier, likely overfitting the 238 training samples (about 340 epochs in total since `monthly_pf_qm`). `monthly_pf_qm_ema` stays the v0.4.0 candidate (it meets the September target, this one doesn't). Next: new information for the model rather than more training on the same data, the summer-gap inputs (thaw-day degree-days, ISS-007; ocean memory).
