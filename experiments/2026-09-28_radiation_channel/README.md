# Downward solar and thermal radiation as inputs

**Date:** 2026-09-27 (trained) · **Status:** in progress (hindcasts pending, ISS-004) · **Model:** `monthly_pf_qm_rad`

## Question
Summer melt is driven mostly by absorbed sunlight, and monthly air temperature over melting ice sits near 0 °C whether the month was sunny or cloudy. Does giving the model radiation fix the Pacific-shelf melt deficit?

## Setup
- Recipe `configs/experiment/pf_qm_rad.yaml`: seeded from `monthly_pf_qm`, QM targets and pushforward, plus `ssrd` and `strd` at *t* and *t+1* (28 inputs, zero-initialised), 20k steps.
- Two changes so new inputs can be learned: EMA rate 0.999 (was 0.9999), and the new channels fed in 10× larger (encoder std / 10) so they learn ~10× faster under Adam.

## Commands
```bash
python train.py +experiment=pf_qm_rad
python scripts/evaluate/channel_weights.py data/models/monthly_pf_qm_rad/step_10000.ckpt
python scripts/project/freerun_monthly.py --ckpt data/models/monthly_pf_qm_rad/last.ckpt --forcing era5 --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members --datacube data/train_data/monthly_datacube_era5x --era5-ext data/train_data/era5_forcing_datacube_era5x [--climatology-forcing ssrd strd]
```

## Results so far
Input-weight norm relative to the other forcings, inference weights: `ssrd` 4–5% at 10k steps and 6–8% at 15.5k; `strd` 3–4% and 5–6%. Growing slowly, the same pattern as SST.

## Next
Run the two hindcasts (real and climatological radiation) and score them against `monthly_pf_qm`, per member and by sector.
