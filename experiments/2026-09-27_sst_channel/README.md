# ERA5 sea-surface temperature as an input

**Date:** 2026-09-25 to 2026-09-27 · **Status:** done · **Model:** `monthly_pf_qm_sst`

## Question
The emulator sees only the atmosphere; neXtSIM had an ocean. Does ERA5 SST, as a fifth forcing, give it the ocean heat that drives Pacific-shelf melt?

## Setup
- Recipe `configs/experiment/pf_qm_sst.yaml`: the `monthly_pf_qm` recipe plus SST at months *t* and *t+1* (26 inputs). The new input weights start at zero, so step 0 reproduces the seed exactly (tested).
- A/B partner: `monthly_pf_qm`. Leak test: the same model with SST replaced by its 1995–2014 monthly climatology (`--climatology-forcing sst`).
- Known risk: ERA5 sets SST to 271.46 K under the *observed* ice, so SST at freezing already places the September edge within 1.1 M km² of NSIDC, better than the model's 1.9.

## Commands
```bash
python scripts/download/era5.py --extra --years 1995-2025
python scripts/data/add_forcing_channels.py --leak-check --era5-dir data/train_data/era5_extra_1995_2025 --vars sst
python train.py +experiment=pf_qm_sst
python scripts/project/freerun_monthly.py --ckpt data/models/monthly_pf_qm_sst/last.ckpt --forcing era5 --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members --datacube data/train_data/monthly_datacube_era5x --era5-ext data/train_data/era5_forcing_datacube_era5x [--climatology-forcing sst]
```

## Runs
| what | ledger run_id |
| --- | --- |
| real SST | `746d3defa31e` |
| climatological SST | `24a126bd2129` |
| A/B partner `monthly_pf_qm` | `f3d9c5bc56e2` |

## Results
| per member, 2019–25 | September bias | September ice-edge error | summer ice-edge error | r |
| --- | --- | --- | --- | --- |
| `monthly_pf_qm` | +0.37 | 1.53 | 1.83 | 0.86 |
| + real SST | +0.42 | 1.52 | 1.80 | 0.86 |
| + climatological SST | +0.43 | 1.53 | 1.83 | 0.86 |

The model barely uses the channel: SST's input-weight norm is 7% (t) / 17% (t+1) of the other forcings in the training weights and 2% / 5% in the inference weights (`scripts/evaluate/channel_weights.py`).

## Conclusion
No effect, and no leak, because the channel was hardly learned: either the fine-tune was too gentle (ISS-005) or SST adds little beyond ERA5 air temperature, which already carries the observed ice. Not pursued further; see [radiation_channel](../2026-09-28_radiation_channel/).
