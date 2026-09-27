# Data and models — inventory

What each model is, what each dataset does, the years it covers, and how much it actually shapes the model. Written 2026-09-24. For the pipeline diagram see [STATUS.md](STATUS.md); for why the model sits above observations see [OBS_GAP.md](OBS_GAP.md).

## Models

All share one architecture: a flow-matching Transformer (~4.2 M trainable parameters, 7.9 M including the EMA copy) on a 512×512 polar grid, predicting the 6 ice states one month ahead from the previous state + atmospheric forcing. Checkpoints live in `data/models/<name>/`.

| model | lineage | training | status and headline numbers |
| --- | --- | --- | --- |
| `monthly` | trained from scratch | ~250k steps, original neXtSIM targets | The base/reference model. `monthly.ckpt`. |
| `monthly_pf` | `monthly` + pushforward (drift correction) | +55k steps: 20k gentle (prob 0.5, substeps 8), then 35k harder (prob 0.8, substeps 10); original targets | Best forecast skill. ERA5 hindcast Sept 2013–17: 5.69 M km², +0.71 vs obs — the iciest, because pushforward locks onto the icy targets. The 20k version is kept at `step_20000.ckpt`. |
| `monthly_qm` | `monthly` fine-tuned on QM de-iced targets | +20k steps, lr 1e-4, no pushforward | The QM correction barely imprinted. ERA5 hindcast Sept 5.48, +0.49 vs obs. |
| `monthly_pf_qm` | `monthly_pf` (55k) + pushforward on de-iced targets | 40k steps, prob 0.8, substeps 10, `data.suffix=_sicqm` | **Best model out-of-sample** (2019–25, per member): Sept bias +0.37, July +0.28, August +0.23, Sept r 0.86. Recommended for the animation. |
| `monthly_pf_qm_sst` | `monthly_pf_qm` recipe + ERA5 SST as a 5th forcing (26 inputs; new weights zero-initialised) | 40k steps, `config_finetune_pf_qm_sst_mac.yaml` | Trained 2026-09-26 (40k). **No effect**: 2019–25 scores match `monthly_pf_qm` with real or climatological SST, because the channel was barely learned (inference weights 2–5% of the other forcings). See OBS_GAP.md 2026-09-27. |
| `monthly_pf_qm_rad` | `monthly_pf_qm` + ERA5 downward solar and thermal radiation (28 inputs) | 20k steps, `config_finetune_pf_qm_rad_mac.yaml`: EMA 0.999, new inputs fed in 10× larger so they learn faster | Configured 2026-09-27. Check at 10k steps with `channel_weights.py`; attribute with `--climatology-forcing ssrd strd`. Can't run CMIP projections until CMIP radiation (`rsds`, `rlds`) is added. |

## Datasets

"Weight" = how much the dataset determines what the model does.

| dataset (path) | years | what it is / does | weight on the model |
| --- | --- | --- | --- |
| **neXtSIM-OPA** (`train_data/nextsim_opa_monthly/` raw → states in `monthly_datacube/`) | 1995–2018 | The 6 ice states — sit (thickness), sic (concentration), sid (damage), siu/siv (drift), snt (snow). The training targets: what the model learns to produce. | **Total.** The only source of target physics, so its biases are the model's (Sept ~0.37 M km² too icy on the 1995–2018 average, ~0.15 in recent years). The simulation does not exist outside 1995–2018. |
| **ERA5 monthly means** (`train_data/data_stream-moda_stepType-avgua.nc` raw → forcings in `monthly_datacube/`) | 1994–2018 (1994 is degree-day spin-up) | Atmospheric forcing: t2m, d2m, sp, u10, v10 → tus, huss, uas, vas (+ rhus) and 4 degree-day features. What the model is conditioned on. | **Total for forcing.** The model has only ever seen ERA5; any other forcing (CMIP) is out-of-distribution. |
| **ERA5 extension** (`train_data/era5_2018_2025/` raw → `train_data/era5_forcing_datacube/`) | 2018–2025 (2018 is spin-up; cube holds 2019–2025) | Forcing-only cube for a true out-of-sample hindcast, set up 2026-09-24. No states — neXtSIM ends in 2018. | **None on training.** Evaluation only: run from the Dec 2018 state under real 2019–2025 weather and compare to NSIDC. |
| **Train/validation zarrs** (`train_monthly.zarr`, `validation_monthly.zarr`) | train 1995–2014 (240 months), val 2015–2018 (48 months) | The training-ready split of `monthly_datacube` (14 channels: 6 states + 4 forcings + 4 degree-days, 127,152 ocean cells). | This *is* the training set: **288 months total**, which is small. That's why skill fades after a few months and drift exists. Validation picks the best checkpoint, so 2015–18 is not fully independent. |
| **QM targets** (`monthly_datacube_sicqm/`, `train_sicqm.zarr`, `validation_sicqm.zarr`) | 1995–2018 | neXtSIM with `sic` quantile-mapped per cell and per month onto NSIDC-0051. Only `sic` differs. | Only for `monthly_qm` and `monthly_pf_qm`. |
| **ERA5 extra fields** (`train_data/era5_extra_1995_2025/` raw → `*_era5x/` cubes: `monthly_datacube_era5x`, `monthly_datacube_sicqm_era5x`, `era5_forcing_datacube_era5x`; zarr `train_sicqm_sst.zarr`) | 1995–2025 | `sst` (K), `ssrd` and `strd` (W m⁻²) appended to the datacubes by `add_forcing_channels.py`. The `_era5x` cubes are supersets: the model's config picks which channels it sees. | Only for models that list them in `data.forcing_variables` (so far `monthly_pf_qm_sst`: sst). **ERA5 SST is set to 271.46 K under the observed sea ice**, so it nearly *is* the observed ice mask (September edge disagreement with NSIDC 1.1 M km², better than the model's own ~1.9). |
| Mean-shift targets (`monthly_datacube_siccorr/`) | 1995–2018 | Additive `sic` correction — parked, it didn't move extent. | None. 1.2 GB, safe to delete. |
| **CMIP6 forcing** (`cmip_forcing.nc` raw → `cmip_datacube/`), MPI-ESM1-2-LR ssp245, one member | 2015–2100 | Future atmosphere for projections and the animation. Bias-corrected per calendar month against ERA5 1995–2014 by `preprocess_cmip.py`. | **Never trained on** — inference only. One climate trajectory: wrong weather for any specific year, and runs ~0.36 K colder than ERA5 over 2010–18 (~+0.13 M km² of ice). |
| **NSIDC-0051 gridded** (`obs/nsidc0051/` raw → `obs/nsidc0051_grid/` on the model grid) | 1995–2025 | Observed passive-microwave concentration, regridded to the model grid. | Fits the QM correction (all 12 months) and is the obs line in comparisons. Agrees with the official index to ~0.1 M km². The only data in the repo that covers 2019–2025. |
| **NSIDC Sea Ice Index** (`obs/nsidc/N_MM_extent_v4.0.csv`, all 12 months) | 1979–2025 | Official pan-Arctic extent and area. | Evaluation only (`ice_area.py --nsidc`). |
| **Auxiliary grid** (`auxiliary/ds_auxiliary.nc`) | — | Ocean mask (127,152 cells, 27.6 M km²), cell areas, x/y and lon/lat coordinates. | Structural — used everywhere. |
| `auxiliary/ds_demo.nc` | — | Leftover from the original GenSIM demo. | None. |

## Caveats

0. **Fine-tunes mostly kept their seed's inference weights.** With `ema_rate 0.9999` and `ema_update_every 4`, the EMA copy used for inference still holds 61% of the seed model after 20k steps and 37% after 40k. So `monthly_qm`, `monthly_pf_qm` and `monthly_pf_qm_sst` are partly their seeds, and their gains are probably understated. New fine-tunes use `ema_rate 0.999`.

1. **Until the ERA5 extension runs, there is no true out-of-sample test.** The 2013–17 hindcast straddles training years (2013–14) and validation years (2015–17), and validation data chooses checkpoints. Every "vs obs" number so far is at least partly in-sample.
2. **The QM models have seen the obs they're scored against.** The QM maps were fitted on 1995–2018 NSIDC, including 2015–18. So `monthly_qm` and `monthly_pf_qm` look somewhat better against 2015–18 obs than they really are. The 2019–2025 hindcast is clean for them too: nothing was fitted on those years.
3. **ERA5 SST carries the observed ice mask, and ERA5 air temperature carries a weaker imprint of it.** ERA5 prescribes SST and sea ice from satellite analyses. Any gain from SST must survive the `--climatology-forcing sst` test before it counts as physics.
4. **The CMIP free-run is not a hindcast.** Compare it to observations on trend and distribution only, never year by year.

## Housekeeping (safe to delete)

- `data/train_data/monthly_datacube_siccorr/` (1.2 GB) — the parked mean-shift correction.
- Older `step_*.ckpt` in `data/models/*/` — training history. Keep `step_20000.ckpt` in `monthly_pf` (the gentle-pushforward model).
- Loose `plots/freerun/snapshots/state_*.nc` at the top level — an old un-namespaced CMIP free-run (Jan/Jul/Sep, 2015–2095) from before snapshot folders were namespaced.
