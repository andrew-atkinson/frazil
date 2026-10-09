# Weekly step

**Date:** 2026-10-09 · **Branch:** `exp/weekly` · **Status:** planned · **Issue:** ISS-014

## Question
Does a model trained on weekly steps predict *monthly* sea ice better than the monthly model? Weekly data gives ~4.3× more training samples (1,043 weeks vs 240 months in 1995–2014) and finer melt and freeze timing; against that, one month takes ~4.4 autoregressive steps, so drift can compound, and runs are ~4.3× slower.

## Data source (checked 2026-10-08)
- SASIP's `OPA-neXtSIM_CREG025-ILBOXE140-MEAN/1d/` and `6h/` folders are **empty**. The source is the 6-hourly snapshots `Moorings_YYYYmMM.nc` in `OPA-neXtSIM_CREG025-ILBOXE140-S/<year>/nextsim/` on the same THREDDS server: 1995–2018, all 6 targets (`sit`, `sic`, `damage`, `siu`, `siv`, `snt`), 124 time steps and ~1.3 GB per month.
- One snapshot per day (OPeNDAP time stride 4, the 6 variables only) is ~240 MB per month, **~68 GB in total**. Reduce each month to weekly means as it streams and discard the daily data: ~4 GB kept.
- Forcing needs daily ERA5 (`t2m`, `d2m`, `sp`, `u10`, `v10`) for 1994–2022, reduced per year to weekly means; degree-days are then computed from daily temperatures (this also addresses ISS-007). Projections would need daily CMIP, but this experiment doesn't.
- Time one month first (`--dry-run`) before committing to the full download.

## Storage (prerequisite)
The laptop has ~37 GB free, so the data goes on an external drive and Frazil stays on the laptop. Drive speed barely matters: the training zarr (~4 GB weekly) stays in the 32 GB of RAM after the first pass, steps are GPU-bound and downloads are network-bound. Use an external SSD or hard disk formatted **APFS** (not FAT32/exFAT sticks: 4 GB file limit, no journal, slow sustained writes), and exclude it from Spotlight.

`frazil/paths.py` uses relative paths, so a symlink is enough. With the drive named `FrazilData`, from the repo root:

```bash
rsync -a --progress data/train_data/ /Volumes/FrazilData/train_data/
mv data/train_data data/train_data.old && ln -s /Volumes/FrazilData/train_data data/train_data
pytest --slow                     # then: rm -rf data/train_data.old
```

To free ~11 GB more, move the old step checkpoints to the drive, keeping `monthly_pf/step_20000.ckpt`. **This experiment needs `monthly/step_{25000..250000}.ckpt`** as its comparison curve, so they must stay reachable on the drive:

```bash
cd data/models && rsync -aR --remove-source-files $(ls */step_*.ckpt | grep -v '^monthly_pf/step_20000') /Volumes/FrazilData/old_checkpoints/ && cd ../..
```

`monthly_datacube_siccorr` (1.2 GB) and `data/wandb` (0.5 GB) can go too. Keep `data/models`, `data/obs` and `data/auxiliary` on the laptop so free-runs and animations work without the drive.

## Setup
- **Model:** `weekly`, trained from scratch with the same architecture and config as `monthly` (no pushforward, original targets), so the only difference is the time step. A training step costs the same at either cadence.
- **Comparison:** `monthly` at matched training steps, using its saved checkpoints (`monthly/step_*.ckpt`, every 25k) and its skill curve (`archive/skill_vs_step.csv`).
- **Scoring on calendar months**, the output we use: roll `weekly` forward, average into calendar months, and score both models the same way. Measure one-month skill against persistence for 2015–18 against neXtSIM monthly means. For promising checkpoints, also run the 2019–22 ERA5 hindcast scored per member against NSIDC (September and summer bias, July–September ice-edge error, drift growth).
- **New code:** a streaming weekly downloader, weekly preprocessing and zarr, and a monthly-aggregating evaluation for a weekly model.

## Minimum training
The monthly model trained at ~16k steps per hour (25k steps every ~95 minutes), so the full 250k steps is ~16 hours. Training is the cheap part; the download and the new code are the expensive parts.

1. **Train to 100k steps (~6.5 h, one night),** saving every 25k as the monthly run did.
2. **Score at 50k and 100k** against `monthly` at the same steps. The monthly curve passes zero skill around 60k and reaches 0.23 at 100k (0.49 at 225k), so 100k is the earliest point where the comparison is meaningful; 50k is an early warning only.
3. **Go / no-go:** if `weekly` at 100k matches or beats `monthly` at 100k on one-month skill and its 2019–22 drift growth is no worse, train on to 250k (another ~10 h) and score it fully. If it is clearly behind at 100k, stop. Between-seed noise hasn't been measured (the two 25k monthly checkpoints differ by 0.07), so treat smaller gaps as a tie.

## Commands
To be written with the code.

## Runs
| what | ledger run_id | output |
| --- | --- | --- |

## Results
None yet.

## Conclusion
None yet.
