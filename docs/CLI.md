# GenSIM monthly model — CLI reference

Command-by-command reference for the monthly sea-ice pipeline: download →
preprocess → train → evaluate → project. Every script is a plain
`python <script>.py` with `--flags` (except `train.py`, which is Hydra-driven,
and `download/era5.py`, which is edit-the-script). Defaults are shown so most
commands run with no arguments at all.

- **Environment:** activate the project env first (e.g. `conda activate gensim`).
  All the model scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1` and auto-detect the
  device (Apple-Silicon **MPS**, else CPU; CUDA if present).
- **Run from the repo root** (`seaIce/gensim/`) — default paths are relative to it.

---

## The pipeline at a glance

```
download/nextsim.py ─┐                        (neXtSIM-OPA sea-ice TRUTH, 1995-2018)
download/era5.py  ───┼─▶ preprocess_monthly.py ─▶ data/train_data/monthly_datacube/  ─┐
                     │                             (states + ERA5 forcing, on-grid)    │
                     │                                                                 ├─▶ train.py ─▶ data/models/monthly/monthly.ckpt
download/cmip.py ────┴─▶ preprocess_cmip.py ────▶ data/train_data/cmip_datacube/       │
                          (regrid + BIAS-CORRECT)  (future forcing, 2015-2100)          │
                                                                                        ▼
   eval_monthly.py · predict_monthly.py · rollout_monthly.py · sweep_skill_monthly.py   (skill / maps)
   freerun_monthly.py                                                                   (decadal projection)
```

Data volumes: the model trains on **240 monthly samples** (1995–2014) and is
validated on **2015–2018**. The trained checkpoint is `data/models/monthly/monthly.ckpt`.

---

# 1 · Download

### `download/nextsim.py` — sea-ice targets (the truth)
neXtSIM-OPA monthly fields from the SASIP THREDDS/OPeNDAP server. These are what
the model learns to predict.

```bash
python download/nextsim.py                      # all years to nextsim_opa_monthly/
python download/nextsim.py --dry-run            # inspect one file's variables, exit
```

| Flag | Default | Meaning |
|---|---|---|
| `--start-year` | first available | first year to fetch |
| `--end-year` | last available | last year to fetch |
| `--out-dir` | `nextsim_opa_monthly` | output directory |
| `--overwrite` | off | re-download years even if the file exists |
| `--dry-run` | off | open one file, print its variables, exit |

### `download/era5.py` — atmospheric forcing (ERA5)
**Edit-the-script**, not flag-driven: the CDS `request` dict at the top holds the
years/variables. Needs credentials in `~/.cdsapirc` (never commit them — see
`download/cdsapirc.example`). Pulls the 5 raw fields later turned into
`tus/huss/uas/vas` + degree-days.

```bash
python download/era5.py                          # downloads per the request dict
```

### `download/cmip.py` — future scenario forcing (CMIP6)
Fetches one model/member/scenario from the analysis-ready CMIP6 zarr on Google
Cloud (`gs://cmip6`, anonymous), stitching historical (≤2014) + scenario (≥2015).
Gives `tas/huss/uas/vas` directly (no humidity derivation needed).

```bash
python download/cmip.py --dry-run                # resolve stores + report, no download
python download/cmip.py                          # MPI-ESM1-2-LR ssp245, 1994-2100
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `MPI-ESM1-2-LR` | CMIP6 `source_id` |
| `--member` | `r1i1p1f1` | ensemble member id |
| `--scenario` | `ssp245` | future `experiment_id` (e.g. `ssp245`, `ssp585`) |
| `--grid` | `gn` | `grid_label` (`gn`/`gr`/…) |
| `--start-year` | `1994` | first year (needs ≤1994 for degree-day lead-in) |
| `--end-year` | `2100` | last year |
| `--out` | `data/train_data/cmip_forcing.nc` | output NetCDF |
| `--dry-run` | off | open lazily, report shapes/spans, write nothing |

> **Caveat:** CMIP is a *scenario*, not a forecast, and its atmosphere is biased
> vs ERA5. The bias is removed in the next step, not here.

### `download/nsidc.py` — observed sea-ice extent (validation)
Fetches NSIDC Sea Ice Index monthly extent CSVs (NOAA@NSIDC G02135 v4.0, 1979–
present) — the observational baseline for validating GenSIM past the 2018 neXtSIM
window. Extent = ocean area with SIC ≥ 15%, in 10⁶ km². Open access, stdlib only.

```bash
python download/nsidc.py                 # September, North -> data/obs/nsidc/
python download/nsidc.py --month 0       # all 12 months
```
| Flag | Default | Meaning |
|---|---|---|
| `--month` | `9` | calendar month 1–12, or `0` for all twelve |
| `--hemisphere` | `north` | `north` or `south` |
| `--out-dir` | `data/obs/nsidc` | output directory |

Overlay it on `ice_area.py --nsidc …` to compare model vs observed extent.

### `download/nsidc0051.py` — gridded observed concentration (for target correction)
Fetches **NSIDC-0051 v2** monthly gridded sea-ice concentration (25 km NH polar
grid) via NASA **Earthdata** — the observational field used to bias-correct the
neXtSIM `sic` training targets. Needs `pip install earthaccess` and an Earthdata
login (no EULA; token/user-pass via env or `~/.netrc` — see the file header;
never commit or pass credentials on the command line).

```bash
python download/nsidc0051.py --list                        # search + list granule names
python download/nsidc0051.py                               # download monthly 1995-2025
```
| Flag | Default | Meaning |
|---|---|---|
| `--start-year` / `--end-year` | `1995` / `2025` | 1995–2018 = the neXtSIM overlap for target correction; 2019–2025 = gridded obs for validating the projection |
| `--out-dir` | `data/obs/nsidc0051` | output directory |
| `--daily` | off | fetch daily instead of monthly |
| `--list` | off | search + print granule names, download nothing |

---

# 2 · Preprocess (onto the 512×512 NPS grid)

### `experiments/preprocess_monthly.py` — build the training datacube
Regrids neXtSIM targets + ERA5 forcing onto the model grid (nearest-neighbour),
rotates winds into the grid frame, derives humidity and the four degree-day
features, and writes one file per year: `monthly_datacube_YYYY.nc`.

```bash
python experiments/preprocess_monthly.py                 # all years
python experiments/preprocess_monthly.py --years 2015 2016
```

| Flag | Default | Meaning |
|---|---|---|
| `--aux-path` | `data/auxiliary/ds_auxiliary.nc` | grid geometry + land/sea mask |
| `--era5-path` | `data/train_data/data_stream-moda_...nc` | raw ERA5 file |
| `--nextsim-dir` | `data/train_data/nextsim_opa_monthly` | downloaded targets |
| `--out-dir` | `data/train_data/monthly_datacube` | output datacube dir |
| `--years` | all | restrict to these years (space-separated) |
| `--overwrite` | off | rebuild years whose file exists |

### `experiments/preprocess_cmip.py` — future forcing, **bias-corrected**
Regrids the CMIP fields onto the same grid, rotates winds, then **bias-corrects
each field against ERA5** (additive per-calendar-month, per-cell delta so CMIP's
baseline climatology matches ERA5's), and recomputes degree-days from the
*corrected* temperature. Writes `cmip_datacube_YYYY.nc` stamped with the
correction baseline.

```bash
python experiments/preprocess_cmip.py --selfcheck        # unit-test the bias math, exit
python experiments/preprocess_cmip.py                    # regrid + bias-correct (default)
python experiments/preprocess_cmip.py --no-bias          # raw (off-distribution) version
```

| Flag | Default | Meaning |
|---|---|---|
| `--cmip` | `data/train_data/cmip_forcing.nc` | raw CMIP from `download/cmip.py` |
| `--aux` | `data/auxiliary/ds_auxiliary.nc` | grid geometry + mask |
| `--out-dir` | `data/train_data/cmip_datacube` | output datacube dir |
| `--start-year` | `2015` | first year written |
| `--end-year` | `2100` | last year written |
| `--bias-ref` | `data/train_data/monthly_datacube` | on-grid ERA5 to correct against |
| `--bias-years` | `1995-2014` | baseline overlap period (historical CMIP vs ERA5) |
| `--no-bias` | off | skip correction (stamps datacube `bias_corrected=none`) |
| `--selfcheck` | off | run the `monthly_bias` unit check and exit |

> **Why bias correction matters:** raw CMIP pushes the model off-distribution
> (it sees an atmosphere unlike its ERA5 training). Corrected, the mean seasonal
> cycle matches ERA5 while the scenario's warming trend rides on top — so a
> future rollout can legitimately show declining ice. It's a **mean/delta**
> correction only (fixes the mean cycle, not variance/tails; upgrade path =
> quantile mapping, noted in the source). Verify an existing datacube with:
> ```bash
> python3 - <<'PY'
> import xarray as xr, glob
> def m(p,v): return float(xr.open_mfdataset(sorted(glob.glob(p)),combine="by_coords")["datacube"].sel(var_names=v).mean().compute())
> cm=m("data/train_data/cmip_datacube/cmip_datacube_201[5-8].nc","tus")
> er=m("data/train_data/monthly_datacube/monthly_datacube_201[5-8].nc","tus")
> print(f"diff {cm-er:+.2f}K :: "+("corrected" if abs(cm-er)<1.5 else "looks RAW"))
> PY
> ```

### `experiments/estimate_normalization_monthly.py` — normalization stats
Computes the 18-channel mean/std the `GaussianEncoder` standardizes with (6 states
+ 4 forcings×2 + 4 degree-days) from the training datacube, and writes them into
the monthly Hydra configs. **Run after `preprocess_monthly.py`, before `train.py`.**

```bash
python experiments/estimate_normalization_monthly.py            # compute + wire into configs
python experiments/estimate_normalization_monthly.py --no-configs   # print stats only
```
| Flag | Default | Meaning |
|---|---|---|
| `--datacube` | `data/train_data/monthly_datacube/monthly_datacube_*.nc` | glob of datacube files |
| `--out-json` | `experiments/normalization_monthly.json` | stats output |
| `--no-configs` | off | only compute/print stats, don't edit the configs |

### `experiments/build_zarr_monthly.py` — pack train/validation zarr
Converts the per-year datacube NetCDFs into the consolidated `train_monthly.zarr`
/ `validation_monthly.zarr` stores the data module actually loads (a single
continuous monthly sequence, `delta_t=1`). **Run after normalization, before
`train.py`.**

```bash
python experiments/build_zarr_monthly.py
```
| Flag | Default | Meaning |
|---|---|---|
| `--datacube-dir` | `data/train_data/monthly_datacube` | input datacube dir |
| `--out-dir` | `data/train_data` | where the zarr stores are written |
| `--suffix` | `_monthly` | names them `train<suffix>.zarr` / `validation<suffix>.zarr` |
| `--val-start-year` | `2015` | first year in the validation store (rest are train) |
| `--aux-path` | `data/auxiliary/ds_auxiliary.nc` | grid + mask |
| `--overwrite` | off | rebuild stores that already exist |
| `--no-verify` | off | skip the post-write read-back check |

### `experiments/preprocess_nsidc0051.py` — regrid observed concentration
Regrids downloaded NSIDC-0051 monthly concentration onto the 512×512 GenSIM grid
(projects the EPSG:3411 grid with `pyproj`, handles the pole-hole/land flags,
nearest-neighbour via `preprocess_monthly`'s helpers). Output is a `sic`-only
datacube (`data/obs/nsidc0051_grid/`) in the same layout as `monthly_datacube`,
for correcting the `sic` targets and for spatial validation.

```bash
python experiments/preprocess_nsidc0051.py                 # 1995-2025 -> obs sic on grid
```
| Flag | Default | Meaning |
|---|---|---|
| `--in-dir` | `data/obs/nsidc0051` | downloaded NSIDC-0051 files (North only used) |
| `--aux` | `data/auxiliary/ds_auxiliary.nc` | target grid + mask |
| `--out-dir` | `data/obs/nsidc0051_grid` | gridded obs output |
| `--start-year` / `--end-year` | `1995` / `2025` | years to regrid |

It prints each year's September extent — cross-check against NSIDC's number
(`ice_area.py … --nsidc data/obs/nsidc`) to confirm the regrid.

---

# 3 · Train

### `train.py` — train the monthly model (Hydra)
Hydra-driven (`config_name='config_train'`, `config_path='.'`). On Apple Silicon
use the Mac config, which shrinks the network (`n_features=256`, `n_blocks=4`),
sets `accelerator: mps`, `precision: 32`, and writes checkpoints to
`data/models/monthly/`.

```bash
python train.py --config-name config_train_monthly_mac          # train on Mac/MPS
python train.py --config-name config_train_monthly_mac ckpt_path=data/models/monthly/monthly.ckpt   # resume
```
Override any config value inline, Hydra-style: `key=value` (e.g.
`surrogate.optimizer.lr=1e-4`). Checkpoints rotate as `last.ckpt`, `last-v1.ckpt`,
… ; the trained model is consolidated to `monthly.ckpt` (see *Checkpoints* below).

**Auto-resume:** re-running any `train.py` command continues its run if
`data/models/<exp_name>/last.ckpt` exists (optimizer, scheduler, step, RNG, EMA
all restored) — so you can stop (`Ctrl-C`) and pick up any time. Pass an explicit
`ckpt_path=…` to override, or delete the run dir to start fresh.

**Pushforward (rollout) fine-tune** — teach the model to correct its own
free-running drift, resumable in short bursts:
```bash
python train.py --config-name config_finetune_pushforward_mac
```
Starts from `monthly.ckpt`, writes to `data/models/monthly_pf/`. Full guide,
knobs, and cost in **[docs/PUSHFORWARD.md](PUSHFORWARD.md)**.

### `experiments/smoke_train_monthly.py` — training smoke test
Runs a handful of steps to confirm the training loop, data, and device all work
before committing to a full run.

```bash
python experiments/smoke_train_monthly.py
```
| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config_train_monthly.yaml` | training config to smoke-test |
| `--seed` | `0` | RNG seed |

### `experiments/ab_optimizer_monthly.py` — optimizer A/B micro-benchmark
Short comparison run (loss-vs-steps) to sanity-check optimizer settings.

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | `200` | steps per variant |
| `--batch-size` | `2` | batch size |
| `--cpu` | off | force CPU |
| `--out-png` | `experiments/ab_optimizer.png` | output figure |

---

# 4 · Evaluate & forecast (validation window, has truth)

These share a loader and read the trained checkpoint. If `--ckpt` is omitted they
resolve `data/models/monthly/monthly.ckpt` (else the newest `last*.ckpt`).

### `experiments/eval_monthly.py` — one-month-ahead skill table
Predicts month *t+1* from **true** month *t* + forcings over the validation
window, and reports RMSE over ocean per variable and the **skill vs persistence**
(`1 − RMSE_model/RMSE_persistence`; >0 beats "no change").

```bash
python experiments/eval_monthly.py --n-ens 8
```
| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | `monthly.ckpt` | checkpoint (auto-resolved if omitted) |
| `--config` | `config_forecast_monthly.yaml` | forecast/inference config |
| `--train-config` | `config_train_monthly_mac.yaml` | architecture the ckpt was trained with |
| `--datacube` | `data/train_data/monthly_datacube` | states + forcing |
| `--aux` | `data/auxiliary/ds_auxiliary.nc` | grid + mask |
| `--val-start-year` | `2015` | first validation year |
| `--max-pairs` | `0` | cap month-pairs (0 = all) |
| `--n-ens` | `1` | ensemble members; the mean is scored (see *Ensemble mean*) |

### `experiments/predict_monthly.py` — one forecast, as maps
Predicts one target month and plots, per variable, **truth | GenSIM | persistence**
maps. The visual "does the forecast look like reality" check.

```bash
python experiments/predict_monthly.py --target-year 2018 --target-month 3
```
| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` / `--config` / `--train-config` / `--datacube` / `--aux` | as above | — |
| `--target-year` | `2018` | year to forecast |
| `--target-month` | `3` | month to forecast (input = previous month) |
| `--n-ens` | `8` | ensemble members (mean) |
| `--out-png` | `plots/forecast_maps.png` | output figure |

### `experiments/rollout_monthly.py` — multi-start skill vs lead
Free-runs from several start months (feeding the model its **own** output back
with real forcings), capped at `--max-lead`, and pools squared error across
starts by lead. Reports skill vs **persistence** *and* vs **climatology**
(average-that-month) — the honest long-lead bar.

```bash
python experiments/rollout_monthly.py --fast                                   # 4 starts, 12-mo cap
python experiments/rollout_monthly.py --fast --max-lead 47 --starts 2015-01,2015-07   # multi-year
python experiments/rollout_monthly.py --fast --starts 2015-01,2016-01,2017-01         # robustness
```
| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` / `--config` / `--train-config` / `--datacube` / `--aux` | as above | — |
| `--starts` | `2015-01,2015-04,2015-07,2015-10` | comma-sep `YYYY-MM` starts; skill averaged over them |
| `--max-lead` | `12` | cap trajectory length (months) |
| `--n-ens` | `8` | ensemble members (mean) |
| `--fast` | off | first-order sampler + 12 substeps (~3× faster; for curves, not art) |
| `--out-png` | `plots/rollout_multistart.png` | figure; a `.csv` is written alongside |

Output tables carry an `n` column = starts averaged at that lead (drops at long
lead when only early starts reach it).

### `experiments/sweep_skill_monthly.py` — skill vs training step
Evaluates checkpoints at roughly every `--interval` steps to show how skill
improved during training. Caches to CSV so it isn't recomputed.

```bash
python experiments/sweep_skill_monthly.py
```
| Flag | Default | Meaning |
|---|---|---|
| `--ckpt-dir` | `data/models/monthly` | directory of checkpoints |
| `--config` / `--train-config` / `--datacube` / `--aux` | as above | — |
| `--interval` | `25000` | evaluate a checkpoint near every N steps |
| `--max-pairs` | `12` | month-pairs per checkpoint (speed) |
| `--out-csv` | `plots/skill_vs_step.csv` | cache/output CSV |
| `--out-png` | `plots/skill_vs_step.png` | output figure |

---

# 5 · Decadal projection (past 2018 — no truth)

### `experiments/freerun_monthly.py` — the primary use case
Free-runs for years, driven by either the **cyclic** 2015–2018 atmosphere (a
"groundhog decade" that isolates model drift; **no warming trend**) or the
**CMIP** scenario forcing (real trend; use the bias-corrected datacube). Past
2018 there is no truth, so it logs **self-consistency** diagnostics (domain-mean
state, ice area, a sharpness/spectral-collapse proxy, physical-bound health),
saves spatial **snapshots**, and draws a **final-state map**.

```bash
python experiments/freerun_monthly.py --years 15 --fast                        # 15-yr cyclic stability
python experiments/freerun_monthly.py --forcing cmip --years 5 --fast          # 5-yr CMIP smoke test
python experiments/freerun_monthly.py --forcing cmip --years 80 --fast --snapshot-every 1   # decadal, every month saved
```
| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` / `--config` / `--train-config` / `--datacube` / `--aux` | as above | — |
| `--cmip-datacube` | `data/train_data/cmip_datacube` | bias-corrected future forcing |
| `--start` | `2015-01` | start month `YYYY-MM` (initial state from truth) |
| `--years` | `15` | length of the free run (× 12 = months). Capped by available forcing |
| `--n-ens` | `8` | ensemble members (mean) |
| `--fast` | off | first-order sampler + 12 substeps (~3× fewer net evals; keep OFF for pretty art frames) |
| `--forcing` | `cyclic` | `cyclic` (repeat 2015–18) or `cmip` (scenario, bias-corrected) |
| `--out-dir` | `plots/freerun` | outputs (csv, diagnostics png, map, snapshots/) |
| `--snapshot-every` | `12` | save the full spatial state every N months (0=off); final always saved |
| `--no-map` | off | skip the final-state map figure |

**Outputs:** `freerun_<forcing>.csv` (diagnostics), `freerun_<forcing>.png`
(4-panel diagnostics), `freerun_<forcing>_map.png` (6-variable map of the final
month), and `snapshots/state_YYYYMM.nc` (full spatial states — what maps/animations
are built from). With a January start, `--snapshot-every 12` lands on winter
maxima; use `--snapshot-every 1` (or `--start 2015-09`) to capture the September
minimum for a shrinking-ice animation.

**Reading it:** the *sharpness* panel is the key failure detector — decay toward 0
= smoothing to mush (classic autoregressive collapse). Under cyclic forcing a
stable repeating sawtooth is the good outcome *and* the ceiling (no trend to
respond to). Only the CMIP run can show climate decline.

### `experiments/extend_snapshots.py` — restart from snapshots, step forward
Reuses existing snapshots as restart points instead of re-running a whole
rollout. Built for: a long run saved only Jan/Jul snapshots, but you want the
**September** minimum every year — seed each July state, step 2 months with CMIP
forcing. 79 Julys × 2 steps ≈ 160 steps (~15 min at `--n-ens 1`) vs a ~12 h full
re-run. Writes `state_YYYY09.nc` into the snapshots folder so `ice_area.py
--month 9` / `animate_snapshots.py --month 9` pick them up.

```bash
python experiments/extend_snapshots.py --fast --n-ens 1 --dry-run   # print the plan, exit
python experiments/extend_snapshots.py --fast --n-ens 1             # July -> September
```
| Flag | Default | Meaning |
|---|---|---|
| `--snap-dir` | `plots/freerun/snapshots` | folder of `state_*.nc` to restart from |
| `--out-dir` | = `--snap-dir` | where to write the new states |
| `--from-month` | `7` | restart from this month (7 = July) |
| `--steps` | `2` | months to advance (7 + 2 = September) |
| `--cmip-datacube` | `data/train_data/cmip_datacube` | forcing for the extra steps |
| `--n-ens` | `1` | ensemble members (mean) |
| `--fast` | off | first-order sampler + 12 substeps |
| `--overwrite` | off | redo targets that already exist |
| `--dry-run` | off | print which seeds → which outputs, then exit |

> **Approximation:** snapshots hold the ensemble *mean* (spread was discarded), so
> restarting from them isn't identical to a continuous rollout — but over 2 months
> the error is small, fine for a September extent estimate.

### `experiments/animate_snapshots.py` — snapshots → GIF/MP4
Turns a `freerun` `snapshots/` folder into an animation of one variable on a
**fixed** colour scale (so the trend shows, not per-frame rescaling), land masked.
Filter to one calendar month for a yearly-cadence view (e.g. the September
minimum) or animate every month. Reuses no model — pure plotting from the `.nc`
files, so it's instant and re-runnable.

```bash
python experiments/animate_snapshots.py --var sic --month 9        # Sept minimum, yearly
python experiments/animate_snapshots.py --var sit --out ice.mp4 --fps 12
python experiments/animate_snapshots.py --selfcheck                # test frame selection
```
| Flag | Default | Meaning |
|---|---|---|
| `--snap-dir` | `plots/freerun/snapshots` | folder of `state_*.nc` |
| `--aux` | `data/auxiliary/ds_auxiliary.nc` | grid + land mask |
| `--var` | `sic` | variable (`sit`/`sic`/`sid`/`siu`/`siv`/`snt`) |
| `--month` | all | keep only this calendar month (1–12) |
| `--out` | `plots/freerun/anim_<var>[_mMM].<gif>` | `.gif` (pillow) or `.mp4` (needs ffmpeg) |
| `--fps` | `6` | frames per second |
| `--dpi` | `110` | frame resolution |
| `--vmin` / `--vmax` | pooled 2/98 pctile | override the fixed colour scale |
| `--selfcheck` | off | unit-test frame ordering/filtering, exit |

> For the decadal shrinking-ice animation: run the projection with
> `--snapshot-every 1` (every month saved), then `--var sic --month 9` for the
> summer minimum over the century. `.gif` needs only pillow; `.mp4` needs ffmpeg.

### `experiments/ice_area.py` — sea-ice area/extent vs time
Graphs total sea-ice **extent** (area where SIC > threshold) or **area**
(SIC-weighted) over time, from any state-bearing cube — the neXtSIM truth
datacube or a `freerun` snapshots folder — and **overlays** several as separate
lines (e.g. truth 1995–2018 beside a projection). Pure plotting, no model.
`cmip_datacube` won't work (forcing only, no ice states).

```bash
python experiments/ice_area.py data/train_data/monthly_datacube               # truth extent
python experiments/ice_area.py data/train_data/monthly_datacube plots/freerun/snapshots  # truth + projection
python experiments/ice_area.py plots/freerun/snapshots --month 9 --metric area --csv
```
| Flag | Default | Meaning |
|---|---|---|
| `sources` (positional) | — | one or more cube dirs; each becomes a line |
| `--labels` | basenames | comma-sep line labels |
| `--aux` | `data/auxiliary/ds_auxiliary.nc` | grid + mask + `cell_area` |
| `--metric` | `extent` | `extent` (SIC>threshold) or `area` (SIC-weighted) |
| `--threshold` | `0.15` | SIC extent threshold (standard 15%) |
| `--month` | all | keep only this calendar month (e.g. 9 = Sept minimum) |
| `--nsidc` | none | NSIDC extent CSV or dir (from `download/nsidc.py`) to overlay as observations; needs `--month`; prints each model line's mean offset vs obs |
| `--out` | `plots/ice_area.png` | output figure |
| `--csv` | off | also write the series as CSV |
| `--selfcheck` | off | unit-test the area + NSIDC-parse logic, exit |

> The offset it prints (`model − NSIDC`, M km²) is the anchor correction: NSIDC is
> pan-Arctic while the model is grid-limited, so treat it as a bias estimate, not
> an exact accounting.

### `experiments/drift_rollout.py` — original demo-model drift
Long free-running rollout for the **original 12-hour demo model** (safetensors
weights, `config_forecast.yaml`, 3.5-day cyclic forcing) — not the monthly model.
Kept for the demo; the monthly analogue is `freerun_monthly.py`.

```bash
python experiments/drift_rollout.py --n-steps 120                 # ~60 simulated days
python experiments/drift_rollout.py --plot-only --out-dir data/drift_run
```
| Flag | Default | Meaning |
|---|---|---|
| `--n-steps` | `120` | number of 12h steps (~60 days) |
| `--n-ens` | `1` | ensemble members |
| `--forcing` | `cyclic` | extend the 3.5-day forcing: `cyclic` or `hold` |
| `--device` | `auto` | `auto`/`mps`/`cpu`/`cuda:0` |
| `--seed` | `42` | RNG seed |
| `--snapshot-every` | `10` | dump full state every N steps |
| `--out-dir` | `data/drift_run` | output dir (CSV, snapshots, figure) |
| `--model-path` | `data/models/model_weights_ema.safetensors` | demo EMA weights |
| `--aux-path` / `--demo-path` | demo files | grid + 3.5-day demo data |
| `--plot-only` | off | only redraw the figure from an existing out-dir |

---

# Shared concepts

### Checkpoints
- **`data/models/monthly/monthly.ckpt`** — the canonical trained model. All eval
  scripts default to it (falling back to the newest `last*.ckpt`).
- Lightning writes `last.ckpt`, `last-v1.ckpt`, … as it rotates; these are stale
  once training ends. To resume training, pass `ckpt_path=…/monthly.ckpt` to
  `train.py`.
- Checkpoints load the **EMA weights** (the inference-quality copy) via the
  `ema_model.module.*` prefix; architecture comes from `--train-config`.

### Ensemble mean (`--n-ens`)
GenSIM is **generative** — it samples a plausible next state with fresh noise each
call. Running it `n_ens` times and **averaging** cancels per-sample noise and
yields the minimum-RMSE estimate. Empirically `--n-ens 8` beats a single sample by
~10 skill points. Cost scales with `n_ens`; drop to 4 to halve it.

### The `--fast` sampler
Switches the flow-matching sampler from second-order (2 model calls/substep) × 20
substeps to **first-order × 12** — ~3× fewer network evals. Fine for skill curves
and diagnostics; keep it **off** when rendering art frames, where sampling quality
matters.

### Performance (Apple Silicon / MPS, measured)
Per-step cost scales **linearly with `--n-ens`** — the ensemble is run through the
model once per member and MPS doesn't parallelize the batch. Measured with
`--fast`: **~5.7 s/step at `n_ens=1`, ~44 s/step at `n_ens=8`.** For a long
`freerun` (960 steps = 80 yr) that's ~90 min at `n_ens=1`, ~6 h at `n_ens=4`,
~12 h at `n_ens=8`. Ensemble mean improves one-step skill (~10 points), so keep
`n_ens=8` for `eval`/`rollout`; drop to 1–4 for long projections where the
seasonal cycle and trend matter more than per-frame noise. The first step of any
run includes a one-off ~100 s MPS kernel compilation — ignore it in the average.
Snapshot frequency and forcing source (`cyclic`/`cmip`) do **not** affect step
cost (both are sub-second I/O).

### Skill and baselines
`skill = 1 − RMSE_model / RMSE_baseline`, per variable, over ocean cells (>0 beats
the baseline). Two baselines:
- **Persistence** — last state held constant. Hard to beat at 1-month lead, weak
  at long lead.
- **Climatology** — the average value for that calendar month. The honest
  long-lead bar (beating it means tracking the specific year, not just the
  seasonal cycle).
RMSE is never averaged across variables (different units); always reported
per-variable.

### Forcing choices in `freerun`
- **cyclic** — repeats the 48 real 2015–2018 months forever. Stationary climate,
  no trend → tests self-consistency/stability only.
- **cmip** — the bias-corrected scenario forcing → carries a warming trend, so it
  can show ice decline over decades. One model / member / scenario = one plausible
  trajectory, not an ensemble projection.

### Variables
| State | Meaning | Unit |
|---|---|---|
| `sit` | sea-ice thickness | m |
| `sic` | sea-ice concentration | fraction (0–1) |
| `sid` | sea-ice damage | dimensionless (0–1) |
| `siu`,`siv` | ice velocity (grid x, y) | m/s |
| `snt` | snow thickness | m |

| Forcing | Meaning | Unit |
|---|---|---|
| `tus` | near-surface air temperature | K |
| `huss` | near-surface specific humidity | kg/kg |
| `uas`,`vas` | near-surface wind (grid frame) | m/s |
| `pdd_month`,`fdd_month` | positive / freezing degree-day rate, current month | from 2 m temp |
| `pdd_year`,`fdd_year` | positive / freezing degree-days, trailing 12-month mean | from 2 m temp |

---

# End-to-end, from scratch

```bash
conda activate gensim

# 1. data
python download/nextsim.py
python download/era5.py                       # after setting request dict + ~/.cdsapirc
python download/cmip.py                        # optional: future scenario

# 2. preprocess
python experiments/preprocess_monthly.py
python experiments/estimate_normalization_monthly.py   # normalization stats -> configs
python experiments/build_zarr_monthly.py               # pack train/validation zarr
python experiments/preprocess_cmip.py          # optional: bias-corrected future forcing

# 3. train (Apple Silicon)
python train.py --config-name config_train_monthly_mac

# 4. evaluate
python experiments/eval_monthly.py --n-ens 8
python experiments/rollout_monthly.py --fast

# 5. project
python experiments/freerun_monthly.py --years 15 --fast                  # stability
python experiments/freerun_monthly.py --forcing cmip --years 80 --fast --snapshot-every 1   # scenario
```
