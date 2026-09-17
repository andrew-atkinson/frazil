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
| `--fast` | off | first-order sampler + 12 substeps (~3.3 s/step; keep OFF for pretty art frames) |
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
