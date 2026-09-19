# Images & animations

How to produce every visual output in the project — spatial maps, time-series
plots, and the decadal sea-ice animations — from the model's data. Written for the
art–science use: getting publishable frames and movies of the ice out of the
`.nc` snapshots and result CSVs.

All commands run from the repo root with the `gensim` env active. Figures land in
`plots/` (or wherever `--out*` points).

---

## What produces what

| Output | Tool | Kind |
|---|---|---|
| Forecast maps (truth \| GenSIM \| persistence) | `predict_monthly.py` | spatial, validation |
| Final-state map of a run | `freerun_monthly.py` (`_map.png`) | spatial |
| Map of **any** saved month | `state_map()` (snippet below) | spatial |
| Ice extent / area over time (+ obs) | `ice_area.py` | time series |
| Skill vs lead | `rollout_monthly.py` | time series |
| Drift diagnostics (area, sharpness, bounds) | `freerun_monthly.py` (`.png`) | time series |
| Skill vs training step | `sweep_skill_monthly.py` | time series |
| **Animation (GIF/MP4)** | `animate_snapshots.py` | movie |
| Spatial state files the maps/movies read | `freerun_monthly.py --snapshot-every` | data (`.nc`) |

---

## 1 · Spatial maps (one moment in time)

### Forecast maps — `predict_monthly.py`
Predicts one validation month and draws, per variable, a row of **truth \| GenSIM \|
persistence**. The visual "does the forecast look like reality" check (validation
window only — it needs truth).

```bash
python experiments/predict_monthly.py --target-year 2018 --target-month 9 --n-ens 8
# -> plots/forecast_maps.png
```

### Final-state map of a run — `freerun_monthly.py`
Every `freerun` writes a 6-panel map (`sit sic sid siu siv snt`) of its **last**
month, land masked:

```bash
python experiments/freerun_monthly.py --forcing cmip --years 80 --fast
# -> plots/freerun/freerun_cmip_map.png  (state at 2095-01)
```

### Map of any saved month — reuse `state_map()`
Snapshots (`state_YYYYMM.nc`) hold the full 6-variable field, so you can draw any
month later without re-running the model:

```python
import xarray as xr, sys; sys.path.insert(0, "experiments")
from freerun_monthly import state_map
ocean = xr.open_dataset("data/auxiliary/ds_auxiliary.nc")["mask"].values.astype(bool)
sm = xr.open_dataset("plots/freerun/snapshots/state_205009.nc")["state"].isel(time=0).values
state_map(sm, ocean, "plots/ice_2050-09.png", "GenSIM sea ice 2050-09")
```

### Colormaps & variables
Maps use these per-variable colormaps (from `predict_monthly.CMAP`), land shown as
NaN (transparent/grey), and a robust 2nd–98th-percentile colour range (symmetric
for the diverging velocity fields):

| Variable | Meaning | Unit | Colormap |
|---|---|---|---|
| `sit` | ice thickness | m | `viridis` |
| `sic` | ice concentration | 0–1 | `Blues_r` |
| `sid` | ice damage | 0–1 | `magma` |
| `siu`,`siv` | ice velocity (x, y) | m/s | `RdBu_r` (diverging) |
| `snt` | snow thickness | m | `viridis` |

To restyle for art, edit the `CMAP` dict in `predict_monthly.py` /
`animate_snapshots.py` (e.g. swap in `cmocean` maps — `cmocean.cm.ice` for `sic`,
`cmocean.cm.dense` for thickness; `import cmocean` and use `cmocean.cm.ice`).

---

## 2 · Time-series plots

### Ice extent / area — `ice_area.py`
The headline "shrinking ice" line chart. Overlays multiple sources and NSIDC
observations, computes model−obs offsets.

```bash
# truth vs a projection, September minimum, against observations
python experiments/ice_area.py data/train_data/monthly_datacube plots/freerun/snapshots \
    --month 9 --nsidc data/obs/nsidc --out plots/ice_area_sept.png
```
`--metric extent` (SIC>0.15, the standard) or `--metric area` (SIC-weighted);
`--csv` also dumps the numbers.

### Skill vs lead — `rollout_monthly.py`
Two panels: RMSE vs lead, and skill vs climatology (solid) & persistence (dashed).

```bash
python experiments/rollout_monthly.py --fast --ckpt data/models/monthly_pf/last.ckpt \
    --out-png plots/rollout_pf.png
```

### Drift diagnostics — `freerun_monthly.py`
The 4-panel self-consistency figure (domain-mean state, ice area, sharpness/
spectral-collapse, physical bounds) — how a free run behaves over years:

```bash
python experiments/freerun_monthly.py --years 15 --fast   # -> plots/freerun/freerun_cyclic.png
```

### Skill vs training step — `sweep_skill_monthly.py`
How skill improved during training (cached to CSV):
```bash
python experiments/sweep_skill_monthly.py   # -> plots/skill_vs_step.png
```

---

## 3 · Animations — `animate_snapshots.py`

Animations are built from **snapshots**, so the workflow is always two steps:
**(a) save snapshots during a run, (b) animate them.** No model is re-run for the
animation — it's pure plotting from the `.nc` files, so it's fast and endlessly
re-styleable.

### Step (a): save the frames
Snapshots come from any `freerun`. For a smooth movie you want **every month**:

```bash
python experiments/freerun_monthly.py --forcing cmip --years 80 --fast --snapshot-every 1 \
    --ckpt data/models/monthly_pf/last.ckpt
# -> plots/freerun/snapshots/monthly_pf_cmip/state_YYYYMM.nc   (960 files ~27 GB; see note)
```
Snapshots are **namespaced by model + forcing** (`snapshots/<model>_<forcing>/`), so
different checkpoints/forcings never overwrite each other's frames — the run prints
the exact folder (`[freerun] snapshots -> …`). `--snapshot-every 1` saves every
month (needed for a monthly animation);
`--snapshot-every 12` saves yearly (each January). **Disk:** monthly over 80 yr is
~27 GB — make sure there's room, or animate a shorter span / the September-only
frames (below).

### Step (b): make the movie
```bash
# September ice minimum, one frame per year (the shrinking-cap movie)
python experiments/animate_snapshots.py --snap-dir plots/freerun/snapshots/monthly_pf_cmip \
    --var sic --month 9 --out plots/ice_sept.mp4 --fps 8

# every month: seasonal pulse + long-term decline together
python experiments/animate_snapshots.py --snap-dir plots/freerun/snapshots/monthly_pf_cmip \
    --var sic --out plots/ice_monthly.mp4 --fps 12
```
Point `--snap-dir` at the namespaced folder the freerun printed.

Key options:

| Flag | Default | Why it matters for a movie |
|---|---|---|
| `--var` | `sic` | which field: `sic` (extent/cap), `sit` (thickness), etc. |
| `--month` | all | keep one calendar month — `9` = September minimum, yearly cadence |
| `--out` | `…/anim_<var>.gif` | `.gif` (needs only pillow) or `.mp4` (needs ffmpeg, better for long/HQ) |
| `--fps` | `6` | frames per second; higher = smoother/faster |
| `--dpi` | `110` | frame resolution; raise (e.g. `200`) for print/HQ |
| `--vmin` / `--vmax` | pooled 2/98 pctile | **fix the colour scale** so decline shows instead of auto-rescaling away |

**The fixed colour scale matters most.** By default the scale is pooled across all
frames (2nd–98th percentile), so a cell that's 90% ice looks the same in 2015 and
2095 — the *retreat* is what moves. If you let each frame auto-scale, the shrinking
becomes invisible. Force an explicit range for concentration with `--vmin 0
--vmax 1`.

### GIF vs MP4
- **GIF** — no install, good for quick previews and web loops; gets large and
  low-quality for long runs (960 frames).
- **MP4** — needs ffmpeg (`conda install -c conda-forge ffmpeg`), stays small and
  crisp, loops cleanly. **Use MP4 for the decadal movie.** Confirm ffmpeg is
  visible: `python -c "import matplotlib.animation as a; print('ffmpeg' in a.writers.list())"`.

---

## 4 · Full recipe: the decadal shrinking-ice movie

End to end, from a trained checkpoint to a September-minimum MP4:

```bash
# 1. project 80 years on bias-corrected CMIP forcing, saving every month
python experiments/freerun_monthly.py --forcing cmip --years 80 --fast --snapshot-every 1 \
    --ckpt data/models/monthly_pf/last.ckpt

# 2. animate the September ice cap, fixed 0-1 scale, high-res MP4
python experiments/animate_snapshots.py --snap-dir plots/freerun/snapshots/monthly_pf_cmip \
    --var sic --month 9 --vmin 0 --vmax 1 --dpi 200 --fps 8 --out plots/ice_cap_2015-2095.mp4
```

Lighter variants (avoid the 27 GB / long run):
- **September only, cheaply:** reuse existing July snapshots →
  `extend_snapshots.py` to step them to September (see CLI.md), then animate with
  `--month 9`.
- **Shorter horizon:** `--years 30`, or `--snapshot-every 3` for a coarser cadence.
- **Lower ensemble** for speed: `--n-ens 1` (noisier frames, fine for a movie).

---

## 5 · Re-plot anytime from saved data

Nothing needs re-running to restyle:
- **Snapshots** (`plots/freerun/snapshots/<model>_<forcing>/state_*.nc`) hold every
  variable's full field — re-draw maps or re-animate with different colormaps,
  scales, fps.
- **Time-series CSVs** (`plots/*_multistart.csv`, `plots/freerun/freerun_*.csv`,
  `ice_area … --csv`) hold the numbers behind the line charts — re-plot in your own
  tool if you want a different style.
- **The results ledger** (`results/experiments.jsonl`) holds the skill/drift
  numbers per run for tables and comparison plots.

So the expensive model runs are done once; all the visual iteration is cheap.
