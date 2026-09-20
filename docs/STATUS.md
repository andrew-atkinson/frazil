# Project status & map

A single place to see the whole process, what works, what's parked, and where the outputs live — so the tweaks stop being a blur. Updated 2026-09-20.

## In one paragraph
Goal: a **small, laptop-trainable model that captures broad monthly Arcticsea-ice evolution** for an art–science piece. We have one — a ~4 M-param monthlymodel (`monthly.ckpt`) trained on neXtSIM targets + ERA5 forcing, plus adrift-corrected fine-tune (`monthly_pf/last.ckpt`). It has **genuine 1–3 monthforecast skill, beats "predict the average month" out to a year on thickness &drift, runs 15 years without blowing up, and declines under a warming scenario.**It is *not* a calibrated climate model, and its training target (neXtSIM) itselfsits ~0.4 M km² above observations.

## The pipeline (the mental image)

```
DOWNLOAD              PREPROCESS                          TRAIN / USE
--------              ----------                          ----------
nextsim.py  (targets) preprocess_monthly.py  \
era5.py     (forcing) estimate_normalization  > monthly_datacube > zarr > train.py > monthly.ckpt
                      build_zarr_monthly.py   /                                        |
                                                                                       v
                                              config_finetune_pushforward  >  monthly_pf/last.ckpt
cmip.py      > preprocess_cmip.py (bias-correct vs ERA5)  > cmip_datacube  ------------.
nsidc.py     > (extent CSV, obs baseline)                                              |
nsidc0051.py > preprocess_nsidc0051.py > nsidc0051_grid  (validation / sic-bias)       |
                                                                                       v
   EVALUATE:  eval_monthly (one-step) · rollout_monthly (vs lead) · sweep_skill (vs step)
   PROJECT:   freerun_monthly (snapshots + drift) · extend_snapshots · ensemble_extent
   VISUALISE: predict_monthly (maps) · animate_snapshots (movies) · ice_area (extent vs obs)
```

Everything a run produces is recorded in **`results/experiments.jsonl`**; see it
tabulated with `python experiments/results_report.py` (→ `results/REPORT.md`).

## What works (proven, with honest numbers)

| Capability                           | Result                                                                               |
|--------------------------------------|--------------------------------------------------------------------------------------|
| One-month forecast                   | beats persistence **+40–60%**, beats climatology **+40–50%** on all 6 vars           |
| Multi-month (to 1 yr)                | beats climatology **+20–30%** on thickness & drift; snow/damage fade to ~climatology |
| 15-yr free run (stationary forcing)  | **stable** — no blow-up, holds the seasonal cycle, no spectral collapse              |
| Future (ssp245, bias-corrected CMIP) | ice **declines** 6.3 → 2.8 M km² Sept by 2075                                        |
| Pushforward fine-tune                | modest, consistent gain at long lead + less drift (see PUSHFORWARD.md)               |

## What's parked / not done (and why)

- **neXtSIM training target is ~0.38 M km² too icy vs NSIDC** (a displaced *ice edge*, not a concentration level). An additive `sic`-target correction was built (`correct_sic_targets.py`, `monthly_datacube_siccorr/`) but **barely moves extent** — a mean shift can't move a sharp edge. Parked; a per-cell **quantile mapping** would be the upgrade if extent-matching matters. **Daily data** (SASIP `1d/`, ~35× more samples) — the biggest untapped lever for a better small model. Parked (≈4 GB download, larger training). **Single CMIP** model/member/scenario (MPI-ESM1-2-LR ssp245) — one trajectory, not an ensemble projection. **Compute:** MPS ~5.7 s/step at n_ens=1, ~44 s/step at n_ens=8 (linear in n_ens). No CUDA. This caps how long/large we train and rollout.
## The two models

- **`data/models/monthly/monthly.ckpt`** — the base (259 k steps). Default for all eval scripts. - **`data/models/monthly_pf/last.ckpt`** — pushforward fine-tune (20 k steps).  Strictly better (drift/long-lead); pass `--ckpt` to use it. Step checkpoints in both dirs are training history (safe to keep or prune the older `step_*`). 

## Open threads / possible next steps
- Fine-tune pushforward **longer / stronger** (resumable) — diminishing returns.
- **Quantile-map** the `sic` targets — the honest fix for the extent bias.
- **Daily-data** retrain — the big data lever.
- **Multi-model CMIP** — turn one trajectory into a spread.
- **The animation** — generate namespaced monthly snapshots and render the shrinking-cap movie (see VISUALS.md).

---

## Outputs & filenames

Where things land, and a **consistent naming scheme** so runs are unique, self-describing, and never overwrite each other.

### Convention
`<kind>_<model>[_<forcing>][_<detail>].<ext>` — lowercase, underscores, and always tag the **model** (and forcing where relevant), because that's what distinguishes runs. `<model>` = `monthly` or `monthly_pf`; `<forcing>` = `cyclic`/`cmip`.

Default output names are now **self-describing** (applied 2026-09-20 — pass an explicit `--out*` to override):

| Output                    | Default                                                     | ✅   |
|---------------------------|-------------------------------------------------------------|-----|
| Rollout skill             | `plots/rollout_<model>.{png,csv}`                           | ✅   |
| Forecast maps             | `plots/forecast_<model>_<YYYY-MM>.png`                      | ✅   |
| Freerun diagnostics / map | `plots/freerun/freerun_<model>_<forcing>[_map].{csv,png}`   | ✅   |
| Snapshots                 | `plots/freerun/snapshots/<model>_<forcing>/state_YYYYMM.nc` | ✅   |
| Animation                 | `plots/anim_<var>_<model_forcing>[_m<MM>].{gif,mp4}`        | ✅   |
| Ensemble-extent           | `plots/ensext_<model>_m<MM>.png`                            | ✅   |
| Ice-area chart            | you name it via `--out` (it overlays *several* models/obs)  | n/a |
| One-step eval             | ledger only (no figure)                                     | n/a |

Common helper: `eval_monthly.model_tag(ckpt)` → `monthly` / `monthly_pf` / `monthly_pf_step_10000`.

### Old ad-hoc files — safe to clean up
`plots/` still holds one-off names from iterative debugging with no provenance (`ice-area-no-model.png`, `sic.mp4`, `sit_m7.mp4`, `ensemble_extent_quick.png`, `rollout_multistart.*`, mixed hyphen/underscore), and `plots/freerun/snapshots/*.nc` (~250 flat files) is the pre-namespacing mix. These are regenerable outputs. **Keep** any `.mp4` you rendered on purpose and any snapshots you still want to animate; delete the rest by hand.
