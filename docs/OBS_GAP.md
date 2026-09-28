# Observation gap — measured decomposition of why the model sits above NSIDC

Written 2026-09-22. Updated the same day after running the ERA5-forced hindcast, which **overturned two of the three hypotheses in the first draft of this doc**. Keep this history: the pre-experiment guesses (forcing and metric dominate) were wrong, and the hindcast is why we know. The lesson stands — settle the measurement with an experiment before tuning the model.

## Update 2026-09-25 — out-of-sample hindcast, averaging artifact, and where the error lives

This supersedes the "drift ~+0.5, fix with pushforward + QM" conclusion below. Evidence comes from the first truly out-of-sample test: `monthly_pf` run from the Dec 2018 state under real ERA5 weather to 2025 (`freerun_monthly.py --forcing era5 --start 2018-12 --years 7 --save-members`), scored against NSIDC-0051.

- **It gets the year-to-year variations right** (September correlation r = 0.82 over 2019–25): 2020 low, 2021–22 high. The physics responds correctly to weather it never saw.
- **Thickness build-up is ruled out.** In the 2013–17 hindcast the model has 2–5% *less* volume than neXtSIM, with no growth, and thinner September ice. In 2019–25 its volume holds at neXtSIM's 2017–18 level.
- **Averaging the ensemble inflates extent, but it's the smaller part.** Snapshots store the mean of 8 members, and averaging smears the edge where members disagree. In September the extent of the averaged field is +0.66 vs obs; the average of each member's own extent is **+0.47**. The artifact is ~+0.19 and roughly constant. The growth over time (+0.27 → +0.63, 2019–21 → 2023–25) is in the members themselves: it is real drift.
- **The error is misplaced ice, not just too much ice.** September ice-edge error (IIEE) is ~1.9 M km² against a net bias of ~0.5: ~1.1 M km² of model ice where obs has none, and ~0.6 M km² of obs ice the model lacks.
- **Where, and whose.** September net bias vs obs by sector, with its source:

| sector | emulator 2019–25 (per member) | neXtSIM teacher 2013–18 | emulator vs teacher, same years 2013–17 | source |
| --- | --- | --- | --- | --- |
| East Siberian | +0.25 | −0.05 | +0.30, growing every year (+0.19 → +0.61) | **emulator's own, growing** |
| Beaufort | +0.22, growing +0.08/yr | +0.06 | +0.10 (mostly 2016) | **mostly the emulator's own** |
| Chukchi | +0.19 | +0.19 | +0.09 | inherited from neXtSIM (QM fixes: +0.19 → +0.06) |
| Laptev | −0.11 | −0.10 | +0.11 | inherited (QM fixes: → +0.04) |
| CAA/Baffin | −0.11 | −0.04 | −0.02 | small |
| Barents/Kara, Greenland/Fram | ~0 | +0.19 / +0.04 | ~0 | fine |

**What this means.** The emulator's own, growing error sits on the Pacific-side shelf seas (East Siberian, Beaufort), where it melts too little in summer. The **leading hypothesis is missing ocean memory**: neXtSIM was coupled to an ocean model, but the emulator sees only the atmosphere, so it cannot carry forward the ocean heat (summer solar absorption in open water, Pacific inflow) that drives melt there. Not proven. The degree-day assumption (melt forcing computed from monthly-mean temperature) is now less likely to be primary, because it would act across all sectors, and Barents/Kara is fine. QM remains a valid fix for the *inherited* Chukchi/Laptev part; pushforward does not address either part.

**Candidate model-level fix:** give the emulator an ocean signal. ERA5 has monthly sea-surface temperature for 1994–2025, so the same download path works. Add it as an input channel, with its weights initialised to zero so the model starts identical to today's, then fine-tune. Test with the same out-of-sample hindcast and sector table.

**Ensemble summaries (for the animation).** `experiments/ensemble_summaries.py` compares ways to turn 8 members into one field (84 months, 2019–25). July–September, vs obs:

| method | extent bias | IIEE (edge error) | partial-ice (15–80%) band bias | RMSE | sharpness (obs = 1) |
| --- | --- | --- | --- | --- | --- |
| mean | +0.69 | 1.92 | +0.78 | **0.173** | 0.44 |
| single member | +0.51 | 1.94 | +0.38 | 0.181 | **0.56** |
| median | +0.53 | 1.89 | +0.48 | 0.176 | 0.47 |
| consensus (mean inside ≥50% ice area) | +0.59 | 1.90 | +0.68 | 0.176 | 0.49 |
| **PMM** (probability-matched mean) | +0.51 | **1.87** | +0.41 | 0.176 | 0.46 |
| LPMM (local PMM) | **+0.49** | 1.89 | +0.41 | 0.177 | 0.46 (visible tile seams) |

PMM keeps the averaging's benefits (best edge placement, near-mean RMSE) and removes the averaging's extent inflation and the widened partial-ice band. It does not restore small-scale texture; a single member does. For the art, that means PMM for faithful frames and a single member for textured, physically consistent frames. No summary method fixes the underlying placement error (every IIEE is ~1.9) — that is the model.

## Update 2026-09-27 — SST channel A/B (result: no effect; the channel was barely learned)

`monthly_pf_qm_sst` (the `monthly_pf_qm` recipe + ERA5 SST as a 5th forcing, zero-initialised, 40k steps) vs its A/B partner `monthly_pf_qm`, plus a leak test (the same model with SST replaced by its 1995–2014 monthly climatology). 2019–25 hindcast, per member, vs NSIDC-0051:

| model | Sep bias | Sep IIEE | Jul–Sep IIEE | Sep r | E. Siberian | Beaufort |
| --- | --- | --- | --- | --- | --- | --- |
| `monthly_pf` | 0.47 | 1.66 | 1.99 | 0.83 | 0.25 | 0.22 |
| `monthly_pf_qm` (A) | **0.37** | 1.53 | 1.83 | 0.86 | 0.20 | 0.21 |
| `monthly_pf_qm_sst`, real SST (B) | 0.42 | 1.52 | 1.80 | 0.86 | 0.21 | 0.21 |
| B, climatological SST | 0.43 | 1.53 | 1.83 | 0.86 | 0.22 | 0.21 |

- **SST changed nothing** (all differences ≤0.05, within noise), and removing its year-specific values changed nothing either. The model barely uses the channel: SST's input-weight norm is 7% (t) and 17% (t+1) of the median other forcing in the training weights, and only 2% / 5% in the EMA weights used for inference. So this is "not learned", not "no value". Likely causes: a gentle fine-tune (lr 1e-4, EMA 0.9999 lagging) and/or SST being largely redundant, since ERA5 air temperature already carries the observed-ice imprint. The fastest-growing channel was SST at t+1, the leaky one.
- **QM targets + pushforward (`monthly_pf_qm`) is the best model out-of-sample.** Against `monthly_pf`: Sept bias 0.47 → 0.37, July 0.60 → 0.28, August 0.45 → 0.23, summer IIEE 1.99 → 1.83, East Siberian 0.25 → 0.20, Chukchi 0.19 → 0.16. This is a clean test: QM was fitted on 1995–2018 and scored on 2019–25.
- **Unchanged:** the emulator's own Pacific-shelf excess (East Siberian, Beaufort ~0.2 each), still growing over the run.
- **Next, if pursued:** radiation (`ssrd`, `strd` — already in the `_era5x` cubes), not SST. Use a recipe that lets new inputs learn: a higher learning rate on the input layer, a faster EMA, and a check of the new channels' weight norms at ~10k steps before committing to a full run.

## TL;DR (2026-09-22 — superseded in part by the update above)

The recent-year September gap between the model free-run and NSIDC is ~+0.8 M km² on a consistent grid, and it decomposes as: **drift ~+0.5 (dominant), target bias ~+0.15, CMIP forcing ~+0.13, metric ~+0.11**. The dominant, fixable term is **autoregressive free-run drift** — the model over-ices relative to its own training truth the longer it rolls out, even under perfect forcing. QM/fine-tune was aimed at the *target bias*, the second-smallest term. For the decadal animation (a long free-run) drift is the whole game — but pushforward *alone* can't fix it: it stabilizes toward whatever targets it trains on, so on the icy original targets it just locks in the icy level (measured — see below). The lever is **pushforward on the de-iced QM targets, together** (`monthly_pf_qm`).

## The measured decomposition (the point of this doc)

All September, model grid, M km², model = `monthly_qm` throughout (so the only thing that changes between the two free-run columns is the forcing). ERA5 free-run and CMIP free-run both start from 2013 truth and roll forward autoregressively.

| year | ERA5 free-run | CMIP free-run | neXtSIM truth | NSIDC gridded | NSIDC official |
| --- | --- | --- | --- | --- | --- |
| 2015 | 5.68 | 5.73 | 5.58 | 4.67 | 4.62 |
| 2016 | 5.23 | 4.73 | 4.53 | 4.72 | 4.53 |
| 2017 | 5.35 | 6.20 | 4.62 | 4.89 | 4.82 |
| **2015–17 mean** | **5.42** | **5.55** | **4.91** | **4.76** | **4.66** |

Decomposition of the CMIP free-run's ~+0.8 M excess over gridded obs (2015–17):

| term | size | what it is | lever |
| --- | --- | --- | --- |
| free-run drift | **+0.51** | ERA5 free-run (5.42) − neXtSIM truth (4.91): the model over-ices vs its own truth as it rolls out, with *correct* forcing | **pushforward training** |
| target bias | +0.15 | neXtSIM truth (4.91) − gridded obs (4.76): neXtSIM-OPA is modestly too icy | QM targets (built; barely imprinted) |
| CMIP forcing | +0.13 | CMIP free-run (5.55) − ERA5 free-run (5.42): a small mean cold bias | bias-correct CMIP vs ERA5 |
| metric | +0.11 | gridded-on-model-grid (4.76) − official Sea Ice Index (4.66), identical years | none needed — it's small |

Drift dominates. Everything else is a rounding error by comparison, and drift grows with rollout length, so over a 15–30 year animation it is even more the story.

## What the hindcast overturned

- **"CMIP forcing is the big problem" — wrong.** The CMIP free-run mean is only +0.13 above the ERA5 hindcast. The *stark year-by-year* CMIP swings (4.73 → 6.20) are wrong-weather scatter, not a large bias: CMIP's "2016" is not the real 2016, so individual years scatter around a mean that is barely offset. So never compare the CMIP free-run to specific observed years — only to the trend/distribution. The systematic forcing bias is real but small (~0.13, worth a proper bias-correction, not a research project).
- **"The extent metric is ~1 M inflated on the model grid" — wrong.** On *identical* years, gridded-on-model-grid vs the official Sea Ice Index differ by only +0.11 M (pole-hole fill + regridding). The earlier "~1 M" was an artifact of comparing the gridded 1995–2018 mean (5.60, inflated by icy 1990s) to the CSV's recent years (~4.6). Different year sets. The metric is fine; gridded and official agree to ~0.1 M on matched years.
- **"Target bias ~0.37 M" — half-right, and it shrinks recently.** It is ~0.37 in the 1995–2018 mean but only ~0.15 in recent low-ice years — and it is the term QM addresses, i.e. the second-smallest.

## The lever is drift + de-icing TOGETHER, not pushforward alone

The model reproduces obs well at one step (one-step Aug→Sep ≈ 5.03 ≈ truth, near-zero drift) and drifts up over a free rollout. Pushforward training corrects that drift — but a subtlety I got wrong at first: **pushforward corrects drift *toward the training distribution*.** If those targets are the icy original neXtSIM, more pushforward just locks the free-run onto the icy attractor more firmly; it stabilizes, it does not de-ice. Pushforward alone can never pull the level below the targets it trains on.

The evidence (2026-09-23): a harder pushforward run (55k steps, prob 0.8, substeps 10, on the *original* icy targets) made the ERA5 hindcast September *icier*, not lighter — 5.69 M km² (+0.71 vs obs), above the `monthly_qm` run (5.48, +0.49). That pf-vs-qm comparison is confounded — `monthly_pf` trains on icy targets, `monthly_qm` on de-iced QM targets, so they differ in two ways — but it kills the "pushforward alone shrinks the gap" idea: pushforward stabilized toward the icy targets exactly as expected.

So the only run that can actually land near obs combines both: **de-iced QM targets (lower level) + pushforward (stable at that lower level).** Launched 2026-09-23 as `monthly_pf_qm` (`init_from` = the 55k `monthly_pf`, `data.suffix` = _sicqm, `pushforward` true, 40k steps). The decisive number is its `--forcing era5` hindcast September extent vs obs — that is what closes the ~+0.5 drift and the ~+0.15 target bias at once, or shows they trade off. For the animation, use whichever of `monthly_pf` / `monthly_pf_qm` has the smaller ERA5-hindcast gap; drift is what reads visually over a decade.

## Results log (ERA5 hindcast September extent, 2013–17 mean, model grid, M km²)

| model | targets | pushforward | free-run | vs obs (4.99) | drift vs own truth |
| --- | --- | --- | --- | --- | --- |
| `monthly_qm` | de-iced | none | 5.48 | +0.49 | — |
| `monthly_pf` (55k) | icy original | 55k, prob 0.8 | 5.69 | +0.71 | +0.50 |
| `monthly_pf_qm` | de-iced | 40k on top of pf | *running* | *TBD* | *TBD* |

## PARKED calibration ideas (demoted — drift comes first)

Only worth doing after a stronger pushforward run, and only if a residual target bias still shows in the ERA5 hindcast:

- **A — harder QM fine-tune.** The 20k-step, lr 1e-4, EMA 0.9999 run barely imprinted (one-step Sept 5.06 → 5.03 vs a ~4.95 target; `val/loss` flat). Retry lr 5e-4, EMA 0.999, 40–50k steps, seeded from `monthly_pf`, then re-check one-step Sept vs the ~4.9 target. Binding constraint is the imprint, not the data.
- **B — from scratch on the QM targets.** Cleanest imprint, ~250k steps. Reserve for if A won't move.
- **C — trend-aware QM.** The current per-cell whole-period CDF match undershoots in low-ice years (moved 2015–18 targets only 5.07 → 4.94 vs obs 4.78). A recent-period-weighted fit would close that — but it's chasing a ~0.15 term.

### Prereqs already built

- `data/train_data/monthly_datacube_sicqm/`, `train_sicqm.zarr`, `validation_sicqm.zarr` — QM targets, training-ready.
- `configs/experiment/sicqm.yaml` — the fine-tune config (edit lr/EMA/steps for A).
- `data/models/monthly_qm/last.ckpt` — the weak fine-tune, baseline to beat.
- `experiments/freerun_monthly.py --forcing era5` — the ERA5 hindcast path (added 2026-09-22); re-run it against any new model to re-measure the drift term.
- `experiments/correct_sic_targets_qm.py` and `archive/qm_calibrate_poc.py` — the correction and its proof-of-concept.

## What not to do

Do not tune QM/fine-tune knobs to close a ~0.15 M target sliver while a ~0.5 M drift term sits untouched. And never read year-by-year agreement between the CMIP free-run and observed years as model skill — that comparison is category-mismatched (wrong weather). Fix drift first (pushforward), compare projections on trend not on individual years, and the September level lands within the noise.
