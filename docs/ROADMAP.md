# Roadmap

Where Frazil is going, version by version: what each release adds, the **performance targets** a model must meet to be released, the **guardrails** it must not break, and how we know a milestone is done. How version numbers work and how bugs are fixed is in [CONTRIBUTING.md](../CONTRIBUTING.md); individual problems and ideas are in [ISSUES.md](ISSUES.md), each tagged with the milestone that addresses it.

Before 1.0 the order of milestones can change. Change this file in the same branch as the decision, and when a target is revised, write down why. Targets are never loosened after seeing a model's results without saying so here.

## Principles

- **Measure on years the model never saw, and keep some unseen.** Model development is judged on the out-of-sample hindcast for **2019–2022**. The years **2023–2025** are a holdout, scored only for release candidates, so repeated tuning can't quietly fit them.
- **Score per member, never the averaged field's extent.** Averaging members smears the ice edge (see [ensemble_averaging](../experiments/2026-09-25_ensemble_averaging/)).
- **A gain must survive the leak test.** Anything that comes from a new input must hold up when that input is replaced by its climatology (`--climatology-forcing`), or it was read from the observations rather than learned.
- **Every release names one recommended model,** with its scorecard and the SHA-256 of its checkpoint (checkpoints aren't in git).

## The scorecard

A fixed protocol, so numbers compare across versions. It becomes a tool in v0.3.0 (`scripts/evaluate/scorecard.py`, writing `results/scorecards/<model>.json` and a ledger entry); until then the numbers below come from the hand-run scoring recorded in [experiments/](../experiments/README.md).

1. **Forecast skill, validation years 2015–2018:** one-month skill against persistence and climatology for each of the six variables (`eval_monthly.py`, 8 members); multi-month skill against climatology at leads of 1, 3, 6 and 12 months (`rollout_monthly.py`, fixed start months).
2. **Out-of-sample hindcast:** a free-run from the December 2018 state under real ERA5 weather to 2025, 8 members saved, scored per member against NSIDC-0051: September, June, July and August extent bias; ice-edge error (IIEE) for September, July–September and December–March; the year-to-year correlation of September extent; drift growth (September bias in the last three years minus the first three); September bias in the East Siberian, Beaufort, Chukchi and Laptev seas. Reported for the development years (2019–22) and, for release candidates, the holdout (2023–25).
3. **Stability:** the 2015–2100 CMIP free-run finishes with no non-finite values.
4. **Cost:** seconds per step at 1 and 8 members.

If the protocol changes, re-score the previous release's model with the new protocol so the comparison stays like-for-like, and give the scorecard a new version number.

## Where we are: v0.2.0, the restructure

The code is reorganised and tested; model behaviour is unchanged from v0.1.0. Recommended model: **`monthly_pf_qm`** (pushforward on quantile-mapped targets).

Baseline, `monthly_pf_qm`, out-of-sample hindcast 2019–25 (per member, M km² unless noted):

| measure | baseline |
| --- | --- |
| September extent bias | +0.37 |
| June / July / August extent bias | +0.52 / +0.28 / +0.23 |
| ice-edge error: September / July–September / December–March | 1.53 / 1.83 / 0.93 |
| September year-to-year correlation | 0.86 |
| drift growth (2023–25 minus 2019–21 September bias) | +0.34 |
| September bias: East Siberian / Beaufort / Chukchi / Laptev | +0.20 / +0.21 / +0.16 / −0.11 |
| stability: 2015–2100 CMIP free-run | no non-finite values |
| cost: 1 / 8 members, fast sampler, Apple Silicon | ~5.5 / ~46 s per step |

Not yet measured for this model: the forecast-skill part of the scorecard. The base model beat persistence by 40–60% and climatology by 40–50% one month ahead (2015–18); `monthly_pf_qm`'s own numbers come with v0.3.0. These baselines span 2019–25; v0.3.0 re-reports them split into development and holdout years.

## v0.3.0 — Measurable and reproducible

**Why:** targets need a scorecard anyone can re-run, and a fresh clone must be able to rebuild every dataset.

**Features**
- The scorecard tool, and the full v0.2 baseline scorecard of `monthly_pf_qm`, including its forecast skill and the development/holdout split.
- The data audit (ISS-017): every file in `data/` produced by a script with its default in `frazil/paths.py`; no leftover zips, unused hemispheres or stray outputs; the SASIP data licence checked (ISS-010).
- The report tells runs of the same model apart (ISS-001).

**Performance:** no model change. Guardrail: the scorecard reproduces the baseline table within sampling noise (±0.03 M km² on extent measures).

**Done when:** the scorecard JSON for `monthly_pf_qm` is committed and the README's numbers cite it; every download script runs in `--dry-run`/`--list` mode from a clean clone; fast and slow tests pass.

## v0.4.0 — First artwork release

**Why:** the art–science deliverable, from the release model, reproducibly.

**Features**
- Final 2015–2100 animations from the recommended model: concentration as PMM and as a single member, and thickness (ISS-011, re-run with `--save-members sic sit`; 4 members are enough, per the [ensemble-size experiment](../experiments/2026-09-28_ensemble_size/)).
- The credit line carried with every rendered file ([CREDITS.md](../CREDITS.md)).
- [VISUALS.md](VISUALS.md) updated with the exact commands; fixed colour scales across all frames and runs.

**Performance:** release model unchanged. Guardrails: the projection run has no non-finite values; re-rendering from the saved snapshots reproduces the frames.

**Done when:** the rendered files and the commands that made them are recorded in an `experiments/` entry, with the checkpoint's SHA-256.

## v0.5.0 — Data for live visuals (p5.js, TouchDesigner)

**Why:** the predictions as playable material for creative-coding tools, not only as rendered video: the fields themselves and their motion, in formats those tools read directly.

**What's already there:** every monthly snapshot holds all six predicted fields, including the ice drift velocity (`siu`, `siv`, in m/s along the grid's x and y axes, so it maps straight onto screen space). On the 512×512 grid (cells ~12 km), the median drift is about 3–4 cells a month and the fastest tenth about 17. The animation shows mostly the *concentration* changing; the drift is in the data but not yet shown.

**Features** (a new `scripts/export/` stage, ISS-020 and ISS-021)
- **Fields per month** from any free-run, as the mean, the PMM or one member: concentration, thickness, damage, snow and drift velocity, plus the land mask and grid coordinates. Formats:
  - 32-bit float **EXR** image sequences for TouchDesigner: one channel per variable, exact values, usable directly as GPU textures (for example, velocity driving a particle system);
  - **PNG** sequences with a stated value range per channel for p5.js `loadImage`, and exact **Float32 binary** files for p5.js `loadBytes`;
  - a **JSON manifest** listing frames, dates, variables, units, scaling and grid size, so a sketch can read any export without hard-coding;
  - optional downsampling (for example to 128×128) for lightweight sketches.
- **Motion:**
  - **drift particles:** carried by the monthly velocity fields with time interpolation for smooth sub-monthly steps, born where ice forms and removed where it melts; exported as trajectories (id, time, x, y) in JSON/CSV and as EXR position maps for GPU particles;
  - **ice-edge contours:** the 15% edge as polylines per month in normalised screen coordinates, for line drawing and morphing between months;
  - **growth and melt:** the month-to-month change in concentration and thickness, showing where ice forms and disappears (as distinct from where it drifts);
  - **edge certainty:** the share of members with ice at each cell, showing where the future edge is uncertain;
  - **smooth in-between frames:** each month's field carried along the drift toward the next, rather than cross-faded.
- **Per-member velocity:** `freerun_monthly.py --save-members sic sit siu siv`, so the motion isn't damped by averaging (averaging members partly cancels their velocities).
- A minimal **p5.js example sketch** (`examples/p5/`) and a TouchDesigner walkthrough in [VISUALS.md](VISUALS.md).

**Targets**
- Exact round trip: values read back from EXR and binary exports equal the model's to float32 precision; PNG exports are within one step of their stated scaling.
- Particles: a month's displacement matches the integrated velocity within 1%; no particle ends up on land.
- Contours: the area inside each month's edge matches the computed extent within 1%.
- Exporting the full 2015–2100 run (1,020 months) takes under 15 minutes on the laptop.

**Guardrails:** exporting never changes model outputs; every exporter has fast tests on synthetic fields.

**Done when:** a p5.js sketch and a TouchDesigner network each play the 2015–2100 export (fields, particles and edge), documented end to end.

## v0.6.0 — Summer melt on the Pacific shelf

**Why:** the model's own largest error: too little summer melt in the East Siberian and Beaufort seas, growing over a run (ISS-002).

**Work**
- Finish the radiation attribution (ISS-004).
- Make new inputs learnable: a fine-tune protocol whose inference weights actually follow training (ISS-005), and a check that new channels reach meaningful weight (ISS-003).
- Degree-day features that see thaw days inside a month (ISS-007).
- A leak-free ocean-memory input: sunlight absorbed by the model's *own* open water, accumulated over the summer.

**Targets** (out-of-sample, per member, development years 2019–22; confirmed on the 2023–25 holdout for the release candidate):

| measure | v0.2 baseline | target |
| --- | --- | --- |
| September extent bias | +0.37 | ≤ +0.20 |
| June extent bias | +0.52 | ≤ +0.30 |
| July–September ice-edge error | 1.83 | ≤ 1.65 |
| East Siberian / Beaufort September bias | +0.20 / +0.21 | ≤ +0.10 each |
| drift growth | +0.34 | ≤ +0.15 |

**Guardrails:** September correlation ≥ 0.85; December–March ice-edge error ≤ 0.95; one-month skill within 2 points of the v0.3 scorecard on every variable; the 2015–2100 CMIP run stays finite; gains survive the leak test.

**Done when:** a model meets the targets and guardrails and becomes the recommended model. If none does, v0.6.0 still ships the tooling and the write-ups, and the targets carry forward, stated as unmet.

## v0.7.0 — Projections with a spread

**Why:** one climate model gives one possible future; a few give a range.

**Features**
- Forcing from at least three CMIP6 models (ISS-015), with a bias correction that holds beyond its baseline years (ISS-009).
- Radiation or SST forcing for CMIP runs, if the recommended model uses them (ISS-008).
- Projection plots and animations that show the spread.

**Targets:** for each CMIP model, the 2015–2024 September level of the free-run within ±0.3 M km² of the same model's ERA5 hindcast (forcing bias removed); the spread reported in every projection output.

## v1.0.0 — Stable

1.0 means someone else can rely on it:
- A clean clone rebuilds every dataset with the documented, scripted downloads.
- The recommended model meets the v0.6.0 targets, or a documented waiver says why not.
- The scorecard is published with the release.
- The command-line tools, configs and `frazil` package are stable: from here on, breaking changes need a new major version.
- Fast and slow tests pass.

## Later

Parked until a milestone needs them: daily-data retraining (ISS-014), a harder quantile-mapping fine-tune (ISS-013), offering GenSIM's bug fixes upstream (ISS-012), running tests on GitHub Actions (ISS-019).
