# Roadmap

Where Frazil is going, version by version: what each release adds, the **performance targets** a model must meet to be released, the **guardrails** it must not break, and how we know a milestone is done. How version numbers work and how bugs are fixed is in [CONTRIBUTING.md](../CONTRIBUTING.md); individual problems and ideas are in [ISSUES.md](ISSUES.md), each tagged with the milestone that addresses it.

Before 1.0 the order of milestones can change. Change this file in the same branch as the decision, and when a target is revised, write down why. Targets are never loosened after seeing a model's results without saying so here.

## Principles

- **Measure on years the model never saw, and keep some unseen.** Model development is judged on the out-of-sample hindcast for **2019–2022**. The years **2023–2025** are a holdout, scored only for release candidates, so repeated tuning can't quietly fit them.
- **Score per member, never the averaged field's extent.** Averaging members smears the ice edge (see [ensemble_averaging](../experiments/2026-09-25_ensemble_averaging/)).
- **A gain must survive the leak test.** Anything that comes from a new input must hold up when that input is replaced by its climatology (`--climatology-forcing`), or it was read from the observations rather than learned.
- **Every release names one recommended model,** with its scorecard and the SHA-256 of its checkpoint (checkpoints aren't in git).

## The scorecard

A fixed protocol, so numbers compare across versions. The tool is `scripts/evaluate/scorecard.py` (scorecard version 3: version 1 added 2026-10-15, version 2 the same day adds June ice-edge error, version 3 on 2026-10-22 adds drift from `drift.py`), writing `results/scorecards/<model>.json` and a ledger entry. It reproduces the hand-run v0.2.0 baseline below exactly. Not yet automated: the multi-month rollout skill and cost.

**Noise between training runs** (two runs of the same recipe, 2026-10-20): about 0.01 M km² on September bias and every ice-edge error, 0.05–0.08 on July and August bias, 0.03 on drift growth. A difference smaller than that between two single training runs isn't evidence either way; compare several seeds for a decision that close.

1. **Forecast skill, validation years 2015–2018:** one-month skill against persistence and climatology for each of the six variables (`eval_monthly.py`, 8 members); multi-month skill against climatology at leads of 1, 3, 6 and 12 months (`rollout_monthly.py`, fixed start months).
2. **Out-of-sample hindcast:** a free-run from the December 2018 state under real ERA5 weather to 2025, 8 members saved, scored per member against NSIDC-0051: September, June, July and August extent bias; ice-edge error (IIEE) for September, June, July–September and December–March; the year-to-year correlation of September extent; drift growth (September bias in the last two development years minus the first two, 2021–22 minus 2019–20; with the holdout, 2023–25 minus 2019–21); September bias in the East Siberian, Beaufort, Chukchi and Laptev seas. Reported for the development years (2019–22) and, for release candidates, the holdout (2023–25).
3. **Drift** (`drift.py`, from scorecard version 3): free-runs from the model's own training-target state each December 2008–2014 to December 2018 under real ERA5 weather, scored against that target month by month so the weather cancels; the drift rate is the trend of the annual-mean extent error over lead years, averaged over starts. The hindcast's drift growth above is dominated by weather and can't separate models (ISS-025).
4. **Stability:** the 2015–2100 CMIP free-run finishes with no non-finite values.
5. **Cost:** seconds per step at 1 and 8 members.

If the protocol changes, re-score the previous release's model with the new protocol so the comparison stays like-for-like, and give the scorecard a new version number.

## Where we are: v0.4.0 (released 2026-10-25)

Recommended model: **`monthly_pf_qm_ema_qmw`**: pushforward on quantile-mapped targets whose correction follows the observed trend, fine-tuned with EMA 0.999 (scorecard `results/scorecards/monthly_pf_qm_ema_qmw.json`, version 3). Out-of-sample hindcast, per member, M km² unless noted:

| measure | 2019–22 (development) | 2023–25 (holdout) | 2019–25 |
| --- | --- | --- | --- |
| September extent bias | +0.08 | +0.34 | +0.19 |
| June / July / August extent bias | +0.66 / +0.29 / +0.08 | +0.58 / +0.10 / +0.38 | +0.62 / +0.21 / +0.21 |
| ice-edge error: September / June / July–September / December–March | 1.13 / 1.33 / 1.38 / 0.80 | 1.28 / 1.39 / 1.51 / 0.83 | 1.20 / 1.36 / 1.44 / 0.81 |
| September year-to-year correlation | 0.95 | 0.89 (3 years) | 0.86 |
| September bias: East Siberian / Beaufort / Chukchi / Laptev | +0.16 / +0.01 / +0.08 / −0.07 | +0.11 / +0.31 / +0.10 / −0.09 | +0.14 / +0.14 / +0.09 / −0.08 |

| other measures | |
| --- | --- |
| one-month skill against persistence, 2015–18, 8 members | concentration 58%, thickness 51%, damage 50%, drift 55% / 55%, snow 61% |
| drift against its own target (`drift.py`, 7 starts 2008–14) | +0.003 ± 0.004 per year: none |
| stability: 2015–2100 CMIP free-run | no non-finite values |
| cost: 1 / 8 members, fast sampler, Apple Silicon | ~5.5 / ~46 s per step |

The v0.2.0 model, `monthly_pf_qm`, is the baseline the v0.4.0 targets were set against (scorecard `results/scorecards/monthly_pf_qm.json`).

## v0.3.0 — Measurable and reproducible

**Why:** targets need a scorecard anyone can re-run, and a fresh clone must be able to rebuild every dataset.

**Features**
- The scorecard tool, and the full v0.2 baseline scorecard of `monthly_pf_qm`, including its forecast skill and the development/holdout split.
- The data audit (ISS-017): every file in `data/` produced by a script with its default in `frazil/paths.py`; no leftover zips, unused hemispheres or stray outputs; the SASIP data licence checked (ISS-010).
- The report tells runs of the same model apart (ISS-001).

**Performance:** no model change. Guardrail: the scorecard reproduces the baseline table within sampling noise (±0.03 M km² on extent measures).

**Done when:** the scorecard JSON for `monthly_pf_qm` is committed and the README's numbers cite it; every download script runs in `--dry-run`/`--list` mode from a clean clone; fast and slow tests pass.

**Status (2026-10-25):** not released on its own. The scorecard tool shipped in v0.4.0 together with the drift protocol and resumable runs, and the baseline scorecard is committed. The data audit (ISS-017), telling runs apart in the report (ISS-001) and the SASIP licence check (ISS-010) remain, and go into the next release.

## v0.4.0 — Close the summer gap

**Why:** the model's largest error, and the one between it and the observations: too little summer melt, mostly its own in the East Siberian and Beaufort seas, growing over a run (ISS-002). It comes before the artwork so the animations show the better model. (Moved up from v0.6.0 on 2026-10-09.)

**Work**, cheapest first:
1. ~~**Confirm the fine-tune gain.**~~ Done 2026-10-10: the control `monthly_pf_qm_ema` (same fine-tune, no radiation) matches the radiation model, so the gain is the fine-tune (ISS-005 closed). It is the v0.4.0 candidate; on 2019–22 it meets the September bias, summer ice-edge and June ice-edge targets (June ice-edge error 1.36), not East Siberian (+0.19) or drift growth (+0.12, target ≤ +0.09). One-month skill holds (within 1.5 points on every variable, concentration +10 against its QM targets). Its 2015–2100 CMIP run is finite. It meets every guardrail, so it becomes the recommended model once the targets are met or waived at release.
2. **Make new inputs learnable** (ISS-003, ISS-005): a fine-tune protocol whose inference weights follow training and whose new channels reach meaningful weight, for example a longer run or training the input layer first. Re-run radiation with it if step 1 is promising but weak.
3. **Degree-days that see thaw days inside a month** (ISS-007), computed from daily ERA5 temperatures.
4. **A leak-free ocean-memory input:** sunlight absorbed by the model's *own* open water, accumulated over the summer.

**Targets** (out-of-sample, per member, development years 2019–22; confirmed on the 2023–25 holdout for the release candidate). The target is what a release needs; the stretch goal is a gap the size of the observational uncertainty (NSIDC products differ by ~0.1 M km²), which is where "as close as the data allow" lies.

| measure | v0.2 baseline | target | stretch |
| --- | --- | --- | --- |
| September extent bias | +0.37 | ≤ +0.20 | ≤ +0.10 |
| June ice-edge error | 1.57 | ≤ 1.41 | ≤ 1.29 |
| July–September ice-edge error | 1.83 | ≤ 1.65 | ≤ 1.50 |
| East Siberian / Beaufort September bias | +0.20 / +0.21 | ≤ +0.10 each | ≤ +0.05 each |
| drift growth (2021–22 minus 2019–20) | +0.21 | ≤ +0.09 | ≤ +0.03 |

*Changed 2026-10-15:* the June target was June extent bias (≤ +0.30, stretch ≤ +0.15). Bias is extra ice minus missing ice, so it can rise while the ice edge gets more accurate: `monthly_pf_qm_ema`'s June bias went +0.52 → +0.61 while its June ice-edge error fell 1.57 → 1.36, because it misses less ice and so less of the error cancels (ISS-023). The ice-edge error counts both. The new target and stretch use the same proportions as the July–September ones (−10% and −18% from the baseline). Set after seeing that result: the candidate meets the target (1.36) but not the stretch. June bias is still reported.

*Changed 2026-10-15:* drift growth is now measured on the development years (2021–22 minus 2019–20), like every other target; it was 2023–25 minus 2019–21, which needs the holdout years. The baseline on the new definition is +0.21 (was +0.34). The target and stretch keep the same proportions of the baseline (−56% and −85%: ≤ +0.09 and ≤ +0.03, were ≤ +0.15 and ≤ +0.05), so the change doesn't make the target easier. `monthly_pf_qm_ema` is at +0.12: it would have met the old absolute threshold, but not the proportional one. The 2023–25 minus 2019–21 value is still reported for release candidates.

**Guardrails:** September correlation ≥ 0.85; December–March ice-edge error ≤ 0.95; one-month skill within 2 points of the v0.3 scorecard on every variable; the 2015–2100 CMIP run stays finite; gains survive the leak test. A model that needs inputs CMIP doesn't have yet (radiation) can still be the release model for hindcasts, but the artwork needs those inputs added first (ISS-008).

**Done when:** a model meets the targets and guardrails and becomes the recommended model. If none does, v0.4.0 still ships the tooling and the write-ups, and the targets carry forward, stated as unmet.

**Outcome (released 2026-10-25).** Recommended model `monthly_pf_qm_ema_qmw`. Every guardrail holds. Against the targets, development years 2019–22 and the 2023–25 holdout:

| measure | target | 2019–22 | 2023–25 | |
| --- | --- | --- | --- | --- |
| September extent bias | ≤ +0.20 | +0.08 | +0.34 | met on development years only |
| June ice-edge error | ≤ 1.41 | 1.33 | 1.39 | **met** |
| July–September ice-edge error | ≤ 1.65 (stretch ≤ 1.50) | 1.38 | 1.51 | **met** (stretch on development years) |
| East Siberian / Beaufort September bias | ≤ +0.10 each | +0.16 / +0.01 | +0.11 / +0.31 | **not met** |
| drift growth (2021–22 minus 2019–20) | ≤ +0.09 | +0.12 | — | not met, and not a usable measure: it is dominated by weather (ISS-025); the drift protocol shows no drift (+0.003 ± 0.004 per year). Re-basing pending. |

What the work found: radiation, thaw-day degree-days and an ocean-memory input add nothing (no effect, no new information, or no information the model lacks); longer training overfits the summer; multi-step pushforward makes the model icier; the model doesn't drift. What helped: the EMA fine-tune (ISS-005) and trend-following QM targets (ISS-026). The unmet targets carry forward to the section below.

## Next: the persistent September and East Siberian excess

**Why:** the two errors that survived every change this cycle. The East Siberian Sea is +0.1 to +0.25 too icy in every September 2019–25, in every model. September bias is near target in 2019–22 but +0.34 in 2023–25, mostly a Beaufort Sea excess of about +0.3 that appears in 2023 and stays (ISS-028). Before any more training, find out whether each comes from the model or from its training target, and what changed in 2023.

**Work** (analysis first; no training until one of these points at a cause):
1. **Model or target, by region.** (Done 2026-10-27, [regional_model_vs_target](../experiments/2026-10-25_regional_model_vs_target/): the model's own; in the East Siberian and Beaufort seas it builds over 4–5 years as the ice thickens.) Re-run the drift free-runs saving September fields, and split the model's steady offset from its target by sector (2009–18, no holdout). If the East Siberian excess is the model's own (model − target), it's a modelling problem; if it's already in the target (target − NSIDC), it's a data problem.
2. **The targets by region and year.** (Done: the windowed QM targets match NSIDC to within 0.01 in all three Pacific-shelf seas over 1995–2018.) The QM target minus NSIDC per sector, 1995–2018: does the windowed correction leave a regional bias in the East Siberian Sea?
3. **When in the season the East Siberian excess appears:** month by month, June to September, model against NSIDC: late melt onset, slow summer melt, or ice that doesn't leave in September.
4. **The Beaufort Sea 2023–25 (ISS-028):** what the observed ice and the ERA5 forcing did in those summers (temperature, the winds of the Beaufort High, radiation) against 2019–22, and how the model responds to similar anomalies in its training years.

**Done when:** each of the two errors has a cause supported by these analyses, and a planned fix; or a documented reason it can't be fixed with this data. Targets carry over from v0.4.0. The holdout (2023–25) has been scored once, so for work that responds to it, it's no longer independent.

## v0.5.0 — First artwork release

**Why:** the art–science deliverable, from the release model, reproducibly.

**Features**
- Final 2015–2100 animations from the recommended model: concentration as PMM and as a single member, and thickness (ISS-011, re-run with `--save-members sic sit`; 4 members are enough, per the [ensemble-size experiment](../experiments/2026-09-28_ensemble_size/)).
- The credit line carried with every rendered file ([CREDITS.md](../CREDITS.md)).
- [VISUALS.md](VISUALS.md) updated with the exact commands; fixed colour scales across all frames and runs.

**Performance:** release model unchanged. Guardrails: the projection run has no non-finite values; re-rendering from the saved snapshots reproduces the frames.

**Done when:** the rendered files and the commands that made them are recorded in an `experiments/` entry, with the checkpoint's SHA-256.

## v0.6.0 — Data for live visuals (p5.js, TouchDesigner)

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
- The recommended model meets the v0.4.0 targets, or a documented waiver says why not.
- The scorecard is published with the release.
- The command-line tools, configs and `frazil` package are stable: from here on, breaking changes need a new major version.
- Fast and slow tests pass.

## Later

Parked until a milestone needs them: daily-data retraining (ISS-014), a harder quantile-mapping fine-tune (ISS-013), offering GenSIM's bug fixes upstream (ISS-012), running tests on GitHub Actions (ISS-019).
