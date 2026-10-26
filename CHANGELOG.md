# Changelog

All notable changes to Frazil. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [semantic versioning](https://semver.org/). Add an entry under **Unreleased** in the same branch as the change; move them under a version number when you tag a release. Open problems and ideas live in [docs/ISSUES.md](docs/ISSUES.md).

## [Unreleased]

## [0.4.0] — 2026-10-25

Close the summer gap, with the misses stated. The recommended model is `monthly_pf_qm_ema_qmw` (checkpoint SHA-256 `d2650b2c…9ffd4`, see [docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md)): on the 2019–22 hindcast, September bias +0.37 → +0.08 M km² and July–September ice-edge error 1.83 → 1.38 against the v0.2.0 model; one-month skill, CMIP stability and no drift all hold. **Targets met:** June and July–September ice-edge error. **Not met:** September bias on the 2023–25 holdout (+0.34, mostly a Beaufort Sea excess from 2023, ISS-028), the East Siberian and Beaufort sector targets, and drift growth (a weather-dominated measure; the drift protocol shows no drift, ISS-025). They carry forward to the roadmap's *Next* section. v0.3.0 wasn't released on its own: its scorecard ships here; its data audit (ISS-017), ISS-001 and ISS-010 remain. The roadmap's milestones were renumbered on 2026-10-09 (summer gap moved up to v0.4.0).

### Added
- Release-candidate holdout (2023–25) for `monthly_pf_qm_ema_qmw`: September bias +0.34 as before, August improved (+0.53 → +0.38); the 2023–25 September excess is mostly a Beaufort Sea shift from 2023 (ISS-028).
- `monthly_pf_qm_ema_qmw` passes the one-month skill and CMIP stability guardrails (projection 0.1–0.25 M km² lower than `monthly_pf_qm_ema` after 2035): the v0.4.0 release candidate.
- `monthly_pf_qm_ema_qmw` trained on the trend-following QM targets: better than its same-seed partner on every September and ice-edge measure (2019–22 Sept bias +0.08, Sept ice-edge error 1.13); no drift.
- `correct_sic_targets_qm.py --window N`: trend-following QM targets (ISS-026). With 9-year windows the target − NSIDC September trend drops from +0.21 to +0.03 ± 0.07 M km² per decade; targets built (`_sicqmw9`) with the recipe `pf_qm_ema_qmw`. `drift.py` now finds a model's datacube from its data suffix.
- [obs_gap_sources](experiments/2026-10-23_obs_gap_sources/) analysis: the growing 2019–25 gap comes from the QM training targets, whose trend is a fifth too shallow (ISS-026); the model's response to summer warmth matches the observations.
- [drift_protocol](experiments/2026-10-22_drift_protocol/) results: `monthly_pf_qm`, `_ema` and `_ema_s1` don't drift against their training target over 2009–18 (−0.005 to −0.009 M km² per year, run-to-run difference 0.002); scorecards re-scored at version 3.
- `scripts/evaluate/drift.py` (ISS-025): drift with the weather averaged out (free-runs from several December starts 2008–14 to 2018, scored against the training target), logged to the ledger; scorecard version 3 includes it.
- `monthly_pf_qm_ema_ms` (multi-step pushforward, K = 3): about 0.05 M km² icier throughout, ice edge unchanged; not adopted. ISS-025: the 4-year drift-growth measure is dominated by weather and can't separate models.
- Multi-step pushforward: `surrogate.pushforward_steps` (default 1, the original behaviour) chains up to K of the model's own generated months before the trained step; tested in `tests/test_pushforward.py`. Recipe `pf_qm_ema_ms` (K = 3).
- [ocean_memory](experiments/2026-10-21_ocean_memory/) information test (no training): an ocean-memory input adds ~1% predictability, within noise and none on the Pacific shelf, so it isn't built. The summer gap is free-run drift, not a missing input.
- `monthly_pf_qm_ema_dd` trained on the thaw-day degree-days (closes ISS-007): no effect beyond run-to-run noise. With a climatological spread the new inputs are a function of the monthly mean already in the inputs.
- Degree-days that count thaw days (ISS-007): `frazil.preprocess.expected_degrees` / `degree_days_gaussian` (monthly mean plus the day-to-day spread of daily temperature, assuming a normal distribution), `scripts/data/degree_day_spread.py` (the spread from WeatherBench2's ERA5, with an accuracy check against real daily degree-days), `scripts/data/spread_degree_days.py` (rebuilds a datacube's degree-day inputs) and the recipe `pf_qm_ema_dd`. Daily ERA5 from the CDS was tried first and dropped: its queue would take days to weeks.
- Noise between training runs, measured with a seeded retrain of `pf_qm_ema` (`monthly_pf_qm_ema_s1`): ~0.01 M km² on September bias and ice-edge errors, 0.05–0.08 on July–August bias; recorded in the roadmap's scorecard section.
- `configs/experiment/pf_qm_ema2.yaml` and the [more_finetune](experiments/2026-10-18_more_finetune/) experiment: another 20k steps of the EMA fine-tune doesn't help (summer icier, ice edge unchanged); `monthly_pf_qm_ema` stays the candidate.
- `scripts/evaluate/scorecard.py` (v0.3.0): the roadmap's scorecard as one command, with development/holdout split, one-month skill and CMIP stability from the ledger, and the checkpoint SHA-256; writes `results/scorecards/<model>.json`. Scorecards for `monthly_pf_qm` (reproduces the v0.2.0 baseline exactly) and `monthly_pf_qm_ema`.
- `configs/experiment/pf_qm_ema.yaml` and model `monthly_pf_qm_ema`: `monthly_pf_qm` fine-tuned 20k more steps at EMA 0.999. Out of sample it cuts September bias +0.37 → +0.22 and summer ice-edge error 1.83 → 1.46, matching the radiation model without radiation (closes ISS-005); the new v0.4.0 candidate.
- `copernicusmarine` in `environment-mac.yml`, and ISS-022: the neXtSIM reanalysis checked as a possible training teacher (better in September, worse in other seasons; parked).
- Radiation experiment scored (closes ISS-004): `monthly_pf_qm_rad` cuts September bias +0.37 → +0.20 and summer ice-edge error 1.83 → 1.45, but climatological radiation scores the same, so the gain comes from the EMA 0.999 fine-tune, not radiation.

### Changed
- Outputs and descriptions name the model Frazil: the skill table's column, forecast-map column and titles, free-run titles, script descriptions. GenSIM stays where it credits the original work (its code and class names, the 512×512 grid, its constants).
- `eval_monthly.py` is resumable: re-running the same command continues from the last finished month pair, with exactly the same result (each pair now has its own seed, so scores differ slightly from earlier runs of the same model).
- `freerun_monthly.py` is resumable: re-running the same command continues from the last finished month, with exactly the same result (each month now has its own seed, so results differ slightly from earlier runs of the same settings).
- Roadmap: drift growth is measured on the development years (2021–22 minus 2019–20; baseline +0.21), with the target scaled to keep its ambition (≤ +0.09, stretch ≤ +0.03).
- Roadmap: the v0.4.0 June target is June ice-edge error (≤ 1.41, stretch ≤ 1.29) instead of June extent bias, which rose for a model whose June ice got more accurate (ISS-023). Scorecard version 2 adds June ice-edge error; both scorecards re-scored.
- Roadmap: the summer-melt milestone moves up to v0.4.0 (*Close the summer gap*), straight after the scorecard, with stretch targets at the size of the observational uncertainty; the artwork and live-visuals milestones become v0.5.0 and v0.6.0.

## [0.2.0] — 2026-10-07

The restructure release: a reorganised, tested and documented codebase with a plan to 1.0. Model behaviour is unchanged from 0.1.0; the recommended model is `monthly_pf_qm` (checkpoint SHA-256 `6208b4a8…ce71a`, see [docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md)). Commands moved: see *Changed*.

### Changed
- Every default data, model, config and output location is defined once in `frazil/paths.py` (was: the grid file hard-coded 16 times, the datacube 10, …); scripts and `train.py` read it. All script defaults are unchanged except the fix below.
- Shared colour maps (`CMAP`, `DIVERGING`) and `snapshot_date` moved into the package; no script imports another script any more.
- Configs moved from the root into `configs/`. `train.py` now defaults to the Mac monthly config, so `python train.py` trains the base model.
- The four fine-tunes are Hydra experiment recipes in `configs/experiment/` (`pushforward`, `sicqm`, `pf_qm_sst`, `pf_qm_rad`), run with `python train.py +experiment=<name>`. Each composes to exactly the same config as before.
- Script defaults, the demo notebook and the docs point to `configs/`; `normalization_monthly.json` moved to `results/`.
- Scripts moved out of `experiments/` into `scripts/` by stage (`download/`, `data/`, `evaluate/`, `project/`, `figures/`), keeping their names. Every command's path changes: `python scripts/<stage>/<name>.py`.
- Shared code moved into a new `frazil/` package (`models`, `preprocess`, `diagnostics`, `ensemble`, `io`, `ledger`), installed with `pip install -e .`; scripts no longer import each other through `sys.path`. Re-running the 2018 forcing through the moved code is bit-identical to the training data.
- The results ledger is `frazil/ledger.py` (`python -m frazil.ledger show`).
- `experiments/` is now the lab notebook: one folder per experiment with a write-up, an index and a template.

### Fixed
- `freerun_monthly.py` crashed when drawing its final-state map (it imported colour maps from a script in another folder after phase 2's move).
- `ensemble_extent.py` only ran from its own folder (it imported `extend_snapshots`).
- The neXtSIM downloader saved to `nextsim_opa_monthly/` at the repository root, but preprocessing and the docs expect `data/train_data/nextsim_opa_monthly/`.

### Added
- Roadmap milestone v0.5.0, *Data for live visuals*: per-month fields and motion (drift particles, ice-edge contours, growth and melt, edge certainty) exported for p5.js and TouchDesigner (ISS-020, ISS-021); the physics and projection milestones move to v0.6.0 and v0.7.0.
- [docs/ROADMAP.md](docs/ROADMAP.md): milestones v0.3.0 to v1.0.0, each with features, performance targets against a measured baseline, guardrails and exit criteria; the scorecard protocol; development (2019–22) vs holdout (2023–25) years.
- Versioning, release checklist and bug-fixing route (severities S1–S3, test-first fixes, errata) in CONTRIBUTING.md; a *Target* milestone for every open issue.
- **Tests** (`tests/`, `pytest`), closing ISS-018. Fast tier (~40 s, no data): self-checks, every script loaded from an unrelated folder, snapshots of all script defaults and composed configs, no script-to-script imports or hard-coded paths, no script locating files from its own folder, unit tests of the package (75% coverage). Slow tier (`--slow`, ~5 min): free-run with map and members, bit-identical forcing rebuild, zero-initialised channel expansion vs its seed, animation frame types, report from another folder. Each past regression was reintroduced and caught.
- A docs test: every relative link in the Markdown files must resolve (catches links to moved or deleted docs).
- `pytest.ini` (pins the test root to the repo), a pre-push hook in `.githooks/` running the fast tier, and `pytest`/`pytest-cov` in `environment-mac.yml`.
- `results_report.py` and `plot_data_timeline.py` take arguments (`--help`; `--out-dir` for the figure) instead of running immediately.
- `CHANGELOG.md`, `docs/ISSUES.md` and `CONTRIBUTING.md` (branch workflow).
- `scripts/evaluate/ensemble_size.py` and the [ensemble-size experiment](experiments/2026-09-28_ensemble_size/): averaging removes only the random part of the error (15% in summer); 4 members give ~85% of the 8-member gain; the plain mean's edge worsens with size, PMM's doesn't.
- README sections: repository layout, *What it can and can't do* (with current results, including the 2019–25 hindcast and the `monthly_pf_qm` projection), and a *Documentation* map. CLI.md gains an *Output names* section.
- `archive/` with a README for finished one-off tools.

### Removed
- `docs/STATUS.md`, folded into the README (*What it can and can't do*, *Documentation*), CLI.md (*Output names*) and the lab notebook, closing ISS-016.
- Personal tooling config (`.mcp.json`, `.ignore`) from version control; still ignored locally.
- One-off tools moved to `archive/`: optimizer A/B, training smoke/minimal/watch runs, `loss_graph.py`, skill-vs-step sweep, the QM proof of concept, the mean-shift `sic` correction, and the original 12-hour drift experiment (`drift_rollout.py`).

## [0.1.0] — 2026-09-27

First release as Frazil, a monthly, laptop-scale adaptation of [GenSIM](https://github.com/cerea-daml/gensim).

### Added
- **Monthly pipeline:** downloaders for neXtSIM-OPA, ERA5, CMIP6 and NSIDC (`download/`); regridding and preprocessing onto the GenSIM grid; normalization; zarr packing; Apple Silicon (MPS) training config.
- **Models:** `monthly` (base), `monthly_pf` (pushforward drift fine-tune), `monthly_qm` and `monthly_pf_qm` (quantile-mapped `sic` targets), `monthly_pf_qm_sst` and `monthly_pf_qm_rad` (extra forcing channels).
- **Evaluation:** one-step skill, multi-month rollouts with persistence and climatology baselines, skill-vs-step sweep.
- **Projection and visualisation:** decadal free-runs under cyclic, CMIP or ERA5 forcing; snapshot restarts; animations with mean, probability-matched-mean or single-member frames; ice extent/area vs NSIDC.
- **Out-of-sample hindcast** 2019–2025 under real ERA5 weather, with per-member extent, saved members and a `--climatology-forcing` leak test.
- **Ensemble summaries** (`ensemble_summaries.py`): mean, member, median, consensus, PMM and local PMM scored on ice-edge error.
- **Extra forcing channels:** `add_forcing_channels.py`, `build_zarr --extra-forcings`, and zero-initialised input expansion so a fine-tune starts identical to its seed; `channel_weights.py` to check whether new inputs are learned.
- **Results ledger** (`results/experiments.jsonl`) with code fingerprinting and a generated report.
- **Docs:** CLI reference, data and model inventory, observation-gap investigation, pushforward notes, dataset timeline figure; attribution to GenSIM, credits for every dataset, `CITATION.cff`.

### Fixed (in GenSIM's code)
- `training_step` rebuilt the optimizer every step, resetting Adam and never running the learning-rate schedule.
- Patch extraction crashed at padded boundaries.
- `neglogcdf` used a function with no MPS kernel (a silent, slow CPU fallback).

### Fixed (in Frazil)
- CMIP free-runs read forcing from the cube's first month whatever `--start` said.
- The results report labelled every `monthly_pf_*` run as `monthly_pf`.
- ERA5 radiation and SST were interleaved on different time stamps when merged.

[Unreleased]: https://github.com/andrew-atkinson/frazil/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/andrew-atkinson/frazil/compare/v0.2.0...v0.4.0
[0.2.0]: https://github.com/andrew-atkinson/frazil/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/andrew-atkinson/frazil/releases/tag/v0.1.0
