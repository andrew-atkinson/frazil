# Changelog

All notable changes to Frazil. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [semantic versioning](https://semver.org/). Add an entry under **Unreleased** in the same branch as the change; move them under a version number when you tag a release. Open problems and ideas live in [docs/ISSUES.md](docs/ISSUES.md).

## [Unreleased]

### Changed
- Repository housekeeping in progress on `chore/housekeeping` (see ISS-016).

### Added
- `CHANGELOG.md`, `docs/ISSUES.md` and `CONTRIBUTING.md` (branch workflow).

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

[Unreleased]: https://github.com/andrew-atkinson/frazil/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/andrew-atkinson/frazil/releases/tag/v0.1.0
