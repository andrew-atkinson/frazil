# Changelog

All notable changes to Frazil. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [semantic versioning](https://semver.org/). Add an entry under **Unreleased** in the same branch as the change; move them under a version number when you tag a release. Open problems and ideas live in [docs/ISSUES.md](docs/ISSUES.md).

## [Unreleased]

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
- **Tests** (`tests/`, `pytest`), closing ISS-018. Fast tier (~40 s, no data): self-checks, every script loaded from an unrelated folder, snapshots of all script defaults and composed configs, no script-to-script imports or hard-coded paths, no script locating files from its own folder, unit tests of the package (75% coverage). Slow tier (`--slow`, ~5 min): free-run with map and members, bit-identical forcing rebuild, zero-initialised channel expansion vs its seed, animation frame types, report from another folder. Each past regression was reintroduced and caught.
- `pytest.ini` (pins the test root to the repo), a pre-push hook in `.githooks/` running the fast tier, and `pytest`/`pytest-cov` in `environment-mac.yml`.
- `results_report.py` and `plot_data_timeline.py` take arguments (`--help`; `--out-dir` for the figure) instead of running immediately.
- `CHANGELOG.md`, `docs/ISSUES.md` and `CONTRIBUTING.md` (branch workflow).
- `scripts/evaluate/ensemble_size.py` and the [ensemble-size experiment](experiments/2026-09-28_ensemble_size/): averaging removes only the random part of the error (15% in summer); 4 members give ~85% of the 8-member gain; the plain mean's edge worsens with size, PMM's doesn't.
- A repository-layout section in the README.
- `archive/` with a README for finished one-off tools.

### Removed
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

[Unreleased]: https://github.com/andrew-atkinson/frazil/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/andrew-atkinson/frazil/releases/tag/v0.1.0
