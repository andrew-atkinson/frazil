# Frazil

**A small, laptop-trainable model of monthly Arctic sea ice, made for art–science collaboration.** It captures the large-scale seasonal advance and retreat of the ice and its decline under a warming scenario, rather than the fine 12-hour dynamics of the model it grew out of.

> **Based on [GenSIM](https://github.com/cerea-daml/gensim)**, the generative sea-ice model by Tobias Sebastian Finn, Marc Bocquet, Pierre Rampal, Charlotte Durand, Flavia Porro, Alban Farchi and Alberto Carrassi ([preprint](https://arxiv.org/abs/2508.14984)), released under the MIT licence. This is an **independent, unofficial adaptation**. It is not affiliated with or endorsed by the GenSIM authors, so please raise questions and issues about it here, not in the GenSIM repository. The original documentation is kept in [docs/ORIGINAL_GENSIM.md](docs/ORIGINAL_GENSIM.md).

## How this differs from GenSIM

- **Monthly, not 12-hourly.** Retrained from scratch at a one-month step on monthly neXtSIM-OPA targets and ERA5 forcing, as a much smaller model that trains on a laptop (Apple Silicon / MPS).
- **Same core.** The `gensim/` package keeps GenSIM's flow-matching Transformer and training code, with the bug fixes listed under *Core bug fixes* below.
- **New around it:** downloaders and preprocessing for monthly data, pushforward (drift) fine-tuning, quantile-mapped targets, extra forcing channels, an out-of-sample hindcast against observations, ensemble summaries and animation tools (`download/`, `experiments/`, `docs/`).
- **Different purpose.** Its results are not comparable to GenSIM's published skill; see *Limitations* below.

## Installation

```bash
git clone https://github.com/andrew-atkinson/frazil.git
cd frazil
conda env create -f environment-mac.yml && conda activate gensim && pip install -e .
```

(`environment.yml` is GenSIM's original CUDA environment.)

## The monthly model

A lightweight adaptation of GenSIM that runs and trains on a laptop (Apple Silicon / MPS) and predicts Arctic sea ice **month to month** instead of every 12 hours. The aim is a small, portable model that gets the **broad seasonal picture** right — the yearly advance and retreat of the ice — for art–science work and for anyone who wants a rough sea-ice forecast without a GPU cluster. The full pipeline (download → preprocess → train → evaluate → forecast) runs end to end on one machine.

**Environment.** Use `environment-mac.yml` (drops the CUDA-only PyTorch wheel and flash-attn; MPS build works out of the box):

```bash
conda env create -f environment-mac.yml && conda activate gensim && pip install -e .
```

`inference_demo.ipynb` auto-detects the device (CUDA → MPS → CPU) and casts to float32 (MPS has no float64), so it runs unchanged on a Mac.

**Monthly retraining pipeline** (downloaders in `download/`, pipeline in `experiments/`; run all from the repo root):

```bash
# 1. download targets (neXtSIM-OPA, 1995–2018) and forcing (ERA5 monthly, needs ~/.cdsapirc)
python download/nextsim.py     # -> data/train_data/nextsim_opa_monthly/
python download/era5.py --years 1994-2018   # ERA5 monthly forcing (1994 = degree-day spin-up)
# (for climate-scenario forcing instead of ERA5: python download/cmip.py, see download/notes.md)
# 2. regrid + rotate onto the GenSIM grid, derive humidity/degree-days
python experiments/preprocess_monthly.py           # -> data/train_data/monthly_datacube/
# 3. normalization stats -> config_train_monthly.yaml
python experiments/estimate_normalization_monthly.py
# 4. pack masked train/validation zarr (delta_t=1 = one-month step)
python experiments/build_zarr_monthly.py
# 5. train (config_train_monthly_mac.yaml = ~3.8M-param model tuned for a laptop GPU)
python train.py --config-name config_train_monthly_mac
```

**Stop/resume.** Lightning writes `data/models/<exp_name>/last.ckpt` every 500 steps; resume with:

```bash
python train.py --config-name config_train_monthly_mac ckpt_path=data/models/monthly/last.ckpt
```

**Evaluation & forecasting** (all reuse `experiments/eval_monthly.py`'s checkpoint loader; scores are computed against the neXtSIM-OPA truth over the held-out validation months 2015–2018, with **persistence** — "next month = this month" — as the baseline):

```bash
python experiments/eval_monthly.py        # one-month-ahead RMSE + skill vs persistence
python experiments/predict_monthly.py     # plot forecast maps: truth | GenSIM | persistence
python experiments/rollout_monthly.py     # free-running autoregressive rollout, error vs lead
python experiments/sweep_skill_monthly.py # skill-vs-training-step curve (cached, post-hoc)
```

Other `experiments/` helpers: `smoke_train_monthly.py` / `minimal_train_monthly.py` (config + shape checks), `train_watch_monthly.py` (live loss curve), and `ab_optimizer_monthly.py` (the optimizer A/B below).

### Data inputs

![Datasets: time span and influence on the model](docs/figures/data_timeline_light.png#gh-light-mode-only)
![Datasets: time span and influence on the model](docs/figures/data_timeline_dark.png#gh-dark-mode-only)

| dataset | years | role | influence on the model |
| --- | --- | --- | --- |
| **neXtSIM-OPA** sea-ice states: sit, sic, sid, siu, siv, snt | 1995–2018 | training target | **Total.** The only source of the model's ice physics, so its biases become the model's. |
| **ERA5** monthly 2 m temperature, dewpoint, pressure, 10 m winds → `tus huss uas vas` + 4 degree-day features | 1994–2018 | forcing | **Total.** Every model's atmosphere (1994 = degree-day spin-up). |
| **QM-corrected targets**: neXtSIM `sic` quantile-mapped onto NSIDC-0051, per cell and month | 1995–2018 | training target | Partial: `sic` only, in the QM models (`monthly_qm`, `monthly_pf_qm`, and the SST/radiation runs). |
| **ERA5 extension** (same 5 fields) | 2018–2025 | forcing | Run time only: drives the 2019–25 out-of-sample hindcast from the Dec 2018 state (2018 = spin-up). |
| **ERA5 SST, downward solar and thermal radiation** | 1995–2025 | forcing | Partial: only models that list them in `data.forcing_variables`. |
| **CMIP6** MPI-ESM1-2-LR, ssp245, bias-corrected to ERA5 | 2015–2100 | forcing | Run time only: projections and the animation. Never trained on; one climate trajectory with its own weather. |
| **NSIDC-0051** passive-microwave sea-ice concentration, regridded to the model grid | 1995–2025 | observations | Indirect: fits the QM targets (to 2018) and scores the 2019–25 hindcast. |
| **NSIDC Sea Ice Index** v4 extent/area, all 12 months | 1979–2025 | observations | Evaluation only (`ice_area.py --nsidc`). |

Train 1995–2014 (240 months), validate 2015–2018 (48 months), test out-of-sample 2019–2025 against NSIDC. Two caveats: ERA5 *prescribes* SST and sea ice from satellite analyses, so its SST under the observed ice is fixed at 271.46 K (nearly the observed ice mask), and its air temperature carries a weaker imprint of the same ice. Every dataset's role, paths and weight are in **[docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md)**; the figure is `experiments/plot_data_timeline.py`.

Getting the additional inputs (NSIDC-0051 needs an Earthdata login in `~/.netrc` or `EARTHDATA_TOKEN` — never in the repo):

```bash
python download/nsidc.py --month 0                          # Sea Ice Index, all 12 months
python download/nsidc0051.py                                # gridded concentration 1995-2025
python experiments/preprocess_nsidc0051.py                  # -> data/obs/nsidc0051_grid/
python experiments/correct_sic_targets_qm.py                # QM targets -> monthly_datacube_sicqm/
python download/era5.py --years 2018-2025                   # ERA5 extension
python experiments/preprocess_monthly.py --forcings-only --years 2019 2020 2021 2022 2023 2024 2025 \
  --era5-path data/train_data/era5_2018_2025/data_stream-moda_stepType-avgua.nc \
  --out-dir data/train_data/era5_forcing_datacube
python download/era5.py --extra --years 1995-2025           # SST + downward solar/thermal radiation
python experiments/add_forcing_channels.py --datacube-dir data/train_data/monthly_datacube_sicqm   # (and the other cubes)
```

> **Full CLI reference:** every script's flags, defaults, outputs, and the decadal-projection tools (`freerun_monthly.py`, bias-corrected CMIP forcing) are documented in **[docs/CLI.md](docs/CLI.md)**.

**What it produces.** A ~3.8M-param monthly model that, trained on a laptop, **beats persistence by ~40–60% on all six sea-ice variables** at one-month-ahead lead over 2015–2018 — a working monthly emulator, not a toy. Note the loss (a flow-matching NLL with a learned scale) plateaus early and is *not* a good progress meter; judge the model by the RMSE-vs-persistence skill instead.

**Core bug fixes** (in `gensim/`, required for any real training run):
- `train_module.py` – `training_step` uses the persistent optimizer/scheduler from `configure_optimizers` (it previously rebuilt AdamW every step, resetting Adam momentum and never running the LR schedule). Adds an `ema_update_every` knob.
- `augmentation.py` – `extract_patches` clamps patch starts to the padded bounds (fixes a boundary size-mismatch crash on some samples).
- `utils.py` – `neglogcdf` uses `ndtr` instead of `log_ndtr` (no MPS kernel for the latter → a silent, slow CPU fallback).
- `train.py` – resilient wandb import, always-on CSV logger (`data/models/<exp_name>/metrics.csv`), and a trusted full-checkpoint load (PyTorch 2.6 defaults `weights_only=True`).

### Limitations

This is a **coarse, exploratory, from-scratch experiment**, not a drop-in upgrade of GenSIM. Be clear-eyed about what it does *not* establish:

- **Different (easier) task than upstream.** A monthly step is not the 12-hour dynamics GenSIM was built for; persistence is a much weaker baseline at monthly lead, so the skill numbers are **not comparable** to the paper's.
- **Feature semantics change.** The degree-day features (rolling 30-/366-day sums of daily temperature) degenerate at monthly cadence; whether they still earn their place is untested.
- **Tiny data, shrunk model.** 240 training months and a 256-feature model (vs. 20 years of 12-hourly data and 512 features on 8 GPUs upstream). It saturates fast.
- **Narrow validation.** One-month skill, multi-month rollouts (`rollout_monthly.py`) and a 2019–25 out-of-sample hindcast scored on ice-edge error against NSIDC. No spectra/dynamics realism or ensemble calibration checks yet.
- **Atmosphere-driven — it cannot forecast the future on its own.** GenSIM *consumes* forcings; it does not generate them. Predicting beyond the data requires supplying future forcings (reanalysis for the recent past, or a climate scenario), regridded through `preprocess_monthly.py`.
- **Data coverage.** neXtSIM-OPA exists only for **1995–2018**, so all training and neXtSIM-based validation stays in that window. Beyond it, models are scored against observations only: NSIDC over 2019–25, under real ERA5 weather.
- **Not the intended scale.** Real training belongs on CUDA (the base config targets 8 GPUs); the Mac path trades model size and speed for portability.

## Credits and data

This project stands on GenSIM and on openly shared data: the **neXtSIM-OPA** simulation (Boutin et al., 2023), **ERA5** (Copernicus Climate Change Service), **NSIDC** passive-microwave observations, and **CMIP6** MPI-ESM1-2-LR output. Several of these make citation a condition of use. Full citations, the subsets used, and a suggested credit line for exhibitions are in **[CREDITS.md](CREDITS.md)**. No data is included in this repository.

## Licence

MIT (see [LICENSE](LICENSE)). The GenSIM code is © 2025 Tobias Finn; the adaptation and additions are © 2026 Andrew Atkinson.

## Citing

If you use this project, please cite it (see [CITATION.cff](CITATION.cff), or GitHub's *Cite this repository*) **and** GenSIM:

```bibtex
@article{finn_preprint_2025,
    author={Finn, Tobias Sebastian and Bocquet, Marc and Rampal, Pierre and Durand, Charlotte and Porro, Flavia and Farchi, Alban and Carrassi, Alberto},
    title={Generative AI models capture realistic sea-ice evolution from days to decades},
    url={http://arxiv.org/abs/2508.14984},
    DOI={10.48550/arXiv.2508.14984},
    note={arXiv:2508.14984 [physics]},
    number={arXiv:2508.14984},
    publisher={arXiv},
    year={2025},
    month=nov
}
```
