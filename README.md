

> ### A small, laptop-scale sea-ice model — for art–science collaboration
> This project is built on a clone of [GenSIM](https://github.com/cerea-daml/gensim) (not a GitHub fork). It repurposes GenSIM's generative approach toward a different goal: a **small model, trainable on a laptop, that captures the broad monthly evolution of Arctic sea ice** — the large-scale seasonal advance and retreat, rather than the fine 12-hour dynamics the original resolves. It was made for **art–science collaboration**, where legible, broad-strokes change is the point. Practical guide and honest limits in **[The monthly model](#the-monthly-model)**; the rest of this README is the original GenSIM documentation.

## Repository Structure

``` 
data/
├─ auxiliary – Auxiliary data contained in the repository
├── ds_auxiliary.nc - Auxiliary data file (grid cells, mask)
├── ds_demo.nc - Demo dataset (available at https://doi.org/10.5281/zenodo.17535317)
├─ models - The pre-trained model checkpoints (available at https://huggingface.co/tobifinn/GenSIM)
├─ train_data – Zarr training data (not contained in the repository and has to be linked. How to get data please see notebooks/data/)

gensim/
├─ augmentation.py – data augmentation (flips, rotations, patch generation)
├─ data_module.py – LightningDataModule for training/validation datasets
├─ dataset.py – PyTorch Dataset that reads Zarr data
├─ deterministic_network.py – U-Net architecture used as reference for a deterministic model
├─ embedding.py – random‑Fourier embeddings and Embedder
├─ encoder_decoder.py – Encoder and Decoder to map to physical space
├─ forecast_module.py – Lightweight PyTorch module for inference
├─ network.py – Transformer architecture (tokenizer, attention, skips)
├─ sampler.py – Flow‑matching sampler with schedule and second‑order update
├─ train_module.py – LightningModule for training with EMA model support
├─ utils.py – helper functions (masking, averaging, param grouping)
└─ wrapper.py – PatchedNetwork wrapper for forecasting with domain decomposition

notebooks/ – Jupyter notebooks to reproduce key results from the manuscript
├─ data/ – Jupyter notebooks for data preprocessing and analysis

config_train.yaml – Training configuration for GenSIMTrainModule
config_forecast.yaml – Forecasting configuration for GenSIMForecastModule
environment.yml – Conda environment definition
setup.py – Package installation script
train.py – Entry point for training (Hydra CLI)
LICENSE – MIT license
README.md – This file
```

## Installation

```bash
git clone https://github.com/cerea-daml/gensim.git
cd gensim
conda env create -f environment.yml
conda activate gensim
pip install -e .
```

Verify installation:

```bash
python -c "import gensim; print(gensim.__version__)"
```

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

## Data Preprocessing

The `notebooks/` folder contains Jupyter notebooks that walk through the full data preparation pipeline required for training and inference:

- `data_01_get_auxiliary.ipynb` – Downloads and prepares auxiliary NetCDF data.
- `data_02_nextsim_to_zarr.ipynb` – Converts neXtSIM output to Zarr format.
- `data_03_split_parts.ipynb` – Splits the Zarr dataset into training and validation parts.
- `data_04_estimate_normalization.ipynb` – Estimates and stores dataset normalization statistics.

<u>It is important to process the data in the **exact order** shown above to ensure the model receives correctly formatted inputs.</u>

## Module Architecture

GenSIM provides two specialized modules for different use cases:

### GenSIMTrainModule (`gensim/train_module.py`)
- **Purpose**: Training with PyTorch Lightning
- **Features**:
  - Full training logic with loss computation
  - EMA (Exponential Moving Average) model support
  - Optimizer and scheduler configuration
  - Validation and logging capabilities
- **Use case**: Model training and development

### GenSIMForecastModule (`gensim/forecast_module.py`)
- **Purpose**: Lightweight inference
- **Features**:
  - Minimal overhead for fast predictions
  - No training-specific components
  - Direct PyTorch module instantiation
  - Optimized for deployment
- **Use case**: Production inference and forecasting

## Configuration

The configuration is split into two files:

- [`config_train.yaml`](config_train.yaml) – Training configuration for GenSIMTrainModule
- [`config_forecast.yaml`](config_forecast.yaml) – Forecasting configuration for GenSIMForecastModule

Key sections:

- `trainer` – Lightning trainer settings (accelerator, devices, precision, max_steps).
- `surrogate.network` – Transformer architecture (n_input, n_output, n_features, n_blocks, etc.).
- `surrogate.encoder` / `decoder` – Encoder/decoder parameters.
- `surrogate.sampler` – Number of steps and schedule parameters for flow matching sampler.
- `surrogate.train_augmentation` – Settings for data augmentation.
- `data` – Paths to data, batch size, number of workers.

You can override any entry from the command line, e.g.:

```bash
python train.py trainer.max_steps=500000 exp_name=my_exp
```

## Training

Ensure the `data/train_data` folder contains the required Zarr files (`train.zarr`, `validation.zarr`) and the auxiliary NetCDF (`auxiliary/ds_auxiliary.nc`).

```bash
python train.py
```

The script logs progress with a tqdm bar, saves checkpoints under `data/models/<exp_name>/`, and (offline) logs to Weights & Biases as configured in `config.yaml`.

To resume training set `ckpt_path` in `config.yaml` to the desired checkpoint.

## Inference

For inference with this repository, we provide pre-trained model checkpoints via [`HuggingFace`](https://huggingface.co/tobifinn/GenSIM). The model checkpoints define the weights of the neural network and are available in two different versions: either as exponential moving average (`model_weights_ema.safetensors`) or as raw weights (`model_weights.safetensors`). To avoid a contamination of the weights with malicious data, the model weights are stored in the [`safetensors`](https://huggingface.co/docs/safetensors/en/index) format.

A [`jupyter notebook `](inference_demo.ipynb) is included to showcase prediction steps with GenSIM over a demo dataset. To use the notebook, ensure the demo dataset `data/auxiliary/ds_demo.nc` is downloaded from [`Zenodo`](https://doi.org/10.5281/zenodo.17535317). The inference demo initialises GenSIM, loads its checkpoint, and makes ensemble predictions of up to four days. These predictions are then compared to a persistence forecast and the targetted neXtSIM-OPA simulation. The code used in the inference notebook can be also used as starting step for other usage.

## Data Layout

Sea‑ice state variables: `sit` (thickness) `sic` (concentration) `sid` (damage) `siu` (x-drift) `siv` (y-drift) `snt` (snow)

Forcing variables:
| CMIP   | var meaning                        | maps to                                        |
|--------|------------------------------------|------------------------------------------------|
| `tas`  | near-surface (2 m) air temperature | tus + the 4 degree-day features (rolling sums) |
| `huss` | near-surface specific humidity     | huss (direct — skip the ERA5 humidity math)    |
| `uas`  | eastward near-surface wind         | uas (after rotation to the grid)               |
| `vas`  | northward near-surface wind        | vas (after rotation)                           |

The Zarr file contains a `datacube` array with dimensions `[time, variable, y, x]` and a `var_names` attribute.

The auxiliary NetCDF provides `mask`, `x_coord`, `y_coord`.

## License

This project is released under the MIT License (see `LICENSE`).

## Citation

If you use GenSIM, please cite the following preprint until publication:

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

## GenSIM – Generative Sea‑Ice Model

The official implementation of **GenSIM**, a generative sea-ice model to learn sea-ice dynamics with neural networks and flow matching is found here:

[![Demo](https://img.shields.io/badge/Demo-Colab-F9AB00?style=flat&logo=googlecolab&color=%23F9AB00)](https://colab.research.google.com/drive/1R3KPE4okFUGRcomI97RODO8IJAZHELJM?usp=sharing) [![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/tobifinn/GenSIM) [![Preprint](https://img.shields.io/badge/Preprint-ArXiv-B31B1B?style=flat&logo=arxiv)](https://arxiv.org/abs/2508.14984) ![Website](https://img.shields.io/badge/Website-Stay_Tuned-lightblue?style=flat) ![Version](https://img.shields.io/badge/Version-1.0-blue?style=flat)

## Contact

Tobias Sebastian Finn – tobias.finn@enpc.fr

*End of README*
