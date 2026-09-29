"""Where Frazil's data, models, configs and outputs live.

Every script takes its default paths from here, so moving a dataset is a
one-line change. Paths are relative to the repository root; run the scripts
from there, as the docs do.
"""
# Model grid: ocean mask, cell areas, coordinates.
AUX = "data/auxiliary/ds_auxiliary.nc"

# Training data: raw downloads, the per-year datacubes and the packed zarr stores.
TRAIN_DATA = "data/train_data"
NEXTSIM_RAW = "data/train_data/nextsim_opa_monthly"
ERA5_RAW = "data/train_data/data_stream-moda_stepType-avgua.nc"
ERA5_EXTRA_RAW = "data/train_data/era5_extra_1995_2025"
DATACUBE = "data/train_data/monthly_datacube"
DATACUBE_SICQM = "data/train_data/monthly_datacube_sicqm"
ERA5_EXTENSION = "data/train_data/era5_forcing_datacube"
CMIP_FORCING_RAW = "data/train_data/cmip_forcing.nc"
CMIP_DATACUBE = "data/train_data/cmip_datacube"

# Observations.
NSIDC0051_RAW = "data/obs/nsidc0051"
NSIDC0051_GRID = "data/obs/nsidc0051_grid"
SEA_ICE_INDEX = "data/obs/nsidc"

# Models and the configs used to load them.
MODELS = "data/models"
BASE_MODEL_DIR = "data/models/monthly"
CONFIG_FORECAST = "configs/config_forecast_monthly.yaml"
CONFIG_TRAIN = "configs/config_train_monthly_mac.yaml"

# Outputs (never committed).
FREERUN_OUT = "plots/freerun"
SNAPSHOTS = "plots/freerun/snapshots"
