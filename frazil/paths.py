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
NEXTSIM_OCEAN_HEAT = "data/train_data/nextsim_ocean_heat"   # neXtSIM-OPA upper-ocean heat content, Pacific/Siberian shelves
NEXTSIM_OCEAN_HEAT_FULL = "data/train_data/nextsim_ocean_heat_full"   # the same, every model ocean cell, to 300 m
TOPAZ_OCEAN_HEAT = "data/train_data/topaz_ocean_heat"       # TOPAZ4b reanalysis upper-ocean heat, same box, on the model grid
TOPAZ_DEPTH = "data/auxiliary/topaz_depth.nc"            # seafloor depth from TOPAZ4b wet levels (topaz_depth.py)
NEXTSIM_REANALYSIS = "data/train_data/nextsim_reanalysis_{var}_monthly.nc"   # Copernicus neXtSIM reanalysis, on the model grid
ERA5_RAW = "data/train_data/data_stream-moda_stepType-avgua.nc"
ERA5_EXTRA_RAW = "data/train_data/era5_extra_1995_2025"
WB2_DAILY_T2M = "data/train_data/wb2_daily_t2m.nc"   # daily 2 m temperature, 1.5 deg, from WeatherBench2
T2M_SPREAD = "data/auxiliary/t2m_daily_spread.nc"   # day-to-day std of daily temperature per calendar month
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
DRIFT_OUT = "plots/freerun/drift"   # drift protocol free-runs (diagnostics CSVs only)

# Scorecards (committed: the record of each model's measured performance).
SCORECARDS = "results/scorecards"
