"""The ocean-heat cubes for 2015-2018 with each year's heat replaced by its monthly
climatology over the training years 1995-2014, per cell; everything else untouched. Run
pf_qm_ema_qmw_ohc on it to see what the year's actual heat does, with the same weights.

Run from the repo root:  python experiments/2026-10-30_summer_melt/make_clim_cube.py
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from frazil import paths

SRC = os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9_ohc")
OUT = SRC + "_clim"
LAYERS = ["ohc_0_20", "ohc_20_100"]

cube = xr.open_mfdataset(sorted(glob.glob(f"{SRC}/*_datacube_*.nc")), combine="by_coords")["datacube"]
t = pd.DatetimeIndex(cube.time.values)
base = cube.sel(var_names=LAYERS).isel(time=np.where(t.year <= 2014)[0]).load()
clim = base.groupby("time.month").mean("time")
os.makedirs(OUT, exist_ok=True)
for y in range(2015, 2019):
    f = f"{SRC}/monthly_datacube_{y}.nc"
    da = xr.open_dataset(f)["datacube"].load()
    arr = da.values.copy()
    for i, m in enumerate(pd.DatetimeIndex(da.time.values).month):
        for k in LAYERS:
            arr[i, list(da.var_names.values).index(k)] = clim.sel(month=m, var_names=k).values
    da.copy(data=arr).to_dataset().to_netcdf(os.path.join(OUT, os.path.basename(f)), encoding={"datacube": {"zlib": True, "complevel": 4}})
    print(f"  {y} -> {OUT}", flush=True)
