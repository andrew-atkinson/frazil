"""Reading and writing: datacube series, snapshot states, NSIDC extent CSVs."""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from frazil import STATES

def load_series(datacube_dir, names):
    """Concatenate every yearly datacube file -> (T, len(names), H, W) + times."""
    files = sorted(glob.glob(f"{datacube_dir}/*_datacube_*.nc"))
    if not files:
        raise SystemExit(f"no datacube files in {datacube_dir}")
    da = xr.open_mfdataset(files, combine="by_coords")["datacube"].load()
    arr = np.nan_to_num(da.sel(var_names=names).transpose("time", "var_names", ...).values)
    return arr.astype(np.float32), pd.DatetimeIndex(da["time"].values)


def save_state(sm, date, path):
    """Save one ensemble-mean spatial state (6,H,W) as NetCDF for later mapping."""
    xr.DataArray(sm[None], dims=("time", "var_names", "y", "x"),
                 coords={"var_names": STATES, "time": [np.datetime64(date)]},
                 name="state").to_dataset().to_netcdf(path)


def load_nsidc(path, month, metric="extent"):
    """NSIDC monthly Sea Ice Index CSV (10^6 km^2) -> {year: value_km2} for `month`.
    The file (named ..._extent_...) carries BOTH columns, so read the one matching
    the model metric: `extent` or `area`. `path` is the CSV or a dir holding
    N_MM_extent_v*.csv. Skips missing (-9999) values."""
    col = 5 if metric == "area" else 4   # header: year, mo, source, region, extent, area
    if os.path.isdir(path):
        cand = (glob.glob(f"{path}/[NS]_{month:02d}_extent_v*.csv")
                or glob.glob(f"{path}/*_{month:02d}_extent*.csv"))
        if not cand:
            raise SystemExit(f"no NSIDC ..._{month:02d}_extent CSV in {path} "
                             f"(run: python scripts/download/nsidc.py --month {month})")
        path = sorted(cand)[-1]
    out = {}
    with open(path) as fh:
        next(fh, None)  # header: year, mo, source_dataset, region, extent, area
        for line in fh:
            p = [c.strip() for c in line.split(",")]
            if len(p) <= col:
                continue
            try:
                yr, val = int(p[0]), float(p[col])
            except ValueError:
                continue
            if val >= 0:
                out[yr] = val * 1e6  # 10^6 km^2 -> km^2
    return out


def load_sic(path):
    """Load the SIC time series (T,H,W) + times from a datacube or snapshots dir."""
    files = sorted(glob.glob(f"{path}/*.nc")) if os.path.isdir(path) else sorted(glob.glob(path))
    if not files:
        raise SystemExit(f"no .nc files at {path}")
    da = xr.open_mfdataset(files, combine="by_coords")
    name = next((n for n in ("datacube", "state") if n in da.data_vars),
                list(da.data_vars)[0])
    var = da[name]
    if "sic" not in [str(v) for v in var["var_names"].values]:
        raise SystemExit(f"no 'sic' in {path} (forcing-only cube like cmip_datacube?)")
    sic = var.sel(var_names="sic").load().values
    times = pd.DatetimeIndex(var["time"].values)
    order = np.argsort(times.values)
    return sic[order], times[order]


def snapshot_date(path):
    """state_YYYYMM.nc -> Timestamp(YYYY-MM-15), or None if it doesn't match."""
    m = re.search(r"state_(\d{4})(\d{2})\.nc$", os.path.basename(path))
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-15") if m else None
