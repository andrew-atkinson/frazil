#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monthly preprocessing for a GenSIM retraining experiment.

Takes the two raw monthly datasets you downloaded:

  * neXtSIM-OPA sea-ice targets (native CREG025 curvilinear grid, 603x528),
    one NetCDF per year, variables sit/sic/sid/siu/siv/snt.
  * ERA5 atmospheric forcing (regular 0.25 deg lat-lon, 40-90 N), one NetCDF
    with u10/v10/d2m/t2m/sp, monthly, 1994-2018.

...and produces a single "datacube" on GenSIM's 512x512 North-Polar-Stereographic
grid (from data/auxiliary/ds_auxiliary.nc), one file per year, with the same
15 var_names layout as the inference demo's ds_demo:

    sit, sic, sid, siu, siv, snt,          # 6 sea-ice states (targets)
    tus, huss, uas, vas, rhus,             # 5 atmospheric forcings
    pdd_month, fdd_month, pdd_year, fdd_year  # 4 degree-day features

What it does, and where it deviates from the repo's (incomplete) data_02:

  1. Regrid (nearest-neighbour on the unit sphere) both datasets onto the
     ds_auxiliary grid.  data_02's neXtSIM was already on-grid; ours is native
     CREG025, so it is regridded here.
  2. Rotate vector fields into the target grid's local frame:
       - ERA5 winds (geographic east/north) -> grid (uas, vas).
       - neXtSIM ice drift (siu, siv on the CREG025 grid) -> target grid frame.
     data_02's rotation helpers are not defined in the repo, so rotation is
     re-implemented here from each grid's lon/lat geometry (projection-agnostic,
     exact 2x2 change of basis; magnitude is checked to be preserved).
  3. Humidity (huss, rhus) via the same Magnus formulation as data_02.
  4. Degree-day features via rolling means of 2 m temperature.

     *** MONTHLY DESIGN DECISION ***  data_02 used 6-hourly data with windows of
     120 steps (30 days) and 1464 steps (366 days). At MONTHLY cadence those
     become 1 and 12 steps. So pdd_month/fdd_month are the current month's
     degree-day rate, and pdd_year/fdd_year are a trailing 12-month mean. This
     is a defensible monthly analogue but is a modelling choice you should
     review -- monthly means of temperature are not the same as accumulated
     daily degree-days. Change AVG_LEN in frazil/preprocess.py to experiment.

Land cells (ds_auxiliary mask == 0) are set to NaN, matching the demo datacube
(which is fillna(0)'d at load time).

Usage
-----
Process one year (quick test)::

    python scripts/data/preprocess_monthly.py --years 1995

Full run::

    python scripts/data/preprocess_monthly.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from frazil.preprocess import (build_nearest_index, regrid_time, local_grid_basis,
                               rotate_to_target, specific_humidity, relative_humidity,
                               degree_days, ym_key)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
STATE_VARS = ["sit", "sic", "sid", "siu", "siv", "snt"]
FORCING_VARS = ["tus", "huss", "uas", "vas", "rhus",
                "pdd_month", "fdd_month", "pdd_year", "fdd_year"]
DATACUBE_VARS = STATE_VARS + FORCING_VARS

def process_forcings(era5_path, ds_aux):
    """Return a dict var -> (time, 512, 512) forcing arrays, plus (y,m) keys."""
    print("[era5] opening", era5_path)
    e = xr.open_dataset(era5_path)
    tname = "valid_time" if "valid_time" in e.coords else "time"
    times = e[tname].values
    lon1d, lat1d = e["longitude"].values, e["latitude"].values
    src_lon, src_lat = np.meshgrid(lon1d, lat1d)  # (nlat, nlon)
    tgt_lon = ds_aux["longitude"].values
    tgt_lat = ds_aux["latitude"].values
    tgt_shape = tgt_lon.shape

    print("[era5] building nearest-neighbour index "
          f"({src_lon.size} src -> {tgt_lon.size} tgt) ...")
    idx = build_nearest_index(src_lon, src_lat, tgt_lon, tgt_lat)

    # Regrid the raw fields first (matches data_02 order: interpolate, then derive).
    raw = {}
    for name in ["t2m", "d2m", "sp", "u10", "v10"]:
        raw[name] = regrid_time(e[name].values, idx, tgt_shape)
    e.close()

    out = {}
    out["tus"] = raw["t2m"]
    out["huss"] = specific_humidity(raw["d2m"], raw["sp"]).astype(np.float32)
    out["rhus"] = relative_humidity(raw["t2m"], raw["d2m"]).astype(np.float32)

    # Winds: ERA5 u10/v10 are geographic east/north -> rotate into target frame.
    tgt_basis = local_grid_basis(tgt_lon, tgt_lat)
    out["uas"], out["vas"] = rotate_to_target(
        raw["u10"], raw["v10"], src_basis=None, tgt_basis=tgt_basis, label="wind")

    # Degree days (rolling over the full time series, incl. the 1994 lead-in).
    dd = degree_days(raw["t2m"], times)
    for k, v in dd.items():
        out[k] = v.values.astype(np.float32)

    return out, ym_key(times)


def process_states_year(nc_path, ds_aux, idx_cache):
    """Regrid + rotate one year of neXtSIM states. Returns dict + (y,m) keys."""
    ds = xr.open_dataset(nc_path)
    times = ds["time"].values
    src_lon, src_lat = ds["longitude"].values, ds["latitude"].values
    tgt_lon, tgt_lat = ds_aux["longitude"].values, ds_aux["latitude"].values
    tgt_shape = tgt_lon.shape

    key = (src_lon.shape,)
    if key not in idx_cache:
        idx_cache[key] = build_nearest_index(src_lon, src_lat, tgt_lon, tgt_lat)
    idx = idx_cache[key]

    out = {}
    for v in STATE_VARS:
        out[v] = regrid_time(ds[v].values, idx, tgt_shape)

    # Ice drift (siu, siv on the CREG025 grid) -> target grid frame.
    src_basis = local_grid_basis(src_lon, src_lat)
    # The source basis must be sampled at the SAME target cells as the data, so
    # regrid each source-basis component with the same index.
    def rg(field):
        return field.reshape(1, -1)[:, idx].reshape(tgt_shape)
    src_basis_on_tgt = tuple(rg(np.asarray(c)) for c in src_basis)
    tgt_basis = local_grid_basis(tgt_lon, tgt_lat)
    out["siu"], out["siv"] = rotate_to_target(
        out["siu"], out["siv"], src_basis=src_basis_on_tgt,
        tgt_basis=tgt_basis, label="drift")

    ds.close()
    return out, ym_key(times)


def build_datacube_year(year, states, s_keys, forcings, f_index, ds_aux):
    """Assemble one year's (time, var_names, y, x) datacube DataArray."""
    ocean = ds_aux["mask"].values.astype(bool)
    ny, nx = ocean.shape
    months, layers = [], []
    for i, (yy, mm) in enumerate(s_keys):
        if yy != year:
            continue
        if (yy, mm) not in f_index:
            print(f"    ! no ERA5 forcing for {yy}-{mm:02d} -- skipping month")
            continue
        fi = f_index[(yy, mm)]
        cube = np.empty((len(DATACUBE_VARS), ny, nx), dtype=np.float32)
        for vk, vname in enumerate(STATE_VARS):
            cube[vk] = states[vname][i]
        for k, vname in enumerate(FORCING_VARS):
            cube[len(STATE_VARS) + k] = forcings[vname][fi]
        cube[:, ~ocean] = np.nan  # land -> NaN (demo fillna(0)'s this at load)
        layers.append(cube)
        months.append(pd.Timestamp(year=yy, month=mm, day=15))

    if not layers:
        return None
    data = np.stack(layers, axis=0)  # (time, var, y, x)
    return xr.DataArray(
        data, dims=("time", "var_names", "y", "x"),
        coords={"time": months, "var_names": DATACUBE_VARS,
                "longitude": (("y", "x"), ds_aux["longitude"].values),
                "latitude": (("y", "x"), ds_aux["latitude"].values)},
        name="datacube",
    )


def write_forcings_only(forcings, f_keys, years, ds_aux, out_dir, overwrite):
    """One (time, var_names, y, x) forcing cube per year, same channels and
    layout as the forcing half of the training datacube."""
    ocean = ds_aux["mask"].values.astype(bool)
    for yr in (years or sorted({y for y, _ in f_keys})):
        out_path = out_dir / f"era5_datacube_{yr}.nc"
        if out_path.exists() and not overwrite:
            print(f"[{yr}] exists -> {out_path.name} (skip)")
            continue
        idx = [i for i, (y, _) in enumerate(f_keys) if y == yr]
        if not idx:
            print(f"[{yr}] no ERA5 months -- skipped")
            continue
        data = np.stack([forcings[v][idx] for v in FORCING_VARS], axis=1)  # (t, var, y, x)
        data[:, :, ~ocean] = np.nan
        cube = xr.DataArray(
            data, dims=("time", "var_names", "y", "x"),
            coords={"time": [pd.Timestamp(year=yr, month=f_keys[i][1], day=15) for i in idx],
                    "var_names": FORCING_VARS,
                    "longitude": (("y", "x"), ds_aux["longitude"].values),
                    "latitude": (("y", "x"), ds_aux["latitude"].values)},
            name="datacube")
        cube.to_dataset().to_netcdf(out_path, encoding={"datacube": {"zlib": True, "complevel": 4}})
        print(f"[{yr}] wrote {len(idx)} forcing months -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aux-path", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--era5-path",
                    default="data/train_data/data_stream-moda_stepType-avgua.nc")
    ap.add_argument("--nextsim-dir",
                    default="data/train_data/nextsim_opa_monthly")
    ap.add_argument("--out-dir", default="data/train_data/monthly_datacube")
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="subset of years to process (default: all found)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--forcings-only", action="store_true",
                    help="write forcing-only cubes (era5_datacube_YYYY.nc, no ice states) "
                         "for years past neXtSIM's end, e.g. the 2019-2025 hindcast")
    args = ap.parse_args(argv)

    ds_aux = xr.open_dataset(args.aux_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Forcings: computed once over the whole ERA5 series (degree days need it).
    forcings, f_keys = process_forcings(args.era5_path, ds_aux)
    f_index = {ym: i for i, ym in enumerate(f_keys)}

    if args.forcings_only:
        write_forcings_only(forcings, f_keys, args.years, ds_aux, out_dir, args.overwrite)
        return

    nextsim_dir = Path(args.nextsim_dir)
    files = sorted(nextsim_dir.glob("nextsim_opa_icemod_*_monthly.nc"))
    years_avail = [int(p.stem.split("_")[3]) for p in files]
    years = args.years or years_avail
    idx_cache: dict = {}

    print(f"\n[run] processing years: {years}\n")
    for path, yr in zip(files, years_avail):
        if yr not in years:
            continue
        out_path = out_dir / f"monthly_datacube_{yr}.nc"
        if out_path.exists() and not args.overwrite:
            print(f"[{yr}] exists -> {out_path.name} (skip)")
            continue
        print(f"[{yr}] {path.name}")
        states, s_keys = process_states_year(path, ds_aux, idx_cache)
        cube = build_datacube_year(yr, states, s_keys, forcings, f_index, ds_aux)
        if cube is None:
            print(f"[{yr}] no complete months -- nothing written")
            continue
        enc = {"datacube": {"zlib": True, "complevel": 4}}
        cube.to_dataset().to_netcdf(out_path, encoding=enc)
        finite = np.isfinite(cube.values).mean() * 100
        print(f"[{yr}] wrote {cube.sizes['time']} months, "
              f"{finite:.1f}% finite -> {out_path}")

    print(f"\nDone. Datacubes in {out_dir}/  "
          f"(open with xarray.open_mfdataset('.../monthly_datacube_*.nc'))")


if __name__ == "__main__":
    main()
