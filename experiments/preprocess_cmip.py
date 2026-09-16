#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess CMIP6 forcing (from download/cmip.py) onto the GenSIM grid, to drive
future / scenario rollouts.

Takes the 4 CMIP monthly fields (tas, huss, uas, vas), regrids them onto the
512x512 NPS grid, rotates the winds into the grid frame, and derives the four
degree-day features from tas. Produces a per-year forcing datacube with
var_names [tus, huss, uas, vas, pdd_month, fdd_month, pdd_year, fdd_year] --
exactly the forcing/degree-day channels rollout_monthly.py reads. No states
(there is no future truth) and no humidity derivation (CMIP gives huss directly).

Reuses the regrid + rotation + degree-day helpers from preprocess_monthly, so the
geometry is identical to the ERA5 path.

*** Bias warning ***  CMIP atmospheres are biased vs the ERA5 forcing GenSIM was
trained on. This script only puts CMIP on the grid; it does NOT bias-correct.
Feeding raw CMIP forcing pushes the model off-distribution -- assess/correct the
bias (vs ERA5 over 1995-2014) before trusting the rollout.

Run:  python experiments/preprocess_cmip.py --cmip data/train_data/cmip_forcing.nc
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import preprocess_monthly as P  # build_nearest_index, local_grid_basis, rotate_to_target, regrid_time, degree_days

DEGREE = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
OUT_VARS = ["tus", "huss", "uas", "vas"] + DEGREE  # what rollout_monthly reads


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmip", default="data/train_data/cmip_forcing.nc")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--out-dir", default="data/train_data/cmip_datacube")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2100)
    args = ap.parse_args(argv)

    ds_aux = xr.open_dataset(args.aux)
    tgt_lon, tgt_lat = ds_aux["longitude"].values, ds_aux["latitude"].values
    tgt_shape = tgt_lon.shape
    ocean = ds_aux["mask"].values.astype(bool)

    cmip = xr.open_dataset(args.cmip)
    lon1d, lat1d = cmip["lon"].values, cmip["lat"].values
    src_lon, src_lat = np.meshgrid(lon1d, lat1d)  # (nlat, nlon)
    times = cmip["time"].values
    years = cmip["time"].dt.year.values   # .dt handles cftime and datetime64
    months = cmip["time"].dt.month.values

    print(f"[cmip] {cmip.sizes['time']} months, src grid {src_lon.shape} "
          f"-> target {tgt_shape}", flush=True)
    idx = P.build_nearest_index(src_lon, src_lat, tgt_lon, tgt_lat)

    # Regrid the 4 raw fields (nearest-neighbour), full time series.
    reg = {}
    for name, cv in [("tus", "tas"), ("huss", "huss"), ("uas", "uas"), ("vas", "vas")]:
        reg[name] = P.regrid_time(cmip[cv].values, idx, tgt_shape).astype(np.float32)

    # Rotate winds (geographic east/north -> target grid frame).
    tgt_basis = P.local_grid_basis(tgt_lon, tgt_lat)
    reg["uas"], reg["vas"] = P.rotate_to_target(
        reg["uas"], reg["vas"], src_basis=None, tgt_basis=tgt_basis, label="cmip-wind")

    # Degree days from tas (rolling; needs the pre-start lead-in in the CMIP file).
    dd = P.degree_days(reg["tus"], times)
    for k, v in dd.items():
        reg[k] = v.values.astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    for yr in range(args.start_year, args.end_year + 1):
        sel = np.where(years == yr)[0]
        if len(sel) == 0:
            continue
        cube = np.stack([reg[v][sel] for v in OUT_VARS], axis=1)  # (t, var, y, x)
        cube[:, :, ~ocean] = np.nan
        da = xr.DataArray(
            cube, dims=("time", "var_names", "y", "x"),
            coords={"var_names": OUT_VARS,
                    "time": [np.datetime64(f"{yr}-{m:02d}-15") for m in months[sel]]},
            name="datacube")
        out = f"{args.out_dir}/cmip_datacube_{yr}.nc"
        da.to_dataset().to_netcdf(out, encoding={"datacube": {"zlib": True, "complevel": 4}})
        finite = np.isfinite(cube).mean() * 100
        print(f"  {yr}: {len(sel)} months, {finite:.0f}% finite -> {out}", flush=True)

    print(f"\nDone -> {args.out_dir}/  "
          f"(forcing datacube; BIAS-CORRECT vs ERA5 before trusting rollouts)")


if __name__ == "__main__":
    main()
