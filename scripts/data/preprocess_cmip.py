#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess CMIP6 forcing (from scripts/download/cmip.py) onto the GenSIM grid, to drive
future / scenario rollouts.

Takes the 4 CMIP monthly fields (tas, huss, uas, vas), regrids them onto the
512x512 NPS grid, rotates the winds into the grid frame, and derives the four
degree-day features from tas. Produces a per-year forcing datacube with
var_names [tus, huss, uas, vas, pdd_month, fdd_month, pdd_year, fdd_year] --
exactly the forcing/degree-day channels rollout_monthly.py reads. No states
(there is no future truth) and no humidity derivation (CMIP gives huss directly).

Reuses the regrid + rotation + degree-day helpers from preprocess_monthly, so the
geometry is identical to the ERA5 path.

Bias correction: CMIP atmospheres are biased vs the ERA5 forcing GenSIM trained
on, which pushes the model off-distribution. This script corrects each raw field
by an additive per-calendar-month, per-cell delta so its 1995-2014 climatology
matches ERA5's (--no-bias to skip; --bias-years to change the baseline). It's a
mean/delta correction only -- it fixes the seasonal-cycle mean, not the
distribution shape (upgrade to quantile mapping if variance/tails matter).

Run:  python scripts/data/preprocess_cmip.py --selfcheck        # test the math
      python scripts/data/preprocess_cmip.py --cmip data/train_data/cmip_forcing.nc
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

import frazil.preprocess as P

RAW = ["tus", "huss", "uas", "vas"]                 # bias-corrected fields
DEGREE = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
OUT_VARS = RAW + DEGREE                              # what rollout_monthly reads


def monthly_bias(cmip, months, base, ref_clim):
    """Additive per-calendar-month bias correction (mutates & returns cmip).

    For each calendar month m, shift every year's field by the difference between
    CMIP's baseline-period climatology and ERA5's (ref_clim[m-1]), so the corrected
    CMIP baseline climatology matches ERA5. ref_clim may be NaN on land -> that
    cell's bias becomes 0 (no correction). Pure/no-IO so it can be self-checked.

    ponytail: mean-delta ('delta change') correction only -- fixes the mean
    seasonal cycle, not the distribution shape. Upgrade to quantile mapping if the
    variance/tails matter for the rollout.
    """
    for m in range(1, 13):
        sel = months == m
        if not sel.any():
            continue
        base_sel = sel & base
        clim = cmip[base_sel if base_sel.any() else sel].mean(axis=0)
        bias = np.nan_to_num(clim - ref_clim[m - 1])   # land (NaN ref) -> 0
        cmip[sel] -= bias
    return cmip


def bias_correct(reg, months, years, ref_dir, baseline):
    """Correct the 4 raw CMIP fields in `reg` against the on-grid ERA5 datacube
    over the baseline years (inclusive). Modifies reg in place."""
    y0, y1 = baseline
    files = sorted(f for f in glob.glob(f"{ref_dir}/*_datacube_*.nc")
                   if y0 <= int(f.split("_")[-1].split(".")[0]) <= y1)
    if not files:
        raise SystemExit(f"no ERA5 datacube files for {y0}-{y1} in {ref_dir}")
    ref = xr.open_mfdataset(files, combine="by_coords")["datacube"].sel(
        var_names=RAW).load()
    ref_mon = ref["time"].dt.month.values
    ref_arr = ref.transpose("time", "var_names", "y", "x").values  # (T,4,H,W), land NaN
    base = (years >= y0) & (years <= y1)
    if not base.any():
        raise SystemExit(f"no CMIP months in baseline {y0}-{y1}")
    print(f"[bias] baseline {y0}-{y1}: {int(base.sum())} CMIP vs {ref_arr.shape[0]} "
          f"ERA5 months; per-month additive correction", flush=True)
    for vi, v in enumerate(RAW):
        ref_clim = np.stack([np.nanmean(ref_arr[ref_mon == m, vi], axis=0)
                             for m in range(1, 13)])          # (12,H,W)
        monthly_bias(reg[v], months, base, ref_clim)
        if v == "huss":
            np.clip(reg[v], 0, None, out=reg[v])              # humidity stays >= 0


def _parse_years(s):
    a, b = s.split("-")
    return int(a), int(b)


def _selfcheck():
    """CMIP = ERA5 climatology + a per-month offset; after correction the
    baseline climatology must equal ERA5 again."""
    rng = np.random.default_rng(0)
    H = W = 4
    months = np.tile(np.arange(1, 13), 3)
    years = np.repeat([2000, 2001, 2002], 12)
    base = years <= 2001
    ref_clim = rng.random((12, H, W))
    offset = rng.random((12, H, W)) * 5
    cmip = np.stack([ref_clim[m - 1] + offset[m - 1] for m in months]).astype(float)
    monthly_bias(cmip, months, base, ref_clim.copy())
    for m in range(1, 13):
        sel = (months == m) & base
        assert np.allclose(cmip[sel].mean(0), ref_clim[m - 1], atol=1e-9), m
    print("[selfcheck] monthly_bias OK")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmip", default="data/train_data/cmip_forcing.nc")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--out-dir", default="data/train_data/cmip_datacube")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2100)
    ap.add_argument("--bias-ref", default="data/train_data/monthly_datacube",
                    help="on-grid ERA5 datacube to bias-correct against")
    ap.add_argument("--bias-years", default="1995-2014",
                    help="baseline overlap YYYY-YYYY (historical CMIP vs ERA5)")
    ap.add_argument("--no-bias", action="store_true", help="skip bias correction")
    ap.add_argument("--selfcheck", action="store_true", help="run the unit check and exit")
    args = ap.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return

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

    # Rotate winds (geographic east/north -> target grid frame) BEFORE bias
    # correction, so both CMIP and ERA5 winds are in the same grid frame.
    tgt_basis = P.local_grid_basis(tgt_lon, tgt_lat)
    reg["uas"], reg["vas"] = P.rotate_to_target(
        reg["uas"], reg["vas"], src_basis=None, tgt_basis=tgt_basis, label="cmip-wind")

    # Bias-correct the raw fields vs ERA5 -- must precede degree-days, which are
    # derived from the (corrected) temperature.
    if not args.no_bias:
        bias_correct(reg, months, years, args.bias_ref, _parse_years(args.bias_years))
    else:
        print("[bias] SKIPPED (--no-bias): raw CMIP is off-distribution vs ERA5")

    # Degree days from (corrected) tas (rolling; needs the pre-start lead-in).
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
        # Self-describing stamp so freerun_monthly can report the real status.
        da.attrs["bias_corrected"] = "none" if args.no_bias else args.bias_years
        da.attrs["bias_ref"] = "" if args.no_bias else args.bias_ref
        out = f"{args.out_dir}/cmip_datacube_{yr}.nc"
        da.to_dataset().to_netcdf(out, encoding={"datacube": {"zlib": True, "complevel": 4}})
        finite = np.isfinite(cube).mean() * 100
        print(f"  {yr}: {len(sel)} months, {finite:.0f}% finite -> {out}", flush=True)

    tag = "raw (NOT bias-corrected)" if args.no_bias else f"bias-corrected vs ERA5 {args.bias_years}"
    print(f"\nDone -> {args.out_dir}/  (forcing datacube; {tag})")


if __name__ == "__main__":
    main()
