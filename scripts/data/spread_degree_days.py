#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rebuild the four degree-day inputs (pdd_month, fdd_month, pdd_year, fdd_year) so
they count the thaw days of months that average below freezing (ISS-007), and write
copies of datacubes with them replaced.

The datacubes' degree-days come from monthly-mean temperature alone. Here each
month's mean temperature (the cube's `tus`) is combined with the typical day-to-day
spread of daily temperature for that calendar month (data/auxiliary/
t2m_daily_spread.nc, from degree_day_spread.py) through
frazil.preprocess.degree_days_gaussian(), which assumes daily temperatures within a
month are normally distributed. Same names, units (K) and threshold, so a model
reads them in the same slots. The trailing 12-month inputs need the 11 months before
the cube's first month: they come from the raw ERA5 monthly file (1994-2018), after
checking it matches the cube's own temperature where both exist.

Prints, per calendar month, the old and new mean pdd_month over ocean and the share
of ocean cells that had no melt and now have some, and writes the new channels'
mean/std (encoder convention) to <out-dir>/degree_day_stats.json.

Run:  python scripts/data/spread_degree_days.py --selfcheck
      python scripts/data/spread_degree_days.py --datacube-dir data/train_data/monthly_datacube_sicqm
          (-> data/train_data/monthly_datacube_sicqm_dd)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.preprocess as P
from frazil import DEGREE, paths


def monthly_temperature(files, spinup_file, tgt_lon, tgt_lat):
    """{(year, month): (y, x)} mean 2 m temperature: the cubes' tus, plus the 11 months
    before the first one from the raw ERA5 monthly file."""
    temp = {}
    for f in files:
        cube = xr.open_dataset(f)["datacube"]
        for t, field in zip(pd.DatetimeIndex(cube["time"].values), cube.sel(var_names="tus").values):
            temp[(t.year, t.month)] = field
    first = min(temp)
    need = [((first[0] * 12 + first[1] - 1 - k) // 12, (first[0] * 12 + first[1] - 1 - k) % 12 + 1) for k in range(11, 0, -1)]
    e = xr.open_dataset(spinup_file)
    tname = "valid_time" if "valid_time" in e.coords else "time"
    have = {(t.year, t.month): i for i, t in enumerate(pd.DatetimeIndex(e[tname].values))}
    missing = [k for k in need if k not in have]
    if missing:
        raise SystemExit(f"spin-up months {missing} not in {spinup_file}")
    lon, lat = np.meshgrid(e["longitude"].values, e["latitude"].values)
    idx = P.build_nearest_index(lon, lat, tgt_lon, tgt_lat)
    for k in need + ([first] if first in have else []):
        temp.setdefault(k, None)
        field = P.regrid_time(e["t2m"].isel({tname: have[k]}).values[None], idx, tgt_lon.shape)[0]
        if k == first:                           # same month from both sources: they must agree
            diff = np.nanmax(np.abs(field - temp[first]))
            print(f"[spin-up] raw ERA5 vs cube tus at {first[0]}-{first[1]:02d}: max difference {diff:.3g} K")
            if diff > 0.01:
                raise SystemExit("the raw ERA5 file doesn't match the cube's tus; spin-up would be inconsistent")
        else:
            temp[k] = field
    return temp


def features_by_month(temp, sigma):
    """The four degree-day inputs for every month in `temp` (consecutive months)."""
    keys = sorted(temp)
    ym = np.array([y * 12 + m for y, m in keys])
    if (np.diff(ym) != 1).any():
        raise SystemExit("monthly temperatures are not consecutive")
    times = pd.to_datetime([f"{y}-{m:02d}-15" for y, m in keys])
    dd = P.degree_days_gaussian(np.stack([temp[k] for k in keys]), times, sigma)
    return {k: i for i, k in enumerate(keys)}, {n: dd[n].values for n in DEGREE}


def replace_degree_days(cube, features, index):
    """cube: (time, var_names, y, x); features: {name: (month, y, x)} indexed by (year, month).
    Keeps the cube's NaN pattern (land) on each replaced channel."""
    rows = [index[(t.year, t.month)] for t in pd.DatetimeIndex(cube["time"].values)]
    out = cube.copy()
    for name in DEGREE:
        old = cube.sel(var_names=name).values
        out.loc[dict(var_names=name)] = np.where(np.isnan(old), np.nan, features[name][rows])
    return out


def _selfcheck():
    thr = P.T2M_THRESHOLD
    temp = {(1999 + (k // 12), k % 12 + 1): np.full((2, 2), thr - 3.0) for k in range(24)}   # every month below freezing
    index, f = features_by_month(temp, np.full((12, 2, 2), 4.0))
    assert np.isnan(f["pdd_year"][:11]).all() and (f["pdd_month"] > 0.5).all()              # thaw days counted
    times = pd.date_range("2000-01-01", periods=12, freq="MS") + pd.Timedelta(days=14)
    cube = xr.DataArray(np.zeros((12, 5, 2, 2), np.float32), dims=("time", "var_names", "y", "x"),
                        coords={"time": times, "var_names": ["tus"] + DEGREE})
    cube.loc[dict(var_names="pdd_month")] = np.where(np.arange(4).reshape(2, 2) == 0, np.nan, 0.0)   # one land cell
    out = replace_degree_days(cube, f, index)
    pm = out.sel(var_names="pdd_month").values
    assert np.isnan(pm[:, 0, 0]).all() and (pm[:, 1, 1] > 0.5).all()                        # land stays NaN
    assert np.isfinite(out.sel(var_names="pdd_year").values[:, 1, 1]).all()                 # 1999 gives the spin-up
    assert (out.sel(var_names="tus").values == 0).all()                                     # other channels untouched
    print("[selfcheck] degree-days from monthly mean + spread replace the cube channels")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacube-dir", default=paths.DATACUBE)
    ap.add_argument("--out-dir", default=None, help="default: <datacube-dir>_dd")
    ap.add_argument("--spread", default=paths.T2M_SPREAD, help="from degree_day_spread.py")
    ap.add_argument("--spinup-era5", default=paths.ERA5_RAW, help="raw ERA5 monthly file for the months before the cubes")
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    sigma = xr.open_dataset(args.spread)["t2m_daily_std"].values
    files = sorted(glob.glob(f"{args.datacube_dir}/*_datacube_*.nc"))
    if not files:
        raise SystemExit(f"no *_datacube_*.nc in {args.datacube_dir}")
    temp = monthly_temperature(files, args.spinup_era5, aux["longitude"].values, aux["latitude"].values)
    index, features = features_by_month(temp, sigma)

    out_dir = args.out_dir or os.path.normpath(args.datacube_dir) + "_dd"
    os.makedirs(out_dir, exist_ok=True)
    acc = {v: [0.0, 0.0, 0] for v in DEGREE}
    change = []
    for f in files:
        cube = xr.open_dataset(f)["datacube"].load()
        new = replace_degree_days(cube, features, index)
        for name in DEGREE:
            x = new.sel(var_names=name).values[:, ocean]
            if np.isnan(x).any():
                raise SystemExit(f"{name}: NaN on ocean in {os.path.basename(f)}")
            x = x.astype(np.float64); acc[name][0] += x.sum(); acc[name][1] += (x * x).sum(); acc[name][2] += x.size
        old_p, new_p = cube.sel(var_names="pdd_month").values[:, ocean], new.sel(var_names="pdd_month").values[:, ocean]
        for i, t in enumerate(pd.DatetimeIndex(cube["time"].values)):
            change.append(dict(month=t.month, old=float(old_p[i].mean()), new=float(new_p[i].mean()),
                               gained_melt=float(((old_p[i] == 0) & (new_p[i] > 0.01)).mean())))
        dst = os.path.join(out_dir, os.path.basename(f))
        if os.path.exists(dst) and not args.overwrite:
            print(f"  exists {os.path.basename(dst)} (skip; --overwrite to rebuild)"); continue
        new.rename("datacube").to_dataset().to_netcdf(dst, encoding={"datacube": {"zlib": True, "complevel": 4}})
        print(f"  {os.path.basename(f)} -> {dst}", flush=True)

    df = pd.DataFrame(change).groupby("month").mean()
    print("\npdd_month over ocean by calendar month (K per day): monthly-mean method, with spread, and the share of "
          "ocean cells that had no melt and now have more than 0.01 K per day")
    print(df.round(3).to_string())
    stats = {v: {"mean": s / c, "std": float(np.sqrt((q - s * s / c) / (c - 1)))} for v, (s, q, c) in acc.items()}
    with open(os.path.join(out_dir, "degree_day_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\n[stats] " + "  ".join(f"{v}: mean {s['mean']:.4g} std {s['std']:.4g}" for v, s in stats.items()))


if __name__ == "__main__":
    main()
