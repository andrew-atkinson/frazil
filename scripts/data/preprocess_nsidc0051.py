#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regrid NSIDC-0051 monthly sea-ice concentration onto the GenSIM 512x512 grid, to
(a) bias-correct the neXtSIM `sic` training targets against observations and
(b) validate the projection spatially past 2018.

Handles NSIDC-0051 v2 quirks: the concentration variable is sensor-named
(F11_ICECON / F13_ / F17_ ...); values 0-1 are concentration; flags are byte/250
(1.004 = pole hole -> set to 1.0 full ice; 1.012 = coast, 1.016 = land -> NaN).
The source grid is EPSG:3411 x/y in metres (no lat/lon), so we project it with
pyproj, then nearest-neighbour regrid with preprocess_monthly's helpers.

Output: one file per year, var_names=["sic"], same layout as monthly_datacube --
so `ice_area.py data/obs/nsidc0051_grid --month 9 --nsidc data/obs/nsidc` validates
the regrid against NSIDC's own extent number.

Run:  python scripts/data/preprocess_nsidc0051.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import xarray as xr

import frazil.preprocess as P

POLE = 251 / 250.0  # 1.004 flag = pole hole (unobserved central Arctic -> full ice)


def clean_conc(a):
    """NSIDC ICECON -> concentration in [0,1]; pole hole -> 1.0; land/coast -> NaN."""
    a = a.astype(np.float32)
    out = np.where(a <= 1.0, a, np.nan)                 # >1 flags (coast/land) -> NaN
    out = np.where(np.abs(a - POLE) < 1e-3, 1.0, out)   # pole hole -> full ice
    return out


def icecon_var(ds):
    v = [k for k in ds.data_vars if k.endswith("ICECON")]
    if not v:
        raise SystemExit("no *_ICECON variable in file")
    return v[0]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="data/obs/nsidc0051")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--out-dir", default="data/obs/nsidc0051_grid")
    ap.add_argument("--start-year", type=int, default=1995)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args(argv)

    # North hemisphere only -- the download also pulls S25km (Antarctic), a
    # different grid we don't want for Arctic sea ice.
    files = sorted(f for f in glob.glob(f"{args.in_dir}/*.nc")
                   if "_N25km_" in os.path.basename(f))
    if not files:
        raise SystemExit(f"no NSIDC-0051 N25km files in {args.in_dir}")
    by_month = defaultdict(list)  # 'YYYYMM' -> [files] (sensors overlap -> average)
    for f in files:
        m = re.search(r"_(\d{6})_v", os.path.basename(f))
        if m:
            by_month[m.group(1)].append(f)
    print(f"[nsidc0051] {len(files)} files, {len(by_month)} distinct months", flush=True)

    ds_aux = xr.open_dataset(args.aux)
    tgt_lon, tgt_lat = ds_aux["longitude"].values, ds_aux["latitude"].values
    tgt_shape = tgt_lon.shape
    ocean = ds_aux["mask"].values.astype(bool)
    ca_km2 = ds_aux["cell_area"].values / 1e6

    # Project the NSIDC EPSG:3411 grid to lat/lon and build the nearest index once.
    from pyproj import Transformer
    ds0 = xr.open_dataset(files[0])
    X, Y = np.meshgrid(ds0["x"].values, ds0["y"].values)
    tf = Transformer.from_crs("EPSG:3411", "EPSG:4326", always_xy=True)
    src_lon, src_lat = tf.transform(X, Y)
    print(f"[nsidc0051] src grid {X.shape} (lat {src_lat.min():.1f}..{src_lat.max():.1f}) "
          f"-> target {tgt_shape}", flush=True)
    idx = P.build_nearest_index(src_lon, src_lat, tgt_lon, tgt_lat)

    os.makedirs(args.out_dir, exist_ok=True)
    for yr in range(args.start_year, args.end_year + 1):
        months, fields = [], []
        for mo in range(1, 13):
            key = f"{yr}{mo:02d}"
            if key not in by_month:
                continue
            per_sensor = []
            for f in by_month[key]:
                d = xr.open_dataset(f)
                per_sensor.append(clean_conc(d[icecon_var(d)].isel(time=0).values))
            if len(per_sensor) == 1:
                conc = per_sensor[0]
            else:  # overlapping sensors: average, ignoring all-NaN cells quietly
                with np.errstate(invalid="ignore"):
                    conc = np.nanmean(per_sensor, axis=0)         # (ys, xs)
            g = P.regrid_time(conc[None], idx, tgt_shape)[0]      # (YT, XT)
            months.append(mo)
            fields.append(np.where(ocean, g, np.nan))
        if not months:
            continue
        cube = np.stack(fields)[:, None]                          # (t, 1, y, x)
        da = xr.DataArray(
            cube, dims=("time", "var_names", "y", "x"),
            coords={"var_names": ["sic"],
                    "time": [np.datetime64(f"{yr}-{m:02d}-15") for m in months]},
            name="datacube")
        out = f"{args.out_dir}/nsidc0051_grid_{yr}.nc"
        da.to_dataset().to_netcdf(out, encoding={"datacube": {"zlib": True, "complevel": 4}})
        # Sanity: September extent should match NSIDC's ~4-5 M km^2.
        sep = next((i for i, m in enumerate(months) if m == 9), None)
        tag = ""
        if sep is not None:
            sic = np.nan_to_num(fields[sep])
            ext = float(((sic > 0.15) * ocean * ca_km2).sum())
            tag = f"  Sep extent {ext/1e6:.2f} M km²"
        print(f"  {yr}: {len(months)} months -> {out}{tag}", flush=True)

    print(f"\nDone -> {args.out_dir}/  (obs sic on the GenSIM grid)")
    print("validate: python scripts/evaluate/ice_area.py data/train_data/monthly_datacube "
          f"{args.out_dir} --month 9 --nsidc data/obs/nsidc")


if __name__ == "__main__":
    main()
