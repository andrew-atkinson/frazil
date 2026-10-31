#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upper-ocean heat content of the TOPAZ4b Arctic ocean reanalysis (Copernicus
ARCTIC_MULTIYEAR_PHY_002_003, the ocean that forces the Copernicus neXtSIM sea-ice
reanalysis), for the same Pacific and Siberian shelf box as nextsim_ocean_heat.py:
north of 64 N between 90 E and 110 W, top 100 m (16 levels).

The same heat content above freezing, in the same two layers (0-20 m, 20-100 m), plus
sea-surface temperature and mixed-layer depth, put straight onto the model grid
(nearest neighbour; NaN outside the box). TOPAZ comes on a 0.125 deg grid; level
bounds are the midpoints between its depths. Unlike NEMO output, 0.0 C is water here.

The monthly data is stored in 50-month chunks, so it is read 50 months at a time
(~2 min each, ~20 min for 1995-2025). One file per year,
data/train_data/topaz_ocean_heat/ohc_YYYY.nc; finished years are skipped. Needs a
Copernicus Marine login (`copernicusmarine login`; never in the repo).

Run:  python scripts/download/topaz_ocean_heat.py --years 1995-2025
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

from frazil import paths
from frazil.preprocess import LAYERS, build_nearest_index, heat_content, regrid_time

DATASET = "cmems_mod_arc_phy_my_topaz4_P1M"
CHUNK = 50
LAT_MIN = 64.0


def level_bounds(depth):
    """[top, bottom] per level: midpoints between level depths, from the surface to the last level."""
    mid = (depth[1:] + depth[:-1]) / 2
    return np.stack([np.r_[0.0, mid], np.r_[mid, depth[-1]]], axis=1)


def main(argv=None):
    import copernicusmarine as cm
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="1995-2025")
    ap.add_argument("--out-dir", default=paths.TOPAZ_OCEAN_HEAT)
    args = ap.parse_args(argv)
    y0, y1 = (int(s) for s in args.years.split("-"))
    os.makedirs(args.out_dir, exist_ok=True)
    done = lambda y: os.path.exists(os.path.join(args.out_dir, f"ohc_{y}.nc"))

    ds = cm.open_dataset(dataset_id=DATASET, variables=["thetao", "mlotst"],
                         minimum_latitude=LAT_MIN, maximum_depth=max(b for _, b in LAYERS.values()) + 5)
    ix = np.where((ds.longitude.values >= 90) | (ds.longitude.values <= -110))[0]
    lon2, lat2 = np.meshgrid(ds.longitude.values[ix], ds.latitude.values)
    aux = xr.open_dataset(paths.AUX)
    alon = (aux.longitude.values + 360) % 360
    in_box = aux.mask.values.astype(bool) & (aux.latitude.values >= LAT_MIN) & (alon >= 90) & (alon <= 250)
    idx = build_nearest_index(lon2.ravel(), lat2.ravel(), aux.longitude.values, aux.latitude.values)
    bounds = level_bounds(ds.depth.values.astype(float))
    times = pd.DatetimeIndex(ds.time.values)
    print(f"[topaz heat] {len(ix)} x {ds.sizes['latitude']} source cells, {len(bounds)} levels to "
          f"{bounds[-1, 1]:.0f} m -> {in_box.sum()} model cells; years {y0}-{y1}")

    pending = {}                                   # year -> {month: {var: (y, x)}}
    for c0 in range(0, len(times), CHUNK):
        block = times[c0:c0 + CHUNK]
        want = [i for i, t in enumerate(block) if y0 <= t.year <= y1 and not done(t.year)]
        if not want:
            continue
        t0 = time.time()
        sub = ds.isel(time=slice(c0, c0 + CHUNK), longitude=ix)
        temp, mld = sub.thetao.values, sub.mlotst.values
        for i in want:
            f = heat_content(temp[i], bounds, zero_is_land=False)
            f.update(sst=temp[i, 0], mld=mld[i])
            pending.setdefault(block[i].year, {})[block[i].month] = {
                k: np.where(in_box, regrid_time(v[None], idx, in_box.shape)[0], np.nan) for k, v in f.items()}
        for y in sorted(pending):
            if len(pending[y]) < 12:
                continue
            months = pending.pop(y)
            out = xr.Dataset({k: (("time", "y", "x"), np.stack([months[m][k] for m in range(1, 13)]).astype(np.float32))
                              for k in months[1]},
                             coords={"time": pd.date_range(f"{y}-01-01", periods=12, freq="MS")},
                             attrs={"source": f"TOPAZ4b {DATASET} (doi:10.48670/moi-00007), on the model grid",
                                    "units": "ohc_*: MJ m-2 above -1.8 C; sst: C; mld: m"})
            dst = os.path.join(args.out_dir, f"ohc_{y}.nc")
            out.to_netcdf(dst + ".part", encoding={k: {"zlib": True, "complevel": 4} for k in out.data_vars})
            os.replace(dst + ".part", dst)
            print(f"  {y} -> {dst}", flush=True)
        print(f"  block {block[0]:%Y-%m}..{block[-1]:%Y-%m}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
