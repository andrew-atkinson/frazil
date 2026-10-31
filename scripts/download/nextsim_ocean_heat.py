#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upper-ocean heat content of the neXtSIM-OPA training simulation, for the Pacific and
Siberian shelf seas (experiments/2026-10-29_pacific_shelf_ocean_heat).

Streams the monthly sea-water temperature (`votemper`, 75 levels) from the SASIP
THREDDS server over OPeNDAP, for the cells north of 64 N between 90 E and 110 W and the
levels in the top 100 m, and keeps only the heat content above freezing (-1.8 C),
rho * c_p * sum(max(T - T_f, 0) * dz), in two layers: 0-20 m (the mixed layer) and
20-100 m (where stored summer heat and Pacific water sit), in MJ m^-2. Land and cells
below the sea floor come back as missing or as exactly 0.0 C and count as no water.

One file per year (data/train_data/nextsim_ocean_heat/ohc_YYYY.nc, native grid with
2-D latitude/longitude); finished years are skipped, so it resumes. About 9 MB is
streamed per month (~2.5 GB for 1995-2018); ~0.1 GB is kept.

Run:  python scripts/download/nextsim_ocean_heat.py --selfcheck
      python scripts/download/nextsim_ocean_heat.py --dry-run
      python scripts/download/nextsim_ocean_heat.py --years 1995-2018
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import xarray as xr

from frazil import paths
from frazil.preprocess import LAYERS, OCEAN_RHO_CP, T_FREEZE, heat_content, layer_weights

SERVER = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/dodsC"
RUN = "meomopendap/extract/SASIP/model-outputs/OPA-neXtSIM_CREG025/OPA-neXtSIM_CREG025-ILBOXE140-MEAN/1m"
PREFIX = "OPA-neXtSIM_CREG025-ILBOXE140"
LAT_MIN, LON_RANGE = 64.0, (90.0, 250.0)


def url(year, month):
    return f"{SERVER}/{RUN}/{year}/{PREFIX}_y{year}m{month:02d}.1m_gridT.nc"


def box_and_levels(ds):
    lat, lon = ds["nav_lat"].values, (ds["nav_lon"].values + 360) % 360
    yy, xx = np.where((lat >= LAT_MIN) & (lon >= LON_RANGE[0]) & (lon <= LON_RANGE[1]))
    bounds = ds["deptht_bounds"].values
    nlev = int((bounds[:, 0] < max(b for _, b in LAYERS.values())).sum())
    return slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1), nlev, bounds[:nlev]


def _selfcheck():
    bounds = np.array([[0, 10], [10, 20], [20, 60], [60, 120]], float)
    temp = np.full((4, 1, 2), T_FREEZE + 1.0)              # 1 K above freezing everywhere
    temp[3, 0, 1] = 0.0                                     # below the sea floor in column 2
    temp[2:, 0, 1] = np.nan
    h = heat_content(temp, bounds)
    one_metre = OCEAN_RHO_CP / 1e6                              # MJ m^-2 per metre per kelvin
    assert np.allclose(h["ohc_0_20"][0], 20 * one_metre)
    assert np.isclose(h["ohc_20_100"][0, 0], 80 * one_metre) and h["ohc_20_100"][0, 1] == 0
    assert np.allclose(layer_weights(bounds, 20, 100), [0, 0, 40, 40])
    assert heat_content(temp, bounds, zero_is_land=False)["ohc_20_100"][0, 1] == 0   # NaN still no water
    print("[selfcheck] heat content by layer; missing and below-floor cells count as no water")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="1995-2018", help="YYYY-YYYY (the simulation covers 1995-2018)")
    ap.add_argument("--out-dir", default=paths.NEXTSIM_OCEAN_HEAT)
    ap.add_argument("--dry-run", action="store_true", help="one month: print the box, levels and sizes, save nothing")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return
    y0, y1 = (int(s) for s in args.years.split("-"))
    first = xr.open_dataset(url(y0, 1))
    ys, xs, nlev, bounds = box_and_levels(first)
    print(f"[ocean heat] box y {ys.start}-{ys.stop - 1}, x {xs.start}-{xs.stop - 1}; top {nlev} levels "
          f"(to {bounds[-1, 1]:.0f} m); {nlev * (ys.stop - ys.start) * (xs.stop - xs.start) * 4 / 1e6:.1f} MB per month")
    if args.dry_run:
        t0 = time.time()
        h = heat_content(first["votemper"].isel(time_counter=0, deptht=slice(0, nlev), y=ys, x=xs).values, bounds)
        print(f"  {y0}-01 in {time.time() - t0:.0f}s: " + ", ".join(f"{k} mean {np.nanmean(v[v > 0]):.0f} MJ/m2" for k, v in h.items()))
        return
    os.makedirs(args.out_dir, exist_ok=True)
    for y in range(y0, y1 + 1):
        dst = os.path.join(args.out_dir, f"ohc_{y}.nc")
        if os.path.exists(dst):
            print(f"  {y}: exists (skip)"); continue
        t0, fields = time.time(), {k: [] for k in LAYERS}
        for m in range(1, 13):
            ds = xr.open_dataset(url(y, m))
            for k, v in heat_content(ds["votemper"].isel(time_counter=0, deptht=slice(0, nlev), y=ys, x=xs).values, bounds).items():
                fields[k].append(v)
        times = np.array([np.datetime64(f"{y}-{m:02d}-15") for m in range(1, 13)])
        out = xr.Dataset({k: (("time", "y", "x"), np.stack(v)) for k, v in fields.items()},
                         coords={"time": times, "latitude": (("y", "x"), first["nav_lat"].values[ys, xs]),
                                 "longitude": (("y", "x"), first["nav_lon"].values[ys, xs])},
                         attrs={"source": f"neXtSIM-OPA {PREFIX} monthly votemper, SASIP THREDDS",
                                "units": "MJ m-2, heat content above -1.8 C"})
        out.to_netcdf(dst + ".part", encoding={k: {"zlib": True, "complevel": 4} for k in LAYERS})
        os.replace(dst + ".part", dst)
        print(f"  {y}: {time.time() - t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
