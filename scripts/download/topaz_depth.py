#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seafloor depth on the model grid, from the TOPAZ4b ocean reanalysis (Copernicus
ARCTIC_MULTIYEAR_PHY_002_003): the deepest of its 40 levels that holds water in one
month of temperature (August 2015), north of 64 N. The product has no static
bathymetry file, so this is the ocean model's own seafloor, resolved only to its levels
(10 m steps to 100 m, 25-50 m to 500 m, coarser below): `depth` is the bottom of the
deepest wet level (the midpoint to the next level), `depth_level` that level's depth.
Each model ocean cell takes the nearest TOPAZ cell that holds water (0.125 deg grid), so
coastlines don't leave gaps; NaN where none is within 30 km (outside TOPAZ), on land
and south of 64 N.

One month, all depths, is about 96 MB streamed. Needs a Copernicus Marine login
(`copernicusmarine login`; never in the repo).

Run:  python scripts/download/topaz_depth.py --selfcheck
      python scripts/download/topaz_depth.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr

from frazil import paths
from scipy.spatial import cKDTree

from frazil.preprocess import level_bounds, lonlat_to_xyz

DATASET, MONTH, LAT_MIN, MAX_KM = "cmems_mod_arc_phy_my_topaz4_P1M", "2015-08-01", 64.0, 30.0


def seafloor(temp, depth):
    """temp (z, y, x) with NaN below the seafloor and on land -> (bottom of the deepest wet
    level, that level's depth), NaN where no level is wet."""
    wet = np.isfinite(temp)
    k = wet.shape[0] - 1 - np.argmax(wet[::-1], axis=0)          # deepest wet level
    any_wet = wet.any(0)
    bottom = level_bounds(depth)[:, 1]
    return np.where(any_wet, bottom[k], np.nan), np.where(any_wet, depth[k], np.nan)


def _selfcheck():
    depth = np.array([0.0, 10.0, 20.0, 50.0])
    t = np.full((4, 1, 3), np.nan)
    t[:2, 0, 0] = 1.0                       # water in levels 0-1: bottom at 15 m
    t[:, 0, 1] = 0.0                        # 0.0 C is water; all levels wet: bottom = last level
    b, lv = seafloor(t, depth)
    assert np.allclose(b[0, :2], [15.0, 50.0]) and np.isnan(b[0, 2]) and lv[0, 0] == 10.0, (b, lv)
    print("[selfcheck] deepest wet level and its bottom; 0.0 C counts as water; land NaN")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=paths.TOPAZ_DEPTH)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return
    import copernicusmarine as cm
    ds = cm.open_dataset(dataset_id=DATASET, variables=["thetao"], minimum_latitude=LAT_MIN,
                         start_datetime=MONTH, end_datetime=MONTH)
    depth = ds.depth.values.astype(float)
    print(f"[topaz depth] {DATASET} {MONTH}: {ds.thetao.nbytes / 1e6:.0f} MB, {len(depth)} levels to {depth[-1]:.0f} m")
    bottom, level = seafloor(ds.thetao.isel(time=0).values, depth)
    lon2, lat2 = np.meshgrid(ds.longitude.values, ds.latitude.values)
    aux = xr.open_dataset(paths.AUX)
    keep = aux.mask.values.astype(bool) & (aux.latitude.values >= LAT_MIN)
    wet = np.isfinite(bottom.ravel())
    dist, idx = cKDTree(lonlat_to_xyz(lon2.ravel()[wet], lat2.ravel()[wet])).query(
        lonlat_to_xyz(aux.longitude.values, aux.latitude.values), workers=-1)
    dist, idx = dist.reshape(keep.shape), idx.reshape(keep.shape)
    near = keep & (dist * 6371.0 <= MAX_KM)                       # chord on the unit sphere ~ angle
    grid = lambda f: np.where(near, f.ravel()[wet][idx], np.nan).astype(np.float32)
    out = xr.Dataset({"depth": (("y", "x"), grid(bottom)), "depth_level": (("y", "x"), grid(level))},
                     attrs={"source": f"TOPAZ4b {DATASET} (doi:10.48670/moi-00007), deepest wet level in {MONTH[:7]}, on the model grid",
                            "units": "m"})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_netcdf(args.out, encoding={k: {"zlib": True} for k in out.data_vars})
    d = out.depth.values
    print(f"-> {args.out}: {np.isfinite(d).sum()} ocean cells; under 50 m {np.mean(d[np.isfinite(d)] < 50):.0%}, "
          f"under 200 m {np.mean(d[np.isfinite(d)] < 200):.0%}; ocean cells north of 64 N with no TOPAZ water: {(keep & ~np.isfinite(d)).sum()}")


if __name__ == "__main__":
    main()
