#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copernicus neXtSIM sea-ice reanalysis (ARCTIC_MULTIYEAR_PHY_ICE_002_016; ISS-022),
monthly, onto the model grid: 3 km fields -> 4x4 block mean -> nearest neighbour at the
block centres (averaging longitudes breaks at the dateline). NaN outside its domain
(it stops short of Hudson Bay, the Labrador and Bering seas).

Stand-alone neXtSIM forced by ERA5 and the TOPAZ4b ocean; assimilates passive-microwave
concentration, and CS2SMOS winter thickness from October 2010.

One file per variable, data/train_data/nextsim_reanalysis_<name>_monthly.nc, with
names as in the training data (siconc -> sic, sithick -> sit, sisnthick -> snt).
Needs a Copernicus Marine login (`copernicusmarine login`; never in the repo).

Run:  python scripts/download/nextsim_reanalysis.py --variables sithick
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import xarray as xr

from frazil import paths
from frazil.preprocess import build_nearest_index, regrid_time

DATASET = "cmems_mod_arc_phy_my_nextsim_P1M-m"
NAMES = {"siconc": "sic", "sithick": "sit", "sisnthick": "snt"}
BLOCK, CHUNK = 4, 50


def main(argv=None):
    import copernicusmarine as cm
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variables", nargs="+", default=["sithick"], choices=sorted(NAMES))
    args = ap.parse_args(argv)
    aux = xr.open_dataset(paths.AUX)
    for var in args.variables:
        out = paths.NEXTSIM_REANALYSIS.format(var=NAMES[var])
        if os.path.exists(out):
            print(f"{out}: exists (skip)"); continue
        ds = cm.open_dataset(dataset_id=DATASET, dataset_part="originalGrid", variables=[var])
        print(f"[reanalysis] {var}: {dict(ds.sizes)}; attrs {ds[var].attrs.get('long_name')!r} [{ds[var].attrs.get('units')}]")
        ny, nx = ds.sizes["y"] // BLOCK, ds.sizes["x"] // BLOCK
        ctr = lambda a: a.isel(y=slice(BLOCK // 2, BLOCK * ny, BLOCK), x=slice(BLOCK // 2, BLOCK * nx, BLOCK)).values
        lat, lon = ctr(ds.latitude), ctr(ds.longitude)
        ok = np.isfinite(lat) & np.isfinite(lon)
        idx = build_nearest_index(lon[ok], lat[ok], aux.longitude.values, aux.latitude.values)
        coarse = lambda a: a.coarsen(y=BLOCK, x=BLOCK, boundary="trim").mean()
        fields = []
        for i in range(0, ds.sizes["time"], CHUNK):
            t0 = time.time()
            f = coarse(ds[var].isel(time=slice(i, i + CHUNK))).values[:, ok]
            fields.append(regrid_time(f, idx, aux.mask.shape))
            print(f"  months {i}-{i + len(f) - 1}: {time.time() - t0:.0f}s", flush=True)
        xr.Dataset({NAMES[var]: (("time", "y", "x"), np.concatenate(fields).astype(np.float32))},
                   coords={"time": ds.time.values},
                   attrs={"source": f"{DATASET} {var}, 4x4 block mean, nearest onto the model grid",
                          "long_name": str(ds[var].attrs.get("long_name", "")), "units": str(ds[var].attrs.get("units", ""))}
                   ).to_netcdf(out + ".part", encoding={NAMES[var]: {"zlib": True, "complevel": 4}})
        os.replace(out + ".part", out)
        print("wrote", out)


if __name__ == "__main__":
    main()
