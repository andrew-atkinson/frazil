#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch monthly CMIP6 forcing for the monthly GenSIM model, from Google Cloud.

Pulls the 4 near-surface fields GenSIM needs -- tas, huss, uas, vas -- for one
model, member and scenario, stitches historical (<=2014) + scenario (>=2015),
subsets the Arctic (lat >= 35, margin below the 40N the preprocessing uses), and
saves one NetCDF. CMIP gives specific humidity (huss) directly, so unlike ERA5
there is no dewpoint/pressure/Magnus step.

Data: the analysis-ready CMIP6 zarr on GCS (gs://cmip6), read anonymously via
gcsfs (already in the environment). The pangeo catalog CSV is used to resolve
each (model, experiment, variable) to its zarr store.

Caveats (see README): CMIP is a *scenario* not a forecast; it is coarse (heavy
regrid to the 12.5 km grid); and it is biased vs the ERA5 atmosphere GenSIM was
trained on, so bias-correction is likely needed before the output is meaningful.

Run:  python cmip_fetch.py --dry-run                  # inspect, no download
      python cmip_fetch.py --start-year 1994 --end-year 2100
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
from frazil import paths

VARS = ["tas", "huss", "uas", "vas"]
CATALOG = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"


def find_zstore(cat, source, member, experiment, variable, grid):
    q = cat[(cat.source_id == source) & (cat.member_id == member) &
            (cat.experiment_id == experiment) & (cat.table_id == "Amon") &
            (cat.variable_id == variable)]
    if grid:
        q = q[q.grid_label == grid]
    if len(q) == 0:
        raise SystemExit(
            f"no CMIP6 store for {source} {member} {experiment} {variable} "
            f"(Amon, grid={grid}). Try a different --model/--member/--grid.")
    # latest version if several
    return sorted(q.zstore.values)[-1]


def open_store(fs, zstore, variable):
    return xr.open_zarr(fs.get_mapper(zstore), consolidated=True)[variable]


def arctic(da, min_lat=35):
    lat = da["lat"]
    return da.isel(lat=np.where(lat.values >= min_lat)[0])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="MPI-ESM1-2-LR", help="CMIP6 source_id")
    ap.add_argument("--member", default="r1i1p1f1")
    ap.add_argument("--scenario", default="ssp245", help="future experiment_id")
    ap.add_argument("--grid", default="gn", help="grid_label (gn/gr/...)")
    ap.add_argument("--start-year", type=int, default=1994)
    ap.add_argument("--end-year", type=int, default=2100)
    ap.add_argument("--out", default=paths.CMIP_FORCING_RAW)
    ap.add_argument("--dry-run", action="store_true",
                    help="open lazily and report, don't download/write")
    args = ap.parse_args(argv)

    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")
    print(f"[cmip] reading catalog ... ({CATALOG})", flush=True)
    cat = __import__("pandas").read_csv(
        CATALOG,
        usecols=["source_id", "member_id", "experiment_id", "table_id",
                 "variable_id", "grid_label", "zstore"])

    das = {}
    for v in VARS:
        hist = open_store(fs, find_zstore(cat, args.model, args.member,
                                          "historical", v, args.grid), v)
        fut = open_store(fs, find_zstore(cat, args.model, args.member,
                                         args.scenario, v, args.grid), v)
        # Filter by integer year (works for datetime64 and cftime; avoids
        # string partial-slicing on the concatenated, possibly non-monotonic axis)
        hist = hist.isel(time=np.where(hist.time.dt.year <= 2014)[0])
        fut = fut.isel(time=np.where(fut.time.dt.year >= 2015)[0])
        da = xr.concat([hist, fut], dim="time").sortby("time")
        yr = da.time.dt.year.values
        da = da.isel(time=np.where((yr >= args.start_year) &
                                   (yr <= args.end_year))[0])
        da = arctic(da)
        # tas/huss carry height=2m, uas/vas height=10m -> conflicting scalar
        # coord on merge; drop it (not needed downstream).
        da = da.drop_vars("height", errors="ignore")
        das[v] = da
        t = da["time"].values
        print(f"  {v:5s} {tuple(da.shape)}  {str(t[0])[:7]}..{str(t[-1])[:7]}  "
              f"lat {float(da.lat.min()):.1f}..{float(da.lat.max()):.1f}", flush=True)

    ds = xr.Dataset(das)
    ds.attrs.update(model=args.model, member=args.member,
                    scenario=f"historical+{args.scenario}",
                    note="Arctic CMIP6 monthly forcing for GenSIM (tas/huss/uas/vas)")
    if args.dry_run:
        print("[cmip] dry run -- nothing written")
        return
    print(f"[cmip] downloading + writing -> {args.out} ...", flush=True)
    ds.to_netcdf(args.out)
    print(f"[cmip] done: {args.out}")


if __name__ == "__main__":
    main()
