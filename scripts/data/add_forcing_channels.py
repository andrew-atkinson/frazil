#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Append extra ERA5 forcing channels to existing datacubes, on the model grid.

Adds any of:
  sst    sea-surface temperature (K)               -- an ocean signal; see the
                                                      leak warning below
  ssrd   surface solar radiation downwards (W m-2) -- the main summer melt driver
  strd   surface thermal radiation downwards (W m-2)

ERA5 monthly means store radiation as the mean DAILY accumulation (J m-2), so it
is divided by 86400 to get a mean flux in W m-2. SST is NaN over land, so it is
regridded from the nearest ERA5 cell that has a value. Same nearest-neighbour
geometry as preprocess_monthly.py.

Works on any folder of *_datacube_*.nc (the training datacube, the QM targets,
the 2019-2025 forcing-only extension): each file is copied to --out-dir with
the new channels appended to var_names. Also writes each new variable's
mean/std (over time and ocean cells, ddof=1 -- the encoder convention of
estimate_normalization_monthly.py) to <out-dir>/extra_forcing_stats.json.

LEAK WARNING (sst): ERA5 does not simulate SST; it prescribes it from satellite
analyses (HadISST2, then OSTIA), and sets it to the freezing point wherever
those analyses have sea ice. So "SST at freezing" is close to an observed ice
mask, and a model given SST at month t+1 can learn to read the observed ice
edge. Check with --leak-check before training on it.

Run:  python scripts/data/add_forcing_channels.py --selfcheck
      python scripts/data/add_forcing_channels.py --era5-dir data/train_data/era5_extra_1995_2025 \\
          --datacube-dir data/train_data/monthly_datacube --out-dir data/train_data/monthly_datacube_era5x
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import frazil.preprocess as P
from frazil import paths

PER_DAY = {"ssrd", "strd"}          # mean daily accumulations -> divide by 86400


def load_era5(era5_dir, names):
    """{name: (time, lat, lon) DataArray} from the NetCDFs in era5_dir. Each
    variable keeps its own time axis: ERA5 stamps accumulated fields (radiation)
    at 06:00 and instantaneous ones (SST) at 00:00, so merging them would
    interleave half-empty months. Match by (year, month) instead."""
    out = {}
    for f in sorted(glob.glob(f"{era5_dir}/*.nc")):
        ds = xr.open_dataset(f)
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        for v in names:
            if v in ds and v not in out:
                da = ds[v].rename({tname: "time"})
                out[v] = da / 86400.0 if v in PER_DAY else da
    missing = [v for v in names if v not in out]
    if missing:
        raise SystemExit(f"{missing} not found in the .nc files in {era5_dir}")
    return out


def month_index(da):
    """(year, month) -> time index for one variable."""
    return {(t.year, t.month): i for i, t in enumerate(pd.DatetimeIndex(da["time"].values))}


def regridder(valid, lon1d, lat1d, tgt_lon, tgt_lat):
    """Nearest-neighbour index into the flattened source grid, restricted to
    source cells marked valid (so coastal SST never picks a land NaN)."""
    src_lon, src_lat = np.meshgrid(lon1d, lat1d)
    valid = np.asarray(valid, bool).reshape(-1)
    sub = P.build_nearest_index(src_lon.reshape(-1)[valid], src_lat.reshape(-1)[valid],
                                tgt_lon, tgt_lat)
    return np.flatnonzero(valid)[sub]


def append_channels(cube, extra):
    """cube: (time, var_names, y, x) DataArray; extra: {name: (time, y, x)}."""
    add = xr.DataArray(np.stack([extra[v] for v in extra], axis=1),
                       dims=cube.dims,
                       coords={**{k: v for k, v in cube.coords.items() if k != "var_names"},
                               "var_names": list(extra)})
    return xr.concat([cube, add], dim="var_names")


def _selfcheck():
    t = pd.date_range("2000-01-15", periods=2, freq="MS")
    cube = xr.DataArray(np.zeros((2, 2, 3, 3), np.float32), dims=("time", "var_names", "y", "x"),
                        coords={"time": t, "var_names": ["a", "b"]}, name="datacube")
    out = append_channels(cube, {"sst": np.ones((2, 3, 3)), "ssrd": 2 * np.ones((2, 3, 3))})
    assert list(out.var_names.values) == ["a", "b", "sst", "ssrd"]
    assert float(out.sel(var_names="ssrd").sum()) == 36.0
    lon, lat = np.array([0.0, 1.0]), np.array([80.0, 81.0])
    field = np.array([[np.nan, 5.0], [7.0, np.nan]])        # NaN = land
    idx = regridder(np.isfinite(field), lon, lat, np.array([[0.0]]), np.array([[80.0]]))
    assert np.isfinite(field.reshape(-1)[idx]).all()           # never picks land
    print("[selfcheck] append_channels / regridder OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era5-dir", default=paths.ERA5_EXTRA_RAW)
    ap.add_argument("--datacube-dir", default=paths.DATACUBE)
    ap.add_argument("--out-dir", default=None, help="default: <datacube-dir>_era5x")
    ap.add_argument("--vars", nargs="+", default=["sst", "ssrd", "strd"])
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--leak-check", action="store_true",
                    help="report how closely 'SST at freezing' matches the NSIDC ice mask, then exit")
    ap.add_argument("--obs", default=paths.NSIDC0051_GRID)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    tgt_lon, tgt_lat = aux["longitude"].values, aux["latitude"].values
    ca = aux["cell_area"].values / 1e6
    e = load_era5(args.era5_dir, args.vars)
    ekey = {v: month_index(e[v]) for v in args.vars}
    first = e[args.vars[0]]
    lon1d, lat1d = first["longitude"].values, first["latitude"].values
    # valid = has a value at ANY time: land is NaN throughout, sea ice is not land
    idx = {v: regridder(e[v].notnull().any("time").values, lon1d, lat1d, tgt_lon, tgt_lat)
           for v in args.vars}
    span = sorted(ekey[args.vars[0]])
    print(f"[era5] {args.vars}  {span[0][0]}-{span[0][1]:02d}..{span[-1][0]}-{span[-1][1]:02d}  from {args.era5_dir}")

    if args.leak_check:
        if "sst" not in args.vars:
            raise SystemExit("--leak-check needs sst in --vars")
        o = xr.open_mfdataset(sorted(glob.glob(f"{args.obs}/*.nc")), combine="by_coords")
        o = o[[n for n in ("datacube", "state") if n in o.data_vars][0]].sel(var_names="sic")
        rows = []
        for i, t in enumerate(pd.DatetimeIndex(o["time"].values)):
            if (t.year, t.month) not in ekey["sst"]:
                continue
            sst = P.regrid_time(e["sst"].isel(time=ekey["sst"][(t.year, t.month)]).values[None],
                                idx["sst"], ocean.shape)[0]
            ice_obs = (np.nan_to_num(o.isel(time=i).values) > 0.15) & ocean
            frozen = (sst < 271.5) & ocean            # at/below seawater freezing point
            km = lambda b: float((ca * b).sum()) / 1e6
            rows.append(dict(month=t.month, obs_ice=km(ice_obs), sst_frozen=km(frozen),
                             disagree=km(frozen ^ ice_obs)))
        df = pd.DataFrame(rows).groupby("month").mean()
        print("\nSST-at-freezing vs NSIDC ice mask (M km^2, monthly means):")
        print(df.round(2).to_string())
        print(f"\nmean disagreement {df.disagree.mean():.2f} M km^2 vs mean ice area "
              f"{df.obs_ice.mean():.2f} -- small disagreement = SST is effectively the obs ice mask")
        return

    out_dir = args.out_dir or os.path.normpath(args.datacube_dir) + "_era5x"
    os.makedirs(out_dir, exist_ok=True)
    acc = {v: [0.0, 0.0, 0] for v in args.vars}      # sum, sum of squares, count (ocean)
    for f in sorted(glob.glob(f"{args.datacube_dir}/*_datacube_*.nc")):
        dst = os.path.join(out_dir, os.path.basename(f))
        cube = xr.open_dataset(f)["datacube"].load()
        months = pd.DatetimeIndex(cube["time"].values)
        miss = [f"{v} {t:%Y-%m}" for v in args.vars for t in months if (t.year, t.month) not in ekey[v]]
        if miss:
            raise SystemExit(f"{os.path.basename(f)}: no ERA5 for {miss}")
        extra = {}
        for v in args.vars:
            sel = [ekey[v][(t.year, t.month)] for t in months]
            fld = P.regrid_time(e[v].isel(time=sel).values, idx[v], ocean.shape)
            if np.isnan(fld[:, ocean]).any():
                raise SystemExit(f"{v}: NaN on ocean cells in {os.path.basename(f)} -- ERA5 leaves "
                                 f"{v} empty somewhere (e.g. under sea ice); decide a fill before training")
            fld[:, ~ocean] = np.nan
            extra[v] = fld
            x = fld[:, ocean].astype(np.float64)
            acc[v][0] += x.sum(); acc[v][1] += (x * x).sum(); acc[v][2] += x.size
        if os.path.exists(dst) and not args.overwrite:
            print(f"  exists {os.path.basename(dst)} (skip; --overwrite to rebuild)")
            continue
        new = append_channels(cube, extra).rename("datacube")
        new.to_dataset().to_netcdf(dst, encoding={"datacube": {"zlib": True, "complevel": 4}})
        print(f"  {os.path.basename(f)} + {args.vars} -> {dst}")

    stats = {v: {"mean": s / n, "std": float(np.sqrt((q - s * s / n) / (n - 1)))}
             for v, (s, q, n) in acc.items()}
    with open(os.path.join(out_dir, "extra_forcing_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\n[stats] " + "  ".join(f"{v}: mean {s['mean']:.4g} std {s['std']:.4g}" for v, s in stats.items()))
    print(f"-> {out_dir}/extra_forcing_stats.json")


if __name__ == "__main__":
    main()
