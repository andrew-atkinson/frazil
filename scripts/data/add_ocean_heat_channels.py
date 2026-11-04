#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Append the training simulation's own upper-ocean heat to the datacubes as two forcing
channels, ohc_0_20 and ohc_20_100 (MJ m^-2 above -1.8 C, from nextsim_ocean_heat.py:
neXtSIM-OPA, north of 64 N between 90 E and 110 W), on the model grid.

For experiments/2026-10-30_summer_melt: the Pacific-shelf melt error sits over the shelf
break and slope, where Pacific Water carries heat Frazil can't see. This tests whether
the ocean the targets actually felt fixes it. Diagnostic only: neXtSIM-OPA ends in 2018,
so a model trained on it can't run the 2019-2025 hindcasts.

LAGGED ONE MONTH, against the leak the SST channel had: the forcings go in at months t
and t+1, and heat at t+1 is near zero wherever the t+1 ice is (water under ice sits at
freezing), so it would hand over the answer. The channel at month k holds the heat of
month k-1, so the model sees months t-1 and t, nothing from the month it predicts.
ponytail: the first month (January 1995) gets its own heat, having no December before it.

Outside the box the channels hold the in-box mean (0 after the encoder normalises), and
so do model cells whose nearest neXtSIM-OPA cell is land (0 heat there anyway, so it's
nearest neighbour as in pacific_shelf_ocean_heat/analysis.py). Mean/std over in-box ocean
cells, all months (ddof=1), written to <out-dir>/extra_forcing_stats.json as
add_forcing_channels.py does.

Then:
    build_zarr_monthly.py --datacube-dir <out> --suffix _sicqmw9_ohc --extra-forcings ohc_0_20 ohc_20_100
    train.py +experiment=pf_qm_ema_qmw_ohc

Run:  python scripts/data/add_ocean_heat_channels.py --selfcheck
      python scripts/data/add_ocean_heat_channels.py
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
from frazil import paths

LAYERS = ("ohc_0_20", "ohc_20_100")


def lag_one_month(field, months):
    """field: (T, ...) monthly, months: (T,) DatetimeIndex, consecutive -> month k gets month k-1."""
    assert (np.diff(months.year * 12 + months.month) == 1).all(), "months must be consecutive"
    return np.concatenate([field[:1], field[:-1]])


def _selfcheck():
    m = pd.date_range("1995-01-01", periods=4, freq="MS")
    f = np.arange(4.0)[:, None]
    assert lag_one_month(f, m)[:, 0].tolist() == [0.0, 0.0, 1.0, 2.0]
    try:
        lag_one_month(f, m[[0, 1, 3, 2]]); raise RuntimeError("gap not caught")
    except AssertionError:
        pass
    print("[selfcheck] lag one month; gaps refused")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heat-dir", default=paths.NEXTSIM_OCEAN_HEAT)
    ap.add_argument("--datacube-dir", default=os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
    ap.add_argument("--out-dir", default=os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9_ohc"))
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return

    aux = xr.open_dataset(args.aux)
    ocean = aux.mask.values.astype(bool)
    lon = (aux.longitude.values + 360) % 360
    in_box = ocean & (aux.latitude.values >= 64) & (lon >= 90) & (lon <= 250)
    heat = xr.open_mfdataset(sorted(glob.glob(f"{args.heat_dir}/ohc_*.nc")), combine="by_coords").load()
    idx = P.build_nearest_index(heat.longitude.values.ravel(), heat.latitude.values.ravel(),
                                aux.longitude.values, aux.latitude.values)
    months = pd.DatetimeIndex(heat.time.values)
    grid = {k: np.where(in_box, lag_one_month(P.regrid_time(heat[k].values, idx, ocean.shape), months), np.nan)
            for k in LAYERS}
    key = {(t.year, t.month): i for i, t in enumerate(months)}
    stats = {k: {"mean": float(np.nanmean(grid[k])), "std": float(np.nanstd(grid[k], ddof=1))} for k in LAYERS}
    print(f"[heat] {months[0]:%Y-%m}..{months[-1]:%Y-%m}, {in_box.sum()} model cells in the box; "
          + "  ".join(f"{k}: mean {s['mean']:.1f} std {s['std']:.1f} MJ m-2" for k, s in stats.items()))

    os.makedirs(args.out_dir, exist_ok=True)
    for f in sorted(glob.glob(f"{args.datacube_dir}/*_datacube_*.nc")):
        cube = xr.open_dataset(f)["datacube"].load()
        t = pd.DatetimeIndex(cube.time.values)
        miss = [f"{x:%Y-%m}" for x in t if (x.year, x.month) not in key]
        if miss:
            raise SystemExit(f"{os.path.basename(f)}: no neXtSIM-OPA heat for {miss}")
        sel = [key[(x.year, x.month)] for x in t]
        extra = {k: np.where(ocean, np.nan_to_num(grid[k][sel], nan=stats[k]["mean"]), np.nan).astype(np.float32)
                 for k in LAYERS}
        dst = os.path.join(args.out_dir, os.path.basename(f))
        P.append_channels(cube, extra).rename("datacube").to_dataset().to_netcdf(
            dst, encoding={"datacube": {"zlib": True, "complevel": 4}})
        print(f"  {os.path.basename(f)} + {list(LAYERS)} -> {dst}")
    with open(os.path.join(args.out_dir, "extra_forcing_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"-> {args.out_dir}/extra_forcing_stats.json")


if __name__ == "__main__":
    main()
