"""What shape is the error in Frazil's one-month step? (Follows one_month.py.)

The step-size ratio in one_month.py assumes Frazil's change is a fixed fraction of the
target's (linear through zero); September contradicts it (a smaller change than August,
a worse ratio). Here, cell by cell, from the saved one-month predictions (2015-2018,
each from the target's previous month):
  1. Frazil's concentration change against the target's, binned: the response curve.
     Polynomials of degree 1-3 fitted to it (cells that change at all).
  2. The same against where the cell starts (the target's concentration at t-1), by
     season: does the error depend on the change, or on the state and the season?

Run from the repo root:  python experiments/2026-10-30_summer_melt/step_shape.py [--fields DIR]
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

ap = argparse.ArgumentParser()
ap.add_argument("--fields", default=os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw", "one_month_fields"))
ap.add_argument("--datacube", default=os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
masks = sc.sector_masks(aux.longitude.values, ocean)
shelf = masks["East Siberian"] | masks["Chukchi"] | masks["Beaufort"]
cube = xr.open_mfdataset(sorted(glob.glob(f"{args.datacube}/*_datacube_*.nc")), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)
SEASON = {6: "melt Jun-Sep", 7: "melt Jun-Sep", 8: "melt Jun-Sep", 9: "melt Jun-Sep", 10: "freeze Oct-Nov", 11: "freeze Oct-Nov"}

rows = []
for f in sorted(glob.glob(f"{args.fields}/pred_??????.nc")):
    p = xr.open_dataset(f); t = pd.Timestamp(p.time.values)
    if t.month not in SEASON:
        continue
    i = int(np.where((ct.year == t.year) & (ct.month == t.month))[0][0])
    prev = np.nan_to_num(cube.isel(time=i - 1).sel(var_names="sic").values)
    tgt = np.nan_to_num(cube.isel(time=i).sel(var_names="sic").values)
    fz = np.nan_to_num(p.state.sel(var_names="sic").values)
    m = ocean & ((prev > 0.01) | (tgt > 0.01) | (fz > 0.01))
    rows.append(pd.DataFrame({"month": t.month, "season": SEASON[t.month], "shelf": shelf[m], "w": area[m],
                              "start": prev[m], "d_tgt": (tgt - prev)[m], "d_fz": (fz - prev)[m]}))
C = pd.concat(rows, ignore_index=True)
C["err"] = C.d_fz - C.d_tgt
wmean = lambda g, c: np.average(g[c], weights=g.w)

print("=== 1. Response curve: Frazil's concentration change against the target's, cell by cell (area-weighted)")
bins = np.round(np.arange(-1.0, 1.01, 0.2), 1)
C["d_bin"] = pd.cut(C.d_tgt, bins)
for (season, sh), g in C.groupby(["season", "shelf"]):
    t = g.groupby("d_bin", observed=True).apply(lambda b: pd.Series({"target": wmean(b, "d_tgt"), "frazil": wmean(b, "d_fz"), "share of cells": b.w.sum() / g.w.sum()}))
    t["frazil / target"] = t.frazil / t.target
    print(f"\n{season}, {'Pacific shelves' if sh else 'elsewhere'}:"); print(t.round(3).to_string())
    x, y, w = g.d_tgt.values, g.d_fz.values, g.w.values
    fits = []
    for deg in (1, 2, 3):
        c = np.polyfit(x, y, deg, w=np.sqrt(w)); r = y - np.polyval(c, x)
        fits.append(f"degree {deg}: RMSE {np.sqrt(np.average(r ** 2, weights=w)):.4f}, coefficients (high to low) " + " ".join(f"{v:+.3f}" for v in c))
    print("  fits of Frazil's change on the target's:\n    " + "\n    ".join(fits))

print("\n=== 2. Error (Frazil - target change) by starting concentration, Pacific shelves, by month (area-weighted mean)")
C["start_bin"] = pd.cut(C.start, [-0.01, 0.15, 0.5, 0.8, 0.95, 1.0])
sh = C[C.shelf]
print(sh.groupby(["start_bin", "month"], observed=True).apply(lambda b: wmean(b, "err")).unstack("month").round(3).to_string())
print("\nshare of shelf ice area by starting concentration and month")
print(sh.groupby(["start_bin", "month"], observed=True).apply(lambda b: (b.w * b.start).sum()).unstack("month").div(
      sh.groupby("month").apply(lambda b: (b.w * b.start).sum())).round(2).to_string())
