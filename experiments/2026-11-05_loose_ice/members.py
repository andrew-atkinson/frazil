"""What kind of error is the loose-ice excess? From the saved members of v04_truth (step_test.py),
by starting concentration (the model's input), Pacific shelves, August and September 2015-2018:
  - how often the members open the cell up (below 15%, and to exactly 0) against how often the target does
  - ensemble spread against the error of the ensemble mean (spread < error = members too alike)
Comparing frequencies within a class conditioned on the input is a calibration check, not a
binning on the truth.

Run from the repo root:  python experiments/2026-11-05_loose_ice/members.py [--run v04_truth]
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
ap.add_argument("--run", default="v04_truth")
ap.add_argument("--datacube", default="monthly_datacube_sicqmw9")
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
regions = {"Pacific shelves": sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"], "Barents/Kara": sec["Barents/Kara"]}
cube = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, args.datacube, "*_datacube_*.nc"))), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)
rows = []
for f in sorted(glob.glob(os.path.join(paths.MODELS, "loose_ice_tests", args.run, "step_??????.npz"))):
    ym = os.path.basename(f)[5:11]; y, m = int(ym[:4]), int(ym[4:])
    i = int(np.where((ct.year == y) & (ct.month == m))[0][0])
    tgt = np.nan_to_num(cube.isel(time=i).sel(var_names="sic").values)
    d = np.load(f); start, pred = np.nan_to_num(d["start"][0]), np.nan_to_num(d["next"])
    for r, mk in regions.items():
        c = mk & (start >= 0.15)
        rows.append(pd.DataFrame({"region": r, "m": m, "w": area[c], "start": start[c], "tgt": tgt[c],
                                  "p_open": (pred[:, c] < 0.15).mean(0), "p_zero": (pred[:, c] <= 0.0).mean(0),
                                  "mean": pred[:, c].mean(0), "spread": pred[:, c].std(0, ddof=1)}))
D = pd.concat(rows, ignore_index=True)
D["cls"] = pd.cut(D.start, [0.15, 0.5, 0.8, 0.95, 1.0], include_lowest=True, labels=["15-50%", "50-80%", "80-95%", ">95%"])
f = lambda g: pd.Series({
    "target open <15%": np.average(g.tgt < 0.15, weights=g.w), "members open": np.average(g.p_open, weights=g.w),
    "target at 0": np.average(g.tgt <= 0, weights=g.w), "members at 0": np.average(g.p_zero, weights=g.w),
    "bias": np.average(g["mean"] - g.tgt, weights=g.w),
    "RMSE of mean": np.sqrt(np.average((g["mean"] - g.tgt) ** 2, weights=g.w)), "spread": np.sqrt(np.average(g.spread ** 2, weights=g.w)),
    "area share": g.w.sum()})
T = D.groupby(["region", "m", "cls"], observed=True).apply(f)
T["area share"] = T["area share"] / T.groupby(level=["region", "m"])["area share"].transform("sum")
print(f"=== {args.run}: opening up, by starting concentration (cells starting at 15% or more)")
print(T.round(3).to_string())
