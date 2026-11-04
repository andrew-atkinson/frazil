"""Cell by cell: does Frazil's one-step error follow the seafloor depth? (Follows proximity.py.)

Depth: data/auxiliary/topaz_depth.nc (scripts/download/topaz_depth.py; TOPAZ4b's own
seafloor, resolved to its levels). The sectors run from the coast to the pole, so each
has shelf and deep basin. From the saved one-month predictions (2015-2018, each from the
target's previous month):
  err    Frazil's concentration change minus the target's (positive = too much ice), cells
         with ice at the start (melt half, predicted May-Sep) or at the end (freeze half,
         Oct-Apr) of the step in either Frazil or the target
  1. Mean error by depth class, Pacific shelves vs the Siberian shelves (Laptev, Barents/Kara,
     which are as shallow) vs elsewhere: if depth were the cause, shallow water would be
     biased everywhere.
  2. Straight lines (area-weighted, leave-one-year-out RMSE) in log10 depth, the sector's
     open share, and both, per sector and shared across the Pacific shelves.

Run from the repo root:  python experiments/2026-10-30_summer_melt/depth.py [--fields DIR]
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
depth = xr.open_dataset(paths.TOPAZ_DEPTH).depth.values
sec = sc.sector_masks(aux.longitude.values, ocean)
sec_id = np.full(ocean.shape, "", object)
for s, m in sec.items():
    sec_id[m] = s
PACIFIC, SIBERIAN = ["East Siberian", "Chukchi", "Beaufort"], ["Laptev", "Barents/Kara"]
cube = xr.open_mfdataset(sorted(glob.glob(f"{args.datacube}/*_datacube_*.nc")), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)

rows = []
for f in sorted(glob.glob(f"{args.fields}/pred_??????.nc")):
    p = xr.open_dataset(f); t = pd.Timestamp(p.time.values)
    i = int(np.where((ct.year == t.year) & (ct.month == t.month))[0][0])
    prev = np.nan_to_num(cube.isel(time=i - 1).sel(var_names="sic").values)
    tgt = np.nan_to_num(cube.isel(time=i).sel(var_names="sic").values)
    fz = np.nan_to_num(p.state.sel(var_names="sic").values)
    melt = t.month in (5, 6, 7, 8, 9)
    active = (prev >= 0.15) if melt else ((tgt >= 0.15) | (fz >= 0.15))
    sel = ocean & active & np.isfinite(depth) & (sec_id != "")
    ice = ocean & (prev >= 0.15)
    open_share = {s: 1 - (area * m * ice).sum() / (area * m).sum() for s, m in sec.items()}
    rows.append(pd.DataFrame({"year": t.year, "month": t.month, "half": "melt May-Sep" if melt else "freeze Oct-Apr",
                              "sector": sec_id[sel], "w": area[sel], "depth": depth[sel],
                              "open": [open_share[s] for s in sec_id[sel]], "err": (fz - tgt)[sel]}))
C = pd.concat(rows, ignore_index=True)
C["logd"] = np.log10(C.depth)
C["group"] = np.select([C.sector.isin(PACIFIC), C.sector.isin(SIBERIAN)], ["Pacific shelves", "Laptev + Barents/Kara"], "elsewhere")
C["d_bin"] = pd.cut(C.depth, [0, 25, 50, 100, 200, 500, 1000, 5000], labels=["<25 m", "25-50", "50-100", "100-200", "200-500", "500-1000", ">1000"])
wm = lambda g: np.average(g.err, weights=g.w)

print("=== 1. Mean error (Frazil minus target concentration change, area-weighted) by seafloor depth")
for half, g in C.groupby("half", sort=False):
    t = g.groupby(["group", "d_bin"], observed=True).apply(lambda b: pd.Series({"error": wm(b), "area": b.w.sum()}))
    t["area"] /= t.groupby(level=0)["area"].transform("sum")
    print(f"\n{half} (area = share of that group's cells in the sample)")
    print(t.unstack(0).round(3).to_string())
print("\nPacific shelves, melt half, by depth class and month (error)")
g = C[(C.group == "Pacific shelves") & (C.half == "melt May-Sep")]
print(g.groupby(["d_bin", "month"], observed=True).apply(wm).unstack("month").round(3).to_string())


def loyo_rmse(g, cols):
    e, w = [], []
    for yr in g.year.unique():
        tr, te = g[g.year != yr], g[g.year == yr]
        A = lambda d: np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        sw = np.sqrt(tr.w.values)
        coef = np.linalg.lstsq(A(tr) * sw[:, None], tr.err.values * sw, rcond=None)[0]
        e.append(te.err.values - A(te) @ coef); w.append(te.w.values)
    return float(np.sqrt(np.average(np.concatenate(e) ** 2, weights=np.concatenate(w))))


print("\n=== 2. Straight lines, leave-one-year-out RMSE x100 (cell noise dominates; compare within a row)")
preds = {"constant": [], "log10 depth": ["logd"], "sector open share": ["open"], "open share + log10 depth": ["open", "logd"]}
out = []
for half, g0 in C.groupby("half", sort=False):
    for s, g in list(g0.groupby("sector", sort=False)) + [("Pacific shelves, shared", g0[g0.group == "Pacific shelves"])]:
        r = {"half": half, "sector": s}
        r.update({k: 100 * loyo_rmse(g, v) for k, v in preds.items()})
        out.append(r)
R = pd.DataFrame(out).set_index(["half", "sector"])
R["best"] = R.idxmin(axis=1)
print(R.round(3).to_string())
print("\nPacific shelves, shared line, all four years: err x100 = a + b x log10(depth m)")
for half, g in C[C.group == "Pacific shelves"].groupby("half", sort=False):
    b, a = np.polyfit(g.logd, g.err, 1, w=np.sqrt(g.w))
    print(f"  {half:15s} a {100 * a:+.2f}  b {100 * b:+.2f}  (at 30 m {100 * (a + b * np.log10(30)):+.2f}, at 1000 m {100 * (a + b * 3):+.2f})")
