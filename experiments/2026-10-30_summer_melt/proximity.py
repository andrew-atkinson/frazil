"""Cell by cell: does Frazil's one-step error follow how close a cell is to open water
(or, in the freeze, to the ice edge), better than the sector's open share or the cell's
own starting concentration? (Follows sector_poly.py.)

From the saved one-month predictions (2015-2018, each from the target's previous month):
  err    Frazil's concentration change minus the target's (positive = too much ice)
  dist   at the start of the step (the target's state at t-1, what the model sees):
           melt half (predicted May-Sep), ice cells (>= 15%): km to the nearest open
           water (< 15%); freeze half (Oct-Apr), open cells: km to the nearest ice
  start  the cell's starting concentration
  open   the sector's open share at the start (sector_poly.py's 1 - cover)
Straight lines in each, per sector and shared across the Pacific shelves, scored by
leave-one-year-out RMSE against a per-sector constant (area-weighted). Distances are
between cell centres in the model's projected coordinates; the nearest cell is found on
the grid (cells are 4-25 km), so they are approximate below one cell.

Run from the repo root:  python experiments/2026-10-30_summer_melt/proximity.py [--fields DIR]
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

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
X, Y = aux.x_coord.values / 1e3, aux.y_coord.values / 1e3
sec = sc.sector_masks(aux.longitude.values, ocean)
sec_id = np.full(ocean.shape, "", object)
for s, m in sec.items():
    sec_id[m] = s
PACIFIC = ["East Siberian", "Chukchi", "Beaufort"]
cube = xr.open_mfdataset(sorted(glob.glob(f"{args.datacube}/*_datacube_*.nc")), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)


def km_to(target):
    """km from every cell to the nearest cell where `target` is True (grid-nearest)."""
    _, (iy, ix) = distance_transform_edt(~target, return_indices=True)
    return np.hypot(X - X[iy, ix], Y - Y[iy, ix])


rows = []
for f in sorted(glob.glob(f"{args.fields}/pred_??????.nc")):
    p = xr.open_dataset(f); t = pd.Timestamp(p.time.values)
    i = int(np.where((ct.year == t.year) & (ct.month == t.month))[0][0])
    prev = np.nan_to_num(cube.isel(time=i - 1).sel(var_names="sic").values)
    tgt = np.nan_to_num(cube.isel(time=i).sel(var_names="sic").values)
    fz = np.nan_to_num(p.state.sel(var_names="sic").values)
    melt = t.month in (5, 6, 7, 8, 9)
    ice, opn = ocean & (prev >= 0.15), ocean & (prev < 0.15)
    if melt:
        sel, dist = ice, km_to(opn)
    else:
        sel, dist = opn & (km_to(ice) <= 500), km_to(ice)          # open water near enough to freeze within a month
    open_share = {s: 1 - (area * m * ice).sum() / (area * m).sum() for s, m in sec.items()}
    sel &= sec_id != ""
    rows.append(pd.DataFrame({"year": t.year, "month": t.month, "half": "melt May-Sep" if melt else "freeze Oct-Apr",
                              "sector": sec_id[sel], "w": area[sel], "dist": dist[sel], "start": prev[sel],
                              "open": [open_share[s] for s in sec_id[sel]], "err": (fz - tgt)[sel]}))
C = pd.concat(rows, ignore_index=True)
wm = lambda g, c="err": np.average(g[c], weights=g.w)

print("=== Mean error (Frazil minus target concentration change, area-weighted) by distance, Pacific shelves vs elsewhere")
C["d_bin"] = pd.cut(C.dist, [-1, 25, 50, 100, 200, 400, 5000], labels=["<25 km", "25-50", "50-100", "100-200", "200-400", ">400"])
for half, g in C.groupby("half", sort=False):
    t = g.groupby([g.sector.isin(PACIFIC).map({True: "Pacific shelves", False: "elsewhere"}), "d_bin"], observed=True).apply(
        lambda b: pd.Series({"error": wm(b), "share of area": b.w.sum()}))
    t["share of area"] /= t.groupby(level=0)["share of area"].transform("sum")
    print(f"\n{half} ({'ice cells: distance to open water' if half.startswith('melt') else 'open cells: distance to ice'})")
    print(t.unstack(0).round(3).to_string())


def loyo_rmse(g, cols):
    """Leave-one-year-out weighted RMSE of a line in `cols` (none = constant)."""
    e, w = [], []
    for yr in g.year.unique():
        tr, te = g[g.year != yr], g[g.year == yr]
        A = lambda d: np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        sw = np.sqrt(tr.w.values)
        coef = np.linalg.lstsq(A(tr) * sw[:, None], tr.err.values * sw, rcond=None)[0]
        e.append(te.err.values - A(te) @ coef); w.append(te.w.values)
    e, w = np.concatenate(e), np.concatenate(w)
    return float(np.sqrt(np.average(e ** 2, weights=w)))


print("\n=== Straight lines, leave-one-year-out RMSE x100 (lower is better; cell-level noise dominates, so differences are small)")
C["dist100"] = C.dist / 100
preds = {"constant": [], "sector open share": ["open"], "own start conc.": ["start"], "distance (per 100 km)": ["dist100"],
         "distance + start": ["dist100", "start"]}
out = []
for half, g0 in C.groupby("half", sort=False):
    for s, g in list(g0.groupby("sector", sort=False)) + [("Pacific shelves, shared", g0[g0.sector.isin(PACIFIC)])]:
        r = {"half": half, "sector": s}
        r.update({k: 100 * loyo_rmse(g, v) for k, v in preds.items()})
        out.append(r)
R = pd.DataFrame(out).set_index(["half", "sector"])
R["best"] = R.idxmin(axis=1)
print(R.round(3).to_string())

print("\n=== Pacific shelves, shared line in distance: err x100 = a + b x (km / 100), fitted on all four years")
for half, g in C[C.sector.isin(PACIFIC)].groupby("half", sort=False):
    b, a = np.polyfit(g.dist100, g.err, 1, w=np.sqrt(g.w))
    print(f"  {half:15s} a {100 * a:+.2f}  b {100 * b:+.2f}")
