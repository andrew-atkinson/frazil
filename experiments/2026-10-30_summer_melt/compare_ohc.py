"""Did the training simulation's own ocean heat fix Frazil's Pacific-shelf melt error?
v0.4.0 (monthly_pf_qm_ema_qmw) against monthly_pf_qm_ema_qmw_ohc (the same recipe and seed
plus lagged neXtSIM-OPA heat at 0-20 m and 20-100 m, in the 90 E-110 W box), paired on the
one-month predictions of 2015-2018 against the same targets.
  1. Did the model use the channels: input-weight norm per channel, EMA weights
  2. Extent error from the members by sector and month; sectors outside the heat box are
     the control (how far two trainings differ with no new information there)
  3. Pacific shelves, per year (only four, so the spread matters more than the mean)
  4. Concentration RMSE by region and season
  5. Error by seafloor depth and by starting concentration, Pacific shelves, side by side
  6. Is the error related to the heat at all? By within-month heat tercile, inside each
     starting-concentration class. Heat is an input, so this conditions on what the model
     can see, never on the truth. Flat in v0.4.0 = the heat holds nothing about this error

Run from the repo root:  python experiments/2026-10-30_summer_melt/compare_ohc.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

import frazil.ledger as RL
from frazil import paths

A, B = "monthly_pf_qm_ema_qmw", "monthly_pf_qm_ema_qmw_ohc"
NAMES = {A: "v0.4.0", B: "ohc"}
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
depth = xr.open_dataset(paths.TOPAZ_DEPTH).depth.values
sec = sc.sector_masks(aux.longitude.values, ocean)
PACIFIC = ["East Siberian", "Chukchi", "Beaufort"]
pac = sec[PACIFIC[0]] | sec[PACIFIC[1]] | sec[PACIFIC[2]]
lon = (aux.longitude.values + 360) % 360
in_box = ocean & (aux.latitude.values >= 64) & (lon >= 90) & (lon <= 250)
cube = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9_ohc", "*_datacube_*.nc"))), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)

print("=== 1. Input-weight norm per channel, EMA weights (inputs are normalised, so comparable)")
sd = torch.load(os.path.join(paths.MODELS, B, "last.ckpt"), map_location="cpu", weights_only=False)["state_dict"]
key = next(k for k in sd if k.startswith("ema_model.module") and k.endswith("tokenizer.in_encoder.weight"))
w = sd[key]; p2 = w.shape[1] // 29                # 28 channels + ones
norm = w.reshape(w.shape[0], 29, p2).norm(dim=(0, 2)).numpy()
for lab, i in [("tus", 12), ("huss", 13), ("uas", 14), ("vas", 15), ("ohc_0_20", 16), ("ohc_20_100", 17)]:
    print(f"  {lab:11s} month t {norm[i]:.3f}   month t+1 {norm[i + 6]:.3f}")

ext, rmse, cells = [], [], []
for f in sorted(glob.glob(os.path.join(paths.MODELS, B, "one_month_fields", "pred_??????.nc"))):
    ym = os.path.basename(f)[5:11]
    t = pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01"); i = int(np.where((ct.year == t.year) & (ct.month == t.month))[0][0])
    sel = lambda k: np.nan_to_num(cube.isel(time=i).sel(var_names=k).values)
    tgt, prev = sel("sic"), np.nan_to_num(cube.isel(time=i - 1).sel(var_names="sic").values)
    pred = {m: xr.open_dataset(os.path.join(paths.MODELS, m, "one_month_fields", f"pred_{ym}.nc")) for m in (A, B)}
    r = {"year": t.year, "month": t.month}
    for s, mask in {**sec, "Pacific shelves": pac, "pan-Arctic": ocean}.items():
        r[(s, "target")] = (area * mask * (tgt > 0.15)).sum() / 1e6
        for m, p in pred.items():
            r[(s, NAMES[m])] = np.mean([(area * mask * (np.nan_to_num(x) > 0.15)).sum() for x in p.sic_members.values]) / 1e6
    ext.append(r)
    fz = {NAMES[m]: np.nan_to_num(p.state.sel(var_names="sic").values) for m, p in pred.items()}
    for lab, mask in {"Pacific shelves": pac, "rest of the heat box": in_box & ~pac, "outside the box": ocean & ~in_box}.items():
        rmse.append({"year": t.year, "month": t.month, "region": lab, **{n: np.average((x - tgt)[mask] ** 2, weights=area[mask]) for n, x in fz.items()}})
    active = pac & (prev >= 0.15) & np.isfinite(depth)
    cells.append(pd.DataFrame({"year": t.year, "month": t.month, "w": area[active], "depth": depth[active], "start": prev[active],
                               "h0": sel("ohc_0_20")[active], "h20": sel("ohc_20_100")[active],      # channel at month k = heat of k-1, the start
                               **{n: (x - tgt)[active] for n, x in fz.items()}}))

E = pd.DataFrame(ext).set_index(["year", "month"]); E.columns = pd.MultiIndex.from_tuples(E.columns)
print("\n=== 2. One-step extent error, M km^2, mean over 2015-2018; change = |ohc| - |v0.4.0| of the means (negative = better). * = outside the heat box")
rows = []
for s in ["Pacific shelves", *PACIFIC, "Laptev", "Barents/Kara", "Canadian/Baffin/Hudson", "Greenland", "pan-Arctic"]:
    for mo in (7, 8, 9, 10):
        g = E[s].xs(mo, level="month"); a, b = (g["v0.4.0"] - g.target).mean(), (g["ohc"] - g.target).mean()
        rows.append({"sector": s + (" *" if s in ("Barents/Kara", "Greenland") else ""), "month": mo, "v0.4.0": a, "ohc": b, "change": abs(b) - abs(a)})
print(pd.DataFrame(rows).pivot(index="sector", columns="month").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False).round(3).to_string())

print("\n=== 3. Pacific shelves by year: extent error, M km^2")
g = E["Pacific shelves"]; Y = pd.DataFrame({"v0.4.0": g["v0.4.0"] - g.target, "ohc": g["ohc"] - g.target})
Y["ohc - v0.4.0"] = Y["ohc"] - Y["v0.4.0"]
print(Y[Y.index.get_level_values("month").isin([8, 9, 10])].unstack("month").round(3).to_string())

R = pd.DataFrame(rmse); R["season"] = np.where(R.month.isin([5, 6, 7, 8, 9]), "melt May-Sep", "freeze Oct-Apr")
print("\n=== 4. Concentration RMSE, ensemble mean, by region and season")
T = np.sqrt(R.groupby(["region", "season"])[["v0.4.0", "ohc"]].mean()); T["change %"] = 100 * (T["ohc"] / T["v0.4.0"] - 1)
print(T.round(4).to_string())
Ty = np.sqrt(R[R.season == "melt May-Sep"].groupby(["region", "year"])[["v0.4.0", "ohc"]].mean()); Ty["change %"] = 100 * (Ty["ohc"] / Ty["v0.4.0"] - 1)
print("\nmelt half, by year:\n" + Ty["change %"].unstack("year").round(2).to_string())

C = pd.concat(cells, ignore_index=True)
wm = lambda g: pd.Series({n: np.average(g[n], weights=g.w) for n in ("v0.4.0", "ohc")})
print("\n=== 5a. Pacific shelves: error in the month's concentration change by seafloor depth (positive = too much ice)")
C["d_bin"] = pd.cut(C.depth, [0, 25, 50, 100, 200, 500, 1000, 5000], labels=["<25 m", "25-50", "50-100", "100-200", "200-500", "500-1000", ">1000"])
print(C[C.month.isin([7, 8, 9])].groupby(["d_bin", "month"], observed=True)[["w", "v0.4.0", "ohc"]].apply(wm).unstack("month").round(3).to_string())
print("\n=== 5b. By starting concentration")
C["s_bin"] = pd.cut(C.start, [0.15, 0.5, 0.8, 0.95, 1.0], include_lowest=True)
print(C[C.month.isin([7, 8, 9])].groupby(["s_bin", "month"], observed=True)[["w", "v0.4.0", "ohc"]].apply(wm).unstack("month").round(3).to_string())

print("\n=== 6. Pacific shelves, August and September: error by heat tercile (within each month), inside each starting-concentration class")
for k, lab in (("h0", "0-20 m"), ("h20", "20-100 m")):
    sub = C[C.month.isin([8, 9])].copy()
    sub["tercile"] = sub.groupby("month")[k].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=False)).map({0: "cool", 1: "mid", 2: "warm"})
    print(f"\nheat at {lab} (MJ m^-2, tercile means " + ", ".join(f"{n} {v:.0f}" for n, v in sub.groupby("tercile")[k].mean().items()) + ")")
    print(sub.groupby(["s_bin", "tercile"], observed=True)[["w", "v0.4.0", "ohc"]].apply(wm).unstack("tercile").round(3).to_string())
