"""Is the September gain from ocean heat (compare_ohc.py) due to each year's actual heat?
July-October 2015-2018, one-month predictions against the same targets, three ways:
  v0.4.0   no heat channels
  real     monthly_pf_qm_ema_qmw_ohc with the year's heat
  clim     the same weights and the same per-month seeds, with the heat replaced by its
           1995-2014 monthly climatology (make_clim_cube.py)
Real and clim differ only in the heat (eval_monthly reproduces a saved month exactly), so
real - clim is the effect of the year's heat, and clim - v0.4.0 is the seasonal pattern
plus the difference between the two trainings. Last section: does the model respond to a
heat anomaly the physical way round (warmer than usual -> less ice)?

Run from the repo root:  python experiments/2026-10-30_summer_melt/compare_clim.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

OHC = os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw_ohc")
RUNS = {"v0.4.0": os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw", "one_month_fields"),
        "real": os.path.join(OHC, "one_month_fields"), "clim": os.path.join(OHC, "one_month_fields_clim")}
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
depth = xr.open_dataset(paths.TOPAZ_DEPTH).depth.values
sec = sc.sector_masks(aux.longitude.values, ocean)
pac = sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"]
regions = {"Pacific shelves": pac, "East Siberian": sec["East Siberian"], "Chukchi": sec["Chukchi"], "Beaufort": sec["Beaufort"], "Laptev": sec["Laptev"],
           "Barents/Kara": sec["Barents/Kara"], "Greenland": sec["Greenland"], "pan-Arctic": ocean}
open_cube = lambda name: xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, name, "*_datacube_*.nc"))), combine="by_coords")["datacube"]
real_cube, clim_cube = open_cube("monthly_datacube_sicqmw9_ohc"), open_cube("monthly_datacube_sicqmw9_ohc_clim")
rt, ctt = pd.DatetimeIndex(real_cube.time.values), pd.DatetimeIndex(clim_cube.time.values)

ext, cells = [], []
for f in sorted(glob.glob(os.path.join(RUNS["clim"], "pred_??????.nc"))):
    ym = os.path.basename(f)[5:11]; t = pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01")
    i = int(np.where((rt.year == t.year) & (rt.month == t.month))[0][0]); j = int(np.where((ctt.year == t.year) & (ctt.month == t.month))[0][0])
    tgt = np.nan_to_num(real_cube.isel(time=i).sel(var_names="sic").values); prev = np.nan_to_num(real_cube.isel(time=i - 1).sel(var_names="sic").values)
    pred = {n: xr.open_dataset(os.path.join(d, f"pred_{ym}.nc")) for n, d in RUNS.items()}
    r = {"year": t.year, "month": t.month}
    for s, m in regions.items():
        r[(s, "target")] = (area * m * (tgt > 0.15)).sum() / 1e6
        for n, p in pred.items():
            r[(s, n)] = np.mean([(area * m * (np.nan_to_num(x) > 0.15)).sum() for x in p.sic_members.values]) / 1e6
    ext.append(r)
    fz = {n: np.nan_to_num(p.state.sel(var_names="sic").values) for n, p in pred.items()}
    active = pac & (prev >= 0.15) & np.isfinite(depth)
    anom = {k: (real_cube.isel(time=i).sel(var_names=k).values - clim_cube.isel(time=j).sel(var_names=k).values)[active] for k in ("ohc_0_20", "ohc_20_100")}
    cells.append(pd.DataFrame({"year": t.year, "month": t.month, "w": area[active], "depth": depth[active], "start": prev[active],
                               **{f"a{k}": v for k, v in anom.items()}, "resp": (fz["real"] - fz["clim"])[active],
                               **{n: (x - tgt)[active] for n, x in fz.items()}}))
E = pd.DataFrame(ext).set_index(["year", "month"]); E.columns = pd.MultiIndex.from_tuples(E.columns)
C = pd.concat(cells, ignore_index=True)
wm = lambda g: pd.Series({n: np.average(g[n], weights=g.w) for n in RUNS})

print("=== 1. One-step extent error, M km^2, mean over 2015-2018")
rows = []
for s in regions:
    for mo in (7, 8, 9, 10):
        g = E[s].xs(mo, level="month")
        rows.append({"sector": s, "month": mo, **{n: (g[n] - g.target).mean() for n in RUNS}})
print(pd.DataFrame(rows).pivot(index="sector", columns="month", values=list(RUNS)).swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False).reindex(list(regions)).round(3).to_string())

print("\n=== 2. Pacific shelves by year: extent error, and the two differences")
g = E["Pacific shelves"]
Y = pd.DataFrame({n: g[n] - g.target for n in RUNS}); Y["year's heat (real - clim)"] = g["real"] - g["clim"]; Y["pattern + training (clim - v0.4.0)"] = g["clim"] - g["v0.4.0"]
print(Y.unstack("month").round(3).to_string())

print("\n=== 3. Pacific shelves: concentration RMSE (ensemble mean)")
rm = lambda g: pd.Series({n: np.sqrt(np.average(g[n] ** 2, weights=g.w)) for n in RUNS})
print(C.groupby("month")[["w", *RUNS]].apply(rm).round(4).to_string())

print("\n=== 4. Pacific shelves: error by seafloor depth, September, and by starting concentration, August and September")
C["d_bin"] = pd.cut(C.depth, [0, 50, 100, 200, 500, 1000, 5000], labels=["<50 m", "50-100", "100-200", "200-500", "500-1000", ">1000"])
print(C[C.month == 9].groupby("d_bin", observed=True)[["w", *RUNS]].apply(wm).round(3).to_string())
C["s_bin"] = pd.cut(C.start, [0.15, 0.5, 0.8, 0.95, 1.0], include_lowest=True)
print(C[C.month.isin([8, 9])].groupby(["s_bin", "month"], observed=True)[["w", *RUNS]].apply(wm).unstack("month").round(3).to_string())

print("\n=== 5. Response to the heat anomaly, Pacific shelves, cells with ice at the start: real - clim concentration (negative = less ice with the year's heat), by anomaly (MJ m^-2)")
for k in ("ohc_0_20", "ohc_20_100"):
    b = pd.cut(C[f"a{k}"], [-1e4, -100, -30, -10, 10, 30, 100, 1e4])
    print(f"\n{k}:")
    print(C[C.month.isin([8, 9])].groupby([b, "month"], observed=True).apply(lambda g: pd.Series({"resp": np.average(g.resp, weights=g.w), "share": g.w.sum()})).unstack("month").round(4).to_string())
    x, y, w = C[f"a{k}"].values, C.resp.values, C.w.values
    for mo in (7, 8, 9, 10):
        s = C.month.values == mo
        print(f"  month {mo:2d}: slope {np.polyfit(x[s], y[s], 1, w=np.sqrt(w[s]))[0] * 100:+.4f} per 100 MJ m^-2, r {np.corrcoef(x[s], y[s])[0, 1]:+.3f}, mean anomaly {np.average(x[s], weights=w[s]):+.1f}")
