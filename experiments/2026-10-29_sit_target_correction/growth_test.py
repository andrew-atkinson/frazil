"""Does a growth-based thickness correction hold up out of sample, and does it follow the
ice regime over time where a fixed ratio can't?

Methods, each fitted on winters 2011-14 against the CS2SMOS-constrained neXtSIM reanalysis
and tested on 2015-18 (the teacher is the v0.4.0 thickness target = neXtSIM-OPA):
  teacher       no correction
  static ratio  one ratio per cell and month (correct_sit_targets.fit_ratios, 2011-14)
  growth, flat  winter growth x alpha for all ice
  growth        winter growth x alpha for first-year ice (< 1 m in November), 1 above 2 m
The growth versions rebuild thickness from 1995, so 2011-18 carries 16 years of corrected
history. Scored on April and on November-April: thickness minus the reanalysis by region,
and whether the year-to-year ratio to the teacher drifts the way the reanalysis's does.

Run from the repo root:  python experiments/2026-10-29_sit_target_correction/growth_test.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

spec = importlib.util.spec_from_file_location("cst", os.path.join(RL.ROOT, "scripts/data/correct_sit_targets.py"))
cst = importlib.util.module_from_spec(spec); spec.loader.exec_module(cst)
pd.set_option("display.width", 200)

cube = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9", "*_datacube_*.nc"))),
                         combine="by_coords")["datacube"]
T = cube.sel(var_names="sit").load().values; tt = pd.DatetimeIndex(cube.time.values)
rd = xr.open_dataset(paths.NEXTSIM_REANALYSIS.format(var="sit"))["sit"]; rt = pd.DatetimeIndex(rd.time.values)
R = np.stack([rd.values[np.where((rt.year == t.year) & (rt.month == t.month))[0][0]] for t in tt])
aux = xr.open_dataset(paths.AUX)
oc = aux.mask.values.astype(bool) & np.isfinite(R[0]); area = aux.cell_area.values
lat, lon = aux.latitude.values, (aux.longitude.values + 360) % 360
REG = {"ESS+Chukchi": oc & (lat >= 64) & (lon >= 140) & (lon < 205), "Beaufort": oc & (lat >= 64) & (lon >= 205) & (lon < 240),
       "Laptev": oc & (lat >= 64) & (lon >= 90) & (lon < 140), "central >82N": oc & (lat > 82), "all": oc}
FIT, TEST = (2011, 2014), (2015, 2018)
winter = np.isin(tt.month, [11, 12, 1, 2, 3, 4])
yrs = lambda span: (tt.year >= span[0]) & (tt.year <= span[1])
wyear = np.where(tt.month >= 11, tt.year + 1, tt.year)                          # winter labelled by its April


def rmse(h, sel):
    """Area-weighted cell RMSE vs the reanalysis over the selected months, all valid ocean."""
    d = (np.nan_to_num(h[sel]) - np.nan_to_num(R[sel])) ** 2
    return float(np.sqrt((d * area * oc).sum() / (area * oc).sum() / sel.sum()))


def regional(h, sel, w):
    ww = area * w
    return (np.nan_to_num(h[sel]) * ww).sum((1, 2)) / ww.sum()


# Fit alpha by grid search on April and Nov-Apr RMSE over the fit winters
fit_sel = winter & (wyear >= FIT[0]) & (wyear <= FIT[1])
alphas = np.round(np.arange(0.55, 1.001, 0.05), 2)
best = {}
for name, kw in [("growth, flat", dict(h1=99, h2=100)), ("growth", {})]:
    scores = {a: rmse(cst.growth_corrected(T, tt.month.values, a, **kw), fit_sel) for a in alphas}
    best[name] = min(scores, key=scores.get)
    print(f"{name}: Nov-Apr RMSE vs reanalysis over winters {FIT[0]}-{FIT[1]} by alpha: "
          + "  ".join(f"{a:.2f}:{s:.3f}" for a, s in scores.items()) + f"  -> alpha {best[name]}")
ratios = cst.fit_ratios(T[yrs(FIT)], R[yrs(FIT)], tt[yrs(FIT)].month.values)
M = {"teacher": np.nan_to_num(T),
     "static ratio": np.stack([np.nan_to_num(T[i]) * ratios[m] for i, m in enumerate(tt.month)]),
     "growth, flat": cst.growth_corrected(T, tt.month.values, best["growth, flat"], h1=99, h2=100),
     "growth": cst.growth_corrected(T, tt.month.values, best["growth"])}

test_w = winter & (wyear >= TEST[0]) & (wyear <= TEST[1])
apr = lambda span: (tt.month == 4) & yrs(span)
print(f"\n=== Out of sample, winters {TEST[0]}-{TEST[1]}: thickness minus reanalysis (m), sector means")
rows = []
for n, h in M.items():
    r = {"method": n, "cell RMSE Nov-Apr": rmse(h, test_w)}
    for g, w in REG.items():
        r[f"Apr {g}"] = float(np.mean(regional(h, apr(TEST), w) - regional(R, apr(TEST), w)))
    rows.append(r)
print(pd.DataFrame(rows).set_index("method").round(3).to_string())

print("\n=== April ratio to the teacher by year (reanalysis = what to match; 1995-2010 is where the methods differ unseen)")
for g in ("ESS+Chukchi", "Beaufort", "all"):
    w = REG[g]; base = regional(M["teacher"], tt.month == 4, w)
    D = pd.DataFrame({n: regional(h, tt.month == 4, w) / base for n, h in M.items() if n != "teacher"}, index=tt[tt.month == 4].year)
    D["reanalysis"] = regional(R, tt.month == 4, w) / base
    tr = {c: 10 * np.polyfit(D.loc[2011:2018].index.values.astype(float), D[c].loc[2011:2018].values, 1)[0] for c in D}
    tr95 = {c: 10 * np.polyfit(D.loc[1995:2018].index.values.astype(float), D[c].loc[1995:2018].values, 1)[0] for c in D}
    print(f"\n{g}:"); print(D.loc[[1996, 2000, 2005, 2010, 2011, 2013, 2015, 2017, 2018]].round(2).T.to_string())
    print("  trend per decade 2011-18: " + "  ".join(f"{c} {v:+.2f}" for c, v in tr.items()))
    print("  trend per decade 1995-2018: " + "  ".join(f"{c} {v:+.2f}" for c, v in tr95.items() if c != "reanalysis"))

print("\n=== April volume trend of the targets, 1995-2018 (km^3 per decade; the teacher's is what a correction should not flatten or steepen without cause)")
vol = {n: (np.nan_to_num(h[tt.month == 4]) * area * aux.mask.values.astype(bool)).sum((1, 2)) / 1e9 for n, h in M.items()}
years = tt[tt.month == 4].year.values.astype(float)
print("  " + "  ".join(f"{n}: {np.mean(v):.0f} km^3, {10 * np.polyfit(years, v, 1)[0]:+.0f}" for n, v in vol.items()))
