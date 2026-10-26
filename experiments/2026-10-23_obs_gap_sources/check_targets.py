"""Check corrected training targets against NSIDC-0051 before any retraining (ISS-026).

For each target datacube: the September extent trend of (target - NSIDC) per decade and
its mean in 1995-2006 and 2007-2018 (should be ~0 and equal for a trend-preserving
correction), the mean extent bias and ice-edge error by calendar month, and how close
each year is (RMS of the yearly September difference).

Run from the repo root:  python experiments/2026-10-23_obs_gap_sources/check_targets.py
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from frazil import paths

CUBES = {"neXtSIM": paths.DATACUBE, "QM fixed map": paths.DATACUBE_SICQM,
         "QM 13-yr windows": paths.DATACUBE_SICQM + "w13", "QM 9-yr windows": paths.DATACUBE_SICQM + "w9"}
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e12   # M km2


def sic_by_month(folder):
    out = {}
    for f in sorted(glob.glob(f"{folder}/*_datacube_*.nc")):
        c = xr.open_dataset(f)["datacube"]
        for t, x in zip(pd.DatetimeIndex(c.time.values), c.sel(var_names="sic").values):
            out[(t.year, t.month)] = (np.nan_to_num(x) > 0.15) & ocean
    return out


o = xr.open_mfdataset(sorted(glob.glob(f"{paths.NSIDC0051_GRID}/*.nc")), combine="by_coords")
v = o[[n for n in ("datacube", "state") if n in o.data_vars][0]].sel(var_names="sic")
obs = {(t.year, t.month): (np.nan_to_num(v.isel(time=i).values) > 0.15) & ocean
       for i, t in enumerate(pd.DatetimeIndex(v.time.values)) if 1995 <= t.year <= 2018}
rows, sep = [], {}
for name, folder in CUBES.items():
    if not os.path.isdir(folder):
        print(f"skip {name}: no {folder}"); continue
    tg = sic_by_month(folder)
    for (y, m), x in tg.items():
        if (y, m) in obs:
            rows.append(dict(cube=name, year=y, month=m, bias=float((area * x).sum() - (area * obs[(y, m)]).sum()),
                             iiee=float((area * (x ^ obs[(y, m)])).sum())))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("September: target minus NSIDC (M km2)")
for name, d in df[df.month == 9].groupby("cube", sort=False):
    d = d.sort_values("year"); x, yv = d.year.values.astype(float), d.bias.values
    b, a = np.polyfit(x, yv, 1); se = np.sqrt((yv - (a + b * x)).var(ddof=2) / ((x - x.mean()) ** 2).sum())
    print(f"  {name:17s} trend {10 * b:+.2f} +/- {10 * se:.2f} per decade | mean 1995-2006 {d[d.year <= 2006].bias.mean():+.2f}, "
          f"2007-2018 {d[d.year >= 2007].bias.mean():+.2f} | RMS of yearly difference {np.sqrt((yv ** 2).mean()):.2f}")
print("\nMean extent bias by month, 1995-2018 (M km2)")
print(df.pivot_table(index="cube", columns="month", values="bias", sort=False).round(2).to_string())
print("\nMean ice-edge error by month, 1995-2018 (M km2)")
print(df.pivot_table(index="cube", columns="month", values="iiee", sort=False).round(2).to_string())
