"""Where does the 2019-25 gap to observations come from, if not drift? (no training)

Test 2: does the training target drift away from the observations? September extent of
the QM target (and the original neXtSIM) against NSIDC-0051, 1995-2018, on the model
grid: the yearly difference and its trend.
Test 3: does the model under-respond to warm summers? June-August mean 2 m temperature
over the ocean north of 70 N (the model's own forcing input `tus`), and the September
extent's response to it, for the model (drift free-runs 2009-18, mean over starts, plus
the 2019-25 hindcast) and for NSIDC over the same years.

Run from the repo root:  python experiments/2026-10-23_obs_gap_sources/analysis.py
"""
import glob

import numpy as np
import pandas as pd
import xarray as xr

from frazil import paths
from frazil.diagnostics import ice_area_km2, SIC_THRESHOLD

MODEL = "monthly_pf_qm_ema"
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
north = ocean & (aux.latitude.values >= 70)
ext = lambda sic: ice_area_km2(sic[None], ocean, area, "extent", SIC_THRESHOLD)[0] / 1e6


def cube_series(folder, var):
    out = {}
    for f in sorted(glob.glob(f"{folder}/*_datacube_*.nc")):
        c = xr.open_dataset(f)["datacube"]
        for t, x in zip(pd.DatetimeIndex(c.time.values), c.sel(var_names=var).values):
            out[(t.year, t.month)] = x
    return out


def fit(x, y):
    """slope, its standard error and r."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    se = np.sqrt(res.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return b, se, np.corrcoef(x, y)[0, 1]


o = xr.open_mfdataset(sorted(glob.glob(f"{paths.NSIDC0051_GRID}/*.nc")), combine="by_coords")
v = o[[n for n in ("datacube", "state") if n in o.data_vars][0]].sel(var_names="sic")
obs_sep = {t.year: ext(np.nan_to_num(v.isel(time=i).values)) for i, t in enumerate(pd.DatetimeIndex(v.time.values)) if t.month == 9}
qm = cube_series(paths.DATACUBE_SICQM, "sic"); orig = cube_series(paths.DATACUBE, "sic")
years = range(1995, 2019)
df = pd.DataFrame({"NSIDC": [obs_sep[y] for y in years], "QM target": [ext(qm[(y, 9)]) for y in years],
                   "neXtSIM": [ext(orig[(y, 9)]) for y in years]}, index=list(years))
df["QM - NSIDC"] = df["QM target"] - df["NSIDC"]; df["neXtSIM - NSIDC"] = df["neXtSIM"] - df["NSIDC"]
print("=== Test 2: September extent, training targets vs NSIDC (M km2)")
print(df.round(2).to_string())
for col in ["NSIDC", "QM target", "neXtSIM", "QM - NSIDC", "neXtSIM - NSIDC"]:
    b, se, _ = fit(df.index, df[col])
    print(f"  trend {col:16s} {10 * b:+.2f} +/- {10 * se:.2f} per decade")
for a, b in [(1995, 2006), (2007, 2018), (2013, 2018)]:
    print(f"  mean QM - NSIDC {a}-{b}: {df.loc[a:b, 'QM - NSIDC'].mean():+.2f}")

# Test 3
tus = {**cube_series(paths.DATACUBE, "tus"), **cube_series(paths.ERA5_EXTENSION, "tus")}
jja = pd.Series({y: np.mean([tus[(y, m)][north].mean() for m in (6, 7, 8)]) for y in range(1995, 2026)}) - 273.15
train_max = jja.loc[1995:2014].max()
model = {}
for y in range(2009, 2019):                    # drift free-runs, mean over the starts that cover year y
    vals = []
    for f in sorted(glob.glob(f"{paths.DRIFT_OUT}/freerun_{MODEL}_era5_*.csv")):
        s = int(f[-8:-4]); d = pd.read_csv(f)
        k = (y - s) * 12 - 3                        # row k = start Dec s + k months: Sept of year y
        if 0 < k < len(d):
            vals.append(d.ext_members_km2[k] / 1e6)
    model[y] = np.mean(vals)
hc = sorted(glob.glob(f"{paths.SNAPSHOTS}/{MODEL}_era5_2018/members_*09.nc"))
for f in hc:
    y = int(f[-9:-5]); m = np.nan_to_num(xr.open_dataset(f).sic.values)
    model[y] = np.mean([ext(x) for x in m])
yrs = sorted(model)
t3 = pd.DataFrame({"JJA T north of 70N (C)": jja.loc[yrs], "model Sept": [model[y] for y in yrs],
                   "NSIDC Sept": [obs_sep[y] for y in yrs]}, index=yrs)
t3["model - NSIDC"] = t3["model Sept"] - t3["NSIDC Sept"]
t3["warmer than training"] = t3["JJA T north of 70N (C)"] > train_max
print(f"\n=== Test 3: summer warmth (training years 1995-2014: max {train_max:.2f} C, mean {jja.loc[1995:2014].mean():.2f} C)")
print(t3.round(2).to_string())
for name, col in [("model", "model Sept"), ("NSIDC", "NSIDC Sept"), ("model - NSIDC", "model - NSIDC")]:
    b, se, r = fit(t3["JJA T north of 70N (C)"], t3[col])
    print(f"  {name:14s} Sept extent per +1 C of JJA temperature: {b:+.2f} +/- {se:.2f} M km2  (r {r:+.2f})")
dt = t3.copy(); X = np.vstack([np.ones(len(dt)), dt.index.values - 2017]).T
for col in ["JJA T north of 70N (C)", "model Sept", "NSIDC Sept"]:          # year-to-year only: remove linear trends
    dt[col] = dt[col] - X @ np.linalg.lstsq(X, dt[col].values, rcond=None)[0]
for name, col in [("model", "model Sept"), ("NSIDC", "NSIDC Sept")]:
    b, se, r = fit(dt["JJA T north of 70N (C)"], dt[col])
    print(f"  detrended {name:6s} Sept extent per +1 C: {b:+.2f} +/- {se:.2f} M km2  (r {r:+.2f})")
w, c = t3[t3["warmer than training"]], t3[~t3["warmer than training"]]
print(f"  model - NSIDC: {w['model - NSIDC'].mean():+.2f} in the {len(w)} summers warmer than any training year, "
      f"{c['model - NSIDC'].mean():+.2f} in the other {len(c)}")
