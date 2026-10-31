"""Two checks before choosing how to give Frazil ocean information (route A or B, README).

Per sector (East Siberian, Chukchi, Beaufort; Laptev as a contrast), cells north of 64 N
covered by every source:
  1. Ocean: does TOPAZ4b (the ocean an input would come from, and the one that forces the
     Copernicus neXtSIM reanalysis) show the same upper-ocean heat as the training
     simulation's own ocean (neXtSIM-OPA) over 1995-2018: level, trend, year-to-year
     agreement, and the rise from the training years (1995-2014) to 2009-18?
  2. Teacher: September extent against NSIDC-0051 for the reanalysis, neXtSIM-OPA (raw and
     the v0.4.0 targets) and Frazil v0.4.0's hindcast; and winter (November-April) thickness
     of neXtSIM-OPA and Frazil against the reanalysis, which assimilates CS2SMOS winter
     thickness from October 2010 (so it stands in for observed thickness, winters 2011-18
     and 2020-22). Years after 2018 are the development years only (2019-22); the 2023-25
     holdout is left alone.

Run from the repo root:  python experiments/2026-10-29_ocean_input_and_teacher_checks/analysis.py [--model M]
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
import frazil.preprocess as P
from frazil import paths
from frazil.io import load_series, load_sic

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="monthly_pf_qm_ema_qmw", help="whose 2019-22 hindcast to compare (Frazil v0.4.0 by default)")
MODEL = ap.parse_args().model
SECTORS = ["East Siberian", "Chukchi", "Beaufort", "Laptev"]
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 200)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6      # km^2
lat, lon = aux.latitude.values, (aux.longitude.values + 360) % 360
in_box = ocean & (lat >= 64) & (lon >= 90) & (lon <= 250)


def trend(s):
    b, a = np.polyfit(s.index.values, s.values, 1)
    se = np.sqrt((s.values - (a + b * s.index.values)).var(ddof=2) / ((s.index.values - s.index.values.mean()) ** 2).sum())
    return 10 * b, 10 * se


def sector_means(field_tyx, times, sec, name):
    rows = []
    for s, m in sec.items():
        w = area * m
        v = (np.nan_to_num(field_tyx) * w).sum((1, 2)) / w.sum()
        rows += [dict(var=name, sector=s, year=t.year, month=t.month, value=float(x)) for t, x in zip(times, v)]
    return rows


# ---- 1. Ocean: TOPAZ4b vs neXtSIM-OPA ----
tp = xr.open_mfdataset(sorted(glob.glob(f"{paths.TOPAZ_OCEAN_HEAT}/ohc_*.nc")), combine="by_coords")
tt = pd.DatetimeIndex(tp.time.values)
nx = xr.open_mfdataset(sorted(glob.glob(f"{paths.NEXTSIM_OCEAN_HEAT}/ohc_*.nc")), combine="by_coords")
nt = pd.DatetimeIndex(nx.time.values)
idx = P.build_nearest_index(nx.longitude.values.ravel(), nx.latitude.values.ravel(), aux.longitude.values, aux.latitude.values)
nx_grid = {k: np.where(in_box, P.regrid_time(nx[k].values.reshape(len(nt), -1), idx, ocean.shape), np.nan)
           for k in ("ohc_0_20", "ohc_20_100")}
both = in_box & np.isfinite(tp.ohc_20_100.isel(time=0).values) & np.isfinite(nx_grid["ohc_20_100"][0]) \
    & (nx_grid["ohc_20_100"].max(0) > 0)                      # water in neXtSIM-OPA's top 20-100 m too
sec = {s: m & both for s, m in sc.sector_masks(aux.longitude.values, ocean).items() if s in SECTORS}
rows = []
for k in ("ohc_0_20", "ohc_20_100"):
    rows += [dict(r, source="neXtSIM-OPA") for r in sector_means(nx_grid[k], nt, sec, k)]
    rows += [dict(r, source="TOPAZ4b") for r in sector_means(tp[k].values, tt, sec, k)]
for k in ("sst", "mld"):
    rows += [dict(r, source="TOPAZ4b") for r in sector_means(tp[k].values, tt, sec, k)]
O = pd.DataFrame(rows)
ann = O.groupby(["source", "var", "sector", "year"]).value.mean()

print("=== 1. Upper-ocean heat above freezing (MJ m^-2), annual means; cells in both oceans, north of 64 N")
for k in ("ohc_0_20", "ohc_20_100"):
    out = []
    for s in SECTORS:
        a, b = ann.loc[("neXtSIM-OPA", k, s)].loc[1995:2018], ann.loc[("TOPAZ4b", k, s)]
        ta, sa = trend(a); tb, sb = trend(b.loc[1995:2018]); tc, scc = trend(b.loc[1995:2025])
        da, db = a.loc[2009:2018].mean() - a.loc[1995:2014].mean(), b.loc[2009:2018].mean() - b.loc[1995:2014].mean()
        out.append({"sector": s, "OPA mean": a.mean(), "TOPAZ mean": b.loc[1995:2018].mean(),
                    "OPA trend/dec": f"{ta:+.0f}±{sa:.0f}", "TOPAZ trend/dec": f"{tb:+.0f}±{sb:.0f}",
                    "TOPAZ 1995-2025": f"{tc:+.0f}±{scc:.0f}", "OPA rise": da, "TOPAZ rise": db,
                    "r annual": np.corrcoef(a.values, b.loc[1995:2018].values)[0, 1],
                    "r detrended": np.corrcoef(a.values - np.polyval(np.polyfit(a.index, a.values, 1), a.index),
                                               b.loc[1995:2018].values - np.polyval(np.polyfit(a.index, b.loc[1995:2018].values, 1), a.index))[0, 1]})
    print(f"\n{k} (rise = 2009-18 mean minus the 1995-2014 training mean)")
    print(pd.DataFrame(out).set_index("sector").round(2).to_string())
print("\nTOPAZ4b sea-surface temperature (C) and mixed-layer depth (m), annual mean and trend per decade, 1995-2025")
print(pd.DataFrame([{"sector": s, "SST": ann.loc[("TOPAZ4b", "sst", s)].mean(), "SST trend": "%+.2f±%.2f" % trend(ann.loc[("TOPAZ4b", "sst", s)]),
                     "MLD": ann.loc[("TOPAZ4b", "mld", s)].mean(), "MLD trend": "%+.1f±%.1f" % trend(ann.loc[("TOPAZ4b", "mld", s)])}
                    for s in SECTORS]).set_index("sector").round(2).to_string())

# ---- 2. Teacher: September extent vs NSIDC-0051, winter thickness vs the reanalysis ----
ra_sic = xr.open_dataset(paths.NEXTSIM_REANALYSIS.format(var="sic")).sic
ra_sit = xr.open_dataset(paths.NEXTSIM_REANALYSIS.format(var="sit")).sit
print("\n=== 2. Teacher (both thicknesses are grid-cell means: near the ice edge both fall to a few cm)")
rt = pd.DatetimeIndex(ra_sic.time.values)
cover = in_box & np.isfinite(ra_sic.isel(time=0).values)
sec2 = {s: m & cover for s, m in sc.sector_masks(aux.longitude.values, ocean).items() if s in SECTORS}
obs, ot = load_sic(paths.NSIDC0051_GRID)
opa, opt = load_series(paths.DATACUBE, ["sit", "sic"])
qmw, qt = load_sic(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
st = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.SNAPSHOTS, f"{MODEL}_era5_2018", "state_*.nc"))), combine="by_coords").state
ft = pd.DatetimeIndex(st.time.values)
fz_sit, fz_sic = st.sel(var_names="sit").values, st.sel(var_names="sic").values
mem = {(int(f[-9:-5]), int(f[-5:-3])): np.nan_to_num(xr.open_dataset(f).sic.values)
       for f in glob.glob(os.path.join(paths.SNAPSHOTS, f"{MODEL}_era5_2018", "members_????09.nc"))}
pick = lambda arr, times, y, m: arr[np.where((times.year == y) & (times.month == m))[0][0]]
ext = lambda f, w: float((area * ((np.nan_to_num(f) > 0.15) & w)).sum()) / 1e6

rows = []
for y in range(1995, 2023):
    o = pick(obs, ot, y, 9)
    srcs = {"reanalysis": pick(ra_sic.values, rt, y, 9)}
    if y <= 2018:
        srcs.update({"neXtSIM-OPA": pick(opa[:, 1], opt, y, 9), "v0.4.0 targets (QM)": pick(qmw, qt, y, 9)})
    elif (y, 9) in mem:
        srcs[MODEL] = None
    for s, w in sec2.items():
        for n, f in srcs.items():
            e = np.mean([ext(x, w) for x in mem[(y, 9)]]) if f is None else ext(f, w)
            rows.append(dict(source=n, sector=s, year=y, bias=e - ext(o, w)))
S = pd.DataFrame(rows)
S["period"] = np.where(S.year <= 2018, "1995-2018", "2019-22")
print("\nSeptember extent minus NSIDC-0051 (M km^2), mean over years")
print(S.pivot_table(index=["period", "source"], columns="sector", values="bias", sort=False)[SECTORS].round(3).to_string())

rows = []
winters = [(y, m) for y in range(2010, 2023) for m in (11, 12, 1, 2, 3, 4)]
for y, m in winters:
    wy = y + 1 if m >= 11 else y                                    # winter labelled by its April
    if not (2011 <= wy <= 2018 or 2020 <= wy <= 2022):
        continue
    r = pick(ra_sit.values, rt, y, m)
    srcs = {"neXtSIM-OPA (= the v0.4.0 target)": pick(opa[:, 0], opt, y, m)} if wy <= 2018 else {MODEL: pick(fz_sit, ft, y, m)}
    for s, w in sec2.items():
        ww = area * w
        mean = lambda f: float((np.nan_to_num(f) * ww).sum() / ww.sum())
        for n, f in srcs.items():
            rows.append(dict(source=n, sector=s, winter=wy, month=m, reanalysis=mean(r), model=mean(f), diff=mean(f) - mean(r)))
T = pd.DataFrame(rows)
print("\nWinter (Nov-Apr) mean thickness (m, area mean over the sector), and model minus reanalysis")
print(T.groupby(["source", "sector"], sort=False)[["model", "reanalysis", "diff"]].mean().round(2).to_string())
print("\nApril only, model minus reanalysis (m)")
print(T[T.month == 4].pivot_table(index="source", columns="sector", values="diff", sort=False)[SECTORS].round(2).to_string())
