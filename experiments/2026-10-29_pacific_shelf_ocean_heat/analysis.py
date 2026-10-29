"""Step 1: does the training simulation's ocean heat explain Frazil's own Pacific-shelf error?

Uses the upper-ocean heat content of neXtSIM-OPA (scripts/download/nextsim_ocean_heat.py)
on the model grid, per sector (East Siberian, Chukchi, Beaufort; Laptev as a contrast):
  A. its seasonal cycle and its 1995-2018 trend, in the mixed layer (0-20 m) and below
     (20-100 m), and how far 2009-18 sits above the 1995-2014 training years;
  B. year by year in 2009-18, whether Frazil's own error in a sector (model - target
     September extent, June thickness; drift runs with fields, drift.py --fields) is larger
     when the simulation's ocean heat there is larger. The error grows with years since
     the start of a run, so each error is first adjusted for that (minus the mean error at
     the same lead), then averaged per year.

Run from the repo root:  python experiments/2026-10-29_pacific_shelf_ocean_heat/analysis.py [--model M]
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

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="monthly_pf_qm_ema_qmw")
ap.add_argument("--runs-dir", default=os.path.join(paths.DRIFT_OUT, "fields"))
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
spec = importlib.util.spec_from_file_location("drift", os.path.join(RL.ROOT, "scripts/evaluate/drift.py"))
drift = importlib.util.module_from_spec(spec); spec.loader.exec_module(drift)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values
lat, lon = aux.latitude.values, (aux.longitude.values + 360) % 360
in_box = ocean & (lat >= 64) & (lon >= 90) & (lon <= 250)
SECTORS = ["East Siberian", "Chukchi", "Beaufort", "Laptev"]
sec = {s: m & in_box for s, m in sc.sector_masks(aux.longitude.values, ocean).items() if s in SECTORS}

# Ocean heat on the model grid, sector means by month
oh = xr.open_mfdataset(sorted(glob.glob(f"{paths.NEXTSIM_OCEAN_HEAT}/ohc_*.nc")), combine="by_coords")
src_lat, src_lon = oh.latitude.values, oh.longitude.values
idx = P.build_nearest_index(src_lon.ravel(), src_lat.ravel(), aux.longitude.values[in_box], aux.latitude.values[in_box])
times = pd.DatetimeIndex(oh.time.values)
rows = []
for layer in ("ohc_0_20", "ohc_20_100"):
    field = oh[layer].values.reshape(len(times), -1)[:, idx]               # (time, cells in box)
    full = np.zeros((len(times),) + ocean.shape, np.float32); full[:, in_box] = field
    for s, m in sec.items():
        w = area * m
        series = (full * w).sum((1, 2)) / w.sum()
        rows += [dict(layer=layer, sector=s, year=t.year, month=t.month, ohc=float(v)) for t, v in zip(times, series)]
H = pd.DataFrame(rows)

print(f"=== A. neXtSIM-OPA upper-ocean heat above freezing, MJ m^-2, {times[0]:%Y}-{times[-1]:%Y}")
for layer in ("ohc_0_20", "ohc_20_100"):
    d = H[H.layer == layer]
    print(f"\n{layer}: monthly climatology")
    print(d.pivot_table(index="sector", columns="month", values="ohc").loc[SECTORS].round(0).to_string())
    out = []
    for s in SECTORS:
        ann = d[d.sector == s].groupby("year").ohc.mean()
        b, a = np.polyfit(ann.index, ann.values, 1)
        se = np.sqrt((ann.values - (a + b * ann.index.values)).var(ddof=2) / ((ann.index.values - ann.index.values.mean()) ** 2).sum())
        out.append(dict(sector=s, mean=ann.mean(), trend_per_decade=10 * b, trend_se=10 * se,
                        training_1995_2014=ann.loc[1995:2014].mean(), run_years_2009_2018=ann.loc[2009:2018].mean()))
    print(f"{layer}: annual mean, trend and level"); print(pd.DataFrame(out).set_index("sector").round(1).to_string())

# Frazil's own error per (start, year, sector) from the drift runs with fields
target = {}
for f in sorted(glob.glob(f"{drift.model_datacube(args.model)}/*_datacube_*.nc")):
    c = xr.open_dataset(f)["datacube"]
    for t, sic, sit in zip(pd.DatetimeIndex(c.time.values), c.sel(var_names="sic").values, c.sel(var_names="sit").values):
        target[(t.year, t.month)] = (np.nan_to_num(sic), np.nan_to_num(sit))
tag = RL.model_tag(os.path.join(paths.MODELS, args.model, "last.ckpt"))
err = []
for snap in sorted(glob.glob(os.path.join(args.runs_dir, "snapshots", f"{tag}_era5_*"))):
    s0 = int(snap[-4:])
    for y in range(s0 + 1, 2019):
        mem, st = os.path.join(snap, f"members_{y}09.nc"), os.path.join(snap, f"state_{y}06.nc")
        if not (os.path.exists(mem) and os.path.exists(st)):
            continue
        members = np.nan_to_num(xr.open_dataset(mem).sic.values)
        sit = xr.open_dataset(st); da = sit[list(sit.data_vars)[0]]
        sit_m = np.nan_to_num(da.sel(var_names="sit").values) if "var_names" in da.dims else np.nan_to_num(sit["sit"].values)
        for s, m in sec.items():
            ext = lambda f: float((area * ((f > 0.15) & m)).sum()) / 1e12
            w = area * m
            err.append(dict(start=s0, year=y, lead=y - s0, sector=s,
                            sep_extent=np.mean([ext(x) for x in members]) - ext(target[(y, 9)][0]),
                            june_thickness=float((sit_m * w).sum() / w.sum() - (target[(y, 6)][1] * w).sum() / w.sum())))
E = pd.DataFrame(err)
if E.empty:
    raise SystemExit(f"no drift fields in {args.runs_dir}: run drift.py --model {args.model} --run --fields")
for v in ("sep_extent", "june_thickness"):                     # remove the growth with years since the start
    E[v + "_adj"] = E[v] - E.groupby(["sector", "lead"])[v].transform("mean")
per_year = E.groupby(["sector", "year"])[["sep_extent_adj", "june_thickness_adj"]].mean()

# Ocean-heat predictors per (sector, year): the autumn before (Oct-Dec of year-1) below 20 m, and the same summer (Jul-Aug) above 20 m
def season(layer, months, shift=0):
    d = H[(H.layer == layer) & (H.month.isin(months))].copy(); d["year"] += shift
    return d.groupby(["sector", "year"]).ohc.mean()
pred = pd.DataFrame({"autumn_before_20_100m": season("ohc_20_100", [10, 11, 12], shift=1),
                     "summer_0_20m": season("ohc_0_20", [7, 8])})
J = per_year.join(pred, how="inner").reset_index()
J[["autumn_before_20_100m", "summer_0_20m"]] = J.groupby("sector")[["autumn_before_20_100m", "summer_0_20m"]].transform(lambda x: x - x.mean())

print(f"\n=== B. {args.model}'s own error (lead-adjusted) against the simulation's ocean heat, 2009-18")
out = []
for s in SECTORS + ["Pacific shelf (3 sectors)"]:
    d = J[J.sector.isin(SECTORS[:3])] if s.startswith("Pacific") else J[J.sector == s]
    for e in ("sep_extent_adj", "june_thickness_adj"):
        for p in ("autumn_before_20_100m", "summer_0_20m"):
            r = np.corrcoef(d[p], d[e])[0, 1]
            out.append(dict(sector=s, error=e.replace("_adj", ""), heat=p, r=r, n=len(d)))
print(pd.DataFrame(out).pivot_table(index=["sector", "error"], columns="heat", values="r", sort=False).round(2).to_string())
print(f"(n = years per sector, {J.year.nunique()}; |r| above ~0.63 for 10 points, ~0.36 for 30, is significant at 5%)")
print("\nMean own error over all runs (not lead-adjusted), September extent (M km^2) and June thickness (m):")
print(E.groupby("sector")[["sep_extent", "june_thickness"]].mean().loc[SECTORS].round(3).to_string())
