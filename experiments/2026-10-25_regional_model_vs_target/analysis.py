"""Model or training target? The summer excess by region (roadmap "Next", analyses 1 and 2).

Splits each region's extent error against NSIDC-0051 into the model's own part (model
minus its training target) and the target's part (target minus NSIDC):
  1. From the drift free-runs with fields (drift.py --fields: Dec 2008-2014 starts to
     Dec 2018, June-October sic of every member), per region and summer month, 2009-18:
     model - target, target - NSIDC, model - NSIDC, and model - target by years since
     the start (does the model's own regional error grow?).
  2. The training targets alone, every September 1995-2018: target - NSIDC per region,
     for neXtSIM, the fixed-map QM targets and the 9-year-window QM targets.
Cells where NSIDC has no data in that month are left out of every term (ISS-027).

Run from the repo root (after drift.py --model <model> --run --fields):
    python experiments/2026-10-25_regional_model_vs_target/analysis.py [--model M] [--starts 2008-2014]
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
ap.add_argument("--model", default="monthly_pf_qm_ema_qmw")
ap.add_argument("--starts", default="2008-2014")
ap.add_argument("--runs-dir", default=os.path.join(paths.DRIFT_OUT, "fields"))
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("drift", os.path.join(RL.ROOT, "scripts/evaluate/drift.py"))
drift = importlib.util.module_from_spec(spec); spec.loader.exec_module(drift)
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e12          # M km^2
sectors = {"all": ocean, **sc.sector_masks(aux.longitude.values, ocean)}
ice = lambda x: np.nan_to_num(x) > 0.15


def by_month(folder):
    out = {}
    for f in sorted(glob.glob(f"{folder}/*_datacube_*.nc")):
        c = xr.open_dataset(f)["datacube"]
        for t, x in zip(pd.DatetimeIndex(c.time.values), c.sel(var_names="sic").values):
            out[(t.year, t.month)] = x
    return out


o = xr.open_mfdataset(sorted(glob.glob(f"{paths.NSIDC0051_GRID}/*.nc")), combine="by_coords")
v = o[[n for n in ("datacube", "state") if n in o.data_vars][0]].sel(var_names="sic")
obs = {(t.year, t.month): v.isel(time=i).values for i, t in enumerate(pd.DatetimeIndex(v.time.values)) if t.year <= 2018}


def sector_extents(field, valid):
    f = ice(field) & valid
    return {s: float((area * (f & m)).sum()) for s, m in sectors.items()}


pd.set_option("display.width", 220)
target_cube = drift.model_datacube(args.model)
target = by_month(target_cube)

# 1. Model against its target, from the free-runs
y0, y1 = (int(s) for s in args.starts.split("-"))
rows = []
for s0 in range(y0, y1 + 1):
    snap = os.path.join(args.runs_dir, "snapshots", f"{RL.model_tag(os.path.join(paths.MODELS, args.model, 'last.ckpt'))}_era5_{s0}")
    files = sorted(glob.glob(f"{snap}/members_*.nc"))
    if not files:
        raise SystemExit(f"no member fields in {snap}: run drift.py --model {args.model} --run --fields")
    for f in files:
        y, m = int(f[-9:-5]), int(f[-5:-3])
        if m not in (6, 7, 8, 9, 10) or (y, m) not in obs:
            continue
        valid = ocean & np.isfinite(obs[(y, m)])
        tg, ob = sector_extents(target[(y, m)], valid), sector_extents(obs[(y, m)], valid)
        for k, mem in enumerate(xr.open_dataset(f).sic.values):
            md = sector_extents(mem, valid)
            for s in sectors:
                rows.append(dict(start=s0, year=y, month=m, lead=y - s0, member=k, sector=s,
                                 model_minus_target=md[s] - tg[s], target_minus_obs=tg[s] - ob[s], model_minus_obs=md[s] - ob[s]))
df = pd.DataFrame(rows)
terms = ["model_minus_target", "target_minus_obs", "model_minus_obs"]
per_start = df.groupby(["sector", "month", "start"])[terms].mean()
mean = per_start.groupby(["sector", "month"]).mean()
se = per_start.groupby(["sector", "month"]).std(ddof=1) / np.sqrt(per_start.index.get_level_values("start").nunique())
print(f"=== 1. {args.model} against its target ({target_cube}) and NSIDC, {df.year.min()}-{df.year.max()}, "
      f"{df.start.nunique()} starts x {df.member.nunique()} members (M km^2; +/- = standard error over starts)")
print("\nSeptember, by region:"); print(mean.xs(9, level="month").loc[list(sectors)].round(3).to_string())
print("\n  standard error over starts:"); print(se.xs(9, level="month").loc[list(sectors)].round(3).to_string())
for s in ("East Siberian", "Beaufort", "all"):
    print(f"\n{s}, by month:"); print(mean.xs(s, level="sector").round(3).to_string())
print("\nSeptember model - target by years since the start (does the model's own regional error grow?):")
lead = df[df.month == 9].groupby(["sector", "lead"]).model_minus_target.mean().unstack("lead").round(3)
print(lead.loc[["all", "East Siberian", "Beaufort", "Chukchi", "Laptev"]].to_string())

# 2. The training targets against NSIDC, every September
print("\n=== 2. Training targets minus NSIDC, September, by region (M km^2)")
cubes = {"neXtSIM": paths.DATACUBE, "QM fixed map": paths.DATACUBE_SICQM, "QM 9-yr windows": paths.DATACUBE_SICQM + "w9"}
rows = []
for name, folder in cubes.items():
    if not os.path.isdir(folder):
        continue
    tc = by_month(folder)
    for y in range(1995, 2019):
        valid = ocean & np.isfinite(obs[(y, 9)])
        tg, ob = sector_extents(tc[(y, 9)], valid), sector_extents(obs[(y, 9)], valid)
        rows += [dict(targets=name, year=y, sector=s, diff=tg[s] - ob[s]) for s in sectors]
t2 = pd.DataFrame(rows)
for period, (a, b) in {"1995-2018": (1995, 2018), "2009-2018": (2009, 2018)}.items():
    print(f"\n{period}:"); print(t2[(t2.year >= a) & (t2.year <= b)].pivot_table(index="targets", columns="sector", values="diff", sort=False)[list(sectors)].round(3).to_string())
print("\nEast Siberian, QM 9-yr windows minus NSIDC by year:")
print(t2[(t2.targets == "QM 9-yr windows") & (t2.sector == "East Siberian")].set_index("year")["diff"].round(2).to_string())
