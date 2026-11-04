"""One month at a time from the target's own state: is Frazil's August shelf ice cover
already too high in a single step, or does it build up over the summer?

Reads the one-month predictions saved by
    eval_monthly.py --ckpt data/models/<model>/last.ckpt --datacube <its target> --n-ens 8 --save-fields <dir>
(validation years 2015-2018; each month predicted from the target's previous month,
so no error carries over) and compares each with the target's month, by sector:
extent (> 15%) and ice area from each member's sic (averaged over members, as the
scorecard does), the extent of loose ice (15-80%), and area-mean thickness (ensemble mean).
For comparison, the drift runs' first summer (free runs from the target's December
state, same sectors) is in output.txt: Pacific shelves +0.15 M km^2 in August-September.

Run from the repo root:
    python experiments/2026-10-30_summer_melt/one_month.py [--model M] [--fields DIR]
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
ap.add_argument("--datacube", default=os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
ap.add_argument("--fields", default=None, help="default: data/models/<model>/one_month_fields")
args = ap.parse_args()
fields = args.fields or os.path.join(paths.MODELS, args.model, "one_month_fields")
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = {s: m for s, m in sc.sector_masks(aux.longitude.values, ocean).items() if s in ("East Siberian", "Chukchi", "Beaufort", "Laptev")}
sec["Pacific shelves"] = sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"]
sec["pan-Arctic"] = ocean
cube = xr.open_mfdataset(sorted(glob.glob(f"{args.datacube}/*_datacube_*.nc")), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)


def stats(sit, sics):
    out = {}
    for s, m in sec.items():
        w = area * m
        c = [np.nan_to_num(x) for x in sics]
        out[(s, "ext")] = np.mean([(w * (x > 0.15)).sum() for x in c]) / 1e6
        out[(s, "area")] = np.mean([(w * x).sum() for x in c]) / 1e6
        out[(s, "loose")] = np.mean([(w * (x > 0.15) * (x < 0.8)).sum() for x in c]) / 1e6
        out[(s, "sit")] = float((np.nan_to_num(sit) * w).sum() / w.sum())
    return out


rows = []
files = sorted(glob.glob(f"{fields}/pred_??????.nc"))
if not files:
    raise SystemExit(f"no predictions in {fields}: run eval_monthly.py --save-fields {fields} first")
for f in files:
    p = xr.open_dataset(f); t = pd.Timestamp(p.time.values)
    i = int(np.where((ct.year == t.year) & (ct.month == t.month))[0][0])
    tg = cube.isel(time=i)
    a = stats(p.state.sel(var_names="sit").values, list(p.sic_members.values))
    b = stats(tg.sel(var_names="sit").values, [tg.sel(var_names="sic").values])
    rows += [dict(year=t.year, month=t.month, sector=g, var=v, frazil=x, target=b[(g, v)]) for (g, v), x in a.items()]
D = pd.DataFrame(rows); D["diff"] = D.frazil - D.target
print(f"{args.model}: {len(files)} one-month predictions, {D.year.min()}-{D.year.max()} (each from the target's previous month)\n")

order = ["East Siberian", "Chukchi", "Beaufort", "Pacific shelves", "Laptev", "pan-Arctic"]
for v, unit in (("ext", "extent > 15%, M km^2"), ("area", "ice area, M km^2"), ("loose", "extent of 15-80% ice, M km^2"), ("sit", "thickness, m, area mean")):
    print(f"=== one-step prediction minus target: {unit}, mean over years, by predicted month")
    print(D[D["var"] == v].pivot_table(index="sector", columns="month", values="diff").reindex(order).round(3).to_string(), "\n")

print("=== Pacific shelves, by year: August and September extent, one step minus target (M km^2)")
print(D[(D.sector == "Pacific shelves") & (D["var"] == "ext") & D.month.isin([7, 8, 9])].pivot_table(index="year", columns="month", values="diff").round(3).to_string())

print("\n=== Extent change over the month (M km^2): the target's, Frazil's one step, and their ratio (1 = right size of step)")
E = D[D["var"] == "ext"].sort_values(["sector", "year", "month"]).copy()
E["prev"] = E.groupby("sector").target.shift(1)              # the target's previous month = the step's start
E = E.dropna(subset=["prev"])
E["target_step"], E["frazil_step"] = E.target - E.prev, E.frazil - E.prev
S = E[E.sector.isin(["Pacific shelves", "pan-Arctic"])].groupby(["sector", "month"])[["target_step", "frazil_step"]].mean()
S["ratio"] = S.frazil_step / S.target_step
print(S.round(3).unstack("sector").to_string())
