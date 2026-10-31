"""Does Frazil melt as much ice over summer as its targets do?

Free runs already on disk: the drift runs (drift.py --run --fields), 2 members each,
started every December 2008-2014 and run to December 2018, compared month by month
with the targets the model was trained on (v0.4.0: monthly_datacube_sicqmw9; the
thickness sensitivity test: monthly_datacube_sicqmw9_sitra).
  1. Pan-Arctic: monthly mean thickness and concentration (the run's own CSV, an
     unweighted mean over ocean cells; the target computed the same way), and the
     April-to-September loss, by the summer's lead year (1 = the first summer after the start).
  2. By sector, June-September (the months the drift runs saved as fields):
     area-mean thickness and extent, and the June-to-September loss.

Run from the repo root:  python experiments/2026-10-30_summer_melt/analysis.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)
MODELS = {"monthly_pf_qm_ema_qmw": "monthly_datacube_sicqmw9", "monthly_pf_qm_ema_qmw_sit": "monthly_datacube_sicqmw9_sitra"}
STARTS, END = range(2008, 2015), 2018
FIELDS = os.path.join(paths.DRIFT_OUT, "fields")
SECTORS = ["East Siberian", "Chukchi", "Beaufort", "Laptev"]

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = {s: m for s, m in sc.sector_masks(aux.longitude.values, ocean).items() if s in SECTORS}
sec["pan-Arctic"] = ocean


def target(cube_dir):
    c = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, cube_dir, "*_datacube_*.nc"))), combine="by_coords")["datacube"]
    t = pd.DatetimeIndex(c.time.values); keep = t.year >= STARTS[0]
    return {v: c.sel(var_names=v).isel(time=np.where(keep)[0]).load().values for v in ("sit", "sic")}, t[keep]


def sector_stats(sit, sic):
    out = {}
    for s, m in sec.items():
        w = area * m
        out[(s, "sit")] = float((np.nan_to_num(sit) * w).sum() / w.sum())
        out[(s, "ext")] = float((area * m * (np.nan_to_num(sic) > 0.15)).sum()) / 1e6
    return out


pan, reg = [], []
for model, cube in MODELS.items():
    T, tt = target(cube)
    idx = {(t.year, t.month): i for i, t in enumerate(tt)}
    for y0 in STARTS:
        d = pd.read_csv(os.path.join(FIELDS, f"freerun_{model}_era5_{y0}.csv"))
        months = pd.date_range(f"{y0}-12-01", periods=len(d), freq="MS")
        for k, t in enumerate(months):
            i = idx[(t.year, t.month)]
            pan.append(dict(model=model, start=y0, lead=t.year - y0, year=t.year, month=t.month,
                            sit=d.mean_sit[k], sit_t=float(np.nanmean(T["sit"][i][ocean])),
                            sic=d.mean_sic[k], sic_t=float(np.nanmean(T["sic"][i][ocean]))))
        snap = os.path.join(FIELDS, "snapshots", f"{model}_era5_{y0}")
        for f in sorted(glob.glob(f"{snap}/state_??????.nc")):
            y, m = int(f[-9:-5]), int(f[-5:-3])
            if m > 9:
                continue
            s = xr.open_dataset(f).state
            st = sector_stats(s.sel(var_names="sit").values.squeeze(), s.sel(var_names="sic").values.squeeze())
            tg = sector_stats(T["sit"][idx[(y, m)]], T["sic"][idx[(y, m)]])
            for (g, v), x in st.items():
                reg.append(dict(model=model, start=y0, lead=y - y0, year=y, month=m, sector=g, var=v, frazil=x, target=tg[(g, v)]))
P, R = pd.DataFrame(pan), pd.DataFrame(reg)

# sanity: the run starts from the target's December state
d0 = P[(P.month == 12) & (P.year == P.start)]
print(f"start-state check, mean |sit - target| at lead 0: {np.abs(d0.sit - d0.sit_t).max():.4f} m\n")

print("=== 1. Pan-Arctic monthly mean thickness (m), all summers 2009-2018: Frazil minus target, by month")
P["dsit"], P["dsic"] = P.sit - P.sit_t, P.sic - P.sic_t
print(P[P.lead >= 1].pivot_table(index="model", columns="month", values="dsit").round(3).to_string())
print("\nsame for mean concentration")
print(P[P.lead >= 1].pivot_table(index="model", columns="month", values="dsic").round(3).to_string())

print("\nApril-to-September loss (m): Frazil vs target, by lead year (n = number of start-summers)")
rows = []
for (model, y0, lead), g in P[P.lead >= 1].groupby(["model", "start", "lead"]):
    a, s = g[g.month == 4], g[g.month == 9]
    if len(a) and len(s):
        rows.append(dict(model=model, lead=min(lead, 4), frazil=float(a.sit.iloc[0] - s.sit.iloc[0]),
                         target=float(a.sit_t.iloc[0] - s.sit_t.iloc[0]), apr_err=float(a.dsit.iloc[0]), sep_err=float(s.dsit.iloc[0])))
L = pd.DataFrame(rows); L["ratio"] = L.frazil / L.target
out = L.groupby(["model", "lead"])[["frazil", "target", "ratio", "apr_err", "sep_err"]].mean().round(3)
out["n"] = L.groupby(["model", "lead"]).size()
print(out.rename(index={4: "4+"}, level=1).to_string())

print("\n=== 2. By sector, June-September, all summers (lead >= 1): Frazil minus target")
R["diff"] = R.frazil - R.target
for v, unit in (("sit", "m, area mean"), ("ext", "M km^2 extent")):
    print(f"\n{v} ({unit})")
    print(R[(R["var"] == v) & (R.lead >= 1)].pivot_table(index=["model", "sector"], columns="month", values="diff", sort=False).round(3).to_string())
print("\nJune-to-September thickness loss (m): Frazil / target / ratio")
J = R[(R["var"] == "sit") & (R.lead >= 1)].pivot_table(index=["model", "sector", "start", "year"], columns="month", values=["frazil", "target"])
loss = pd.DataFrame({"frazil": J[("frazil", 6)] - J[("frazil", 9)], "target": J[("target", 6)] - J[("target", 9)]})
o = loss.groupby(["model", "sector"], sort=False).mean(); o["ratio"] = o.frazil / o.target
print(o.round(3).to_string())
