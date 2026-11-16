"""Does v0.4.0's one-month error sit where QM moved the model's input (ISS-030)?

Cells are grouped by the model's inputs only, never by the target:
  1. by how far QM moved the starting sic (QM - teacher, the month before), within each starting class;
  2. by the starting thickness per unit of ice (sit / sic of the input), within each starting class.
For each group: area share, the target's and the members' mean change of sic over the month, the bias, and the
share that opens (ends below 15%). August and September, Pacific shelves and Barents/Kara, from
eval_monthly --save-fields (each month predicted from the target's previous month).
Run from the repo root:  python experiments/2026-11-14_assumptions_review/qm_error_location.py
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
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
regions = {"Pacific shelves": sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"], "Barents/Kara": sec["Barents/Kara"]}


def cube(d):
    return xr.open_mfdataset(sorted(glob.glob(os.path.join(d, "*_datacube_*.nc"))), combine="by_coords")["datacube"]


qm, raw = cube(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9")), cube(paths.DATACUBE)
ct, rt = pd.DatetimeIndex(qm.time.values), pd.DatetimeIndex(raw.time.values)
val = lambda da, i, v: np.nan_to_num(da.isel(time=i).sel(var_names=v).values)
rows = []
for sub in ("one_month_fields_2011", "one_month_fields"):
    for f in sorted(glob.glob(os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw", sub, "pred_??????.nc"))):
        ym = os.path.basename(f)[5:11]; y, m = int(ym[:4]), int(ym[4:])
        if m not in (8, 9):
            continue
        i = int(np.where((ct.year == y) & (ct.month == m))[0][0]); j = int(np.where(rt == ct[i - 1])[0][0])
        start, tgt, sit0, start_raw = val(qm, i - 1, "sic"), val(qm, i, "sic"), val(qm, i - 1, "sit"), val(raw, j, "sic")
        mem = np.nan_to_num(xr.open_dataset(f).sic_members.values)
        for r, mk in regions.items():
            c = mk & (start >= 0.15) & (start < 0.8)
            rows.append(pd.DataFrame({"years": "2011-14 (trained on)" if y < 2015 else "2015-18", "region": r, "m": m, "w": area[c],
                                      "start": start[c], "moved": start[c] - start_raw[c], "h": sit0[c] / start[c],
                                      "target change": tgt[c] - start[c], "members change": mem[:, c].mean(0) - start[c],
                                      "target opens": (tgt[c] < 0.15).astype(float), "members open": (mem[:, c] < 0.15).mean(0)}))
D = pd.concat(rows, ignore_index=True)
D["cls"] = pd.cut(D.start, [0.15, 0.5, 0.8], right=False, labels=["15-50%", "50-80%"])
D["QM moved start"] = pd.cut(D.moved, [-1, -0.15, -0.05, 0.05, 0.15, 1], labels=["down >15", "down 5-15", "within 5", "up 5-15", "up >15"])
D["ice thickness"] = pd.cut(D.h, [0, 1, 2, 100], labels=["<1 m", "1-2 m", ">2 m"])
COLS = ["target change", "members change", "target opens", "members open"]


def table(by):
    def agg(g):
        s = pd.Series({k: np.average(g[k], weights=g.w) for k in COLS}); s["bias"] = s["members change"] - s["target change"]; s["area"] = g.w.sum()
        return s
    T = D.groupby(["years", "region", "m", "cls", by], observed=True).apply(agg)
    T["area share"] = T.area / T.groupby(level=[0, 1, 2, 3]).area.transform("sum")
    return T.drop(columns="area")[["area share"] + COLS + ["bias"]]


pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
print("=== 1. by how far QM moved the starting sic (QM - teacher), cells starting at 15-80%")
print(table("QM moved start").round(3).to_string())
print("\n=== 2. by the input's thickness per unit of ice (sit / sic)")
print(table("ice thickness").round(3).to_string())

# 3. No model: in the training data itself (1995-2014), does thin loose ice open more often than thick loose ice?
#    Teacher cube against QM cube, each from its own starting sic; sit is the teacher's in both.
rows = []
for name, da, tt in (("teacher", raw, rt), ("QM", qm, ct)):
    for i in np.where(np.isin(tt.month, (8, 9)) & (tt.year >= 1995) & (tt.year <= 2014))[0]:
        start, end, sit0 = val(da, i - 1, "sic"), val(da, i, "sic"), val(da, i - 1, "sit")
        for r, mk in regions.items():
            c = mk & (start >= 0.15) & (start < 0.8)
            rows.append(pd.DataFrame({"cube": name, "region": r, "m": tt[i].month, "w": area[c], "start": start[c], "h": sit0[c] / start[c],
                                      "opens": (end[c] < 0.15).astype(float), "change": end[c] - start[c]}))
T = pd.concat(rows, ignore_index=True)
T["cls"] = pd.cut(T.start, [0.15, 0.5, 0.8], right=False, labels=["15-50%", "50-80%"])
T["ice thickness"] = pd.cut(T.h, [0, 1, 2, 100], labels=["<1 m", "1-2 m", ">2 m"])
g = T.groupby(["region", "m", "cls", "ice thickness", "cube"], observed=True)
out = pd.DataFrame({"opens": g.apply(lambda x: np.average(x.opens, weights=x.w)), "change": g.apply(lambda x: np.average(x.change, weights=x.w)), "area": g.w.sum()})
out["area share"] = out.area / out.groupby(level=[0, 1, 2, 4]).area.transform("sum")
print("\n=== 3. training data 1995-2014, no model: share of loose ice that opens within a month, by the start's thickness per unit of ice")
print(out.drop(columns="area").unstack("cube").round(3).to_string())
