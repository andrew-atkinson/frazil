"""Is the loose-ice gap a training limit or rarity in the data? Two no-training checks.

1. How often the training targets (1995-2014) open loose ice in one month: by starting class
   (the previous month's target, which is also the model's input), the share of cells that end
   below 15%, split into "partly open" (0-15%) and "open water" (at 0).
2. The same split for v0.4.0's members on years it trained on (2011-14, eval_monthly
   --save-fields, each month from the target's previous month), against the target.
If the members match the target in 2011-14 but not in 2015-18, it is generalisation; if they
miss the same way on training years, the model cannot fit it.

Run from the repo root:  python experiments/2026-11-05_loose_ice/train_years.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

ZERO = 1e-4   # members land ~1e-7 above the bound (sampler clamp + float error)
MONTHS = (8, 9)
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
regions = {"Pacific shelves": sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"], "Barents/Kara": sec["Barents/Kara"]}
cube = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9", "*_datacube_*.nc"))),
                         combine="by_coords")["datacube"].sel(var_names="sic")
ct = pd.DatetimeIndex(cube.time.values)
sic = lambda i: np.nan_to_num(cube.isel(time=i).values)


def cells(start, end, members=None):
    """Per-cell frame for cells starting at 15% or more; end outcomes as shares (members: over members)."""
    out = []
    for r, mk in regions.items():
        c = mk & (start >= 0.15)
        e = end[:, c] if end.ndim == 3 else end[None, c]
        out.append(pd.DataFrame({"region": r, "w": area[c], "start": start[c],
                                 "partly open": ((e > ZERO) & (e < 0.15)).mean(0), "open water": (e <= ZERO).mean(0)}))
    return pd.concat(out, ignore_index=True)


def table(D):
    D["cls"] = pd.cut(D.start, [0.15, 0.5, 0.8], include_lowest=True, labels=["15-50%", "50-80%"])
    D = D.dropna(subset=["cls"])
    return D.groupby(["region", "m", "cls"], observed=True).apply(
        lambda g: pd.Series({k: np.average(g[k], weights=g.w) for k in ("partly open", "open water")}))


rows = []
for i in np.where(np.isin(ct.month, MONTHS) & (ct.year <= 2014))[0]:
    d = cells(sic(i - 1), sic(i)); d["m"], d["y"] = ct[i].month, ct[i].year; rows.append(d)
T = pd.concat(rows, ignore_index=True)
print("=== 1. training targets: share of starting cells that end partly open (0-15%) or at open water")
print("--- 1995-2014"); print(table(T.copy()).round(3).to_string())
print("--- 2011-2014 only"); print(table(T[T.y >= 2011].copy()).round(3).to_string())

tgt, mem = [], []
for f in sorted(glob.glob(os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw", "one_month_fields_2011", "pred_??????.nc"))):
    ym = os.path.basename(f)[5:11]; y, m = int(ym[:4]), int(ym[4:])
    if m not in MONTHS:
        continue
    i = int(np.where((ct.year == y) & (ct.month == m))[0][0])
    start = sic(i - 1)
    members = np.nan_to_num(xr.open_dataset(f).sic_members.values)
    for out, end in ((tgt, sic(i)), (mem, members)):
        d = cells(start, end); d["m"] = m; out.append(d)
M = pd.concat([table(pd.concat(tgt, ignore_index=True)).add_prefix("target "),
               table(pd.concat(mem, ignore_index=True)).add_prefix("members ")], axis=1)
print("\n=== 2. v0.4.0 on its training years 2011-14 (one month from the target), members vs target")
print(M.round(3).to_string())
