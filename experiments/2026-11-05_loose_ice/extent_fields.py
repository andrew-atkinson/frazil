"""One-month summer extent error by sector, from eval_monthly --save-fields (members vs the QM target),
for v0.4.0 and pf_qm_ema_qmw_sic0, on 2015-18 (one_month_fields) and training years 2011-14 (one_month_fields_2011).

Run from the repo root:  python experiments/2026-11-05_loose_ice/extent_fields.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

RUNS = {"v0.4.0": "monthly_pf_qm_ema_qmw", "sic0": "monthly_pf_qm_ema_qmw_sic0"}
REGIONS = ["Pacific shelves", "Laptev", "Barents/Kara", "Canadian/Baffin/Hudson", "Greenland"]
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
sec["Pacific shelves"] = sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"]
cube = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9", "*_datacube_*.nc"))),
                         combine="by_coords")["datacube"].sel(var_names="sic")
ct = pd.DatetimeIndex(cube.time.values)
rows = []
for name, run in RUNS.items():
    for sub in ("one_month_fields", "one_month_fields_2011"):
        for f in sorted(glob.glob(os.path.join(paths.MODELS, run, sub, "pred_??????.nc"))):
            ym = os.path.basename(f)[5:11]; y, m = int(ym[:4]), int(ym[4:])
            if m not in (6, 7, 8, 9):
                continue
            i = int(np.where((ct.year == y) & (ct.month == m))[0][0])
            tgt = np.nan_to_num(cube.isel(time=i).values)
            mem = np.nan_to_num(xr.open_dataset(f).sic_members.values)
            for r in REGIONS:
                mk = sec[r]
                rows.append({"run": name, "years": "2011-14" if y < 2015 else "2015-18", "region": r, "m": m,
                             "err": np.mean([(area * mk * (p > 0.15)).sum() for p in mem]) / 1e6 - (area * mk * (tgt > 0.15)).sum() / 1e6})
E = pd.DataFrame(rows)
print("=== One-month extent error, M km^2 (members vs QM target), mean over years. 2011-14: Aug-Sep only for sic0")
print(E.groupby(["years", "region", "run", "m"]).err.mean().unstack("m").round(3).to_string())
