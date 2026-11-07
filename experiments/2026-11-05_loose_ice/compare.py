"""Compare the loose-ice runs (step_test.py), August and September 2015-2018.
  1. Extent error by sector, from the members, as the scorecard does
  2. Error in concentration by starting concentration: each member against its own start, so a
     generated start is judged as an input. Both sides of that conditioning are what the model
     sees, never the truth. Flat near zero = calibrated from that kind of start

Run from the repo root:  python experiments/2026-11-05_loose_ice/compare.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

O = os.path.join(paths.MODELS, "loose_ice_tests")
RUNS = {"v04_truth": "monthly_datacube_sicqmw9", "v04_generated": "monthly_datacube_sicqmw9", "v04_truth_n40": "monthly_datacube_sicqmw9",
        "qm_nopf_truth": "monthly_datacube_sicqm", "qm_pf_truth": "monthly_datacube_sicqm"}
CLASSES = [0.15, 0.5, 0.8, 0.95, 1.0]
LABELS = ["15-50%", "50-80%", "80-95%", ">95%"]
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
sec["Pacific shelves"] = sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"]
REGIONS = ["Pacific shelves", "Laptev", "Barents/Kara", "Canadian/Baffin/Hudson", "Greenland"]

ext, cls = [], []
for run, cube_name in RUNS.items():
    cube = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, cube_name, "*_datacube_*.nc"))), combine="by_coords")["datacube"]
    ct = pd.DatetimeIndex(cube.time.values)
    for f in sorted(glob.glob(os.path.join(O, run, "step_??????.npz"))):
        ym = os.path.basename(f)[5:11]; y, m = int(ym[:4]), int(ym[4:])
        i = int(np.where((ct.year == y) & (ct.month == m))[0][0])
        tgt = np.nan_to_num(cube.isel(time=i).sel(var_names="sic").values)
        d = np.load(f); start, pred = np.nan_to_num(d["start"]), np.nan_to_num(d["next"])
        for r in REGIONS:
            mk = sec[r]
            ext.append({"run": run, "region": r, "y": y, "m": m, "err": np.mean([(area * mk * (p > 0.15)).sum() for p in pred]) / 1e6 - (area * mk * (tgt > 0.15)).sum() / 1e6})
            for k, lab in enumerate(LABELS):
                hit = (start >= CLASSES[k]) & (start <= CLASSES[k + 1] if k == 3 else start < CLASSES[k + 1]) & mk
                w = area[None] * hit
                cls.append({"run": run, "region": r, "y": y, "m": m, "cls": lab, "e": ((pred - tgt) * w).sum(), "w": w.sum()})
E, C = pd.DataFrame(ext), pd.DataFrame(cls)
print("=== 1. One-step extent error, M km^2, mean over 2015-2018 (one-month runs from the start named)")
print(E.groupby(["region", "run", "m"]).err.mean().unstack("m").round(3).reindex(RUNS, level="run").reindex(REGIONS, level="region").to_string())
print("\nPacific shelves by year:")
print(E[E.region == "Pacific shelves"].pivot_table(index="run", columns=["m", "y"], values="err").reindex(list(RUNS)).round(3).to_string())
print("\n=== 2. Concentration error by the member's starting concentration (positive = too much ice)")
for r in REGIONS:
    g = C[C.region == r].groupby(["run", "m", "cls"])[["e", "w"]].sum()
    t = (g.e / g.w).unstack("cls")[LABELS].unstack("m")
    print(f"\n{r}:\n" + t.reindex(list(RUNS)).round(3).to_string())
g = C[C.region == "Pacific shelves"].groupby(["run", "m", "cls"]).w.sum().unstack("cls")[LABELS]
print("\nPacific shelves, share of area in each class:\n" + g.div(g.sum(1), axis=0).unstack("m").reindex(list(RUNS)).round(2).to_string())
