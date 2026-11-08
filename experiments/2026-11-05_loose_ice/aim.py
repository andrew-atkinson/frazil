"""How far below 0 do the members aim? From the --no-clamp run (step_test.py), which keeps the
decoder's unclamped sic (next_raw): for cells starting at 15-80%, the share of members that
aim below 0, and how far, by region, month and starting class.

Run from the repo root:  python experiments/2026-11-05_loose_ice/aim.py [--run v04_truth_noclamp]
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
ap.add_argument("--run", default="v04_truth_noclamp")
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
regions = {"Pacific shelves": sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"], "Barents/Kara": sec["Barents/Kara"]}
rows = []
for f in sorted(glob.glob(os.path.join(paths.MODELS, "loose_ice_tests", args.run, "step_??????.npz"))):
    d = np.load(f); start, raw = np.nan_to_num(d["start"][0]), np.nan_to_num(d["next_raw"])
    for r, mk in regions.items():
        for cls, lo, hi in (("15-50%", 0.15, 0.5), ("50-80%", 0.5, 0.8)):
            c = mk & (start >= lo) & (start < hi)
            v, w = raw[:, c].ravel(), np.tile(area[c], raw.shape[0])
            neg = v < 0
            rows.append({"region": r, "m": int(os.path.basename(f)[9:11]), "cls": cls, "w": w, "v": v, "neg": neg})
out = []
for (r, m, cls), g in pd.DataFrame(rows).groupby(["region", "m", "cls"]):
    v, w, neg = np.concatenate(list(g.v)), np.concatenate(list(g.w)), np.concatenate(list(g.neg))
    below = v[neg]
    out.append({"region": r, "m": m, "cls": cls, "aims below 0": np.average(neg, weights=w),
                "median below": np.median(below) if below.size else np.nan,
                "10th pct below": np.percentile(below, 10) if below.size else np.nan,
                "lowest": below.min() if below.size else np.nan})
print(f"=== {args.run}: unclamped sic of members, cells starting at 15-80%")
print(pd.DataFrame(out).set_index(["region", "m", "cls"]).round(3).to_string())
