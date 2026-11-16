"""C0, out of sample: the scorecard's hindcast measures for the teacher-trained monthly_pf with QM applied to its
predictions, beside v0.4.0 (trained on QM targets). Development years 2019-22 only, against NSIDC-0051.

Uses the saved members of both ERA5 hindcasts (free-run from December 2018), no model calls. The maps are the ones
v0.4.0's last training targets were made with: per cell and month, fitted on 2010-2018. No observation from
2019-22 enters them. Scoring is scorecard.py's own member_rows and scores.
Run from the repo root:  python experiments/2026-11-14_assumptions_review/teacher_space_hindcast.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(RL.ROOT, rel))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


sc, Q = load("sc", "scripts/evaluate/scorecard.py"), load("qmmod", "scripts/data/correct_sic_targets_qm.py")
Y0, Y1, WINDOW = 1995, 2018, 9
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
d = xr.open_mfdataset(sorted(glob.glob(f"{paths.NSIDC0051_GRID}/*.nc")), combine="by_coords")
obs = d[[n for n in ("datacube", "state") if n in d.data_vars][0]].sel(var_names="sic")
obs_i = {(t.year, t.month): i for i, t in enumerate(pd.DatetimeIndex(obs.time.values))}
nx, ob = Q.load_month_sic(paths.DATACUBE, Y0, Y1), Q.load_month_sic(paths.NSIDC0051_GRID, Y0, Y1)
lo, hi = Q.window_years(Y1, Y0, Y1, WINDOW)
maps = {m: Q.fit_qm(nx[m][lo - Y0:hi - Y0 + 1], ob[m][lo - Y0:hi - Y0 + 1]) for m in range(1, 13)}
RUNS = {"v0.4.0 (QM targets)": ("monthly_pf_qm_ema_qmw", False), "monthly_pf (teacher targets)": ("monthly_pf", False),
        "monthly_pf + QM of predictions": ("monthly_pf", True),
        # C2: v0.4.0's last fine-tune (20k steps, EMA 0.999, seed 1) on teacher targets, from monthly_pf
        "monthly_pf_ema (teacher targets)": ("monthly_pf_ema", False), "monthly_pf_ema + QM of predictions": ("monthly_pf_ema", True)}
out = {}
for name, (model, do_qm) in RUNS.items():
    rows = []
    for f in sorted(glob.glob(os.path.join(paths.SNAPSHOTS, f"{model}_era5_2018", "members_*.nc"))):
        y, m = int(f[-9:-5]), int(f[-5:-3])
        if not (sc.DEV[0] <= y <= sc.DEV[1]) or (y, m) not in obs_i:
            continue
        mem = xr.open_dataset(f)["sic"].values
        if do_qm:
            mem = np.stack([Q.apply_qm(p, *maps[m]) for p in mem])
        rows += sc.member_rows(mem, obs.isel(time=obs_i[(y, m)]).values, ocean, area, sec, y, m)
    df = pd.DataFrame(rows)
    out[name] = {k: v for k, v in sc.scores(df).items() if not isinstance(v, str)}
    out[name].update({f"{k}_bias": round(float(df[df.month == mo].ext.mean()), 3) for k, mo in (("mar", 3), ("oct", 10), ("dec", 12))})
pd.set_option("display.width", 200)
print(f"=== hindcast 2019-22 against NSIDC-0051, M km^2 unless noted (maps fitted on {lo}-{hi})")
print(pd.DataFrame(out).to_string())
