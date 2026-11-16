"""C0: does the loose-ice gap exist in teacher space, and does correcting the predictions beat correcting the targets?

`monthly_pf` was trained on the teacher's own states. From eval_monthly --save-fields on the teacher cube
(each month from the teacher's previous month), August and September, 2011-14 and 2015-18:
  1. Teacher space: one-month extent error against the teacher target, and opening of loose ice by the start's
     thickness per unit of ice, members against the teacher.
  2. QM applied to each member's sic, scored against the QM target beside v0.4.0 (trained on QM targets).
     "same maps" uses the 9-year maps the targets were made with; those include the target year, so with 9 samples
     per map they carry that year's observations. "year left out" refits each map without the target year (8 samples).
     "teacher, year left out" maps the teacher's own target that way: the floor for a perfect teacher-space model.
C2 adds `monthly_pf_ema`: v0.4.0's last fine-tune (20k steps, EMA 0.999, seed 1) on teacher targets, from monthly_pf.
Run from the repo root:  python experiments/2026-11-14_assumptions_review/teacher_space.py
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
Y0, Y1, WINDOW, ZERO = 1995, 2018, 9, 1e-4
aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
regions = {"Pacific shelves": sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"], "Barents/Kara": sec["Barents/Kara"]}
cube = lambda d: xr.open_mfdataset(sorted(glob.glob(os.path.join(d, "*_datacube_*.nc"))), combine="by_coords")["datacube"]
raw, qm = cube(paths.DATACUBE), cube(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
rt, ct = pd.DatetimeIndex(raw.time.values), pd.DatetimeIndex(qm.time.values)
val = lambda da, i, v="sic": np.nan_to_num(da.isel(time=i).sel(var_names=v).values)
nx, ob = Q.load_month_sic(paths.DATACUBE, Y0, Y1), Q.load_month_sic(paths.NSIDC0051_GRID, Y0, Y1)


def maps(month, year, leave_out):
    lo, hi = Q.window_years(year, Y0, Y1, WINDOW)
    yrs = [y for y in range(lo, hi + 1) if not (leave_out and y == year)]
    k = [y - Y0 for y in yrs]
    return Q.fit_qm(nx[month][k], ob[month][k])


def mapped(x, m):
    return np.where(ocean, Q.apply_qm(x, *m), 0.0)


def fields(run, sub):
    for f in sorted(glob.glob(os.path.join(paths.MODELS, run, sub, "pred_??????.nc"))):
        ym = os.path.basename(f)[5:11]; y, m = int(ym[:4]), int(ym[4:])
        if m in (8, 9):
            yield y, m, np.nan_to_num(xr.open_dataset(f).sic_members.values)


extent = lambda x, mk: (area * mk * (x > 0.15)).sum() / 1e6
ext, cells, check = [], [], 0.0
for sub in ("one_month_fields_2011", "one_month_fields"):
    v04 = {(y, m): mem for y, m, mem in fields("monthly_pf_qm_ema_qmw", sub)}
    ema = {(y, m): mem for y, m, mem in fields("monthly_pf_ema", sub)}
    for y, m, mem in fields("monthly_pf", sub):
        yrs = "2011-14" if y < 2015 else "2015-18"
        i = int(np.where((rt.year == y) & (rt.month == m))[0][0]); j = int(np.where(ct == rt[i])[0][0])
        t_start, t_tgt, q_start, q_tgt, sit0 = val(raw, i - 1), val(raw, i), val(qm, j - 1), val(qm, j), val(raw, i - 1, "sit")
        same, loo = maps(m, y, False), maps(m, y, True)
        check = max(check, float(np.abs(mapped(t_tgt, same) - q_tgt)[ocean].max()))
        runs = {"monthly_pf, teacher space": (mem, t_tgt, t_start),
                "v0.4.0": (v04[(y, m)], q_tgt, q_start),
                "monthly_pf + QM, same maps": (np.stack([mapped(p, same) for p in mem]), q_tgt, q_start),
                "monthly_pf + QM, year left out": (np.stack([mapped(p, loo) for p in mem]), q_tgt, q_start),
                "monthly_pf_ema, teacher space": (ema[(y, m)], t_tgt, t_start),
                "monthly_pf_ema + QM, year left out": (np.stack([mapped(p, loo) for p in ema[(y, m)]]), q_tgt, q_start),
                "teacher + QM, year left out": (mapped(t_tgt, loo)[None], q_tgt, q_start)}
        for name, (p, tgt, start) in runs.items():
            for r, mk in regions.items():
                ext.append({"years": yrs, "region": r, "m": m, "run": name, "extent error": np.mean([extent(x, mk) for x in p]) - extent(tgt, mk),
                            "se": ((p.mean(0) - tgt)[mk] ** 2 * area[mk]).sum(), "w": area[mk].sum()})
                c = mk & (start >= 0.15) & (start < 0.8)
                cells.append(pd.DataFrame({"years": yrs, "region": r, "m": m, "run": name, "w": area[c], "start": start[c], "h": sit0[c] / start[c],
                                           "target opens": (tgt[c] < 0.15).astype(float), "members open": (p[:, c] < 0.15).mean(0),
                                           "target partly open": ((tgt[c] > ZERO) & (tgt[c] < 0.15)).astype(float),
                                           "members partly open": ((p[:, c] > ZERO) & (p[:, c] < 0.15)).mean(0),
                                           "bias": p[:, c].mean(0) - tgt[c]}))
E, C = pd.DataFrame(ext), pd.concat(cells, ignore_index=True)
C["cls"] = pd.cut(C.start, [0.15, 0.5, 0.8], right=False, labels=["15-50%", "50-80%"])
C["ice thickness"] = pd.cut(C.h, [0, 1, 2, 100], labels=["<1 m", "1-2 m", ">2 m"])
ORDER = ["monthly_pf, teacher space", "monthly_pf_ema, teacher space", "v0.4.0", "monthly_pf + QM, same maps", "monthly_pf + QM, year left out",
         "monthly_pf_ema + QM, year left out", "teacher + QM, year left out"]
avg = lambda g, cols: pd.Series({k: np.average(g[k], weights=g.w) for k in cols})
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
print(f"check: teacher target through the same maps reproduces the QM cube to within {check:.1e}")
print("\n=== 1. one-month extent error, M km^2 (each run against its own target: teacher for the teacher-space rows, QM for the rest)")
print(E.groupby(["region", "run", "years", "m"])["extent error"].mean().unstack(["years", "m"]).round(3).reindex(ORDER, level="run").to_string())
g = E.groupby(["region", "run", "years", "m"])[["se", "w"]].sum()
print("\n=== 2. RMSE of the ensemble-mean sic over the region")
print(np.sqrt(g.se / g.w).unstack(["years", "m"]).round(3).reindex(ORDER, level="run").to_string())
COLS = ["target opens", "members open", "target partly open", "members partly open", "bias"]
print("\n=== 3. loose ice by starting class (the start as each run sees it)")
print(C.groupby(["region", "m", "cls", "years", "run"], observed=True).apply(avg, COLS).round(3).reindex(ORDER, level="run").to_string())
print("\n=== 4. by the start's thickness per unit of ice, starting at 50-80%")
T = C[C.cls == "50-80%"].groupby(["region", "m", "years", "run", "ice thickness"], observed=True).apply(avg, ["target opens", "members open", "bias"])
print(T.unstack("ice thickness").round(3).reindex(ORDER[:-1], level="run").to_string())
