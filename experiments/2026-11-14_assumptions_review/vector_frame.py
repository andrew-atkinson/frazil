"""How much does the unrotated-vector augmentation scramble (ISS-029), and does QM weaken the link between
drift and concentration (ISS-030)?

Ice transport by the drift: predicted change = -div(sic * drift) * one month, from monthly means, on the grid's
own index axes. It is compared (Spearman rank correlation, no linearity assumed) with the actual change of sic to
the next month, over marginal-ice cells (5-95%), 1995-2014.
  1. Eight ways of reading the two components: the 4 rotations and 4 mirror images. The true frame should stand
     out. "+90" and "-90" are what the network is shown in rotated training samples.
  2. The same, with the wind (uas, vas) in place of the drift.
  3. True frame only: teacher sic against QM sic (drift is the teacher's in both cubes).
Run from the repo root:  python experiments/2026-11-14_assumptions_review/vector_frame.py
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

from frazil import paths

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); dx, dy = aux.dx.values, aux.dy.values; lat = aux.latitude.values
DT = 30 * 86400.


def cube(d):
    fs = sorted(f for f in glob.glob(os.path.join(d, "*.nc")) if f[-7:-3].isdigit() and 1995 <= int(f[-7:-3]) <= 2014)
    return xr.open_mfdataset(fs, combine="by_coords")["datacube"]


raw, qm = cube(paths.DATACUBE), cube(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
get = lambda da, v: np.nan_to_num(da.sel(var_names=v).values)
months = pd.DatetimeIndex(raw.time.values).month.values
c_raw, c_qm = get(raw, "sic"), get(qm, "sic")
VEC = {"drift": (get(raw, "siu"), get(raw, "siv")), "wind": (get(raw, "uas"), get(raw, "vas"))}
READ = {"true (u, v)": lambda u, v: (u, v), "+90 (-v, u)": lambda u, v: (-v, u), "180 (-u, -v)": lambda u, v: (-u, -v),
        "-90 (v, -u)": lambda u, v: (v, -u), "mirror (u, -v)": lambda u, v: (u, -v), "mirror (-u, v)": lambda u, v: (-u, v),
        "mirror (v, u)": lambda u, v: (v, u), "mirror (-v, -u)": lambda u, v: (-v, -u)}
SEASONS = {"Oct-Mar": (10, 11, 12, 1, 2, 3), "Apr-Jun": (4, 5, 6), "Jul-Sep": (7, 8, 9)}


def transport(c, a, b):
    """-div(c * vector) * one month; a along +x index (columns), b along +y index (rows)."""
    return -(np.gradient(c * a, axis=-1) / dx + np.gradient(c * b, axis=-2) / dy) * DT


def corr(c, u, v, read):
    out = {}
    cm, um, vm = c[:-1], (u[:-1] + u[1:]) / 2, (v[:-1] + v[1:]) / 2   # drift over the step: mean of the two months
    pred, dc = transport(cm, *read(um, vm)), c[1:] - c[:-1]
    for s, ms in SEASONS.items():
        k = np.isin(months[1:], ms)[:, None, None] & (cm > 0.05) & (cm < 0.95) & ocean[None] & (lat > 60)[None]
        out[s] = spearmanr(pred[k], dc[k]).statistic
    return out


pd.set_option("display.width", 200)
for name, (u, v) in VEC.items():
    print(f"=== {name}: rank correlation of -div(sic * {name}) with the actual change of teacher sic, by how the components are read")
    print(pd.DataFrame({r: corr(c_raw, u, v, f) for r, f in READ.items()}).T.round(3).to_string())
u, v = VEC["drift"]
print("\n=== true frame, drift: teacher sic against QM sic (same drift in both cubes)")
print(pd.DataFrame({"teacher sic": corr(c_raw, u, v, READ["true (u, v)"]), "QM sic": corr(c_qm, u, v, READ["true (u, v)"])}).T.round(3).to_string())
