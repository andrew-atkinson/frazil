#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scale the neXtSIM-OPA `sit` (thickness) training targets towards the Copernicus neXtSIM
reanalysis, which assimilates CS2SMOS winter thickness from October 2010
(experiments/2026-10-29_ocean_input_and_teacher_checks: the teacher's Pacific-shelf ice
is 0.26-0.29 m too thick by April, and the concentration-only QM leaves that in).

Per calendar month and grid cell, one ratio: the reanalysis mean thickness over the
baseline years (default 2011-2018, the CryoSat era) divided by the targets' mean over the
same months, applied to every year's `sit`. A ratio keeps open water at zero and needs no
more than 8 years per cell (a quantile map would need more). Identity where either mean
is under 5 cm or the reanalysis has no data (Hudson Bay, the Labrador and Bering seas);
ratios clipped to [0.5, 2].

Only October-April is constrained by CS2SMOS. In summer the two models differ in melt
(the teacher grows too much in winter and melts more, so by September they agree on the
Pacific shelves), and the reanalysis's weaker melt is unconstrained (its September is too
icy there). So May-September ratios are interpolated per cell from April's to October's:
the teacher's summer physics is kept, starting from the corrected spring ice, and there is
no jump in May or October.
WARNING: that interpolation was a mistake. It made up a summer melt about 15% weaker than
the teacher's, and Frazil learned it (September bias +0.08 -> +0.22). Kept only to
reproduce the sensitivity test.
ONLY the sit channel changes. Thickness is a grid-cell mean in both.

Status: a sensitivity test only. One ratio for all years assumes the teacher's excess is
proportional to its thickness before 2011 too, which the data can't support; the
growth-based alternative (growth_corrected) was tested and rejected
(experiments/2026-10-29_sit_target_correction).

Then:
    build_zarr_monthly.py --datacube-dir <out> --suffix _sicqmw9_sitra
    train.py +experiment=pf_qm_ema_qmw_sit

Run:  python scripts/data/correct_sit_targets.py --selfcheck
      python scripts/data/correct_sit_targets.py
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from frazil import paths

MIN_SIT, CLIP = 0.05, (0.5, 2.0)


def fit_ratios(target, ref, months):
    """target, ref: (T, H, W) over the baseline, months: (T,) -> {month: (H, W) ratio}."""
    out = {}
    for m in range(1, 13):
        sel = months == m
        t, r = np.nanmean(target[sel], 0), np.nanmean(ref[sel], 0)
        ok = np.isfinite(t) & np.isfinite(r) & (t >= MIN_SIT) & (r >= MIN_SIT)
        out[m] = np.where(ok, np.clip(r / np.where(ok, t, 1.0), *CLIP), 1.0).astype(np.float32)
    for m in range(5, 10):                      # summer: April -> October, no CS2SMOS
        out[m] = out[4] + (out[10] - out[4]) * (m - 4) / 6
    return out


GROWTH_MONTHS = (12, 1, 2, 3, 4)      # increments Nov->Dec ... Mar->Apr (CS2SMOS-constrained)


def growth_factor(h_nov, alpha, h1=1.0, h2=2.0):
    """Growth scale by the ice's November thickness: alpha below h1 (first-year ice),
    1 above h2 (older ice), linear between."""
    return alpha + (1 - alpha) * np.clip((h_nov - h1) / (h2 - h1), 0, 1)


def growth_corrected(teacher, months, alpha, h1=1.0, h2=2.0):
    """Rebuild thickness month by month from the teacher's (T, H, W) thickness: winter
    gains (Dec-Apr) scaled by growth_factor of this winter's corrected November
    thickness; other gains added as they are; losses (melt, export) proportional to the
    teacher's, so open water stays open water. The correction carries over from year to
    year through the corrected state, so it follows the ice regime, not the calendar."""
    T = np.nan_to_num(teacher).astype(np.float32)
    h = np.empty_like(T); h[0] = T[0]
    f = np.ones_like(T[0])
    for t in range(1, len(T)):
        d = T[t] - T[t - 1]
        scale = f if months[t] in GROWTH_MONTHS else 1.0
        shrink = np.divide(T[t], T[t - 1], out=np.zeros_like(T[t]), where=T[t - 1] > 0)
        h[t] = np.maximum(np.where(d > 0, h[t - 1] + scale * d, h[t - 1] * shrink), 0)
        if months[t] == 11:
            f = growth_factor(h[t], alpha, h1, h2)
    return np.where(np.isfinite(teacher), h, np.nan)


def _selfcheck():
    # growth: a cell freezing from open water grows alpha x the teacher's winter gain,
    # then melts back proportionally; thick ice (> h2) is left as the teacher has it
    months = np.array([9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    thin = np.array([0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.4, 1.5, 1.4, 0.8, 0.2, 0, 0])
    T = np.stack([thin, thin + 3.0], 1)[:, None, :].astype(np.float32)
    h = growth_corrected(T, months, alpha=0.5)
    assert np.isclose(h[7, 0, 0], 0.3 + 0.5 * 1.2) and np.allclose(h[7, 0, 1], T[7, 0, 1])
    assert h[11, 0, 0] == 0 and np.isclose(h[8, 0, 0] / h[7, 0, 0], 1.4 / 1.5)
    assert np.isclose(growth_factor(np.array(1.5), 0.5), 0.75)
    months = np.tile(np.arange(1, 13), 2)
    target = np.full((24, 1, 4), 2.0, np.float32)
    ref = np.full((24, 1, 4), 1.5, np.float32)
    ref[:, 0, 1] = np.nan                       # outside the reanalysis domain
    target[:, 0, 2] = 0.0                       # open water stays open water
    ref[:, 0, 3] = 10.0                         # clipped
    r = fit_ratios(target, ref, months)
    assert np.allclose(r[4][0], [0.75, 1.0, 1.0, 2.0]), r[4]
    assert (2.0 * r[1][0, 0]) == 1.5 and (0.0 * r[1][0, 2]) == 0.0
    ref[months == 10, 0, 0] = 2.0                # October ratio 1.0: summer ramps 0.75 -> 1.0
    r = fit_ratios(target, ref, months)
    assert np.allclose([r[m][0, 0] for m in (4, 5, 7, 10)], [0.75, 0.75 + 0.25 / 6, 0.75 + 0.75 / 6, 1.0])
    print("[selfcheck] per-month ratio; identity outside the reference and on open water; clipped; summer interpolated")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacube", default=os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
    ap.add_argument("--reference", default=paths.NEXTSIM_REANALYSIS.format(var="sit"))
    ap.add_argument("--baseline", default="2011-2018")
    ap.add_argument("--out-dir", default=os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9_sitra"))
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return

    y0, y1 = (int(s) for s in args.baseline.split("-"))
    files = sorted(glob.glob(f"{args.datacube}/*_datacube_*.nc"))
    if not files:
        raise SystemExit(f"no datacube files in {args.datacube}")
    cube = xr.open_mfdataset(files, combine="by_coords")["datacube"]
    t_all = pd.DatetimeIndex(cube.time.values)
    base = (t_all.year >= y0) & (t_all.year <= y1)
    target = cube.sel(var_names="sit").isel(time=np.where(base)[0]).load().values
    ref_da = xr.open_dataset(args.reference)["sit"]
    ref_t = pd.DatetimeIndex(ref_da.time.values)
    ref = np.stack([ref_da.values[np.where((ref_t.year == t.year) & (ref_t.month == t.month))[0][0]] for t in t_all[base]])
    ratios = fit_ratios(target, ref, t_all[base].month.values)

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool); area = aux["cell_area"].values / 1e6
    lon = (aux["longitude"].values + 360) % 360
    shelves = ocean & (aux["latitude"].values >= 64) & (lon >= 140) & (lon < 205)   # East Siberian + Chukchi
    print(f"[sit] ratios fitted on {y0}-{y1}; share of ocean cells changed (|ratio-1| > 0.02) and mean ratio, by month:")
    for m in (1, 4, 6, 7, 9, 11):
        ch = ocean & (np.abs(ratios[m] - 1) > 0.02)
        print(f"  month {m:2d}: {ch.sum() / ocean.sum():.0%} of cells, mean ratio there {ratios[m][ch].mean():.2f}, "
              f"East Siberian + Chukchi {np.average(ratios[m][shelves], weights=area[shelves]):.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    vol = lambda f: float(np.nansum(np.where(ocean, f, 0) * area)) / 1e3      # km^2 * m -> km^3
    for f in files:
        da = xr.open_dataset(f)["datacube"]
        vn = [str(v) for v in da["var_names"].values]
        si = vn.index("sit")
        arr = da.values.copy()
        months = pd.DatetimeIndex(da["time"].values).month
        before = [vol(arr[i, si]) for i in range(len(months))]
        for i, mo in enumerate(months):
            arr[i, si] = np.where(ocean, arr[i, si] * ratios[int(mo)], np.nan)
        out = da.copy(data=arr)
        out.attrs["sit_scaled"] = f"neXtSIM reanalysis (ARCTIC_MULTIYEAR_PHY_ICE_002_016) {y0}-{y1}, per month and cell"
        out.to_dataset().to_netcdf(os.path.join(args.out_dir, os.path.basename(f)),
                                   encoding={"datacube": {"zlib": True, "complevel": 4}})
        apr = np.where(months == 4)[0]
        if len(apr):
            print(f"  {os.path.basename(f)}: April volume {before[apr[0]]:.0f} -> {vol(arr[apr[0], si]):.0f} km^3")
    print(f"\nDone -> {args.out_dir}/  (only sit scaled)")
    print(f"next: python scripts/data/build_zarr_monthly.py --datacube-dir {args.out_dir} --suffix _sicqmw9_sitra")
    print("      python train.py +experiment=pf_qm_ema_qmw_sit")


if __name__ == "__main__":
    main()
