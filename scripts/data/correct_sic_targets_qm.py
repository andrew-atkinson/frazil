#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantile-map the neXtSIM `sic` training targets onto NSIDC-0051 observations, so
the model learns an obs-consistent ice distribution (the real fix for the ~0.38 M
km² September over-icing that the additive mean-shift couldn't touch).

Per calendar month, per grid cell, it matches the neXtSIM sic distribution to the
observed one (CDF mapping) over the 1995-2018 overlap, then applies that map to
every target month's sic. Unlike a mean shift, this changes the *frequency* of ice
at each cell, so the ice edge actually retreats (proven in qm_calibrate_poc.py).
ONLY the sic channel changes; the other 5 states, forcings and degree-days pass
through untouched. Cells with no valid observation are left as-is.

Then feed the corrected datacube to the training prep + a fine-tune:
    build_zarr_monthly.py --datacube-dir <out> --suffix _sicqm
    train.py +experiment=sicqm

Run:  python scripts/data/correct_sic_targets_qm.py --selfcheck
      python scripts/data/correct_sic_targets_qm.py
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

QS = np.linspace(0.0, 1.0, 21)   # quantile knots


def fit_qm(ref, obs):
    """Per-cell quantile knots (ref->obs) + a validity mask. ref/obs: (T,H,W) with
    NaN allowed. Cells lacking valid obs get an identity map."""
    rq = np.nanquantile(ref, QS, axis=0)
    oq = np.nanquantile(obs, QS, axis=0)
    valid = np.isfinite(rq).all(0) & np.isfinite(oq).all(0)
    return np.nan_to_num(rq).astype(np.float32), np.nan_to_num(oq).astype(np.float32), valid


def apply_qm(x, rq, oq, valid):
    """Map one (H,W) field through the per-cell monotone quantile transform;
    identity where obs was missing."""
    xf = np.nan_to_num(x)
    xc = np.clip(xf, rq[0], rq[-1])
    out = oq[0].copy()
    for k in range(1, rq.shape[0]):
        lo, hi, olo, ohi = rq[k - 1], rq[k], oq[k - 1], oq[k]
        span = np.where(hi > lo, hi - lo, 1.0)
        frac = np.clip((xc - lo) / span, 0.0, 1.0)
        out = np.where((xc > lo) & (xc <= hi), olo + frac * (ohi - olo), out)
    return np.where(valid, np.clip(out, 0.0, 1.0), xf)


def _selfcheck():
    ref = np.array([[[0.9]], [[0.8]], [[0.7]]])       # icy cell in ref
    obs = np.array([[[0.0]], [[0.1]], [[0.0]]])       # ~ice-free in obs
    rq, oq, valid = fit_qm(ref, obs)
    assert valid[0, 0]
    assert apply_qm(np.array([[0.85]]), rq, oq, valid)[0, 0] < 0.15   # pulled below edge
    # cell with no obs -> identity
    rq2, oq2, v2 = fit_qm(np.array([[[0.5]]]), np.array([[[np.nan]]]))
    assert not v2[0, 0] and apply_qm(np.array([[0.5]]), rq2, oq2, v2)[0, 0] == 0.5
    print("[selfcheck] quantile map OK")


def _year(p):
    return int(p.split("_")[-1].split(".")[0])


def load_month_sic(datacube_dir, y0, y1):
    """{month: (T,H,W) sic} over years [y0,y1], from a datacube or grid dir."""
    files = [f for f in sorted(glob.glob(f"{datacube_dir}/*.nc"))
             if _year_safe(f) and y0 <= _year(f) <= y1]
    da = xr.open_mfdataset(files, combine="by_coords")
    name = next((n for n in ("datacube", "state") if n in da.data_vars), None)
    v = da[name].sel(var_names="sic").load()
    months = pd.DatetimeIndex(v["time"].values).month
    arr = v.values.astype(np.float32)
    return {m: arr[months == m] for m in range(1, 13)}


def _year_safe(p):
    try:
        _year(p); return True
    except ValueError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--obs", default="data/obs/nsidc0051_grid")
    ap.add_argument("--out-dir", default="data/train_data/monthly_datacube_sicqm")
    ap.add_argument("--baseline", default="1995-2018")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return

    y0, y1 = (int(s) for s in args.baseline.split("-"))
    nx = load_month_sic(args.datacube, y0, y1)
    ob = load_month_sic(args.obs, y0, y1)
    qm = {m: fit_qm(nx[m], ob[m]) for m in range(1, 13)}   # per-month maps
    print(f"[qm] fit per-cell September..-map on {y0}-{y1} "
          f"({nx[9].shape[0]} yrs/month)")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    ca = aux["cell_area"].values / 1e6

    os.makedirs(args.out_dir, exist_ok=True)
    de = df = 0.0
    for f in sorted(glob.glob(f"{args.datacube}/*.nc")):
        da = xr.open_dataset(f)["datacube"]
        vn = [str(v) for v in da["var_names"].values]
        si = vn.index("sic")
        arr = da.values.copy()
        months = pd.DatetimeIndex(da["time"].values).month
        for i, mo in enumerate(months):
            rq, oq, valid = qm[int(mo)]
            arr[i, si] = np.where(ocean, apply_qm(arr[i, si], rq, oq, valid), np.nan)
        out = da.copy(data=arr)
        out.attrs["sic_quantile_mapped"] = f"NSIDC-0051 {y0}-{y1}"
        dst = f"{args.out_dir}/{os.path.basename(f)}"
        out.to_dataset().to_netcdf(dst, encoding={"datacube": {"zlib": True, "complevel": 4}})
        sep = np.where(months == 9)[0]
        if len(sep):
            de += float(((np.nan_to_num(da.values[sep[0], si]) > 0.15) * ocean * ca).sum())
            df += float(((np.nan_to_num(arr[sep[0], si]) > 0.15) * ocean * ca).sum())
    if de:
        print(f"[qm] September extent (summed over years): "
              f"{de/1e6:.1f} -> {df/1e6:.1f} M km²  ({(df-de)/1e6:+.1f})")
    print(f"\nDone -> {args.out_dir}/  (only sic quantile-mapped)")
    print("next: python scripts/data/estimate_normalization_monthly.py "
          f"--datacube '{args.out_dir}/monthly_datacube_*.nc' --no-configs")
    print(f"      python scripts/data/build_zarr_monthly.py --datacube-dir {args.out_dir} --suffix _sicqm")
    print("      python train.py +experiment=sicqm")


if __name__ == "__main__":
    main()
