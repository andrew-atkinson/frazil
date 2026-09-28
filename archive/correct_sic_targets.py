#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bias-correct the neXtSIM `sic` training targets toward NSIDC-0051 observations,
so the model learns obs-consistent concentration -- anchored in the training data,
not a post-hoc offset.

Per calendar month, per grid cell:  bias = clim(neXtSIM sic) - clim(obs sic) over
the 1995-2018 overlap. Subtract that from every target month's sic and clip to
[0,1]. ONLY the sic channel changes; the other 5 states, forcings and degree-days
are copied through untouched (obs give us concentration, not damage/velocity/snow).

Writes a corrected copy of the datacube. Then re-run the training prep on it:
    estimate_normalization_monthly.py  ->  build_zarr_monthly.py  ->  fine-tune.

Run:  python experiments/correct_sic_targets.py --selfcheck
      python experiments/correct_sic_targets.py
"""
from __future__ import annotations

import argparse
import glob
import os

import warnings

import numpy as np
import pandas as pd
import xarray as xr


def _year(path):
    return int(path.split("_")[-1].split(".")[0])


def load_sic_series(datacube_dir, y0, y1):
    files = []
    for f in sorted(glob.glob(f"{datacube_dir}/*.nc")):  # accepts *_datacube_ and nsidc0051_grid_
        try:
            yr = _year(f)
        except ValueError:
            continue
        if y0 <= yr <= y1:
            files.append(f)
    if not files:
        raise SystemExit(f"no yearly .nc for {y0}-{y1} in {datacube_dir}")
    da = xr.open_mfdataset(files, combine="by_coords")["datacube"].sel(var_names="sic").load()
    return da.values.astype(np.float32), pd.DatetimeIndex(da["time"].values).month


def monthly_bias(nx, nx_mo, obs, obs_mo):
    """Per calendar month: (H,W) additive bias = clim(nx) - clim(obs) over years.
    NaN (land, or obs coastal gaps) -> 0 (no correction there). Pure/testable."""
    bias = {}
    with warnings.catch_warnings():  # all-NaN land cells -> handled by nan_to_num
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for m in range(1, 13):
            an, ao = nx_mo == m, obs_mo == m
            if not an.any() or not ao.any():
                bias[m] = np.float32(0.0)
                continue
            cn = np.nanmean(nx[an], axis=0)
            co = np.nanmean(obs[ao], axis=0)
            bias[m] = np.nan_to_num(cn - co).astype(np.float32)
    return bias


def _selfcheck():
    H = W = 3
    nx_mo = np.array([1, 1, 2])
    obs_mo = np.array([1, 2])
    nx = np.stack([np.full((H, W), 0.8), np.full((H, W), 0.6), np.full((H, W), 0.5)])
    obs = np.stack([np.full((H, W), 0.5), np.full((H, W), 0.4)])
    b = monthly_bias(nx, nx_mo, obs, obs_mo)
    # month 1: mean(0.8,0.6)=0.7 - 0.5 = 0.2 ; month 2: 0.5 - 0.4 = 0.1
    assert np.allclose(b[1], 0.2) and np.allclose(b[2], 0.1), (b[1].mean(), b[2].mean())
    # NaN handling: land cell NaN in both -> bias 0
    nx2 = nx.copy(); obs2 = obs.copy(); nx2[:, 0, 0] = np.nan; obs2[:, 0, 0] = np.nan
    b2 = monthly_bias(nx2, nx_mo, obs2, obs_mo)
    assert b2[1][0, 0] == 0.0
    print("[selfcheck] monthly_bias OK")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--obs", default="data/obs/nsidc0051_grid")
    ap.add_argument("--out-dir", default="data/train_data/monthly_datacube_siccorr")
    ap.add_argument("--baseline", default="1995-2018", help="overlap years YYYY-YYYY")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return

    y0, y1 = (int(s) for s in args.baseline.split("-"))
    nx, nx_mo = load_sic_series(args.datacube, y0, y1)
    obs, obs_mo = load_sic_series(args.obs, y0, y1)
    bias = monthly_bias(nx, nx_mo, obs, obs_mo)

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    ca = aux["cell_area"].values / 1e6
    print(f"[sic-corr] baseline {y0}-{y1}: per-month mean sic bias over ocean "
          "(neXtSIM - obs, + = model too icy):")
    for m in range(1, 13):
        bm = bias[m]
        val = float(np.mean(bm[ocean])) if np.ndim(bm) else 0.0
        print(f"    {m:2d}: {val:+.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(f"{args.datacube}/*_datacube_*.nc"))
    de = df = 0.0
    for f in files:
        da = xr.open_dataset(f)["datacube"]
        vn = [str(v) for v in da["var_names"].values]
        si = vn.index("sic")
        arr = da.values.copy()  # (T, V, H, W)
        months = pd.DatetimeIndex(da["time"].values).month
        for i, mo in enumerate(months):
            corr = np.clip(arr[i, si] - bias[int(mo)], 0.0, 1.0)  # NaN land stays NaN
            arr[i, si] = corr
        out = da.copy(data=arr)
        out.attrs["sic_bias_corrected"] = f"NSIDC-0051 {y0}-{y1}"
        dst = f"{args.out_dir}/{os.path.basename(f)}"
        out.to_dataset().to_netcdf(dst, encoding={"datacube": {"zlib": True, "complevel": 4}})
        # Track the September extent shift as a sanity signal.
        sep = np.where(months == 9)[0]
        if len(sep):
            before = np.nan_to_num(da.values[sep[0], si])
            after = np.nan_to_num(arr[sep[0], si])
            de += float(((before > 0.15) * ocean * ca).sum())
            df += float(((after > 0.15) * ocean * ca).sum())
    if de:
        print(f"[sic-corr] September extent (summed over years): "
              f"{de/1e6:.1f} -> {df/1e6:.1f} M km²  ({(df-de)/1e6:+.1f})")
    print(f"\nDone -> {args.out_dir}/  (only sic corrected)")
    print("next: python experiments/estimate_normalization_monthly.py "
          f"--datacube '{args.out_dir}/monthly_datacube_*.nc'")
    print("      python experiments/build_zarr_monthly.py "
          f"--datacube-dir {args.out_dir} --suffix _siccorr")


if __name__ == "__main__":
    main()
