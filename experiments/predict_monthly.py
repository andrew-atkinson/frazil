#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict one month with a trained monthly GenSIM checkpoint and plot the maps.

Predicts state[target] from the true previous month + forcings, then plots, per
state variable, a row of [truth, GenSIM, persistence] maps (persistence = the
previous month, the naive baseline). Reuses eval_monthly.load_model.

Run:  python experiments/predict_monthly.py --target-year 2018 --target-month 3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import eval_monthly as E  # load_model, STATES/FORCINGS/DEGREE, UNITS

CMAP = {"sit": "viridis", "sic": "Blues_r", "sid": "magma",
        "siu": "RdBu_r", "siv": "RdBu_r", "snt": "viridis"}
DIVERGING = {"siu", "siv"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="default: newest last*.ckpt")
    ap.add_argument("--config", default="configs/config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="configs/config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--target-year", type=int, default=2018)
    ap.add_argument("--target-month", type=int, default=3)
    ap.add_argument("--n-ens", type=int, default=8, help="ensemble members (mean)")
    ap.add_argument("--out-png", default=None,
                    help="default: plots/forecast_<model>_<YYYY-MM>.png")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = args.ckpt or E.default_ckpt()
    args.out_png = args.out_png or (
        f"plots/forecast_{E.model_tag(ckpt)}_{args.target_year}-{args.target_month:02d}.png")
    model = E.load_model(ckpt, args.config, device, args.train_config)

    cube = xr.open_mfdataset(sorted(glob.glob(f"{args.datacube}/monthly_datacube_*.nc")),
                             combine="by_coords")["datacube"].load()
    times = pd.DatetimeIndex(cube["time"].values)
    hits = np.where((times.year == args.target_year) &
                    (times.month == args.target_month))[0]
    if len(hits) == 0 or hits[0] == 0:
        raise SystemExit(f"target {args.target_year}-{args.target_month:02d} "
                         "not found (or has no preceding month)")
    ti = int(hits[0])
    prev = ti - 1
    print(f"[predict] input {str(times[prev])[:7]} -> forecast {str(times[ti])[:7]}")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)

    def rep(x):  # tile (1, ...) along batch to n_ens members
        return x.repeat(args.n_ens, *([1] * (x.ndim - 1)))

    mesh = rep(torch.as_tensor(
        (aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None],
        device=device, dtype=torch.float32))
    mask = rep(torch.as_tensor(aux["mask"].values[None, None], device=device,
                               dtype=torch.float32))
    resolution = torch.full((args.n_ens, 1), 12.5, device=device, dtype=torch.float32)

    def arr(t, names):
        return np.nan_to_num(cube.isel(time=t).sel(var_names=names).values)

    states = rep(torch.as_tensor(arr(prev, E.STATES)[None], device=device, dtype=torch.float32))
    forc = np.stack([arr(prev, model.forcing_names), arr(ti, model.forcing_names)])
    forc = rep(torch.as_tensor(forc[None], device=device, dtype=torch.float32))
    dd = rep(torch.as_tensor(arr(prev, E.DEGREE)[None], device=device, dtype=torch.float32))
    with torch.no_grad():
        pred = model(states[:, None], forc, resolution=resolution, mesh=mesh,
                     mask=mask, degree_days=dd).mean(0).cpu().numpy()  # ensemble mean (6,H,W)
    assert np.isfinite(pred).all(), "non-finite prediction"

    truth = arr(ti, E.STATES)
    persist = arr(prev, E.STATES)

    # Plot: rows = variables, cols = truth | GenSIM | persistence (land -> NaN).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def show(ax, field, v, vmin, vmax):
        img = np.where(ocean, field, np.nan)
        ax.imshow(img, origin="lower", cmap=CMAP[v], vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])

    cols = ["truth", "GenSIM", "persistence"]
    fig, axes = plt.subplots(len(E.STATES), 3, figsize=(9, 2.6 * len(E.STATES)))
    for r, v in enumerate(E.STATES):
        t = truth[r][ocean]
        if v in DIVERGING:
            m = np.nanmax(np.abs(t)); vmin, vmax = -m, m
        else:
            vmin, vmax = np.nanpercentile(t, 2), np.nanpercentile(t, 98)
        rmse_g = np.sqrt(np.mean((pred[r] - truth[r])[ocean] ** 2))
        rmse_p = np.sqrt(np.mean((persist[r] - truth[r])[ocean] ** 2))
        for c, field in enumerate([truth[r], pred[r], persist[r]]):
            show(axes[r, c], field, v, vmin, vmax)
        axes[r, 0].set_ylabel(f"{v} ({E.UNITS[v]})", fontsize=10)
        axes[r, 1].set_title(f"RMSE {rmse_g:.3f} vs persist {rmse_p:.3f}", fontsize=8)
    for c, name in enumerate(cols):
        axes[0, c].set_title(name + (f"\n{axes[0, c].get_title()}"
                                     if axes[0, c].get_title() else ""), fontsize=10)
    fig.suptitle(f"GenSIM monthly forecast: {str(times[ti])[:7]} "
                 f"(from {str(times[prev])[:7]})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out_png, dpi=130)
    print(f"[plot] {args.out_png}")


if __name__ == "__main__":
    main()
