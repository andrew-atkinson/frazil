#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a trained monthly GenSIM checkpoint by prediction skill, not loss.

Loads the EMA weights from a Lightning .ckpt into a GenSIMForecastModule and does
one-month-ahead predictions over the validation months (2015-2018): predict
state[t+1] from truth state[t] + forcings, compare to truth. Baseline =
persistence (state[t+1] == state[t]). Reports RMSE over ocean cells per variable
and the skill score 1 - rmse_model/rmse_persistence (>0 means beats persistence).

Run:  python experiments/eval_monthly.py --ckpt data/models/monthly/last-v3.ckpt
"""
from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf
from hydra.utils import instantiate

import gensim
import gensim.network

STATES = ["sit", "sic", "sid", "siu", "siv", "snt"]
FORCINGS = ["tus", "huss", "uas", "vas"]
DEGREE = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
UNITS = {"sit": "m", "sic": "1", "sid": "1", "siu": "m/s", "siv": "m/s", "snt": "m"}


def load_model(ckpt_path, config, device, train_config):
    cfg = OmegaConf.load(config)
    # The checkpoint's architecture comes from the train config (e.g. the mac
    # config shrinks n_features); use it so shapes match.
    cfg.surrogate.network = OmegaConf.load(train_config).surrogate.network
    model = instantiate(cfg.surrogate)  # GenSIMForecastModule (recursive)
    gensim.network.USE_FLASH_ATTN = (
        gensim.network.USE_FLASH_ATTN and device.type == "cuda")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    # Prefer the EMA weights (the inference-quality copy); fall back to network.*
    prefix = "ema_model.module." if any(k.startswith("ema_model.module.")
                                        for k in sd) else "network."
    net_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.network.load_state_dict(net_sd, strict=False)
    print(f"[ckpt] loaded {len(net_sd)} tensors from '{prefix}*' "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")

    model = model.to(device)
    model.set_inference_model(compile_model=device.type == "cuda")
    model.eval()
    return model


def load_val_cube(datacube_dir, val_start_year):
    files = sorted(f for f in glob.glob(f"{datacube_dir}/monthly_datacube_*.nc")
                   if int(f.split("_")[-1].split(".")[0]) >= val_start_year)
    return xr.open_mfdataset(files, combine="by_coords")["datacube"].load()


def eval_skill(model, cube, aux, device, max_pairs=0, progress=True):
    """One-month-ahead RMSE over ocean per state var. Returns {var: (rmse_model,
    rmse_persistence)}."""
    ocean = aux["mask"].values.astype(bool)
    mesh = torch.as_tensor(
        (aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None],
        device=device, dtype=torch.float32)
    mask = torch.as_tensor(aux["mask"].values[None, None], device=device,
                           dtype=torch.float32)
    resolution = torch.full((1, 1), 12.5, device=device, dtype=torch.float32)

    def arr(t, names):
        return np.nan_to_num(cube.isel(time=t).sel(var_names=names).values)

    n = cube.sizes["time"] - 1
    if max_pairs:
        n = min(n, max_pairs)
    se_model = {v: 0.0 for v in STATES}
    se_pers = {v: 0.0 for v in STATES}
    count = 0
    for t in range(n):
        states = torch.as_tensor(arr(t, STATES)[None], device=device, dtype=torch.float32)
        forc = np.stack([arr(t, FORCINGS), arr(t + 1, FORCINGS)])  # (2,4,H,W)
        forc = torch.as_tensor(forc[None], device=device, dtype=torch.float32)
        dd = torch.as_tensor(arr(t, DEGREE)[None], device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(states[:, None], forc, resolution=resolution, mesh=mesh,
                         mask=mask, degree_days=dd)[0].cpu().numpy()  # (6,H,W)
        truth = arr(t + 1, STATES)
        persist = arr(t, STATES)
        for i, v in enumerate(STATES):
            se_model[v] += float(((pred[i] - truth[i]) ** 2)[ocean].sum())
            se_pers[v] += float(((persist[i] - truth[i]) ** 2)[ocean].sum())
        count += int(ocean.sum())
        if progress:
            print(f"  month-pair {t+1}/{n} done", end="\r", flush=True)
    return {v: ((se_model[v] / count) ** 0.5, (se_pers[v] / count) ** 0.5)
            for v in STATES}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/monthly/last-v3.ckpt")
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--val-start-year", type=int, default=2015)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all val pairs")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.ckpt, args.config, device, args.train_config)
    cube = load_val_cube(args.datacube, args.val_start_year)
    aux = xr.open_dataset(args.aux)
    print(f"[data] {cube.sizes['time']} val months")
    rmse = eval_skill(model, cube, aux, device, args.max_pairs)

    print(f"\n\nOne-month-ahead skill (RMSE over ocean, lower is better):\n")
    print(f"  {'var':5s} {'GenSIM':>10s} {'persistence':>12s} {'skill':>8s}")
    beats = 0
    for v in STATES:
        rm, rp = rmse[v]
        skill = 1 - rm / rp if rp > 0 else float("nan")
        beats += skill > 0
        print(f"  {v:5s} {rm:10.4f} {rp:12.4f} {skill:+8.1%}   ({UNITS[v]})")
    print(f"\n=> beats persistence on {beats}/{len(STATES)} variables "
          f"(skill > 0 = model better than 'no change')")


if __name__ == "__main__":
    main()
