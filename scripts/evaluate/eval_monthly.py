#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a trained monthly GenSIM checkpoint by prediction skill, not loss.

Loads the EMA weights from a Lightning .ckpt into a GenSIMForecastModule and does
one-month-ahead predictions over the validation months (2015-2018): predict
state[t+1] from truth state[t] + forcings, compare to truth. Baseline =
persistence (state[t+1] == state[t]). Reports RMSE over ocean cells per variable
and the skill score 1 - rmse_model/rmse_persistence (>0 means beats persistence).

Run:  python scripts/evaluate/eval_monthly.py --ckpt data/models/monthly/last-v3.ckpt
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
from frazil import STATES, FORCINGS, DEGREE, UNITS  # noqa: F401
from frazil.models import load_model, default_ckpt, load_val_cube, model_tag  # noqa: F401


def eval_skill(model, cube, aux, device, max_pairs=0, progress=True, n_ens=1):
    """One-month-ahead RMSE over ocean per state var. Returns {var: (rmse_model,
    rmse_persistence)}. With n_ens>1, GenSIM is a generative ensemble -- draw
    n_ens members (the batch dim) and use the ensemble MEAN, which cancels
    per-sample sampling noise."""
    ocean = aux["mask"].values.astype(bool)

    def rep(x):  # tile a (1, ...) tensor along the batch dim to n_ens
        return x.repeat(n_ens, *([1] * (x.ndim - 1)))

    mesh = rep(torch.as_tensor(
        (aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None],
        device=device, dtype=torch.float32))
    mask = rep(torch.as_tensor(aux["mask"].values[None, None], device=device,
                               dtype=torch.float32))
    resolution = torch.full((n_ens, 1), 12.5, device=device, dtype=torch.float32)

    def arr(t, names):
        return np.nan_to_num(cube.isel(time=t).sel(var_names=names).values)

    n = cube.sizes["time"] - 1
    if max_pairs:
        n = min(n, max_pairs)
    se_model = {v: 0.0 for v in STATES}
    se_pers = {v: 0.0 for v in STATES}
    count = 0
    for t in range(n):
        states = rep(torch.as_tensor(arr(t, STATES)[None], device=device, dtype=torch.float32))
        forc = np.stack([arr(t, model.forcing_names), arr(t + 1, model.forcing_names)])  # (2,F,H,W)
        forc = rep(torch.as_tensor(forc[None], device=device, dtype=torch.float32))
        dd = rep(torch.as_tensor(arr(t, DEGREE)[None], device=device, dtype=torch.float32))
        with torch.no_grad():
            pred = model(states[:, None], forc, resolution=resolution, mesh=mesh,
                         mask=mask, degree_days=dd).mean(0).cpu().numpy()  # ensemble mean (6,H,W)
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
    ap.add_argument("--ckpt", default=None, help="default: newest last*.ckpt")
    ap.add_argument("--config", default="configs/config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="configs/config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--val-start-year", type=int, default=2015)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all val pairs")
    ap.add_argument("--n-ens", type=int, default=1, help="ensemble members (mean)")
    ap.add_argument("--note", default="", help="annotation stored with the logged result")
    ap.add_argument("--no-log", action="store_true", help="don't append to the results ledger")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = args.ckpt or default_ckpt()
    print(f"[ckpt] {ckpt}")
    model = load_model(ckpt, args.config, device, args.train_config)
    cube = load_val_cube(args.datacube, args.val_start_year)
    aux = xr.open_dataset(args.aux)
    print(f"[data] {cube.sizes['time']} val months  (n_ens={args.n_ens})")
    rmse = eval_skill(model, cube, aux, device, args.max_pairs, n_ens=args.n_ens)

    print(f"\n\nOne-month-ahead skill (RMSE over ocean, lower is better):\n")
    print(f"  {'var':5s} {'GenSIM':>10s} {'persistence':>12s} {'skill':>8s}")
    beats = 0
    skill_v, rm_v, rp_v = {}, {}, {}
    for v in STATES:
        rm, rp = rmse[v]
        skill = 1 - rm / rp if rp > 0 else float("nan")
        skill_v[v], rm_v[v], rp_v[v] = skill, rm, rp
        beats += skill > 0
        print(f"  {v:5s} {rm:10.4f} {rp:12.4f} {skill:+8.1%}   ({UNITS[v]})")
    print(f"\n=> beats persistence on {beats}/{len(STATES)} variables "
          f"(skill > 0 = model better than 'no change')")

    if not args.no_log:
        import frazil.ledger as RL
        RL.log_result("eval_monthly",
                      {**vars(args), "ckpt_resolved": ckpt, "seed": 42,
                       "val_months": int(cube.sizes["time"])},
                      {"skill_vs_persistence": skill_v,
                       "rmse_model": rm_v, "rmse_persistence": rp_v,
                       "beats_persistence": beats},
                      note=args.note)


if __name__ == "__main__":
    main()
