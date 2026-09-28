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
from results_log import model_tag  # noqa: F401  (re-exported: E.model_tag)

STATES = ["sit", "sic", "sid", "siu", "siv", "snt"]
FORCINGS = ["tus", "huss", "uas", "vas"]
DEGREE = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
UNITS = {"sit": "m", "sic": "1", "sid": "1", "siu": "m/s", "siv": "m/s", "snt": "m"}


def load_model(ckpt_path, config, device, train_config):
    cfg = OmegaConf.load(config)
    # Architecture and forcing variables come from the run's own saved config
    # (data/models/<run>/.hydra/config.yaml) when present, else --train-config.
    run_cfg = os.path.join(os.path.dirname(ckpt_path), ".hydra", "config.yaml")
    tcfg = OmegaConf.load(run_cfg if os.path.exists(run_cfg) else train_config)
    cfg.surrogate.network = tcfg.surrogate.network
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    if "encoder.mean" in sd:  # input normalization exactly as trained
        cfg.surrogate.encoder.mean = sd["encoder.mean"].flatten().tolist()
        cfg.surrogate.encoder.std = sd["encoder.std"].flatten().tolist()
    model = instantiate(cfg.surrogate)  # GenSIMForecastModule (recursive)
    model.forcing_names = list(tcfg.data.get("forcing_variables", FORCINGS))
    gensim.network.USE_FLASH_ATTN = (
        gensim.network.USE_FLASH_ATTN and device.type == "cuda")

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


def default_ckpt(ckpt_dir="data/models/monthly"):
    """The canonical monthly.ckpt if present, else the newest last*.ckpt.
    (Lightning's -v rotation makes plain last.ckpt an unreliable name, so we
    consolidate the trained model to monthly.ckpt and prefer that.)"""
    canonical = f"{ckpt_dir}/monthly.ckpt"
    if os.path.exists(canonical):
        return canonical
    cks = glob.glob(f"{ckpt_dir}/last*.ckpt")
    if not cks:
        raise SystemExit(f"no monthly.ckpt or last*.ckpt in {ckpt_dir}")
    return max(cks, key=os.path.getmtime)


def load_val_cube(datacube_dir, val_start_year):
    files = sorted(f for f in glob.glob(f"{datacube_dir}/monthly_datacube_*.nc")
                   if int(f.split("_")[-1].split(".")[0]) >= val_start_year)
    return xr.open_mfdataset(files, combine="by_coords")["datacube"].load()


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
        import results_log as RL
        RL.log_result("eval_monthly",
                      {**vars(args), "ckpt_resolved": ckpt, "seed": 42,
                       "val_months": int(cube.sizes["time"])},
                      {"skill_vs_persistence": skill_v,
                       "rmse_model": rm_v, "rmse_persistence": rp_v,
                       "beats_persistence": beats},
                      note=args.note)


if __name__ == "__main__":
    main()
