#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free-running (autoregressive) rollout of the monthly GenSIM model over 2015-2018.

Starts from the true Jan-2015 sea-ice state, then predicts each next month by
feeding the model its OWN previous prediction, driven by the REAL monthly
forcings (perfect-forcing rollout). Compares the drifting trajectory to truth at
every lead and to persistence (Jan-2015 held constant), so you can see how fast
error accumulates -- the multi-month skill the one-step eval can't show.

Reuses eval_monthly.load_model + the datacube helpers.

Run:  python experiments/rollout_monthly.py --ckpt data/models/monthly/last.ckpt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import eval_monthly as E


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/monthly/last.ckpt")
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--out-png", default="plots/rollout_2015_2018.png")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = E.load_model(args.ckpt, args.config, device, args.train_config)

    cube = E.load_val_cube(args.datacube, args.start_year)
    n = cube.sizes["time"]
    print(f"[rollout] {n} months from {args.start_year}")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    mesh = torch.as_tensor(
        (aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None],
        device=device, dtype=torch.float32)
    mask = torch.as_tensor(aux["mask"].values[None, None], device=device,
                           dtype=torch.float32)
    resolution = torch.full((1, 1), 12.5, device=device, dtype=torch.float32)

    def arr(t, names):
        return np.nan_to_num(cube.isel(time=t).sel(var_names=names).values)

    truth0 = arr(0, E.STATES)                      # persistence forecast (held)
    state = torch.as_tensor(truth0[None], device=device, dtype=torch.float32)

    # per-lead RMSE over ocean, per variable
    leads, rmse_m, rmse_p = [], {v: [] for v in E.STATES}, {v: [] for v in E.STATES}
    for t in range(n - 1):
        forc = np.stack([arr(t, E.FORCINGS), arr(t + 1, E.FORCINGS)])
        forc = torch.as_tensor(forc[None], device=device, dtype=torch.float32)
        dd = torch.as_tensor(arr(t, E.DEGREE)[None], device=device, dtype=torch.float32)
        with torch.no_grad():
            state = model(state[:, None], forc, resolution=resolution, mesh=mesh,
                          mask=mask, degree_days=dd)  # feed own output back
        pred = state[0].cpu().numpy()
        truth = arr(t + 1, E.STATES)
        leads.append(t + 1)
        for i, v in enumerate(E.STATES):
            rmse_m[v].append(float(np.sqrt(np.mean((pred[i] - truth[i])[ocean] ** 2))))
            rmse_p[v].append(float(np.sqrt(np.mean((truth0[i] - truth[i])[ocean] ** 2))))
        if not np.isfinite(pred).all():
            print(f"[rollout] non-finite at lead {t+1} -- stopping"); break
        print(f"  lead {t+1:2d} mo", end="\r", flush=True)

    leads = np.array(leads)
    # Per-variable skill (variables have different units, so never average RMSE
    # across them). skill = 1 - rmse_model/rmse_persistence, per lead.
    skill = {v: 1 - np.array(rmse_m[v]) / np.array(rmse_p[v]) for v in E.STATES}

    print("\n\nper-variable skill vs persistence (Jan-2015 held):")
    print("  lead " + "  ".join(f"{v:>6s}" for v in E.STATES))
    for L in (1, 3, 6, 12, 24, len(leads)):
        if L <= len(leads):
            i = L - 1
            print(f"  {L:3d}mo " + "  ".join(f"{skill[v][i]:+6.0%}" for v in E.STATES))

    # Save per-variable RMSE + skill so it isn't recomputed.
    import csv
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    with open(os.path.splitext(args.out_png)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lead"] + [f"rmse_{v}" for v in E.STATES]
                   + [f"skill_{v}" for v in E.STATES])
        for i, L in enumerate(leads):
            w.writerow([L] + [f"{rmse_m[v][i]:.5f}" for v in E.STATES]
                       + [f"{skill[v][i]:.4f}" for v in E.STATES])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for v in E.STATES:
        ax1.plot(leads, rmse_m[v], "-", lw=1.2, label=v)
    ax1.set_xlabel("lead (months)"); ax1.set_ylabel("RMSE over ocean")
    ax1.set_title("Free-running rollout error vs lead"); ax1.legend(fontsize=8, ncol=2)
    for v in E.STATES:
        ax2.plot(leads, skill[v], "-", lw=1.2, label=v)
    ax2.axhline(0, color="0.6", ls=":")
    ax2.set_xlabel("lead (months)"); ax2.set_ylabel("skill vs persistence")
    ax2.set_title("Per-variable skill vs lead (>0 beats persistence)")
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
    print(f"[plot] {args.out_png}")


if __name__ == "__main__":
    main()
