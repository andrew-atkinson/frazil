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
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import eval_monthly as E
import results_log as RL


def build_climatology(datacube_dir, before_year):
    """Per-calendar-month mean of each state var over the TRAINING years
    (< before_year). Returns {month: (6,H,W)} or None if no training files.
    The honest long-lead baseline: 'predict the average July', which persistence
    (frozen January) can't touch but a real forecast should."""
    files = sorted(f for f in glob.glob(f"{datacube_dir}/monthly_datacube_*.nc")
                   if int(f.split("_")[-1].split(".")[0]) < before_year)
    if not files:
        return None
    c = xr.open_mfdataset(files, combine="by_coords")["datacube"].load()
    months = pd.DatetimeIndex(c["time"].values).month
    clim = {}
    for m in range(1, 13):
        sel = np.where(months == m)[0]
        clim[m] = np.nan_to_num(
            c.isel(time=sel).sel(var_names=E.STATES).mean("time").values)
    return clim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="default: newest last*.ckpt")
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--starts", default="2015-01,2015-04,2015-07,2015-10",
                    help="comma-sep YYYY-MM start months; skill is averaged over them")
    ap.add_argument("--max-lead", type=int, default=12, help="cap trajectory length (months)")
    ap.add_argument("--n-ens", type=int, default=8, help="ensemble members (mean)")
    ap.add_argument("--fast", action="store_true",
                    help="first-order sampler + fewer substeps (for skill curves, not art)")
    ap.add_argument("--out-png", default=None,
                    help="default: plots/rollout_<model>.png (self-describing)")
    ap.add_argument("--note", default="", help="annotation stored with the logged result")
    ap.add_argument("--no-log", action="store_true", help="don't append to the results ledger")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = args.ckpt or E.default_ckpt()
    args.out_png = args.out_png or f"plots/rollout_{E.model_tag(ckpt)}.png"
    model = E.load_model(ckpt, args.config, device, args.train_config)

    if args.fast:  # ~4x fewer net evals: drop 2nd-order (2 calls->1) and subsample schedule
        s = model.sampler
        s.second_order = False
        idx = torch.linspace(0, len(s.schedule) - 1, 13).round().long()  # 20 -> 12 substeps
        s.schedule = s.schedule[idx]
        s.n_steps = len(s.schedule) - 1
        print(f"[fast] first-order sampler, {s.n_steps} substeps")

    starts = [s.strip() for s in args.starts.split(",") if s.strip()]
    min_year = min(int(s[:4]) for s in starts)
    cube = E.load_val_cube(args.datacube, min_year)
    n = cube.sizes["time"]
    print(f"[rollout] {len(starts)} starts {starts}  max-lead {args.max_lead}mo  (n_ens={args.n_ens})")

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

    clim = build_climatology(args.datacube, min_year)  # {month: (6,H,W)} or None
    months = pd.DatetimeIndex(cube["time"].values)     # per-step timestamps
    if clim is None:
        print(f"[rollout] no pre-{min_year} training files -> climatology baseline skipped")

    def start_index(ym):  # "YYYY-MM" -> position in the val cube
        y, m = int(ym[:4]), int(ym[5:7])
        hit = np.where((months.year == y) & (months.month == m))[0]
        if len(hit) == 0:
            raise SystemExit(f"start {ym} not in datacube")
        return int(hit[0])

    def run_one(t0):
        """Free-running rollout from cube index t0. Returns per-lead squared-error
        sums over ocean per var for model/persistence/climatology (pooled later)."""
        truth0 = arr(t0, E.STATES)                  # persistence = start held
        state = rep(torch.as_tensor(truth0[None], device=device, dtype=torch.float32))
        out = {}  # lead -> {v: (sse_model, sse_pers, sse_clim or None)}
        for lead in range(1, args.max_lead + 1):
            t = t0 + lead - 1
            if t + 1 >= n:
                break
            forc = np.stack([arr(t, model.forcing_names), arr(t + 1, model.forcing_names)])
            forc = rep(torch.as_tensor(forc[None], device=device, dtype=torch.float32))
            dd = rep(torch.as_tensor(arr(t, E.DEGREE)[None], device=device, dtype=torch.float32))
            with torch.no_grad():
                state = model(state[:, None], forc, resolution=resolution, mesh=mesh,
                              mask=mask, degree_days=dd)  # feed own output back
            pred = state.mean(0).cpu().numpy()
            if not np.isfinite(pred).all():
                print(f"\n[rollout] non-finite at lead {lead} (start {t0}) -- stopping start")
                break
            truth = arr(t + 1, E.STATES)
            cm = clim[int(months.month[t + 1])] if clim is not None else None
            out[lead] = {v: (
                float(((pred[i] - truth[i]) ** 2)[ocean].sum()),
                float(((truth0[i] - truth[i]) ** 2)[ocean].sum()),
                float(((cm[i] - truth[i]) ** 2)[ocean].sum()) if cm is not None else None,
            ) for i, v in enumerate(E.STATES)}
            print(f"  start {months[t0].strftime('%Y-%m')}  lead {lead:2d} mo", end="\r", flush=True)
        return out

    # Pool squared errors across starts, per lead (RMSE isn't linear, so sum SSE
    # then sqrt -- never average per-start RMSEs). count = ocean cells * n_starts.
    ncell = int(ocean.sum())
    sse = {}  # lead -> {v: [sse_m, sse_p, sse_c]}, nstart -> {lead: count}
    nstart = {}
    for ym in starts:
        for lead, d in run_one(start_index(ym)).items():
            sl = sse.setdefault(lead, {v: [0.0, 0.0, 0.0] for v in E.STATES})
            nstart[lead] = nstart.get(lead, 0) + 1
            for v in E.STATES:
                m, p, c = d[v]
                sl[v][0] += m; sl[v][1] += p
                sl[v][2] += (c if c is not None else 0.0)

    have_clim = clim is not None
    leads = np.array(sorted(sse))
    def rmse_of(j):
        return {v: np.array([np.sqrt(sse[L][v][j] / (ncell * nstart[L])) for L in leads])
                for v in E.STATES}
    rmse_m, rmse_p, rmse_c = rmse_of(0), rmse_of(1), rmse_of(2)
    skill = {v: 1 - rmse_m[v] / rmse_p[v] for v in E.STATES}
    skill_c = ({v: 1 - rmse_m[v] / rmse_c[v] for v in E.STATES} if have_clim else None)

    pos = {int(L): i for i, L in enumerate(leads)}  # lead -> row index (leads may skip)
    def table(title, sk):
        print(f"\n\n{title}:")
        print("  lead " + "  ".join(f"{v:>6s}" for v in E.STATES) + "   n")
        for L in sorted({1, 3, 6, 12, int(leads[-1])} & set(pos)):
            i = pos[L]
            print(f"  {L:3d}mo " + "  ".join(f"{sk[v][i]:+6.0%}" for v in E.STATES)
                  + f"   {nstart[L]}")

    table("per-variable skill vs persistence (start-month held)", skill)
    if have_clim:
        table("per-variable skill vs climatology (average-that-month)", skill_c)
        print("\n(climatology is the honest bar: >0 means the model beats just"
              "\n predicting the monthly average. n = starts averaged at that lead.)")

    # Save per-variable RMSE + skill so it isn't recomputed.
    import csv
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    with open(os.path.splitext(args.out_png)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = (["lead"] + [f"rmse_{v}" for v in E.STATES]
                + [f"skillP_{v}" for v in E.STATES]
                + ([f"skillC_{v}" for v in E.STATES] if have_clim else []))
        w.writerow(cols)
        for i, L in enumerate(leads):
            row = ([L] + [f"{rmse_m[v][i]:.5f}" for v in E.STATES]
                   + [f"{skill[v][i]:.4f}" for v in E.STATES]
                   + ([f"{skill_c[v][i]:.4f}" for v in E.STATES] if have_clim else []))
            w.writerow(row)

    # Append a structured record to the results ledger (per-variable arrays
    # across leads), pinned to the exact checkpoint + code state.
    if not args.no_log:
        ckpt = args.ckpt or E.default_ckpt()
        params = {**vars(args), "ckpt_resolved": ckpt, "seed": 42}
        metrics = {
            "leads_months": [int(L) for L in leads],
            "n_starts_per_lead": [nstart[int(L)] for L in leads],
            "skill_vs_persistence": {v: skill[v].tolist() for v in E.STATES},
            "skill_vs_climatology": ({v: skill_c[v].tolist() for v in E.STATES}
                                     if have_clim else None),
            "rmse_model": {v: [rmse_m[v][i] for i in range(len(leads))] for v in E.STATES},
        }
        RL.log_result("rollout_monthly", params, metrics, note=args.note)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for v in E.STATES:
        ax1.plot(leads, rmse_m[v], "-", lw=1.2, label=v)
    ax1.set_xlabel("lead (months)"); ax1.set_ylabel("RMSE over ocean")
    ax1.set_title(f"Free-running rollout error vs lead\n(avg over {len(starts)} starts)")
    ax1.legend(fontsize=8, ncol=2)
    # Right panel: skill vs climatology (solid, the honest bar) with persistence
    # (dashed) for reference. Falls back to persistence-only if no training data.
    for i, v in enumerate(E.STATES):
        col = f"C{i}"
        ax2.plot(leads, (skill_c if have_clim else skill)[v], "-", lw=1.4, color=col, label=v)
        if have_clim:
            ax2.plot(leads, skill[v], "--", lw=0.8, color=col, alpha=0.6)
    ax2.axhline(0, color="0.6", ls=":")
    ax2.set_xlabel("lead (months)")
    ax2.set_ylabel("skill vs climatology" if have_clim else "skill vs persistence")
    ax2.set_title(("Skill vs climatology (solid) & persistence (dashed)"
                   if have_clim else "Per-variable skill vs persistence")
                  + "\n(>0 beats the baseline)")
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
    print(f"[plot] {args.out_png}")


if __name__ == "__main__":
    main()
