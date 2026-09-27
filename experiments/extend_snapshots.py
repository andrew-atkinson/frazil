#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extend existing freerun snapshots forward a few months, reusing them as restart
points instead of re-running the whole rollout.

Use case: a long CMIP freerun saved only Jan/Jul snapshots, but you want the
SEPTEMBER ice minimum for every year. Rather than the full 960-step re-run, seed
the model from each saved July state and step it forward 2 months (Aug -> Sep)
with the CMIP forcing -- 80 Julys x 2 steps = 160 steps (~15 min at n_ens=1)
instead of ~12 h.

*** Approximation ***  The saved snapshot is the ensemble MEAN state (the 8
members' spread was discarded). Restarting from the mean and re-drawing members
is not identical to a continuous rollout (mean-of-rollout != rollout-of-mean),
but over 2 months the error is small -- fine for a September extent estimate, not
for anything needing the exact trajectory.

Run:  python experiments/extend_snapshots.py --fast --n-ens 1 --dry-run   # list plan
      python experiments/extend_snapshots.py --fast --n-ens 1             # do it
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import eval_monthly as E
from freerun_monthly import save_state  # reuse the snapshot writer


def snapshot_date(path):
    """state_YYYYMM.nc -> Timestamp(YYYY-MM-15), or None if it doesn't match."""
    m = re.search(r"state_(\d{4})(\d{2})\.nc$", os.path.basename(path))
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-15") if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap-dir", default="plots/freerun/snapshots",
                    help="folder of state_*.nc to restart from")
    ap.add_argument("--out-dir", default=None, help="where to write (default: --snap-dir)")
    ap.add_argument("--from-month", type=int, default=7, help="restart from this month (7=Jul)")
    ap.add_argument("--steps", type=int, default=2, help="months to advance (7+2 = Sep)")
    ap.add_argument("--cmip-datacube", default="data/train_data/cmip_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--n-ens", type=int, default=1, help="ensemble members (mean)")
    ap.add_argument("--fast", action="store_true", help="first-order sampler + 12 substeps")
    ap.add_argument("--overwrite", action="store_true", help="redo targets that already exist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or args.snap_dir

    # Restart snapshots for the requested month.
    seeds = sorted(f for f in glob.glob(f"{args.snap_dir}/state_*.nc")
                   if (d := snapshot_date(f)) is not None and d.month == args.from_month)
    if not seeds:
        raise SystemExit(f"no state_*.nc with month {args.from_month:02d} in {args.snap_dir}")

    # CMIP forcing (lazy per-step reads; only a few months touched per seed).
    files = sorted(glob.glob(f"{args.cmip_datacube}/*_datacube_*.nc"))
    if not files:
        raise SystemExit(f"no datacube files in {args.cmip_datacube}")
    cds = xr.open_mfdataset(files, combine="by_coords")["datacube"]
    times_f = pd.DatetimeIndex(cds["time"].values)
    dda = cds.sel(var_names=E.DEGREE).transpose("time", "var_names", "y", "x")

    def find_k(date):  # forcing index for a snapshot date
        hit = np.where((times_f.year == date.year) & (times_f.month == date.month))[0]
        return int(hit[0]) if len(hit) else None

    # Build the plan, skipping seeds without forcing coverage or already-done targets.
    plan = []
    for f in seeds:
        d = snapshot_date(f)
        k0 = find_k(d)
        tgt_date = d + pd.DateOffset(months=args.steps)
        tgt = f"{out_dir}/state_{tgt_date:%Y%m}.nc"
        if k0 is None or k0 + args.steps >= len(times_f):
            print(f"  skip {d:%Y-%m}: forcing not covered"); continue
        if os.path.exists(tgt) and not args.overwrite:
            print(f"  skip {d:%Y-%m}: {os.path.basename(tgt)} exists (use --overwrite)"); continue
        plan.append((f, d, k0, tgt_date, tgt))

    print(f"[extend] {len(plan)} seeds  {args.from_month:02d}+{args.steps}mo "
          f"-> month {(args.from_month + args.steps - 1) % 12 + 1:02d}  n_ens={args.n_ens}")
    if args.dry_run:
        for f, d, k0, td, tgt in plan[:5]:
            print(f"    {d:%Y-%m} (fidx {k0}) -> {td:%Y-%m}  {tgt}")
        if len(plan) > 5:
            print(f"    ... +{len(plan)-5} more")
        return
    if not plan:
        return

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = E.load_model(args.ckpt or E.default_ckpt(), args.config, device, args.train_config)
    fda = cds.sel(var_names=model.forcing_names).transpose("time", "var_names", "y", "x")
    if args.fast:
        s = model.sampler
        s.second_order = False
        idx = torch.linspace(0, len(s.schedule) - 1, 13).round().long()
        s.schedule = s.schedule[idx]
        s.n_steps = len(s.schedule) - 1
        print(f"[fast] first-order sampler, {s.n_steps} substeps")

    aux = xr.open_dataset(args.aux)
    N = args.n_ens
    rep = lambda x: x.repeat(N, *([1] * (x.ndim - 1)))
    mesh = rep(torch.as_tensor(
        (aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None],
        device=device, dtype=torch.float32))
    mask = rep(torch.as_tensor(aux["mask"].values[None, None], device=device, dtype=torch.float32))
    resolution = torch.full((N, 1), 12.5, device=device, dtype=torch.float32)

    def at(da, k):
        return np.nan_to_num(da.isel(time=k).values).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    for i, (f, d, k0, tgt_date, tgt) in enumerate(plan):
        sm0 = np.nan_to_num(xr.open_dataset(f)["state"].isel(time=0)
                            .sel(var_names=E.STATES).values).astype(np.float32)
        state = rep(torch.as_tensor(sm0[None], device=device, dtype=torch.float32))
        for j in range(args.steps):
            k = k0 + j
            forc = rep(torch.as_tensor(np.stack([at(fda, k), at(fda, k + 1)])[None],
                                       device=device, dtype=torch.float32))
            ddt = rep(torch.as_tensor(at(dda, k)[None], device=device, dtype=torch.float32))
            with torch.no_grad():
                state = model(state[:, None], forc, resolution=resolution, mesh=mesh,
                              mask=mask, degree_days=ddt)
        sm = state.mean(0).cpu().numpy()
        save_state(sm, tgt_date, tgt)
        rate = (time.time() - t_start) / (i + 1)
        print(f"  {i+1}/{len(plan)}  {d:%Y-%m}->{tgt_date:%Y-%m}  finite={np.isfinite(sm).all()}"
              f"  {rate*args.steps:.1f}s/seed  ETA {rate*(len(plan)-i-1)/60:4.0f}m", flush=True)

    print(f"[extend] done: {len(plan)} September states -> {out_dir}/  "
          f"({(time.time()-t_start)/60:.1f} min)")
    print(f"  now: python experiments/ice_area.py data/train_data/monthly_datacube "
          f"{args.snap_dir} --month {(args.from_month + args.steps - 1) % 12 + 1}")


if __name__ == "__main__":
    main()
