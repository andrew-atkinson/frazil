#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recompute September sea-ice extent the CORRECT way for a generative ensemble, to
see how much of GenSIM's apparent 'drift' is an ice-edge averaging artifact.

  extent-of-mean  = extent(mean of the members)   <- what we plotted (over-counts
                    the soft, disagreeing ice edge -> inflated summer extent)
  mean-of-extents = mean over members of extent(member)   <- the correct quantity

Seeds each July snapshot with n_ens members, steps to September with CMIP forcing,
and reports both, overlaid with NSIDC obs and (optionally) the neXtSIM truth.

*** CAVEAT ***  Snapshots hold the ensemble MEAN (spread was discarded), so the
members here only diverge over the 2 restart months -> this UNDER-estimates the
edge spread of a continuous rollout. Treat it as a fast lower bound; the proper
number comes from a continuous n_ens>1 freerun that logs both metrics.

Run:  python experiments/ensemble_extent.py --fast --stride 8      # ~10 Julys, quick look
      python experiments/ensemble_extent.py --fast                 # every July (slow)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
import eval_monthly as E
from ice_area import ice_area_km2, load_nsidc
from extend_snapshots import snapshot_date

SIC = 1  # index of 'sic' in E.STATES


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap-dir", default="plots/freerun/snapshots")
    ap.add_argument("--from-month", type=int, default=7)
    ap.add_argument("--steps", type=int, default=2, help="months to advance (7+2=Sep)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth July (speed)")
    ap.add_argument("--cmip-datacube", default="data/train_data/cmip_datacube")
    ap.add_argument("--truth", default="data/train_data/monthly_datacube",
                    help="neXtSIM datacube for the truth baseline (or '' to skip)")
    ap.add_argument("--nsidc", default="data/obs/nsidc", help="NSIDC dir/CSV ('' to skip)")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--n-ens", type=int, default=8)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--out", default="plots/ensemble_extent_sept.png")
    args = ap.parse_args(argv)

    tgt_month = (args.from_month + args.steps - 1) % 12 + 1
    seeds = sorted(f for f in glob.glob(f"{args.snap_dir}/state_*.nc")
                   if (d := snapshot_date(f)) is not None and d.month == args.from_month)
    seeds = seeds[::args.stride]
    if not seeds:
        raise SystemExit(f"no month-{args.from_month:02d} snapshots in {args.snap_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(42)
    model = E.load_model(args.ckpt or E.default_ckpt(), args.config, device, args.train_config)
    if args.fast:
        s = model.sampler
        s.second_order = False
        idx = torch.linspace(0, len(s.schedule) - 1, 13).round().long()
        s.schedule = s.schedule[idx]
        s.n_steps = len(s.schedule) - 1
        print(f"[fast] first-order sampler, {s.n_steps} substeps")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    ca_km2 = aux["cell_area"].values / 1e6
    N = args.n_ens
    rep = lambda x: x.repeat(N, *([1] * (x.ndim - 1)))
    mesh = rep(torch.as_tensor((aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None],
                               device=device, dtype=torch.float32))
    mask = rep(torch.as_tensor(aux["mask"].values[None, None], device=device, dtype=torch.float32))
    resolution = torch.full((N, 1), 12.5, device=device, dtype=torch.float32)

    cds = xr.open_mfdataset(sorted(glob.glob(f"{args.cmip_datacube}/*_datacube_*.nc")),
                            combine="by_coords")["datacube"]
    times_f = pd.DatetimeIndex(cds["time"].values)
    fda = cds.sel(var_names=E.FORCINGS).transpose("time", "var_names", "y", "x")
    dda = cds.sel(var_names=E.DEGREE).transpose("time", "var_names", "y", "x")
    at = lambda da, k: np.nan_to_num(da.isel(time=k).values).astype(np.float32)

    print(f"[ens-extent] {len(seeds)} seeds  {args.from_month:02d}+{args.steps}->{tgt_month:02d}  "
          f"n_ens={N}  (~{len(seeds)*args.steps*(44 if N==8 else 6*N/8):.0f}s at n_ens={N})")
    years, eom, moe = [], [], []
    t0 = time.time()
    for i, f in enumerate(seeds):
        d = snapshot_date(f)
        hit = np.where((times_f.year == d.year) & (times_f.month == d.month))[0]
        if not len(hit) or int(hit[0]) + args.steps >= len(times_f):
            continue
        k0 = int(hit[0])
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
        sic = state[:, SIC].cpu().numpy()                       # (N, H, W) members
        e_mean = float(ice_area_km2(sic.mean(0)[None], ocean, ca_km2, "extent", args.threshold)[0])
        e_perm = float(ice_area_km2(sic, ocean, ca_km2, "extent", args.threshold).mean())
        years.append(d.year + args.steps / 12); eom.append(e_mean); moe.append(e_perm)
        print(f"  {d.year}: extent-of-mean {e_mean/1e6:.2f}  mean-of-extents {e_perm/1e6:.2f} "
              f"M km²  (Δ {(e_mean-e_perm)/1e6:+.2f})  [{time.time()-t0:.0f}s]", flush=True)

    years, eom, moe = np.array(years), np.array(eom), np.array(moe)
    if len(years) == 0:
        raise SystemExit("no seeds had forcing coverage")
    print(f"\n[ens-extent] mean inflation from extent-of-mean: "
          f"{(eom-moe).mean()/1e6:+.2f} M km²  ({100*(eom-moe).mean()/eom.mean():.0f}% of extent)")

    # Obs + truth baselines for context / offsets.
    obs = load_nsidc(args.nsidc, tgt_month) if args.nsidc else {}
    if obs:
        common = sorted({int(y) for y in years} & set(obs))
        for name, series in [("extent-of-mean", eom), ("mean-of-extents", moe)]:
            by = {int(y): v for y, v in zip(years, series)}
            off = np.mean([by[y] - obs[y] for y in common if y in by])
            print(f"[offset] {name:16s} - NSIDC ({len(common)}yr): {off/1e6:+.2f} M km²")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(years, eom/1e6, "-o", ms=3, color="C1", label="model: extent-of-mean (old)")
    ax.plot(years, moe/1e6, "-o", ms=3, color="C2", label="model: mean-of-extents (correct)")
    if obs:
        oy = sorted(obs); ax.plot(oy, [obs[y]/1e6 for y in oy], "k--s", ms=4, label="NSIDC obs")
    if args.truth:
        ts = xr.open_mfdataset(sorted(glob.glob(f"{args.truth}/*_datacube_*.nc")),
                               combine="by_coords")["datacube"]
        tt = pd.DatetimeIndex(ts["time"].values)
        keep = tt.month == tgt_month
        tsic = np.nan_to_num(ts.sel(var_names="sic").isel(time=np.where(keep)[0]).values)
        ta = ice_area_km2(tsic, ocean, ca_km2, "extent", args.threshold)
        ax.plot([y.year for y in tt[keep]], ta/1e6, "-", color="C0", lw=1, label="neXtSIM truth")
    ax.set_ylabel("September extent (M km²)"); ax.set_xlabel("year")
    ax.set_title("Ensemble extent: extent-of-mean vs mean-of-extents"); ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"[plot] {args.out}")


if __name__ == "__main__":
    main()
