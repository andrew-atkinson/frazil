#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decadal free-running rollout of the monthly GenSIM model -- the PRIMARY use case.

Past the 2015-2018 validation window there is NO truth, so this measures
SELF-CONSISTENCY, not skill: does the free-running model hold a stable seasonal
cycle over 10-20 years, or drift / smooth to mush? Reuses drift_rollout's
diagnostics (domain-mean state, ice area, a sharpness / spectral-collapse proxy,
physical-bound health) and eval_monthly's monthly-model loader.

Forcing past 2018 is either the cyclically-repeated 2015-2018 real forcing
(default; a "groundhog decade" that isolates model drift from the atmosphere) or
the CMIP scenario forcing (--forcing cmip; a real future trend but
BIAS-UNCORRECTED vs the ERA5 the model was trained on -- see preprocess_cmip.py).

Run:  python experiments/freerun_monthly.py --years 15 --fast
"""
from __future__ import annotations

import argparse
import csv
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
from drift_rollout import step_diagnostics, SIC_THRESHOLD  # reuse the diagnostics


def load_series(datacube_dir, names):
    """Concatenate every yearly datacube file -> (T, len(names), H, W) + times."""
    files = sorted(glob.glob(f"{datacube_dir}/*_datacube_*.nc"))
    if not files:
        raise SystemExit(f"no datacube files in {datacube_dir}")
    da = xr.open_mfdataset(files, combine="by_coords")["datacube"].load()
    arr = np.nan_to_num(da.sel(var_names=names).transpose("time", "var_names", ...).values)
    return arr.astype(np.float32), pd.DatetimeIndex(da["time"].values)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--cmip-datacube", default="data/train_data/cmip_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--start", default="2015-01", help="start month YYYY-MM (initial truth)")
    ap.add_argument("--years", type=int, default=15, help="length of the free run")
    ap.add_argument("--n-ens", type=int, default=8, help="ensemble members (mean)")
    ap.add_argument("--fast", action="store_true",
                    help="first-order sampler + fewer substeps (for diagnostics, not art)")
    ap.add_argument("--forcing", choices=["cyclic", "cmip"], default="cyclic")
    ap.add_argument("--out-dir", default="plots/freerun")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = E.load_model(args.ckpt or E.default_ckpt(), args.config, device, args.train_config)

    if args.fast:  # ~3x fewer net evals; fine for self-consistency diagnostics
        s = model.sampler
        s.second_order = False
        idx = torch.linspace(0, len(s.schedule) - 1, 13).round().long()
        s.schedule = s.schedule[idx]
        s.n_steps = len(s.schedule) - 1
        print(f"[fast] first-order sampler, {s.n_steps} substeps")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    cell_area = aux["cell_area"].values

    # Real 2015-2018 states+forcing: initial condition, forcing source, truth anchor.
    states_r, times_r = load_series(args.datacube, E.STATES)
    forc_r, _ = load_series(args.datacube, E.FORCINGS)
    dd_r, _ = load_series(args.datacube, E.DEGREE)
    n_real = len(times_r)

    hit = np.where((times_r.year == int(args.start[:4])) &
                   (times_r.month == int(args.start[5:7])))[0]
    if len(hit) == 0:
        raise SystemExit(f"start {args.start} not in {args.datacube}")
    t0 = int(hit[0])
    nsteps = args.years * 12

    # Pick the forcing source. cyclic: repeat the 48 real months. cmip: real trend.
    if args.forcing == "cmip":
        forc_f, times_f = load_series(args.cmip_datacube, E.FORCINGS)
        dd_f, _ = load_series(args.cmip_datacube, E.DEGREE)
        if len(times_f) < nsteps + 1:
            raise SystemExit(f"cmip forcing has {len(times_f)} months < {nsteps+1} needed")
        print("[freerun] *** CMIP forcing is BIAS-UNCORRECTED vs ERA5 -- see preprocess_cmip.py ***")

        def forc_pair(k):  # (t, t+1) forcing pair + degree-days at step k
            return forc_f[k], forc_f[k + 1], dd_f[k]
        month_of = lambda k: times_f[k].month
    else:  # cyclic groundhog decade over the 48 real months
        def forc_pair(k):
            a, b = (t0 + k) % n_real, (t0 + k + 1) % n_real
            return forc_r[a], forc_r[b], dd_r[a]
        month_of = lambda k: times_r[(t0 + k) % n_real].month

    print(f"[freerun] start {args.start}  {args.years}yr ({nsteps} steps)  "
          f"forcing={args.forcing}  n_ens={args.n_ens}", flush=True)

    def rep(x):
        return x.repeat(args.n_ens, *([1] * (x.ndim - 1)))

    mesh = rep(torch.as_tensor(
        (aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None],
        device=device, dtype=torch.float32))
    mask = rep(torch.as_tensor(aux["mask"].values[None, None], device=device,
                               dtype=torch.float32))
    resolution = torch.full((args.n_ens, 1), 12.5, device=device, dtype=torch.float32)

    state = rep(torch.as_tensor(states_r[t0][None], device=device, dtype=torch.float32))

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    d0 = step_diagnostics(states_r[t0], ocean, cell_area)  # step 0 (initial truth)
    d0["year"] = 0.0
    rows.append(d0)
    for k in range(nsteps):
        fa, fb, dd = forc_pair(k)
        forc = rep(torch.as_tensor(np.stack([fa, fb])[None], device=device, dtype=torch.float32))
        ddt = rep(torch.as_tensor(dd[None], device=device, dtype=torch.float32))
        with torch.no_grad():
            state = model(state[:, None], forc, resolution=resolution, mesh=mesh,
                          mask=mask, degree_days=ddt)
        sm = state.mean(0).cpu().numpy()  # ensemble-mean state (6,H,W)
        d = step_diagnostics(sm, ocean, cell_area)
        d["year"] = (k + 1) / 12
        rows.append(d)
        print(f"  yr {d['year']:5.2f}  mon {month_of(k):2d}  "
              f"sit={d['mean_sit']:.3f} sic={d['mean_sic']:.3f} "
              f"area={d['ice_area_km2']:.2e} sharp={d['sharpness_sit']:.2e} "
              f"nan={d['n_nonfinite']}", end="\r", flush=True)
        if d["n_nonfinite"] > 0:
            print(f"\n[freerun] non-finite at year {d['year']:.2f} -- stopping"); break

    # Truth anchor: real-state diagnostics over the 2015-2018 window (where it exists).
    anchor = []
    for t in range(t0, n_real):
        da = step_diagnostics(states_r[t], ocean, cell_area)
        da["year"] = (t - t0) / 12
        anchor.append(da)

    fields = ["year", "mean_sit", "mean_sic", "ice_area_km2", "mean_speed",
              "sharpness_sit", "sharpness_sic", "frac_sic_saturated",
              "frac_sic_zero", "frac_sit_zero", "n_nonfinite"]
    csv_path = os.path.join(args.out_dir, f"freerun_{args.forcing}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    plot(rows, anchor, args, csv_path)


def plot(rows, anchor, args, csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yr = np.array([r["year"] for r in rows])
    ayr = np.array([a["year"] for a in anchor]) if anchor else None
    col = lambda rs, k: np.array([r[k] for r in rs])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"GenSIM monthly free run -- {args.years}yr, forcing={args.forcing} "
                 f"(grey = truth window)", fontsize=13)
    truth_end = ayr[-1] if ayr is not None and len(ayr) else 0

    def shade(a):
        a.axvspan(0, truth_end, color="0.9", zorder=0)
        a.set_xlabel("year of rollout")

    a = ax[0, 0]
    a.plot(yr, col(rows, "mean_sit"), c="#E65007", label="SIT")
    if ayr is not None: a.plot(ayr, col(anchor, "mean_sit"), "o", c="#E65007", ms=3)
    a.set_ylabel("mean thickness (m)"); a.set_title("Domain-mean SIT"); shade(a); a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(yr, col(rows, "ice_area_km2"), c="k")
    if ayr is not None: a.plot(ayr, col(anchor, "ice_area_km2"), "o", c="k", ms=3)
    a.set_ylabel("ice area (km²)"); a.set_title(f"Ice area (SIC>{SIC_THRESHOLD}) -- seasonal cycle"); shade(a)

    a = ax[1, 0]
    s0 = col(rows, "sharpness_sit")[0] or 1.0
    a.plot(yr, col(rows, "sharpness_sit") / s0, c="#2ca02c", label="SIT")
    a.plot(yr, col(rows, "sharpness_sic") / (col(rows, "sharpness_sic")[0] or 1.0),
           c="#9467bd", label="SIC")
    a.axhline(1, c="0.6", lw=0.8, ls=":")
    a.set_ylabel("small-scale energy (rel. step 0)")
    a.set_title("Sharpness -- decay = smoothing to mush"); shade(a); a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(yr, col(rows, "frac_sic_saturated"), label="SIC=1")
    a.plot(yr, col(rows, "frac_sic_zero"), label="SIC=0")
    a.plot(yr, col(rows, "frac_sit_zero"), label="SIT=0")
    a.set_ylabel("fraction of ocean"); a.set_title("Physical-bound health"); shade(a); a.legend(fontsize=8)

    for x in ax.flat: x.margins(x=0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.splitext(csv_path)[0] + ".png"
    fig.savefig(png, dpi=130)
    print(f"\n[plot] {png}")


if __name__ == "__main__":
    main()
