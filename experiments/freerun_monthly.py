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
import results_log as RL
from drift_rollout import step_diagnostics, SIC_THRESHOLD  # reuse the diagnostics


def snap_tag(ckpt, forcing):
    """Namespace outputs by model + forcing so runs never clobber each other:
    monthly.ckpt+cmip -> 'monthly_cmip'; monthly_pf/last.ckpt+cmip -> 'monthly_pf_cmip'."""
    return f"{E.model_tag(ckpt)}_{forcing}"


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
    ap.add_argument("--config", default="configs/config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="configs/config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--cmip-datacube", default="data/train_data/cmip_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--start", default="2015-01", help="start month YYYY-MM (initial truth)")
    ap.add_argument("--years", type=int, default=15, help="length of the free run")
    ap.add_argument("--n-ens", type=int, default=8, help="ensemble members (mean)")
    ap.add_argument("--fast", action="store_true",
                    help="first-order sampler + fewer substeps (for diagnostics, not art)")
    ap.add_argument("--forcing", choices=["cyclic", "cmip", "era5"], default="cyclic",
                    help="era5: the real ERA5 forcing of each successive month (a true "
                         "hindcast); past 2018 it continues from --era5-ext")
    ap.add_argument("--era5-ext", default="data/train_data/era5_forcing_datacube",
                    help="forcing-only ERA5 cube beyond neXtSIM's 2018 end "
                         "(preprocess_monthly.py --forcings-only)")
    ap.add_argument("--out-dir", default="plots/freerun")
    ap.add_argument("--snapshot-every", type=int, default=12,
                    help="save the full spatial state every N months (0=off); the "
                         "final state is always saved. These are what maps are drawn from.")
    ap.add_argument("--no-map", action="store_true", help="skip the final-state map figure")
    ap.add_argument("--climatology-forcing", nargs="*", default=[],
                    help="era5 only: replace these forcing channels (e.g. sst) with their "
                         "1995-2014 calendar-month mean, removing year-specific information "
                         "(e.g. the observed-ice imprint in ERA5 SST). Tagged '_clim-<vars>'.")
    ap.add_argument("--save-members", nargs="*", default=None, metavar="VAR",
                    help="with each snapshot also write every member's fields "
                         "(members_YYYYMM.nc, dims member,y,x). No VAR = sic only; "
                         "e.g. --save-members sic sit for thickness frames too")
    ap.add_argument("--note", default="", help="annotation stored with the logged result")
    ap.add_argument("--no-log", action="store_true", help="don't append to the results ledger")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = args.ckpt or E.default_ckpt()
    model = E.load_model(ckpt, args.config, device, args.train_config)

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

    # Real 2015-2018 states: initial condition + truth anchor (always needed).
    states_r, times_r = load_series(args.datacube, E.STATES)
    n_real = len(times_r)
    # The cyclic forcing is only used by the cyclic path -- don't load its ~2.4GB
    # in CMIP mode (that plus the CMIP forcing was thrashing swap).
    forc_r = dd_r = None
    if args.forcing in ("cyclic", "era5"):
        forc_r, _ = load_series(args.datacube, model.forcing_names)
        dd_r, _ = load_series(args.datacube, E.DEGREE)

    hit = np.where((times_r.year == int(args.start[:4])) &
                   (times_r.month == int(args.start[5:7])))[0]
    if len(hit) == 0:
        raise SystemExit(f"start {args.start} not in {args.datacube}")
    t0 = int(hit[0])
    nsteps = args.years * 12

    # Pick the forcing source. cyclic: repeat the 48 real months. cmip: real trend.
    if args.forcing == "cmip":
        # LAZY: read forcing per-step instead of materializing all ~86 years
        # (~8.6GB) up front, which swap-thrashes a 16GB laptop.
        files = sorted(glob.glob(f"{args.cmip_datacube}/*_datacube_*.nc"))
        if not files:
            raise SystemExit(f"no datacube files in {args.cmip_datacube}")
        cds = xr.open_mfdataset(files, combine="by_coords")["datacube"]
        times_f = pd.DatetimeIndex(cds["time"].values)
        # Forcing must start at the same month as the initial state.
        hit_f = np.where((times_f.year == int(args.start[:4])) &
                         (times_f.month == int(args.start[5:7])))[0]
        if len(hit_f) == 0:
            raise SystemExit(f"start {args.start} not in {args.cmip_datacube} "
                             f"({times_f[0]:%Y-%m}..{times_f[-1]:%Y-%m})")
        f0 = int(hit_f[0])
        if len(times_f) - f0 < nsteps + 1:
            raise SystemExit(f"cmip forcing has {len(times_f) - f0} months from {args.start} "
                             f"< {nsteps+1} needed")
        fda = cds.sel(var_names=model.forcing_names).transpose("time", "var_names", "y", "x")
        dda = cds.sel(var_names=E.DEGREE).transpose("time", "var_names", "y", "x")
        # Read the bias-correction stamp preprocess_cmip writes onto the datacube.
        bc = None
        try:
            one = sorted(glob.glob(f"{args.cmip_datacube}/*_datacube_*.nc"))[0]
            bc = xr.open_dataset(one)["datacube"].attrs.get("bias_corrected")
        except Exception:
            pass
        if bc in (None, ""):
            print("[freerun] CMIP forcing: bias-correction UNSTAMPED (older preprocess) -- "
                  "check it was corrected vs ERA5, or re-run preprocess_cmip.py")
        elif bc == "none":
            print("[freerun] *** CMIP forcing is RAW / BIAS-UNCORRECTED vs ERA5 ***")
        else:
            print(f"[freerun] CMIP forcing bias-corrected vs ERA5 (baseline {bc})")

        def _at(da, k):  # materialize just month k (one small slice)
            return np.nan_to_num(da.isel(time=k).values).astype(np.float32)

        def forc_pair(k):  # (t, t+1) forcing pair + degree-days at step k
            return _at(fda, f0 + k), _at(fda, f0 + k + 1), _at(dda, f0 + k)
        month_of = lambda k: times_f[f0 + k].month
    elif args.forcing == "era5":  # true hindcast: real forcing of each successive month
        times_e = times_r
        if t0 + nsteps >= n_real and os.path.isdir(args.era5_ext):
            # Past neXtSIM's end: append the forcing-only ERA5 extension (no truth there).
            fx, tx = load_series(args.era5_ext, model.forcing_names)
            dx, _ = load_series(args.era5_ext, E.DEGREE)
            keep = np.asarray(tx > times_r[-1])
            forc_r = np.concatenate([forc_r, fx[keep]])
            dd_r = np.concatenate([dd_r, dx[keep]])
            times_e = times_r.append(tx[keep])
            print(f"[freerun] ERA5 extension: +{int(keep.sum())} months from {args.era5_ext}")
        ym = times_e.year * 12 + times_e.month
        if (np.diff(ym) != 1).any():
            raise SystemExit("era5 forcing has a gap or overlap between the datacube and "
                             f"{args.era5_ext} -- months must be consecutive")
        if t0 + nsteps >= len(times_e):
            raise SystemExit(f"era5 hindcast: start {args.start} + {args.years}yr runs past "
                             f"{times_e[-1]:%Y-%m} (only {(len(times_e)-1-t0)/12:.1f}yr of real forcing left); "
                             f"reduce --years, move --start earlier, or build {args.era5_ext}")
        base = np.asarray((times_e.year >= 1995) & (times_e.year <= 2014))
        for name in args.climatology_forcing:
            j = model.forcing_names.index(name)
            for mo in range(1, 13):
                sel = np.asarray(times_e.month == mo)
                forc_r[sel, j] = forc_r[base & sel, j].mean(0)
            print(f"[freerun] {name}: replaced by its 1995-2014 monthly climatology")
        def forc_pair(k):
            return forc_r[t0 + k], forc_r[t0 + k + 1], dd_r[t0 + k]
        month_of = lambda k: times_e[t0 + k].month
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
    # Namespace ALL outputs by model+forcing so runs never overwrite each other
    # (e.g. freerun_monthly_pf_cmip.*, snapshots/monthly_pf_cmip/).
    # era5 runs carry their start year: different starts are different runs.
    tag = snap_tag(ckpt, f"era5_{args.start[:4]}" if args.forcing == "era5" else args.forcing)
    if args.climatology_forcing:
        tag += "_clim-" + "-".join(args.climatology_forcing)
    snap_dir = os.path.join(args.out_dir, "snapshots", tag)
    if args.snapshot_every:
        os.makedirs(snap_dir, exist_ok=True)
        print(f"[freerun] snapshots -> {snap_dir}", flush=True)
    base_date = pd.Timestamp(f"{args.start}-01")   # forward calendar for naming

    rows = []
    d0 = step_diagnostics(states_r[t0], ocean, cell_area)  # step 0 (initial truth)
    d0["year"] = 0.0
    d0["wall_s"] = 0.0
    d0["ext_members_km2"], d0["ext_members_std_km2"] = d0["ice_area_km2"], 0.0  # members start identical
    rows.append(d0)
    sm = states_r[t0]
    last_date = base_date
    t_start = time.time()
    for k in range(nsteps):
        t_step = time.time()
        fa, fb, dd = forc_pair(k)
        forc = rep(torch.as_tensor(np.stack([fa, fb])[None], device=device, dtype=torch.float32))
        ddt = rep(torch.as_tensor(dd[None], device=device, dtype=torch.float32))
        with torch.no_grad():
            state = model(state[:, None], forc, resolution=resolution, mesh=mesh,
                          mask=mask, degree_days=ddt)
        sm = state.mean(0).cpu().numpy()  # ensemble-mean state (6,H,W)
        sic_m = state[:, 1].cpu().numpy()  # each member's sic (n_ens,H,W)
        last_date = base_date + pd.DateOffset(months=k + 1)
        d = step_diagnostics(sm, ocean, cell_area)
        # Extent of the MEAN over-counts the soft edge where members disagree;
        # the mean of each member's own extent is the unbiased ensemble number.
        ext_m = (((sic_m > SIC_THRESHOLD) & ocean) * cell_area).sum((1, 2)) / 1e6
        d["ext_members_km2"] = float(ext_m.mean())
        d["ext_members_std_km2"] = float(ext_m.std())
        d["year"] = (k + 1) / 12
        d["wall_s"] = time.time() - t_step
        rows.append(d)
        elapsed = time.time() - t_start
        rate = elapsed / (k + 1)                       # mean s/step so far
        eta = rate * (nsteps - k - 1)
        print(f"  yr {d['year']:5.2f}  mon {month_of(k):2d}  "
              f"sit={d['mean_sit']:.3f} sic={d['mean_sic']:.3f} "
              f"ext mean-field={d['ice_area_km2']:.2e} members={d['ext_members_km2']:.2e} "
              f"nan={d['n_nonfinite']}  "
              f"{d['wall_s']:.1f}s/step  ETA {eta/60:4.0f}m", end="\r", flush=True)
        last_nonfinite = d["n_nonfinite"] > 0
        if args.snapshot_every and ((k + 1) % args.snapshot_every == 0
                                    or k + 1 == nsteps or last_nonfinite):
            save_state(sm, last_date, os.path.join(
                snap_dir, f"state_{last_date:%Y%m}.nc"))
            if args.save_members is not None:
                mvars = args.save_members or ["sic"]
                xr.Dataset({v: (("member", "y", "x"), state[:, E.STATES.index(v)].cpu().numpy())
                            for v in mvars}).to_netcdf(
                    os.path.join(snap_dir, f"members_{last_date:%Y%m}.nc"),
                    encoding={v: {"zlib": True, "complevel": 4} for v in mvars})
        if last_nonfinite:
            print(f"\n[freerun] non-finite at year {d['year']:.2f} -- stopping"); break

    total = time.time() - t_start
    done = len(rows) - 1
    summary = (f"{pd.Timestamp.now():%Y-%m-%d %H:%M}  {args.forcing}  {done} steps  "
               f"n_ens={args.n_ens} fast={args.fast}  {total/60:.1f} min total  "
               f"{total/max(done,1):.2f} s/step")
    print(f"\n[timing] {summary}")
    with open(os.path.join(args.out_dir, "timing.log"), "a") as fh:
        fh.write(summary + "\n")

    # Truth anchor: real-state diagnostics over the 2015-2018 window (where it exists).
    anchor = []
    for t in range(t0, n_real):
        da = step_diagnostics(states_r[t], ocean, cell_area)
        da["year"] = (t - t0) / 12
        anchor.append(da)

    fields = ["year", "wall_s", "mean_sit", "mean_sic", "ice_area_km2",
              "ext_members_km2", "ext_members_std_km2", "mean_speed",
              "sharpness_sit", "sharpness_sic", "frac_sic_saturated",
              "frac_sic_zero", "frac_sit_zero", "n_nonfinite"]
    csv_path = os.path.join(args.out_dir, f"freerun_{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Log a compact drift summary (initial/final/min/max of the stability
    # diagnostics) so free-run behaviour is comparable across checkpoints.
    if not args.no_log:
        def stat(key):
            s = [r[key] for r in rows]
            return {"initial": s[0], "final": s[-1], "min": min(s), "max": max(s)}
        keys = ["mean_sit", "mean_sic", "ice_area_km2", "sharpness_sit", "sharpness_sic"]
        keys += ["ext_members_km2"] if "ext_members_km2" in rows[-1] else []
        drift = {k: stat(k) for k in keys}
        drift["years_completed"] = round((len(rows) - 1) / 12, 3)
        drift["nonfinite_total"] = sum(r["n_nonfinite"] for r in rows)
        drift["sharpness_sit_retention"] = (
            rows[-1]["sharpness_sit"] / (rows[0]["sharpness_sit"] or 1.0))
        RL.log_result("freerun_monthly",
                      {**vars(args), "ckpt_resolved": ckpt, "snap_dir": snap_dir,
                       "seed": 42},
                      drift, note=args.note)

    plot(rows, anchor, args, csv_path)
    if not args.no_map:
        state_map(sm, ocean, os.path.join(args.out_dir, f"freerun_{tag}_map.png"),
                  f"GenSIM sea ice {last_date:%Y-%m}  ({args.years}yr free run, {tag})")


def save_state(sm, date, path):
    """Save one ensemble-mean spatial state (6,H,W) as NetCDF for later mapping."""
    xr.DataArray(sm[None], dims=("time", "var_names", "y", "x"),
                 coords={"var_names": E.STATES, "time": [np.datetime64(date)]},
                 name="state").to_dataset().to_netcdf(path)


def state_map(sm, ocean, path, title):
    """Per-variable maps of one spatial state (land masked). No truth needed."""
    from predict_monthly import CMAP, DIVERGING  # reuse the colormaps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, i, v in zip(axes.flat, range(len(E.STATES)), E.STATES):
        field = np.where(ocean, sm[i], np.nan)
        o = sm[i][ocean]
        if v in DIVERGING:
            m = np.nanmax(np.abs(o)); vmin, vmax = -m, m
        else:
            vmin, vmax = np.nanpercentile(o, 2), np.nanpercentile(o, 98)
        im = ax.imshow(field, origin="lower", cmap=CMAP[v], vmin=vmin, vmax=vmax)
        ax.set_title(f"{v} ({E.UNITS[v]})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    print(f"[map] {path}")


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
