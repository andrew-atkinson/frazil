#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free-running long-rollout drift experiment for GenSIM.

GenSIM is a short-range, atmosphere-driven sea-ice forecast emulator: every
12-hour step it consumes prescribed atmospheric forcing and predicts the next
sea-ice state auto-regressively. This script pushes it far past its intended
horizon to visualise *how* and *how fast* the free-running forecast drifts.

Because the demo dataset only contains 3.5 days (8 timesteps) of forcing, the
atmosphere is extended past that window by cyclically repeating the available
forcing pairs ("groundhog-day" atmosphere). Any drift is therefore attributable
to the model's auto-regressive rollout, not to a degenerate atmosphere.

There is no ground truth beyond 3.5 days, so drift is measured as *self
consistency*: domain-mean state, total ice area, physical-bound health, and a
small-scale-energy (sharpness) proxy that exposes the classic auto-regressive
failure mode of progressive spatial smoothing.

The run checkpoints after every step (diagnostics appended to a CSV, full state
snapshots dumped every ``--snapshot-every`` steps), so it can be interrupted and
resumed, and plotted at any time with ``--plot-only``.

Examples
--------
Smoke test (2 steps)::

    python experiments/drift_rollout.py --n-steps 2 --out-dir data/drift_smoke

Full first run (120 steps ~ 60 simulated days)::

    python experiments/drift_rollout.py --n-steps 120

Re-draw the figure from an existing run::

    python experiments/drift_rollout.py --plot-only --out-dir data/drift_run
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr


# --------------------------------------------------------------------------- #
# Device / model setup
# --------------------------------------------------------------------------- #
def select_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device, model_path, config_name="config_forecast.yaml"):
    """Instantiate GenSIM, load EMA weights, move to device. Mirrors the demo."""
    import torch
    from hydra import compose, initialize
    from hydra.utils import instantiate
    from safetensors import safe_open
    import gensim
    import gensim.network

    # Flash attention is NVIDIA-only; keep it off elsewhere (torch SDPA fallback).
    gensim.network.USE_FLASH_ATTN = (
        gensim.network.USE_FLASH_ATTN and device.type == "cuda"
    )

    with initialize(version_base=None, config_path="..", job_name="drift"):
        cfg = compose(config_name=config_name, overrides=[])
    cfg = cfg["surrogate"]
    model = instantiate(cfg)

    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        sd = model.network.state_dict()
        sd.update({k: f.get_tensor(k) for k in keys})
        model.network.load_state_dict(sd)

    model = model.to(device)
    # torch.compile is unreliable on mps; only compile on cuda.
    model.set_inference_model(compile_model=device.type == "cuda")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
STATE_VARS = ["sit", "sic", "sid", "siu", "siv", "snt"]
FORCING_VARS = ["tus", "huss", "uas", "vas"]
DEGREE_VARS = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
SIC_THRESHOLD = 0.15  # ice-edge threshold for extent/area


def load_inputs(n_ens, aux_path, demo_path):
    """Load auxiliary + demo data and build the numpy inputs used for rollout."""
    ds_aux = xr.open_dataset(aux_path)
    mesh = ds_aux[["x_coord", "y_coord"]].to_dataarray("coord_names").values / 1000.0
    mesh = mesh[None].repeat(n_ens, axis=0)
    mask = ds_aux["mask"].values[None, None].repeat(n_ens, axis=0)

    ds_demo = xr.open_dataset(demo_path)["datacube"].fillna(0)
    ds_initial = ds_demo.isel(time=0).sel(var_names=STATE_VARS)
    initial = ds_initial.values[None].repeat(n_ens, axis=0)

    forcings = ds_demo.sel(var_names=FORCING_VARS).values[None].repeat(n_ens, axis=0)
    degree = ds_demo.sel(var_names=DEGREE_VARS).values[None].repeat(n_ens, axis=0)

    # Static fields for diagnostics (single member, ocean grid).
    ocean = ds_aux["mask"].values.astype(bool)
    cell_area = ds_aux["cell_area"].values
    dx = float(ds_aux["dx"].values.mean()) if "dx" in ds_aux else 12500.0

    return dict(
        mesh=mesh, mask=mask, initial=initial, forcings=forcings, degree=degree,
        ocean=ocean, cell_area=cell_area, dx=dx,
        n_native_pairs=forcings.shape[1] - 1, ds_aux=ds_aux, ds_demo=ds_demo,
    )


def forcing_index(step, n_native_pairs, strategy):
    """Return the start index of the (t, t+12h) forcing pair for a given step."""
    if strategy == "cyclic":
        return step % n_native_pairs
    if strategy == "hold":
        return min(step, n_native_pairs - 1)
    raise ValueError(f"unknown forcing strategy: {strategy}")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def small_scale_energy(field, ocean):
    """Mean squared gradient magnitude over ocean cells (sharpness proxy).

    Progressive auto-regressive smoothing shows up as this value decaying
    toward zero, so it is a sensitive indicator of spectral collapse.
    """
    gy, gx = np.gradient(field.astype(np.float64))
    grad2 = gx ** 2 + gy ** 2
    return float(grad2[ocean].mean())


def step_diagnostics(state, ocean, cell_area):
    """Compute scalar diagnostics for one ensemble-mean state (6, Y, X)."""
    sit, sic = state[0], state[1]
    siu, siv = state[3], state[4]
    ice = sic > SIC_THRESHOLD
    total_area = float((cell_area * ice * ocean).sum())  # m^2
    speed = np.sqrt(siu ** 2 + siv ** 2)
    return {
        "mean_sit": float(sit[ocean].mean()),
        "mean_sic": float(sic[ocean].mean()),
        "max_sit": float(sit[ocean].max()),
        "ice_area_km2": total_area / 1e6,
        "mean_speed": float(speed[ocean].mean()),
        "frac_sic_saturated": float((sic[ocean] >= 0.999).mean()),
        "frac_sic_zero": float((sic[ocean] <= 1e-4).mean()),
        "frac_sit_zero": float((sit[ocean] <= 1e-4).mean()),
        "sharpness_sit": small_scale_energy(sit, ocean),
        "sharpness_sic": small_scale_energy(sic, ocean),
        "n_nonfinite": int((~np.isfinite(state)).sum()),
    }


CSV_FIELDS = [
    "step", "sim_days", "forced",
    "mean_sit", "mean_sic", "max_sit", "ice_area_km2", "mean_speed",
    "frac_sic_saturated", "frac_sic_zero", "frac_sit_zero",
    "sharpness_sit", "sharpness_sic", "n_nonfinite", "wall_s",
]


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def run_rollout(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snap_dir = out / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    csv_path = out / "diagnostics.csv"

    device = select_device(args.device)
    torch.manual_seed(args.seed)
    print(f"[drift] device={device}  n_steps={args.n_steps}  n_ens={args.n_ens}  "
          f"forcing={args.forcing}", flush=True)

    data = load_inputs(args.n_ens, args.aux_path, args.demo_path)
    ocean, cell_area = data["ocean"], data["cell_area"]
    n_pairs = data["n_native_pairs"]

    model = load_model(device, args.model_path)
    print("[drift] model loaded", flush=True)

    F32 = torch.float32
    mesh = torch.as_tensor(data["mesh"], device=device, dtype=F32)
    mask = torch.as_tensor(data["mask"], device=device, dtype=F32)
    resolution = torch.full((args.n_ens, 1), 12.5, device=device, dtype=F32)
    curr = torch.as_tensor(data["initial"], device=device, dtype=F32)

    # Truth anchor for the native (real-forcing) window, ensemble-independent.
    truth = data["ds_demo"].sel(var_names=STATE_VARS).values  # (T, 6, Y, X)

    # Fresh CSV with header (append mode for resumability handled by --plot-only).
    with open(csv_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    # Log the initial condition as step 0.
    init_mean = data["initial"].mean(axis=0)  # ensemble mean -> (6, Y, X)
    _append_row(csv_path, 0, 0.0, True, init_mean, ocean, cell_area, 0.0)

    interrupted = {"flag": False}

    def _handler(signum, frame):
        interrupted["flag"] = True
        print("\n[drift] interrupt received — finishing current step then stopping",
              flush=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    forcings_np = data["forcings"]
    degree_np = data["degree"]

    for step in range(args.n_steps):
        t0 = time.time()
        fstart = forcing_index(step, n_pairs, args.forcing)
        forced = step < n_pairs  # within real-forcing window?

        cf = torch.as_tensor(
            forcings_np[:, fstart:fstart + 2], device=device, dtype=F32
        )
        cdd = torch.as_tensor(degree_np[:, fstart], device=device, dtype=F32)

        with torch.no_grad():
            curr = model(
                curr[:, None], cf,
                resolution=resolution, mesh=mesh, mask=mask, degree_days=cdd,
            )

        state_mean = curr.detach().to("cpu", torch.float32).numpy().mean(axis=0)
        wall = time.time() - t0
        sim_days = (step + 1) * 0.5  # 12h steps
        _append_row(csv_path, step + 1, sim_days, forced,
                    state_mean, ocean, cell_area, wall)

        diag = step_diagnostics(state_mean, ocean, cell_area)
        print(f"[drift] step {step+1:4d}/{args.n_steps}  {sim_days:6.1f} d  "
              f"sit={diag['mean_sit']:.4f}  sic={diag['mean_sic']:.4f}  "
              f"area={diag['ice_area_km2']:.3e} km2  "
              f"sharp_sit={diag['sharpness_sit']:.3e}  "
              f"nan={diag['n_nonfinite']}  ({wall:.1f}s)", flush=True)

        if (step + 1) % args.snapshot_every == 0 or (step + 1) == args.n_steps:
            _save_snapshot(snap_dir, step + 1, sim_days, state_mean, data["ds_aux"])

        if diag["n_nonfinite"] > 0:
            print("[drift] non-finite values encountered — stopping early", flush=True)
            break
        if interrupted["flag"]:
            break

    # Save the truth anchor once for plotting.
    np.savez(out / "truth_anchor.npz",
             truth=_truth_diag_series(truth, ocean, cell_area))
    print(f"[drift] done. diagnostics -> {csv_path}", flush=True)
    make_plot(args.out_dir)


def _append_row(csv_path, step, sim_days, forced, state_mean, ocean, cell_area, wall):
    diag = step_diagnostics(state_mean, ocean, cell_area)
    row = {"step": step, "sim_days": sim_days, "forced": int(forced),
           "wall_s": round(wall, 2), **diag}
    with open(csv_path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)


def _save_snapshot(snap_dir, step, sim_days, state_mean, ds_aux):
    da = xr.DataArray(
        state_mean[None],
        dims=("time", "var_names", "y", "x"),
        coords={"var_names": STATE_VARS, "time": [sim_days]},
        name="state",
    )
    da.attrs["step"] = step
    da.to_netcdf(snap_dir / f"snapshot_step{step:05d}.nc")


def _truth_diag_series(truth, ocean, cell_area):
    """Diagnostics for each real-forcing truth timestep (for anchoring plots)."""
    rows = []
    for t in range(truth.shape[0]):
        d = step_diagnostics(truth[t], ocean, cell_area)
        d["sim_days"] = t * 0.5
        rows.append(d)
    return np.array([(r["sim_days"], r["mean_sit"], r["mean_sic"],
                      r["ice_area_km2"], r["sharpness_sit"]) for r in rows])


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_plot(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    rows = list(csv.DictReader(open(out / "diagnostics.csv")))
    if not rows:
        print("[drift] no diagnostics to plot")
        return
    days = np.array([float(r["sim_days"]) for r in rows])
    forced = np.array([int(r["forced"]) for r in rows], dtype=bool)

    def col(name):
        return np.array([float(r[name]) for r in rows])

    anchor = None
    apath = out / "truth_anchor.npz"
    if apath.exists():
        anchor = np.load(apath)["truth"]  # cols: days, sit, sic, area, sharp

    forced_end = days[forced].max() if forced.any() else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GenSIM free-running rollout — drift diagnostics", fontsize=14)

    def shade(ax):
        ax.axvspan(days.min(), forced_end, color="0.85", zorder=0)
        ax.set_xlabel("Simulated time (days)")

    # Panel A: mean thickness & concentration
    ax = axes[0, 0]
    l1, = ax.plot(days, col("mean_sit"), c="#E65007", label="mean SIT (m)")
    ax.set_ylabel("mean thickness (m)", color="#E65007")
    ax2 = ax.twinx()
    l2, = ax2.plot(days, col("mean_sic"), c="#1f77b4", label="mean SIC")
    ax2.set_ylabel("mean concentration", color="#1f77b4")
    if anchor is not None:
        ax.plot(anchor[:, 0], anchor[:, 1], "o", c="#E65007", ms=4, label="truth SIT")
        ax2.plot(anchor[:, 0], anchor[:, 2], "s", c="#1f77b4", ms=4, label="truth SIC")
    shade(ax)
    ax.set_title("Domain-mean state")
    ax.legend(handles=[l1, l2], loc="best", fontsize=8)

    # Panel B: total ice area
    ax = axes[0, 1]
    ax.plot(days, col("ice_area_km2"), c="black")
    if anchor is not None:
        ax.plot(anchor[:, 0], anchor[:, 3], "o", c="black", ms=4)
    shade(ax)
    ax.set_ylabel("ice area (km²)")
    ax.set_title(f"Total ice area (SIC > {SIC_THRESHOLD})")

    # Panel C: sharpness / small-scale energy (spectral-collapse indicator)
    ax = axes[1, 0]
    s0 = col("sharpness_sit")[0]
    ax.plot(days, col("sharpness_sit") / s0, c="#2ca02c", label="SIT")
    ax.plot(days, col("sharpness_sic") / col("sharpness_sic")[0],
            c="#9467bd", label="SIC")
    shade(ax)
    ax.set_ylabel("small-scale energy (rel. to step 0)")
    ax.set_title("Sharpness proxy — decay = smoothing/collapse")
    ax.axhline(1.0, c="0.6", lw=0.8, ls=":")
    ax.legend(fontsize=8)

    # Panel D: physical-bound health
    ax = axes[1, 1]
    ax.plot(days, col("frac_sic_saturated"), label="SIC saturated (=1)")
    ax.plot(days, col("frac_sic_zero"), label="SIC = 0")
    ax.plot(days, col("frac_sit_zero"), label="SIT = 0")
    shade(ax)
    ax.set_ylabel("fraction of ocean cells")
    ax.set_title("Physical-bound health")
    ax.legend(fontsize=8)

    for a in axes.flat:
        a.margins(x=0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = out / "drift_diagnostics.png"
    fig.savefig(png, dpi=130)
    print(f"[drift] figure -> {png}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-steps", type=int, default=120,
                   help="number of 12h forecast steps (default 120 ~ 60 days)")
    p.add_argument("--n-ens", type=int, default=1, help="ensemble members")
    p.add_argument("--forcing", choices=["cyclic", "hold"], default="cyclic",
                   help="how to extend forcing past the 3.5-day window")
    p.add_argument("--device", default="auto", help="auto|mps|cpu|cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot-every", type=int, default=10,
                   help="dump full state fields every N steps")
    p.add_argument("--out-dir", default="data/drift_run")
    p.add_argument("--model-path", default="data/models/model_weights_ema.safetensors")
    p.add_argument("--aux-path", default="data/auxiliary/ds_auxiliary.nc")
    p.add_argument("--demo-path", default="data/auxiliary/ds_demo.nc")
    p.add_argument("--plot-only", action="store_true",
                   help="only (re)draw the figure from an existing out-dir")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.plot_only:
        make_plot(args.out_dir)
        return
    run_rollout(args)


if __name__ == "__main__":
    main()
