#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Skill-vs-training-step curve, computed post-hoc on saved step checkpoints.

Training saves a checkpoint every 500 steps, so we don't need to slow training
with periodic eval -- just evaluate a spread of the existing step_*.ckpt files.
For each, runs eval_monthly.eval_skill (one-month-ahead RMSE vs persistence over
ocean) on a subset of validation pairs and records the skill per variable.

Run:  python experiments/sweep_skill_monthly.py --n-ckpts 10 --max-pairs 12
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import xarray as xr

import eval_monthly as E  # same dir; reuse load_model + eval_skill


def step_of(path):
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def pick_checkpoints(ckpt_dir, interval):
    # distinct steps (ignore -v1/-v2 dupes -> keep the first per step)
    by_step = {}
    for p in glob.glob(f"{ckpt_dir}/step_*.ckpt"):
        by_step.setdefault(step_of(p), p)
    steps = sorted(by_step)
    if not steps:
        return []
    # Snap a fixed step grid (0, interval, 2*interval, ...) to the nearest
    # available checkpoint, plus the latest. Stable as training grows, so reruns
    # reuse cached steps and only the new high end gets evaluated.
    targets = list(range(0, steps[-1] + 1, interval))
    chosen = sorted({min(steps, key=lambda s: abs(s - t)) for t in targets}
                    | {steps[-1]})
    return [(s, by_step[s]) for s in chosen]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="data/models/monthly")
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--interval", type=int, default=25000,
                    help="eval a checkpoint near every N steps (stable grid)")
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--out-csv", default="plots/skill_vs_step.csv")
    ap.add_argument("--out-png", default="plots/skill_vs_step.png")
    args = ap.parse_args(argv)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cube = E.load_val_cube(args.datacube, 2015)
    aux = xr.open_dataset(args.aux)
    ckpts = pick_checkpoints(args.ckpt_dir, args.interval)
    print(f"[sweep] {len(ckpts)} checkpoints, {args.max_pairs} pairs each, "
          f"steps {ckpts[0][0]}..{ckpts[-1][0]}")

    fields = ["step", "mean"] + E.STATES
    # Reuse previously computed steps so reruns only evaluate new checkpoints.
    cached = {}
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as fh:
            for r in csv.DictReader(fh):
                cached[int(r["step"])] = {"step": int(r["step"]),
                                          **{k: float(r[k]) for k in fields[1:]}}
        print(f"[cache] {len(cached)} steps loaded from {args.out_csv}")

    rows = []
    for step, path in ckpts:
        if step in cached:
            rows.append(cached[step])
            print(f"  step {step:7d}: cached (mean {cached[step]['mean']:+.1%})")
            continue
        torch.manual_seed(42)
        model = E.load_model(path, args.config, device, args.train_config)
        rmse = E.eval_skill(model, cube, aux, device, args.max_pairs, progress=False)
        skill = {v: (1 - rm / rp if rp > 0 else float("nan"))
                 for v, (rm, rp) in rmse.items()}
        skill["mean"] = float(np.mean([skill[v] for v in E.STATES]))
        rows.append({"step": step, **skill})
        print(f"  step {step:7d}: mean skill {skill['mean']:+.1%}  "
              + "  ".join(f"{v}={skill[v]:+.0%}" for v in E.STATES), flush=True)

    # Keep any cached steps outside the current selection too, so the CSV grows.
    seen = {r["step"] for r in rows}
    rows += [r for s, r in cached.items() if s not in seen]
    rows.sort(key=lambda r: r["step"])
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {args.out_csv} ({len(rows)} steps)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [r["step"] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        for v in E.STATES:
            ax.plot(steps, [r[v] for r in rows], lw=1, alpha=0.6, label=v)
        ax.plot(steps, [r["mean"] for r in rows], lw=2.5, color="black", label="mean")
        ax.axhline(0, color="0.6", ls=":", lw=1)  # 0 = ties persistence
        ax.set_xlabel("training step"); ax.set_ylabel("skill vs persistence")
        ax.set_title("Monthly GenSIM — one-month-ahead skill over training")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
        print(f"[plot] {args.out_png}")
    except Exception as e:
        print("[plot] skipped:", e)


if __name__ == "__main__":
    main()
