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


def pick_checkpoints(ckpt_dir, n):
    # distinct steps (ignore -v1/-v2 dupes -> keep the first per step)
    by_step = {}
    for p in glob.glob(f"{ckpt_dir}/step_*.ckpt"):
        by_step.setdefault(step_of(p), p)
    steps = sorted(by_step)
    if len(steps) <= n:
        chosen = steps
    else:  # evenly spaced across the range, always include first + last
        idx = np.linspace(0, len(steps) - 1, n).round().astype(int)
        chosen = [steps[i] for i in sorted(set(idx))]
    return [(s, by_step[s]) for s in chosen]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="data/models/monthly")
    ap.add_argument("--config", default="config_forecast_monthly.yaml")
    ap.add_argument("--train-config", default="config_train_monthly_mac.yaml")
    ap.add_argument("--datacube", default="data/train_data/monthly_datacube")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--n-ckpts", type=int, default=10)
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--out-csv", default="experiments/skill_vs_step.csv")
    ap.add_argument("--out-png", default="experiments/skill_vs_step.png")
    args = ap.parse_args(argv)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cube = E.load_val_cube(args.datacube, 2015)
    aux = xr.open_dataset(args.aux)
    ckpts = pick_checkpoints(args.ckpt_dir, args.n_ckpts)
    print(f"[sweep] {len(ckpts)} checkpoints, {args.max_pairs} pairs each, "
          f"steps {ckpts[0][0]}..{ckpts[-1][0]}")

    rows = []
    for step, path in ckpts:
        torch.manual_seed(42)
        model = E.load_model(path, args.config, device, args.train_config)
        rmse = E.eval_skill(model, cube, aux, device, args.max_pairs, progress=False)
        skill = {v: (1 - rm / rp if rp > 0 else float("nan"))
                 for v, (rm, rp) in rmse.items()}
        skill["mean"] = float(np.mean([skill[v] for v in E.STATES]))
        rows.append({"step": step, **skill})
        print(f"  step {step:7d}: mean skill {skill['mean']:+.1%}  "
              + "  ".join(f"{v}={skill[v]:+.0%}" for v in E.STATES), flush=True)

    fields = ["step", "mean"] + E.STATES
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {args.out_csv}")

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
