"""One-month steps for the loose-ice tests: where a step starts, and how it's sampled.
Saves, per predicted month, each member's starting and predicted concentration, for compare.py.
  --start truth      from the target's previous month (the check in summer_melt)
  --start generated  from the model's own prediction of the previous month, each member from
                     its own start: the condition 80% of the fine-tune trained on
  --n-steps N        sampler steps (trained models use 20)
Each pair draws from eval_monthly's seed (42 + index), so --start truth with 20 steps
reproduces its saved predictions. Resumable: finished months are skipped.

Run from the repo root:
  python experiments/2026-11-05_loose_ice/step_test.py --ckpt C --datacube D --out DIR [--start generated] [--n-steps 40]
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import xarray as xr

from frazil import DEGREE, STATES, paths
from frazil.models import load_model, load_val_cube
from gensim.sampler import get_schedule

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--datacube", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", choices=["truth", "generated"], default="truth")
ap.add_argument("--n-steps", type=int, default=20)
ap.add_argument("--months", type=int, nargs="+", default=[8, 9])
ap.add_argument("--val-start-year", type=int, default=2015)
ap.add_argument("--last-year", type=int, default=2018)
ap.add_argument("--n-ens", type=int, default=8)
args = ap.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = load_model(args.ckpt, paths.CONFIG_FORECAST, device, paths.CONFIG_TRAIN)
if args.n_steps != model.sampler.n_steps:
    model.sampler.n_steps, model.sampler.schedule = args.n_steps, get_schedule(args.n_steps, 3, 0.5)   # as config_forecast_monthly
cube = load_val_cube(args.datacube, args.val_start_year)
aux = xr.open_dataset(paths.AUX)
os.makedirs(args.out, exist_ok=True)
SIC = STATES.index("sic")
rep = lambda x: x.repeat(args.n_ens, *([1] * (x.ndim - 1)))
tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
arr = lambda t, names: np.nan_to_num(cube.isel(time=t).sel(var_names=names).values)
mesh = rep(tensor((aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000)[None]))
mask = rep(tensor(aux["mask"].values[None, None]))
resolution = torch.full((args.n_ens, 1), 12.5, device=device, dtype=torch.float32)


def step(states, t):
    """states (n_ens, 6, H, W) at index t -> members at t + 1, with eval_monthly's seed for that pair."""
    torch.manual_seed(42 + t)
    forc = rep(tensor(np.stack([arr(t, model.forcing_names), arr(t + 1, model.forcing_names)])[None]))
    dd = rep(tensor(arr(t, DEGREE)[None]))
    with torch.no_grad():
        return model(states[:, None], forc, resolution=resolution, mesh=mesh, mask=mask, degree_days=dd)


times = cube.time.dt
for t in range(cube.sizes["time"] - 1):
    y, m = int(times.year.values[t + 1]), int(times.month.values[t + 1])
    dst = os.path.join(args.out, f"step_{y}{m:02d}.npz")
    if m not in args.months or y > args.last_year or os.path.exists(dst):
        continue
    if args.start == "truth":
        start = rep(tensor(arr(t, STATES)[None]))
    else:
        start = step(rep(tensor(arr(t - 1, STATES)[None])), t - 1)
    next_ = step(start, t)
    np.savez_compressed(dst + ".part.npz", start=start[:, SIC].cpu().numpy().astype(np.float32), next=next_[:, SIC].cpu().numpy().astype(np.float32))
    os.replace(dst + ".part.npz", dst)
    print(f"  {y}-{m:02d} -> {dst}", flush=True)
