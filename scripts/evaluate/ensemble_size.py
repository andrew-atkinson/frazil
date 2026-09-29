#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
How much does averaging help, as a function of ensemble size?

Members of a free-run evolve independently, so any k of the saved members is a
fair stand-in for a k-member run. For k = 1..n this draws up to --max-subsets
k-member subsets per month, summarises each with the plain mean and with the
probability-matched mean (PMM), and scores them against NSIDC-0051:
  rmse   sic RMSE over ocean              iiee  ice-edge error (M km^2)
  ext    extent bias vs obs (M km^2)      miz   partial-ice (15-80%) band bias
  sharp  small-scale energy, ratio to obs (1 = as sharp as observed)
Then fits the ensemble-averaging law  rmse^2 = a + b/k  for the mean: a is the
error no amount of averaging removes, b the member-to-member noise.

Needs members_YYYYMM.nc from freerun_monthly.py --save-members.

Run:  python scripts/evaluate/ensemble_size.py --selfcheck
      python scripts/evaluate/ensemble_size.py plots/freerun/snapshots/monthly_pf_qm_era5_2018
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
import xarray as xr

from frazil.diagnostics import small_scale_energy
from frazil.ensemble import pmm
from frazil import paths

SEASON = {1: "JFM", 2: "JFM", 3: "JFM", 7: "JAS", 8: "JAS", 9: "JAS"}


def score(f, o, ocean, ca):
    km = lambda b: float((ca * b).sum()) / 1e6
    fi, oi = (f > 0.15) & ocean, (o > 0.15) & ocean
    return dict(rmse=float(np.sqrt(((f - o)[ocean] ** 2).mean())), iiee=km(fi ^ oi),
                ext=km(fi) - km(oi), miz=km(fi & (f < 0.8)) - km(oi & (o < 0.8)),
                sharp=small_scale_energy(f, ocean) / small_scale_energy(o, ocean))


def subsets(n, k, max_subsets, rng):
    combos = list(itertools.combinations(range(n), k))
    if len(combos) > max_subsets:
        combos = [combos[i] for i in rng.choice(len(combos), max_subsets, replace=False)]
    return combos


def fit_averaging_law(k, rmse):
    """rmse^2 = a + b/k by least squares -> (a, b)."""
    b, a = np.polyfit(1.0 / np.asarray(k, float), np.asarray(rmse) ** 2, 1)
    return a, b


def _selfcheck():
    rng = np.random.default_rng(1)
    ocean = np.ones((64, 64), bool); ca = np.ones((64, 64))
    truth = np.clip(rng.random((64, 64)), 0, 1)
    members = truth + 0.1 * rng.standard_normal((8, 64, 64))   # noise sd 0.1 -> b ~ 0.01
    ks, rm = [], []
    for k in range(1, 9):
        ks.append(k); rm.append(score(members[:k].mean(0), truth, ocean, ca)["rmse"])
    a, b = fit_averaging_law(ks, rm)
    assert rm[-1] < rm[0] and abs(b - 0.01) < 0.003 and a < 0.002, (a, b)
    assert subsets(8, 4, 10, rng).__len__() == 10 and subsets(8, 8, 10, rng) == [tuple(range(8))]
    print(f"[selfcheck] averaging law recovered: a={a:.4f} b={b:.4f} (expected ~0, 0.01)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snap_dir", nargs="?", help="snapshot folder holding members_YYYYMM.nc")
    ap.add_argument("--months", type=int, nargs="+", default=[1, 2, 3, 7, 8, 9])
    ap.add_argument("--max-subsets", type=int, default=10, help="subsets drawn per (month, k)")
    ap.add_argument("--obs", default=paths.NSIDC0051_GRID)
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--out", default=None, help="per-subset csv (default: plots/ensemble_size_<tag>.csv)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return
    if not args.snap_dir:
        raise SystemExit("give a snapshot folder (from freerun_monthly.py --save-members)")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    ca = aux["cell_area"].values / 1e6
    d = xr.open_mfdataset(sorted(glob.glob(f"{args.obs}/*.nc")), combine="by_coords")
    v = d[[n for n in ("datacube", "state") if n in d.data_vars][0]].sel(var_names="sic")
    t = pd.DatetimeIndex(v["time"].values)
    obs = {(tt.year, tt.month): i for i, tt in enumerate(t)}

    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(f"{args.snap_dir}/members_*.nc")):
        y, m = int(f[-9:-5]), int(f[-5:-3])
        if m not in args.months or (y, m) not in obs:
            continue
        mem = np.nan_to_num(xr.open_dataset(f)["sic"].values)
        mem[:, ~ocean] = 0.0
        o = np.nan_to_num(v.isel(time=obs[(y, m)]).values)
        season = SEASON.get(m, f"m{m:02d}")
        for k in range(1, len(mem) + 1):
            for c in subsets(len(mem), k, args.max_subsets, rng):
                sub = mem[list(c)]
                rows.append(dict(year=y, month=m, season=season, k=k, method="mean", **score(sub.mean(0), o, ocean, ca)))
                if k > 1:
                    rows.append(dict(year=y, month=m, season=season, k=k, method="pmm", **score(pmm(sub, ocean), o, ocean, ca)))
    if not rows:
        raise SystemExit(f"no members_*.nc for months {args.months} with observations in {args.snap_dir}")
    df = pd.DataFrame(rows)
    tag = os.path.basename(os.path.normpath(args.snap_dir))
    out = args.out or f"plots/ensemble_size_{tag}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 160)
    for s in sorted(df.season.unique()):
        for meth in ("mean", "pmm"):
            tab = df[(df.season == s) & (df.method == meth)].groupby("k")[["rmse", "iiee", "ext", "miz", "sharp"]].mean()
            print(f"\n=== {s}: {meth} of k members (rmse unitless; iiee/ext/miz M km^2 vs obs; sharp: obs = 1)")
            print(tab.round(3).to_string())
        r = df[(df.season == s) & (df.method == "mean")].groupby("k").rmse.mean()
        a, b = fit_averaging_law(r.index, r.values)
        rm = lambda k: np.sqrt(a + b / k)
        gain = lambda k: (rm(1) - rm(k)) / (rm(1) - np.sqrt(a))
        print(f"  fit: rmse^2 = {a:.5f} + {b:.5f}/k  |  noise share of 1-member error {b / (a + b):.0%}  |  "
              f"rmse 1/4/8/16/inf = {rm(1):.4f}/{rm(4):.4f}/{rm(8):.4f}/{rm(16):.4f}/{np.sqrt(a):.4f}  |  "
              f"share of the possible gain: k=4 {gain(4):.0%}, k=8 {gain(8):.0%}")
    print(f"\n[csv] {out}  ({df[['year', 'month']].drop_duplicates().shape[0]} months)")


if __name__ == "__main__":
    main()
