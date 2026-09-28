#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare ways of summarising a generative ensemble into ONE sea-ice field, scored
against NSIDC-0051 observations. The ensemble mean is the minimum-RMSE field but
smears the ice edge (members disagree where the edge is, so the mean invents a wide
band of partial ice that counts toward extent). The alternatives try to keep the
mean's consensus *placement* while restoring the members' sharp *values*.

  mean       per-cell ensemble mean (current snapshots)
  member     one realization (member 0)
  median     per-cell median across members
  consensus  mean inside the >=50%-of-members ice area, open water outside
  pmm        probability-matched mean (Ebert 2001): the mean's rank pattern, the
             members' pooled value distribution
  lpmm       local PMM (Clark 2017): PMM in overlapping regional patches

Metrics (M km^2 unless noted), per month then averaged by season:
  ext / area   extent (SIC>0.15) and SIC-weighted area, as bias vs obs
  iiee         integrated ice-edge error: area where summary and obs disagree on
               ice vs no ice (lower = edge in the right place)
  miz          width of the partial-ice band (15-80% SIC) as bias vs obs
  rmse         SIC RMSE over ocean (unitless) -- the mean's home turf
  sharp        small-scale energy of SIC (edge granularity), ratio to obs

Needs members_YYYYMM.nc from `freerun_monthly.py --save-members`.

Run:  python scripts/evaluate/ensemble_summaries.py --selfcheck
      python scripts/evaluate/ensemble_summaries.py plots/freerun/snapshots/monthly_pf_era5_2018
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from frazil.diagnostics import small_scale_energy
from frazil.ensemble import THR, METHODS, pmm, lpmm, summaries, metrics

def _selfcheck():
    rng = np.random.default_rng(0)
    mask = np.ones((20, 20), bool)
    # 8 members, each a sharp 0/1 ice edge at a slightly different column
    members = np.stack([(np.arange(20)[None, :] < 10 + s) * np.ones((20, 1))
                        for s in rng.integers(-3, 4, 8)]).astype(float)
    p = pmm(members, mask)
    assert np.allclose(np.sort(p[mask]), np.sort(members[:, mask].ravel())[4::8])  # members' histogram
    m = members.mean(0)[mask]
    assert np.all(np.diff(p[mask][np.argsort(m, kind="stable")]) >= 0)          # mean's ranking
    partial = lambda f: ((f > 0.15) & (f < 0.8)).mean()
    assert partial(p) < partial(members.mean(0))                                # edge re-sharpened
    assert lpmm(members, mask, tile=8, halo=4, min_cells=10).shape == (20, 20)
    print("[selfcheck] pmm/lpmm OK")


def load_obs(obs_dir, years):
    files = [f for f in sorted(glob.glob(f"{obs_dir}/*.nc"))
             if any(str(y) in os.path.basename(f) for y in years)]
    d = xr.open_mfdataset(files, combine="by_coords")
    v = d[[n for n in ("datacube", "state") if n in d.data_vars][0]].sel(var_names="sic")
    t = pd.DatetimeIndex(v["time"].values)
    return {(tt.year, tt.month): np.nan_to_num(v.isel(time=i).values) for i, tt in enumerate(t)}


def panel(fields, obs, ocean, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("Blues_r").copy()  # one hue: dark open water -> white ice
    cmap.set_bad("#9a9a9a")                # land, neutral gray
    items = list(fields.items()) + [("NSIDC obs", obs)]
    # zoom: the box around the obs partial-ice band
    ys, xs = np.where(ocean & (obs > THR) & (obs < 0.8))
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (256, 256)
    z = 90
    zy = slice(max(cy - z, 0), min(cy + z, 512)); zx = slice(max(cx - z, 0), min(cx + z, 512))
    fig, axes = plt.subplots(2, len(items), figsize=(2.3 * len(items), 5.0), layout="constrained")
    for c, (name, f) in enumerate(items):
        show = np.where(ocean, f, np.nan)
        for r, sl in enumerate([(slice(None), slice(None)), (zy, zx)]):
            ax = axes[r, c]
            im = ax.imshow(show[sl], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)
    axes[0, 0].set_ylabel("whole Arctic", fontsize=9); axes[1, 0].set_ylabel("zoom on the ice edge", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="sea-ice concentration")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"[plot] {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snap_dir", nargs="?", help="snapshot folder holding members_YYYYMM.nc")
    ap.add_argument("--obs", default="data/obs/nsidc0051_grid")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--panel-month", default=None, help="YYYY-MM for the map panel (default: latest September)")
    ap.add_argument("--out", default=None, help="csv path (default: plots/ensemble_summaries_<tag>.csv)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return
    if not args.snap_dir:
        raise SystemExit("give a snapshot folder (from freerun_monthly.py --save-members)")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    ca = aux["cell_area"].values / 1e6
    files = sorted(glob.glob(f"{args.snap_dir}/members_*.nc"))
    if not files:
        raise SystemExit(f"no members_*.nc in {args.snap_dir}")
    ym = lambda f: (int(f[-9:-5]), int(f[-5:-3]))
    obs = load_obs(args.obs, sorted({ym(f)[0] for f in files}))

    rows, keep = [], {}
    for f in files:
        y, m = ym(f)
        if (y, m) not in obs:
            continue
        mem = np.nan_to_num(xr.open_dataset(f)["sic"].values)
        mem[:, ~ocean] = 0.0
        S = summaries(mem, ocean)
        for name, fld in S.items():
            rows.append({"year": y, "month": m, "method": name, **metrics(fld, obs[(y, m)], ocean, ca)})
        # the ensemble-average of each member's own scores, for reference
        per = [metrics(mem[i], obs[(y, m)], ocean, ca) for i in range(len(mem))]
        rows.append({"year": y, "month": m, "method": "members(avg)",
                     **{k: float(np.mean([p[k] for p in per])) for k in per[0]}})
        keep[(y, m)] = S

    df = pd.DataFrame(rows)
    tag = os.path.basename(os.path.normpath(args.snap_dir))
    out = args.out or f"plots/ensemble_summaries_{tag}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[csv] {out}  ({df[['year','month']].drop_duplicates().shape[0]} months)")

    season = {1: "JFM", 2: "JFM", 3: "JFM", 4: "AMJ", 5: "AMJ", 6: "AMJ",
              7: "JAS", 8: "JAS", 9: "JAS", 10: "OND", 11: "OND", 12: "OND"}
    df["season"] = df["month"].map(season)
    order = METHODS + ["members(avg)"]
    pd.set_option("display.width", 160)
    for s in ["JAS", "AMJ", "OND", "JFM"]:
        sub = df[df.season == s]
        if sub.empty:
            continue
        t = sub.groupby("method")[["ext", "area", "iiee", "miz", "rmse", "sharp"]].mean().reindex(order)
        print(f"\n--- {s}  (ext/area/miz = bias vs obs; iiee lower=better; sharp = ratio to obs) ---")
        print(t.round(3).to_string())

    pm = args.panel_month
    key = (int(pm[:4]), int(pm[5:7])) if pm else max((k for k in keep if k[1] == 9), default=max(keep))
    panel(keep[key], obs[key], ocean, f"Ensemble summaries vs NSIDC, {key[0]}-{key[1]:02d} ({tag})",
          f"plots/ensemble_summaries_{tag}_{key[0]}{key[1]:02d}.png")


if __name__ == "__main__":
    main()
