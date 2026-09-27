#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Animate freerun_monthly snapshots into a GIF/MP4 -- e.g. the shrinking ice cap
over a decadal CMIP rollout.

Reads the per-month state_*.nc files freerun_monthly.py writes, renders one
variable per frame on a FIXED colour scale (so decline is visible, not
auto-rescaled away), land masked. Filter to one calendar month (e.g. September
minimum) for a yearly-cadence animation, or animate every month.

Frames can be the ensemble MEAN (state_*.nc; smooth, but it smears the ice edge
where members disagree), the probability-matched mean (--summary pmm: the mean's
pattern with the members' sharp values) or one member (--summary member: a
single physically consistent realization with real texture). pmm/member read
members_*.nc (sic only) from a run made with freerun_monthly.py --save-members.

Run:  python experiments/animate_snapshots.py --var sic --month 9
      python experiments/animate_snapshots.py --var sic --month 9 --summary pmm
      python experiments/animate_snapshots.py --var sit --out ice.mp4 --fps 12
      python experiments/animate_snapshots.py --selfcheck        # test frame selection
"""
from __future__ import annotations

import argparse
import glob
import sys
import os

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))

# Same colormaps as predict_monthly / freerun state_map (inlined to avoid pulling
# in torch/gensim just to draw pictures).
CMAP = {"sit": "viridis", "sic": "Blues_r", "sid": "magma",
        "siu": "RdBu_r", "siv": "RdBu_r", "snt": "viridis"}
DIVERGING = {"siu", "siv"}
UNITS = {"sit": "m", "sic": "1", "sid": "1", "siu": "m/s", "siv": "m/s", "snt": "m"}


def select_indices(dates, month=None):
    """Frame order: chronological, optionally only calendar `month`. Pure."""
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    if month is not None:
        order = [i for i in order if dates[i].month == month]
    return order


def _selfcheck():
    d = [pd.Timestamp(x) for x in
         ["2016-09-15", "2015-09-15", "2015-03-15", "2016-03-15"]]
    assert [str(d[i])[:7] for i in select_indices(d)] == \
        ["2015-03", "2015-09", "2016-03", "2016-09"]
    assert [str(d[i])[:7] for i in select_indices(d, 9)] == ["2015-09", "2016-09"]
    print("[selfcheck] frame selection OK")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap-dir", default="plots/freerun/snapshots")
    ap.add_argument("--aux", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--var", default="sic", choices=list(CMAP), help="variable to animate")
    ap.add_argument("--month", type=int, default=None,
                    help="only this calendar month (1-12), e.g. 9 = September minimum")
    ap.add_argument("--out", default=None, help="output .gif (default) or .mp4")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--vmin", type=float, default=None, help="override colour scale")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--summary", default="mean", choices=["mean", "pmm", "member"],
                    help="mean: state_*.nc; pmm / member: from members_*.nc (sic only)")
    ap.add_argument("--member", type=int, default=0, help="which member for --summary member")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return

    ocean = xr.open_dataset(args.aux)["mask"].values.astype(bool)
    fields, dates = [], []
    if args.summary == "mean":
        files = sorted(glob.glob(f"{args.snap_dir}/state_*.nc"))
        if not files:
            raise SystemExit(f"no state_*.nc in {args.snap_dir} "
                             "(run freerun_monthly.py with --snapshot-every)")
        for f in files:  # one variable per file + its date
            ds = xr.open_dataset(f)["state"].isel(time=0)
            fields.append(ds.sel(var_names=args.var).values.astype(np.float32))
            dates.append(pd.Timestamp(ds["time"].values))
    else:
        if args.var != "sic":
            raise SystemExit("--summary pmm/member needs --var sic (members files hold sic only)")
        files = sorted(glob.glob(f"{args.snap_dir}/members_*.nc"))
        if not files:
            raise SystemExit(f"no members_*.nc in {args.snap_dir} "
                             "(run freerun_monthly.py with --save-members)")
        from ensemble_summaries import pmm
        for f in files:
            mem = np.nan_to_num(xr.open_dataset(f)["sic"].values).astype(np.float32)
            fields.append(pmm(mem, ocean) if args.summary == "pmm" else mem[args.member])
            ym = os.path.basename(f)[-9:-3]
            dates.append(pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01"))
    order = select_indices(dates, args.month)
    if not order:
        raise SystemExit(f"no frames match --month {args.month}")
    fields = [fields[i] for i in order]
    dates = [dates[i] for i in order]
    print(f"[anim] {len(fields)} frames  {dates[0]:%Y-%m}..{dates[-1]:%Y-%m}  var={args.var}")

    # FIXED colour scale across all frames (pooled ocean percentiles), so the
    # trend shows instead of being auto-rescaled per frame.
    pool = np.concatenate([f[ocean] for f in fields])
    if args.var in DIVERGING:
        m = args.vmax if args.vmax is not None else np.nanpercentile(np.abs(pool), 99)
        vmin, vmax = -m, m
    else:
        vmin = args.vmin if args.vmin is not None else np.nanpercentile(pool, 2)
        vmax = args.vmax if args.vmax is not None else np.nanpercentile(pool, 98)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    cmap = plt.get_cmap(CMAP[args.var]).copy()
    cmap.set_bad("0.85")  # land

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.where(ocean, fields[0], np.nan), origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{args.var} ({UNITS[args.var]})")
    title = ax.set_title("")

    def update(k):
        im.set_data(np.where(ocean, fields[k], np.nan))
        title.set_text(f"{args.var}   {dates[k]:%Y-%m}" + (f"   ({args.summary})" if args.summary != "mean" else ""))
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(fields), blit=False)

    # Default name carries the variable + the snapshot folder's model_forcing tag,
    # e.g. plots/anim_sic_monthly_pf_cmip_m09.gif
    tag = os.path.basename(os.path.normpath(args.snap_dir))
    kind = {"mean": "", "pmm": "_pmm", "member": f"_member{args.member}"}[args.summary]
    out = args.out or (f"plots/anim_{args.var}_{tag}{kind}"
                       + (f"_m{args.month:02d}" if args.month else "") + ".gif")
    out = os.path.normpath(out)
    writer = "ffmpeg" if out.endswith(".mp4") else "pillow"
    ani.save(out, writer=writer, fps=args.fps, dpi=args.dpi)
    print(f"[anim] -> {out}  ({len(fields)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
