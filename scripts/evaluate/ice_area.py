#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph total sea-ice area (or extent) vs time from one or more state cubes.

Works on any cube that carries the sea-ice states: the neXtSIM truth datacube
(monthly_datacube), or a freerun snapshots/ folder (model output). Overlays
several as separate lines, so you can put truth (1995-2018) next to a projection
on one axis. NOTE: cmip_datacube is forcing-only (no ice states) -- area can't be
computed from it.

  extent = area of cells with SIC > threshold  (the standard sea-ice-extent metric)
  area   = integral of SIC * cell_area          (concentration-weighted)

Run:  python scripts/evaluate/ice_area.py data/train_data/monthly_datacube
      python scripts/evaluate/ice_area.py data/train_data/monthly_datacube plots/freerun/snapshots
      python scripts/evaluate/ice_area.py --selfcheck
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from frazil.diagnostics import ice_area_km2  # noqa: F401
from frazil.io import load_nsidc, load_sic
from frazil import paths


def _selfcheck():
    ocean = np.array([[1, 1], [1, 0]], dtype=bool)
    ca = np.array([[10.0, 10.0], [10.0, 10.0]])          # km^2 per cell
    sic = np.array([[[0.5, 0.1], [0.9, 1.0]]])            # (1,2,2); land cell = 1.0 ignored
    # extent (>0.15): cells (0,0) and (1,0) qualify over ocean -> 10+10 = 20
    assert ice_area_km2(sic, ocean, ca, "extent", 0.15)[0] == 20.0
    # area: 0.5*10 + 0.1*10 + 0.9*10 (ocean only) = 15.0
    assert abs(ice_area_km2(sic, ocean, ca, "area", 0.15)[0] - 15.0) < 1e-9
    print("[selfcheck] ice_area_km2 OK")
    import tempfile
    p = tempfile.mktemp(suffix=".csv")
    with open(p, "w") as fh:
        fh.write("year, mo,source, region, extent,   area\n"
                 "2023,  9, X, N, 4.38, 2.83\n2024,  9, X, N, 4.35, 2.91\n"
                 "1988,  9, X, N, -9999, -9999\n")  # missing year skipped
    d = load_nsidc(p, 9)
    assert d[2024] == 4.35e6 and d[2023] == 4.38e6 and 1988 not in d, d
    da = load_nsidc(p, 9, "area")               # must read the area column, not extent
    assert da[2024] == 2.91e6 and da[2023] == 2.83e6, da
    print("[selfcheck] load_nsidc OK")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*",
                    help="cube dirs (datacube or snapshots); each becomes a line")
    ap.add_argument("--labels", default=None, help="comma-sep labels (default: basenames)")
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--metric", choices=["extent", "area"], default="extent")
    ap.add_argument("--threshold", type=float, default=0.15, help="SIC extent threshold")
    ap.add_argument("--month", type=int, default=None,
                    help="keep only this calendar month (e.g. 9 = Sept minimum)")
    ap.add_argument("--out", default="plots/ice_area.png")
    ap.add_argument("--csv", action="store_true", help="also write <out>.csv")
    ap.add_argument("--nsidc", default=None,
                    help="NSIDC extent CSV or dir to overlay as observations "
                         "(needs --month; compares each model line to obs)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return
    if not args.sources:
        raise SystemExit("give at least one cube dir (e.g. data/train_data/monthly_datacube)")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    cell_area_km2 = aux["cell_area"].values / 1e6

    labels = (args.labels.split(",") if args.labels
              else [os.path.basename(os.path.normpath(s)) for s in args.sources])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))

    plt.grid(color='gray', linestyle='--', linewidth=0.25)

    series = {}
    for src, lab in zip(args.sources, labels):
        sic, times = load_sic(src)
        if args.month is not None:
            keep = times.month == args.month
            sic, times = sic[keep], times[keep]
        a = ice_area_km2(sic, ocean, cell_area_km2, args.metric, args.threshold)
        ax.plot(times, a, "-o", ms=2.5, lw=1.2, label=f"{lab} ({len(a)})")
        series[lab] = (times, a)
        print(f"[ice] {lab}: {len(a)} steps, {times[0]:%Y-%m}..{times[-1]:%Y-%m}, "
              f"{args.metric} {a.min():.2e}..{a.max():.2e} km²")

    # NSIDC observed extent overlay + model-minus-obs offset (the correction factor).
    if args.nsidc:
        if args.month is None:
            raise SystemExit("--nsidc needs --month (NSIDC files are per calendar month)")
        obs = load_nsidc(args.nsidc, args.month, args.metric)
        oy = sorted(obs)
        odates = [pd.Timestamp(f"{y}-{args.month:02d}-15") for y in oy]
        ax.plot(odates, [obs[y] for y in oy], "k--s", ms=4, lw=1.6,
                label=f"NSIDC {args.metric} ({len(oy)})")
        print(f"[nsidc] {len(oy)} obs years {oy[0]}-{oy[-1]} (pan-Arctic; model is grid-limited)")
        for lab, (t, a) in series.items():
            by = {int(pd.Timestamp(ti).year): ai for ti, ai in zip(t, a)}
            common = sorted(set(by) & set(obs))
            if common:
                off = float(np.mean([by[y] - obs[y] for y in common]))
                print(f"[offset] {lab} - NSIDC  {common[0]}-{common[-1]} ({len(common)}yr): "
                      f"{off/1e6:+.2f} M km²  (subtract this to anchor the projection)")

    mlabel = f"{'extent (SIC>%.2f)' % args.threshold if args.metric=='extent' else 'area (SIC-weighted)'}"
    title = f"Sea-ice {mlabel}" + (f" -- month {args.month:02d} only" if args.month else "")
    ax.set_ylabel("km²"); ax.set_xlabel("date"); ax.set_title(title)
    ax.legend(fontsize=9); ax.margins(x=0.01)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"[plot] {args.out}")

    if args.csv:
        import csv
        with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["label", "date", f"{args.metric}_km2"])
            for lab, (t, a) in series.items():
                for ti, ai in zip(t, a):
                    w.writerow([lab, f"{ti:%Y-%m}", f"{ai:.1f}"])


if __name__ == "__main__":
    main()
