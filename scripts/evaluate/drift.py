#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drift: how a model's error grows as it runs on its own predictions, with the weather
averaged out (ISS-025).

Free-runs from the model's own training-target state each December of --starts
(default 2008-2014), each to December 2018 under the real ERA5 weather, scored
against that same target (neXtSIM, or its QM version) month by month. The target
saw the same weather, so year-to-year weather cancels and the error left is the
model's own. For each start, the annual-mean extent error by lead year (year 1 =
months 1-12 after the start; annual means remove the seasonal cycle of the error),
and its trend over lead years. Reported: the mean trend over starts (drift rate,
M km^2 per year) with its standard error, the mean error by lead year, and the same
for September alone. Logged to the ledger, where scorecard.py picks it up.

The runs use freerun_monthly.py with no snapshots (only its diagnostics CSV, in
plots/freerun/drift/), 2 members and the fast sampler: ~590 model months, about 2 h
per model. They resume; finished starts are skipped. The starts are training years:
the model has seen each one-month step but not its own compounding errors, which is
what this measures, and every model shares them, so comparisons stay fair.

Run:  python scripts/evaluate/drift.py --selfcheck
      python scripts/evaluate/drift.py --model monthly_pf_qm_ema --run     # run what's missing, then score
      python scripts/evaluate/drift.py --model monthly_pf_qm_ema           # score existing runs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import frazil.ledger as RL
from frazil import paths
from frazil.diagnostics import ice_area_km2, SIC_THRESHOLD

END_YEAR = 2018                 # the target (neXtSIM) ends in December 2018


def model_datacube(model):
    """The datacube a model was trained on: QM or original targets, with its degree-day
    variant and extra forcing channels."""
    cfg = yaml.safe_load(open(os.path.join(paths.MODELS, model, ".hydra", "config.yaml")))
    suffix = cfg["data"].get("suffix", "")
    extra = [v for v in cfg["data"].get("forcing_variables", []) if v not in ("tus", "huss", "uas", "vas")]
    cube = paths.DATACUBE_SICQM if "sicqm" in suffix else paths.DATACUBE
    return cube + ("_dd" if suffix.endswith("_dd") else "") + ("_era5x" if extra else "")


def target_extent(datacube, aux):
    """{(year, month): extent km^2} of the target's sic over the ocean."""
    ocean = aux["mask"].values.astype(bool)
    area = aux["cell_area"].values / 1e6
    out = {}
    for f in sorted(glob.glob(f"{datacube}/*_datacube_*.nc")):
        cube = xr.open_dataset(f)["datacube"]
        ext = ice_area_km2(cube.sel(var_names="sic").values, ocean, area, "extent", SIC_THRESHOLD)
        out.update({(t.year, t.month): float(e) for t, e in zip(pd.DatetimeIndex(cube.time.values), ext)})
    return out


def start_errors(csv, start_year, truth):
    """Monthly extent error (model mean of member extents minus target), M km^2, by lead month."""
    d = pd.read_csv(csv)
    months = pd.date_range(f"{start_year}-12-01", periods=len(d), freq="MS")
    rows = [dict(lead=k, year=t.year, month=t.month, err=(d.ext_members_km2[k] - truth[(t.year, t.month)]) / 1e6)
            for k, t in enumerate(months) if k > 0 and (t.year, t.month) in truth]
    return pd.DataFrame(rows)


def drift_stats(errors):
    """errors: {start_year: DataFrame(lead, month, err)} -> drift rate and error by lead year."""
    per_start, sep = [], []
    by_lead = {}
    for y, e in errors.items():
        e = e.assign(lead_year=(e.lead - 1) // 12 + 1)
        ann = e.groupby("lead_year").err.agg(["mean", "size"])
        ann = ann[ann["size"] == 12]["mean"]                       # complete lead years only
        if len(ann) >= 2:
            per_start.append(np.polyfit(ann.index.values, ann.values, 1)[0])
        s = e[e.month == 9]
        if len(s) >= 2:
            sep.append(np.polyfit(s.lead_year.values, s.err.values, 1)[0])
        for ly, v in ann.items():
            by_lead.setdefault(int(ly), []).append(float(v))
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
    return {"drift_rate": round(float(np.mean(per_start)), 4), "drift_rate_se": se(per_start) and round(se(per_start), 4),
            "sep_drift_rate": round(float(np.mean(sep)), 4), "sep_drift_rate_se": se(sep) and round(se(sep), 4),
            "error_by_lead_year": {ly: round(float(np.mean(v)), 4) for ly, v in sorted(by_lead.items())},
            "starts_by_lead_year": {ly: len(v) for ly, v in sorted(by_lead.items())},
            "starts": len(errors)}


def _selfcheck():
    rng = np.random.default_rng(0)
    errors = {}
    for y in range(2008, 2015):
        n = (END_YEAR - y) * 12
        lead = np.arange(1, n + 1)
        month = (lead - 1) % 12 + 1                                       # Dec start: lead 1 = January
        err = 0.2 + 0.05 * lead / 12 + 0.3 * np.sin(2 * np.pi * month / 12) + 0.02 * rng.standard_normal(n)
        errors[y] = pd.DataFrame({"lead": lead, "month": month, "err": err})
    st = drift_stats(errors)
    assert abs(st["drift_rate"] - 0.05) < 0.01 and st["drift_rate_se"] < 0.01, st      # seasonal cycle removed
    assert abs(st["sep_drift_rate"] - 0.05) < 0.02, st
    assert st["starts_by_lead_year"][4] == 7 and st["starts_by_lead_year"][10] == 1
    truth = {(t.year, t.month): 1e7 for t in pd.date_range("2008-12-01", "2018-12-01", freq="MS")}
    csv = pd.DataFrame({"ext_members_km2": [1e7] + [1e7 + 5e5] * 12})
    path = os.path.join(os.environ.get("TMPDIR", "."), "drift_selfcheck.csv"); csv.to_csv(path, index=False)
    e = start_errors(path, 2017, truth); os.remove(path)
    assert len(e) == 12 and np.allclose(e.err, 0.5) and e.month.iloc[0] == 1 and e.year.iloc[0] == 2018
    print("[selfcheck] drift rate recovered through a seasonal cycle and noise; CSV rows align to the right months")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="folder name in data/models")
    ap.add_argument("--starts", default="2008-2014", help="December start years, YYYY-YYYY")
    ap.add_argument("--n-ens", type=int, default=2)
    ap.add_argument("--run", action="store_true", help="run the missing free-runs first (resumable)")
    ap.add_argument("--datacube", default=None, help="default: the one the model was trained on")
    ap.add_argument("--out-dir", default=paths.DRIFT_OUT)
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return
    if not args.model:
        raise SystemExit("give --model (its folder name in data/models)")
    cube = args.datacube or model_datacube(args.model)
    ckpt = os.path.join(paths.MODELS, args.model, "last.ckpt")
    y0, y1 = (int(s) for s in args.starts.split("-"))
    starts = range(y0, y1 + 1)
    csv = lambda y: os.path.join(args.out_dir, f"freerun_{RL.model_tag(ckpt)}_era5_{y}.csv")
    complete = lambda y: os.path.exists(csv(y)) and len(pd.read_csv(csv(y))) == (END_YEAR - y) * 12 + 1
    print(f"[drift] {args.model}: starts Dec {y0}-{y1} -> Dec {END_YEAR}, target {cube}")

    if args.run:
        for y in starts:
            if complete(y):
                print(f"  Dec {y}: done"); continue
            cmd = [sys.executable, os.path.join(RL.ROOT, "scripts", "project", "freerun_monthly.py"),
                   "--ckpt", ckpt, "--forcing", "era5", "--start", f"{y}-12", "--years", str(END_YEAR - y),
                   "--n-ens", str(args.n_ens), "--fast", "--snapshot-every", "0", "--no-map", "--no-log",
                   "--datacube", cube, "--out-dir", args.out_dir]
            print(f"  Dec {y}: {END_YEAR - y} years ...", flush=True)
            subprocess.run(cmd, check=True)
    missing = [y for y in starts if not complete(y)]
    if missing:
        raise SystemExit(f"no complete free-run for starts {missing}: run with --run")

    aux = xr.open_dataset(args.aux)
    truth = target_extent(cube, aux)
    st = drift_stats({y: start_errors(csv(y), y, truth) for y in starts})
    st.update({"model": args.model, "target": cube, "n_ens": args.n_ens, "starts_span": args.starts})
    print(f"\nDrift of {args.model} (extent error vs its training target, M km^2; mean over {st['starts']} starts):")
    pm = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"  drift rate: {st['drift_rate']:+.3f} per year (+/- {pm(st['drift_rate_se'])})   "
          f"September: {st['sep_drift_rate']:+.3f} per year (+/- {pm(st['sep_drift_rate_se'])})")
    print("  annual-mean error by lead year: " + "  ".join(
        f"{ly}: {v:+.3f} (n={st['starts_by_lead_year'][ly]})" for ly, v in st["error_by_lead_year"].items()))
    if not args.no_log:
        RL.log_result("drift", {**vars(args), "datacube": cube, "ckpt_resolved": ckpt}, st, note=args.note)


if __name__ == "__main__":
    main()
