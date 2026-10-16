#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The scorecard (docs/ROADMAP.md): one fixed protocol so models compare across versions.

Scores a model's out-of-sample ERA5 hindcast (free-run from the December 2018 state,
members saved) per member against NSIDC-0051 on the model grid:
  extent bias for September, June, July, August (M km^2)
  ice-edge error (IIEE) for September, June, July-September, December-March (M km^2)
  September year-to-year correlation of the members' mean extent with the observed
  drift growth: September bias of the last two/three years minus the first two/three
  September bias by sector: East Siberian, Beaufort, Chukchi, Laptev
Development years 2019-22 always; with --holdout also 2023-25 and 2019-25 (holdout
years are for release candidates only). Adds, from the results ledger, the model's
latest one-month skill (eval_monthly) and CMIP stability (freerun --forcing cmip).
Writes results/scorecards/<model>.json and logs a ledger entry.

Needs:  freerun_monthly.py --ckpt data/models/<model>/last.ckpt --forcing era5 --start 2018-12
        --years 7 --n-ens 8 --save-members ...   (members_YYYYMM.nc with sic)
Run:    python scripts/evaluate/scorecard.py --model monthly_pf_qm_ema --holdout
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

VERSION = 2   # 2: adds June ice-edge error (2026-10-15)
THR = 0.15
SECTORS = {"Barents/Kara": (20, 90), "Laptev": (90, 140), "East Siberian": (140, 180), "Chukchi": (180, 205),
           "Beaufort": (205, 240), "Canadian/Baffin/Hudson": (240, 300), "Greenland": (300, 380)}
REPORTED_SECTORS = ["East Siberian", "Beaufort", "Chukchi", "Laptev"]
DEV, HOLDOUT = (2019, 2022), (2023, 2025)


def sector_masks(lon, ocean):
    lon = (lon + 360) % 360
    return {n: ocean & (((lon >= a) & (lon < b)) | ((lon + 360 >= a) & (lon + 360 < b))) for n, (a, b) in SECTORS.items()}


def member_rows(members, obs, ocean, area_km2, sectors, year, month):
    """One row per member: extent bias, IIEE and per-sector extent bias (M km^2)."""
    km = lambda b: float((area_km2 * b).sum()) / 1e6
    o = (np.nan_to_num(obs) > THR) & ocean
    rows = []
    for k, m in enumerate(members):
        f = (np.nan_to_num(m) > THR) & ocean
        r = dict(year=year, month=month, member=k, ext=km(f) - km(o), model_ext=km(f), obs_ext=km(o), iiee=km(f ^ o))
        r.update({s: km(f & w) - km(o & w) for s, w in sectors.items()})
        rows.append(r)
    return rows


def scores(df):
    """The scorecard measures from member rows covering whole years."""
    g = lambda months: df[df.month.isin(months)]
    sep = g([9])
    yr = sep.groupby("year")[["model_ext", "obs_ext", "ext"]].mean()
    years = sorted(yr.index)
    n = max(1, len(years) // 2) if len(years) < 6 else 3
    out = {f"{name}_bias": round(float(g([m]).ext.mean()), 3) for name, m in [("sep", 9), ("jun", 6), ("jul", 7), ("aug", 8)]}
    out.update({"iiee_sep": float(sep.iiee.mean()), "iiee_jun": float(g([6]).iiee.mean()), "iiee_jas": float(g([7, 8, 9]).iiee.mean()),
                "iiee_djfm": float(g([12, 1, 2, 3]).iiee.mean()),
                "sep_r": float(np.corrcoef(yr.model_ext, yr.obs_ext)[0, 1]) if len(years) > 2 and yr.obs_ext.std() > 0 else None,
                "drift_growth": float(yr.ext.loc[years[-n:]].mean() - yr.ext.loc[years[:n]].mean()),
                "drift_growth_years": f"{years[-n]}-{years[-1]} minus {years[0]}-{years[n - 1]}"})
    out.update({f"sep_bias_{s.lower().replace(' ', '_')}": float(sep[s].mean()) for s in REPORTED_SECTORS})
    out.update({"years": f"{years[0]}-{years[-1]}", "members": int(df.member.nunique()), "months_scored": int(len(df[["year", "month"]].drop_duplicates()))})
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


def latest(records, experiment, model, test=lambda r: True):
    hits = [r for r in records if r.get("experiment") == experiment and test(r)
            and RL.model_tag(r.get("params", {}).get("ckpt_resolved") or r.get("params", {}).get("ckpt") or "") == model]
    return hits[-1] if hits else None


def _selfcheck():
    H = W = 16
    ocean = np.ones((H, W), bool); area = np.full((H, W), 1e4)          # 1e4 km^2 per cell
    lon = np.tile(np.linspace(-179, 179, W), (H, 1))
    sec = sector_masks(lon, ocean)
    obs = np.zeros((H, W)); obs[:8] = 1.0                               # 128 cells of ice = 1.28 M km^2
    rows = []
    for y in range(2019, 2023):
        for m in range(1, 13):
            mem = np.stack([obs.copy(), obs.copy()]); mem[1, 8, :] = 1.0  # member 1: one extra row (0.16)
            rows += member_rows(mem, obs, ocean, area, sec, y, m)
    s = scores(pd.DataFrame(rows))
    assert abs(s["sep_bias"] - 0.08) < 1e-9 and abs(s["iiee_jas"] - 0.08) < 1e-9, s     # mean of 0 and 0.16
    assert s["drift_growth"] == 0 and s["members"] == 2 and s["months_scored"] == 48, s
    assert abs(sum(r[n] for r in rows[:2] for n in SECTORS) - 0.16) < 1e-9             # sectors partition the extra ice
    print("[selfcheck] scorecard measures recovered from a synthetic hindcast")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model name = its folder in data/models, e.g. monthly_pf_qm_ema")
    ap.add_argument("--hindcast", default=None, help=f"members folder (default: {paths.SNAPSHOTS}/<model>_era5_2018)")
    ap.add_argument("--holdout", action="store_true", help="also score the 2023-25 holdout (release candidates only)")
    ap.add_argument("--obs", default=paths.NSIDC0051_GRID)
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--out", default=None, help=f"default: {paths.SCORECARDS}/<model>.json")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return
    if not args.model:
        raise SystemExit("give --model (its folder name in data/models)")
    snap = args.hindcast or os.path.join(paths.SNAPSHOTS, f"{args.model}_era5_2018")
    files = sorted(glob.glob(f"{snap}/members_*.nc"))
    if not files:
        raise SystemExit(f"no members_*.nc in {snap}: run the ERA5 hindcast with --save-members first")

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool); area = aux["cell_area"].values / 1e6
    sec = sector_masks(aux["longitude"].values, ocean)
    d = xr.open_mfdataset(sorted(glob.glob(f"{args.obs}/*.nc")), combine="by_coords")
    v = d[[n for n in ("datacube", "state") if n in d.data_vars][0]].sel(var_names="sic")
    t = pd.DatetimeIndex(v["time"].values)
    obs_i = {(x.year, x.month): i for i, x in enumerate(t)}
    last = HOLDOUT[1] if args.holdout else DEV[1]
    rows = []
    for f in files:
        y, m = int(f[-9:-5]), int(f[-5:-3])
        if not (DEV[0] <= y <= last) or (y, m) not in obs_i:
            continue
        rows += member_rows(xr.open_dataset(f)["sic"].values, v.isel(time=obs_i[(y, m)]).values, ocean, area, sec, y, m)
    df = pd.DataFrame(rows)
    hind = {"development_2019_22": scores(df[df.year <= DEV[1]])}
    if args.holdout:
        hind["holdout_2023_25"] = scores(df[df.year >= HOLDOUT[0]])
        hind["all_2019_25"] = scores(df)

    records = [json.loads(l) for l in open(RL.LEDGER)] if os.path.exists(RL.LEDGER) else []
    ev = latest(records, "eval_monthly", args.model, lambda r: r["params"].get("n_ens", 1) >= 8)
    cm = latest(records, "freerun_monthly", args.model, lambda r: r["params"].get("forcing") == "cmip")
    ckpt = os.path.join(paths.MODELS, args.model, "last.ckpt")
    sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest() if os.path.exists(ckpt) else None
    card = {"scorecard_version": VERSION, "model": args.model, "checkpoint_sha256": sha, "hindcast_dir": snap,
            "hindcast": hind,
            "one_month_skill": None if not ev else {"run_id": ev["run_id"], "datacube": ev["params"].get("datacube"),
                                                     "n_ens": ev["params"].get("n_ens"),
                                                     "skill_vs_persistence": {k: round(x, 3) for k, x in ev["metrics"]["skill_vs_persistence"].items()}},
            "cmip_stability": None if not cm else {"run_id": cm["run_id"], "years_completed": cm["metrics"].get("years_completed"),
                                                   "nonfinite_total": cm["metrics"].get("nonfinite_total")}}
    out = args.out or os.path.join(paths.SCORECARDS, f"{args.model}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(card, fh, indent=1); fh.write("\n")

    table = pd.DataFrame({k: {m: x for m, x in s.items() if not isinstance(x, str)} for k, s in hind.items()})
    print(f"\nScorecard v{VERSION}: {args.model}  (per member vs NSIDC-0051, M km^2 unless noted)\n{table.to_string()}")
    print(f"\none-month skill: {'none in the ledger (eval_monthly --n-ens 8)' if not ev else card['one_month_skill']['skill_vs_persistence']}")
    print(f"CMIP stability:  {'no CMIP free-run in the ledger' if not cm else card['cmip_stability']}")
    print(f"[json] {out}")
    if not args.no_log:
        RL.log_result("scorecard", {**vars(args), "hindcast": snap}, card, note=args.note)


if __name__ == "__main__":
    main()
