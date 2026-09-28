#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PROOF OF CONCEPT — does per-cell quantile mapping close the September extent gap?

*** Not a pipeline tool / not for the final model. *** This only tests whether
distribution-matching (quantile mapping) can do what the additive mean-shift
could not: actually retreat the ice edge so modelled extent lands on observations.

What it does (September only, for clarity):
  1. Fit a per-cell, per-quantile map  sic -> obs  from the 1995-2018 overlap
     (neXtSIM as the model's training distribution vs NSIDC-0051 gridded obs).
  2. PROOF: apply it to neXtSIM itself -> its extent should drop from ~5.97 to the
     observed ~5.60 M km^2 (the mean-shift left it at ~5.92 -- barely moved).
  3. Apply the SAME map to the pushforward model's September projection snapshots
     and report the corrected extent vs obs.

If step 2 lands on obs, the concept is proven and the same mapping goes onto the
training targets for a fine-tune (the real fix). If it doesn't, quantile mapping
isn't the answer either.

Run:  python experiments/qm_calibrate_poc.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

MONTH = 9
THR = 0.15
QS = np.linspace(0.0, 1.0, 21)   # quantile knots for the map


def fit_qm(ref, obs):
    """Per-cell quantile knots for ref and obs -> (Q,H,W) each. ref/obs: (T,H,W)."""
    rq = np.nanquantile(np.nan_to_num(ref), QS, axis=0)
    oq = np.nanquantile(np.nan_to_num(obs), QS, axis=0)
    return rq.astype(np.float32), oq.astype(np.float32)


def apply_qm(x, rq, oq):
    """Map one (H,W) field through the per-cell monotone quantile transform."""
    x = np.clip(np.nan_to_num(x), rq[0], rq[-1])
    out = oq[0].copy()
    for k in range(1, rq.shape[0]):
        lo, hi, olo, ohi = rq[k - 1], rq[k], oq[k - 1], oq[k]
        span = np.where(hi > lo, hi - lo, 1.0)
        frac = np.clip((x - lo) / span, 0.0, 1.0)
        sel = (x > lo) & (x <= hi)
        out = np.where(sel, olo + frac * (ohi - olo), out)
    return np.clip(out, 0.0, 1.0)


def _selfcheck():
    # A cell icy in ref but ice-free in obs: mapping must pull ref values down.
    ref = np.array([[[0.9]], [[0.8]], [[0.7]]])      # (3,1,1)
    obs = np.array([[[0.0]], [[0.1]], [[0.0]]])
    rq, oq = fit_qm(ref, obs)
    mapped = apply_qm(np.array([[0.85]]), rq, oq)[0, 0]
    assert mapped < 0.15, mapped                     # high ref -> below the ice edge
    print("[selfcheck] apply_qm OK (mapped %.3f)" % mapped)


def load_sept(dirn, y0=None, y1=None):
    da = xr.open_mfdataset(sorted(glob.glob(f"{dirn}/*.nc")), combine="by_coords")
    name = next((n for n in ("datacube", "state") if n in da.data_vars), None)
    v = da[name].sel(var_names="sic")
    t = pd.DatetimeIndex(v["time"].values)
    keep = t.month == MONTH
    if y0:
        keep &= (t.year >= y0) & (t.year <= y1)
    sic = np.nan_to_num(v.isel(time=np.where(keep)[0]).values).astype(np.float32)
    return sic, t[keep].year.values


def main():
    if "--selfcheck" in sys.argv:
        _selfcheck(); return

    aux = xr.open_dataset("data/auxiliary/ds_auxiliary.nc")
    ocean = aux["mask"].values.astype(bool)
    ca = aux["cell_area"].values / 1e6
    ext = lambda sic: float(((sic > THR) * ocean * ca).sum()) / 1e6

    nx, nxy = load_sept("data/train_data/monthly_datacube", 1995, 2018)   # ref
    ob, oby = load_sept("data/obs/nsidc0051_grid", 1995, 2018)            # obs
    print(f"[fit] September QM  neXtSIM->obs on {len(nxy)} overlap years")
    rq, oq = fit_qm(nx, ob)

    # --- PROOF: apply to neXtSIM, compare extent to obs ---
    nx_qm = np.stack([apply_qm(f, rq, oq) for f in nx])
    print("\nPROOF (Sept extent, mean over 1995-2018, M km^2):")
    print(f"  neXtSIM raw          {np.mean([ext(f) for f in nx]):.2f}")
    print(f"  neXtSIM quantile-map {np.mean([ext(f) for f in nx_qm]):.2f}   <- should match obs")
    print(f"  NSIDC-0051 obs       {np.mean([ext(f) for f in ob]):.2f}")

    # --- apply the same map to the model's September projection ---
    snaps = sorted(glob.glob("plots/freerun/snapshots/state_*09.nc"))
    if snaps:
        pm, pmy = load_sept("plots/freerun/snapshots")
        # restrict to the obs-overlap years for a fair before/after vs obs
        obs_by_year = {int(y): ext(f) for y, f in zip(oby, ob)}
        pm_qm = np.stack([apply_qm(f, rq, oq) for f in pm])
        print("\nMODEL projection (pushforward Sept snapshots):")
        print("  year   model_raw  model_QM   obs")
        for y, r, q in zip(pmy, pm, pm_qm):
            o = obs_by_year.get(int(y))
            print(f"  {int(y)}   {ext(r):8.2f}  {ext(q):8.2f}   "
                  + (f"{o:.2f}" if o else "  – "))
        ov = [int(y) for y in pmy if int(y) in obs_by_year]
        if ov:
            mr = np.mean([ext(f) for f, y in zip(pm, pmy) if int(y) in obs_by_year])
            mq = np.mean([ext(f) for f, y in zip(pm_qm, pmy) if int(y) in obs_by_year])
            oo = np.mean([obs_by_year[y] for y in ov])
            print(f"  overlap mean  raw {mr:.2f}  QM {mq:.2f}  obs {oo:.2f}  "
                  f"({len(ov)} yr)")
    else:
        print("\n(no plots/freerun/snapshots/state_*09.nc -- skipping model projection)")

    print("\n[poc] output QM is a post-hoc field transform; for the real fix, QM the "
          "targets and fine-tune (see the chat / docs).")


if __name__ == "__main__":
    main()
