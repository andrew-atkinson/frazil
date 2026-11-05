"""Gate before any training: do v0.4.0's errors follow ocean heat on the Atlantic side?
One-month predictions, Feb 2015-Dec 2018, against the targets, with the training simulation's
own heat (nextsim_ocean_heat.py --full-domain: 0-20, 20-100, 100-300 m). Heat is taken at the
start month, as the lagged channels would give it, so it can't leak the month predicted.
Anomaly = minus the cell's 1995-2014 mean for that calendar month.

No shape is assumed. A lookup table of mean error is learned on three years and applied to
the fourth, each year in turn, per sector:
  base        month x starting concentration x predicted concentration (no heat; what the
              model can already see, so any gain beyond it is the heat's own)
  + anomaly   base x quintile of one layer's anomaly, or x tercile of all three (the year's heat)
  + raw heat  the same with the heat itself (the year's heat plus its seasonal pattern)
  control     + anomaly, using the heat of another year (next year, 2018 -> 2015). Anything
              this gains is persistence or noise, not this month's ocean
Gain = % change in RMSE against base, out of sample. The fine tables split the data so thinly
that even the control comes out worse than base, so a coarser version (added after the first
run; same decision rule) learns the heat's effect by season x bin only, on what the base table
leaves, with three controls (each other year) to show the noise. Calibration: the same numbers on the
Pacific shelves, where a real training with heat exists: real minus clim RMSE, July-October.

Decision, written before seeing the result: train with full-domain heat only if Barents/Kara
or Canadian/Baffin/Hudson show an anomaly gain above the control, in at least 3 of 4 held-out
years, and larger than the Pacific's, whose real training moved RMSE by about 1%.

Run from the repo root:  python experiments/2026-10-30_summer_melt/atlantic_heat_gate.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths
from frazil.preprocess import build_nearest_index

LAYERS = ("ohc_0_20", "ohc_20_100", "ohc_100_300")
V04 = os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw", "one_month_fields")
OHC = os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw_ohc")
START = [-1, 0.15, 0.5, 0.8, 0.95, 2]
PRED = [-1, 0.05, 0.3, 0.6, 0.85, 2]
SECTORS = ["Barents/Kara", "Canadian/Baffin/Hudson", "Greenland", "Pacific shelves", "Laptev"]
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
sec["Pacific shelves"] = sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"]
heat = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.NEXTSIM_OCEAN_HEAT_FULL, "ohc_*.nc"))), combine="by_coords").load()
idx = build_nearest_index(heat.longitude.values.ravel(), heat.latitude.values.ravel(), aux.longitude.values[ocean], aux.latitude.values[ocean])
yy, xx = np.unravel_index(idx, heat.latitude.shape)
ht = pd.DatetimeIndex(heat.time.values)
H = {k: heat[k].values[:, yy, xx] for k in LAYERS}                        # (288, ocean cells)
clim = {k: np.stack([H[k][(ht.month == m) & (ht.year <= 2014)].mean(0) for m in range(1, 13)]) for k in LAYERS}
at = lambda y, m: int(np.where((ht.year == y) & (ht.month == m))[0][0])
cube = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9", "*_datacube_*.nc"))), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)

rows, cal = [], []
for f in sorted(glob.glob(os.path.join(V04, "pred_??????.nc"))):
    ym = os.path.basename(f)[5:11]; t = pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01")
    i = int(np.where((ct.year == t.year) & (ct.month == t.month))[0][0])
    sic = lambda j: np.nan_to_num(cube.isel(time=j).sel(var_names="sic").values)
    tgt, prev = sic(i), sic(i - 1)
    pred = np.nan_to_num(xr.open_dataset(f).state.sel(var_names="sic").values)
    s = t - pd.DateOffset(months=1)                                       # start month: what the lagged channel would hold
    others = [2015 + (s.year - 2015 + n) % 4 for n in (1, 2, 3)]
    d = {"y": t.year, "mo": t.month, "w": area[ocean], "start": prev[ocean], "pred": pred[ocean], "err": (pred - tgt)[ocean], "tot": (pred - tgt)[ocean] ** 2}
    for k in LAYERS:
        v, c = H[k][at(s.year, s.month)], clim[k][s.month - 1]
        d[f"h_{k}"], d[f"a_{k}"] = v, v - c
        for n, other in enumerate(others):
            d[f"x{n}_{k}"] = H[k][at(other, s.month)] - c
    D = pd.DataFrame(d)
    for name in SECTORS:
        rows.append(D[sec[name][ocean]].assign(sector=name))
    # calibration: the real training, real vs clim heat, same weights and seeds
    clim_f = os.path.join(OHC, "one_month_fields_clim", f"pred_{ym}.nc")
    if os.path.exists(clim_f):
        p = {n: np.nan_to_num(xr.open_dataset(g).state.sel(var_names="sic").values) for n, g in (("real", os.path.join(OHC, "one_month_fields", f"pred_{ym}.nc")), ("clim", clim_f))}
        m = sec["Pacific shelves"] & ocean & ((prev > 0.01) | (pred > 0.01))
        cal.append({"y": t.year, "mo": t.month, **{n: np.average((x - tgt)[m] ** 2, weights=area[m]) for n, x in p.items()}})
A = pd.concat(rows, ignore_index=True)
total = A.groupby("sector").apply(lambda g: (g.tot * g.w).sum())
A = A[(A.start > 0.01) | (A.pred > 0.01)].copy()                          # ice at the start or in the prediction: both are model inputs/outputs
A["base"] = A.mo * 100 + pd.cut(A.start, START, labels=False) * 10 + pd.cut(A.pred, PRED, labels=False)
A["season"] = np.where(A.mo.isin([5, 6, 7, 8, 9]), "melt May-Sep", "freeze Oct-Apr")


def binned(col, train, test, n):
    """Within-month quantile bin of col, edges from the training years only."""
    tr, te = np.zeros(len(train), int), np.zeros(len(test), int)
    for mo in train.mo.unique():
        a, b = train.mo.values == mo, test.mo.values == mo
        edges = np.quantile(train[col].values[a], np.linspace(0, 1, n + 1)[1:-1])
        tr[a], te[b] = np.digitize(train[col].values[a], edges), np.digitize(test[col].values[b], edges)
    return tr, te


def corrected(G, cols, n=5):
    """Out-of-sample residual: err minus the table learned on the other three years."""
    out = np.array(G.err.values, float)
    for yr in G.y.unique():
        a = G.y.values != yr
        train, test = G[a], G[~a]
        ktr, kte = train.base.values.astype(np.int64), test.base.values.astype(np.int64)
        for c in cols:
            b, e = binned(c, train, test, n)
            ktr, kte = ktr * n + b, kte * n + e
        tab = pd.DataFrame({"k": ktr, "e": train.err.values * train.w.values, "w": train.w.values}).groupby("k").sum()
        out[~a] -= pd.Series(kte).map(tab.e / tab.w).fillna(0).values
    return out


def coarse(G, cols, n=5):
    """Base table on three years, then season x heat bin on what it leaves, applied to the fourth."""
    out = np.array(G.err.values, float)
    for yr in G.y.unique():
        a = G.y.values != yr
        train, test = G[a], G[~a]
        tab = pd.DataFrame({"k": train.base.values, "e": train.err.values * train.w.values, "w": train.w.values}).groupby("k").sum()
        base = tab.e / tab.w
        rtr = train.err.values - pd.Series(train.base.values).map(base).values
        rte = pd.Series(test.base.values).map(base).fillna(0).values
        ktr, kte = (train.season.values == "melt May-Sep").astype(np.int64), (test.season.values == "melt May-Sep").astype(np.int64)
        for c in cols:
            b, e = binned(c, train, test, n)
            ktr, kte = ktr * n + b, kte * n + e
        heat = pd.DataFrame({"k": ktr, "e": rtr * train.w.values, "w": train.w.values}).groupby("k").sum()
        out[~a] -= rte + pd.Series(kte).map(heat.e / heat.w).fillna(0).values
    return out


rmse = lambda r, w: np.sqrt(np.average(r ** 2, weights=w))
res, coarse_rows = [], []
for name, G in A.groupby("sector"):
    G = G.reset_index(drop=True)
    b0 = coarse(G, [])
    sets = {**{f"anom {k}": ([f"a_{k}"], 5) for k in LAYERS}, **{f"raw {k}": ([f"h_{k}"], 5) for k in LAYERS},
            "anom all": ([f"a_{k}" for k in LAYERS], 3), "raw all": ([f"h_{k}" for k in LAYERS], 3)}
    for n in range(3):
        sets.update({f"control{n} {k}": ([f"x{n}_{k}"], 5) for k in LAYERS})
        sets[f"control{n} all"] = ([f"x{n}_{k}" for k in LAYERS], 3)
    for lab, (cols, nb) in sets.items():
        r = coarse(G, cols, nb)
        row = {"sector": name, "test": lab}
        for season, m in [("all", np.ones(len(G), bool)), ("melt", G.season.values == "melt May-Sep"), ("freeze", G.season.values == "freeze Oct-Apr")]:
            row[season] = 100 * (rmse(r[m], G.w.values[m]) / rmse(b0[m], G.w.values[m]) - 1)
        for yr in sorted(G.y.unique()):
            u = G.y.values == yr
            row[yr] = 100 * (rmse(r[u], G.w.values[u]) / rmse(b0[u], G.w.values[u]) - 1)
        coarse_rows.append(row)
K = pd.DataFrame(coarse_rows)
print("=== Coarse version: % change in RMSE against base, out of sample (negative = better); controls use another year's heat")
for name, g in K.groupby("sector"):
    print(f"\n{name}:\n" + g.drop(columns="sector").set_index("test").round(3).to_string())
print()

res = []
for name, G in A.groupby("sector"):
    G = G.reset_index(drop=True)
    R = {"none": G.err.values, "base": corrected(G, [])}
    for k in LAYERS:
        R[f"anom {k}"], R[f"raw {k}"], R[f"control {k}"] = corrected(G, [f"a_{k}"]), corrected(G, [f"h_{k}"]), corrected(G, [f"x0_{k}"])
    R["anom all"] = corrected(G, [f"a_{k}" for k in LAYERS], 3)
    R["raw all"] = corrected(G, [f"h_{k}" for k in LAYERS], 3)
    R["control all"] = corrected(G, [f"x0_{k}" for k in LAYERS], 3)
    for season, s in [("all months", np.ones(len(G), bool)), ("melt May-Sep", G.season.values == "melt May-Sep"), ("freeze Oct-Apr", G.season.values == "freeze Oct-Apr"), ("Jul-Oct", G.mo.isin([7, 8, 9, 10]).values)]:
        b = rmse(R["base"][s], G.w.values[s])
        res.append({"sector": name, "season": season, "RMSE v0.4.0": rmse(R["none"][s], G.w.values[s]), "RMSE base": b, "active share of sq. error": ((G.tot * G.w).sum() / total[name]),
                    **{n: 100 * (rmse(r[s], G.w.values[s]) / b - 1) for n, r in R.items() if n not in ("none", "base")}})
        if season == "all months":
            for yr in sorted(G.y.unique()):
                u = G.y.values == yr; bb = rmse(R["base"][u], G.w.values[u])
                res.append({"sector": name, "season": f"year {yr}", **{n: 100 * (rmse(r[u], G.w.values[u]) / bb - 1) for n, r in R.items() if n not in ("none", "base")}})
    for k in ("ohc_0_20", "ohc_20_100", "ohc_100_300"):                  # the shape: base-corrected error by anomaly quintile
        G[f"q_{k}"] = G.groupby("mo")[f"a_{k}"].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
    G["resid"] = R["base"]
    G["season_"] = G.season
    if name in ("Barents/Kara", "Canadian/Baffin/Hudson", "Pacific shelves"):
        print(f"=== Shape, {name}: error left after base, by within-month anomaly quintile (0 = coolest; positive = too much ice)")
        for k in LAYERS:
            T = G.groupby(["season_", f"q_{k}"]).apply(lambda g: np.average(g.resid, weights=g.w)).unstack(f"q_{k}")
            T.insert(0, "mean anomaly range, MJ/m2", G.groupby("season_")[f"a_{k}"].apply(lambda x: f"{x.quantile(.1):.0f} to {x.quantile(.9):.0f}"))
            print(f"{k}:\n" + T.round(4).to_string())
        print()
T = pd.DataFrame(res).set_index(["sector", "season"])
gain = [c for c in T.columns if c.split()[0] in ("anom", "raw", "control")]
print("=== Gain from heat: % change in RMSE against base, out of sample (negative = better). Base itself is a correction with no heat")
print(T[~T.index.get_level_values("season").str.startswith("year")][["RMSE v0.4.0", "RMSE base", "active share of sq. error", *gain]].round(3).T.to_string())
print("\n=== By held-out year, all months")
print(T[T.index.get_level_values("season").str.startswith("year")][gain].round(3).T.to_string())
C = pd.DataFrame(cal)
print("\n=== Calibration, Pacific shelves, the same cells: the real training, real vs clim heat, RMSE % (negative = year's heat helps)")
print(f"  Jul-Oct 2015-18: {100 * (np.sqrt(C.real.mean() / C.clim.mean()) - 1):+.2f}%   by month: " + "  ".join(f"{m}: {100 * (np.sqrt(g.real.mean() / g.clim.mean()) - 1):+.2f}%" for m, g in C.groupby("mo")))
