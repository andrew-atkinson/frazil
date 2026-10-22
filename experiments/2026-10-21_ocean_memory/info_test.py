"""Information test for an ocean-memory input (no model training).

Does sunlight absorbed by open water since spring -- sum over months k = Apr..t of
(1 - sic_k) * S_k, with S_k the top-of-atmosphere insolation -- help predict next
month's concentration, beyond everything the emulator already sees at step t
(its 6 states, the forcing at t and t+1, the degree-days, the month, its position)?

Teacher: the QM-target datacube (what the model trains on), 1995-2018, every 4th
cell. Rows: seasonal-ice-zone cells, t = June..September (predicting July..October).
A small MLP is fit on 1995-2006, early-stopped on 2007-2010 and scored on 2011-2018,
three seeds each, with: the baseline inputs; + memory; + last two months of sic
(a generic "any history" comparison).

Run from the repo root:  python experiments/2026-10-21_ocean_memory/info_test.py
"""
import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr

from frazil import STATES, FORCINGS, DEGREE, paths

STEP, T_MONTHS, SEEDS = 4, (6, 7, 8, 9), (0, 1, 2)
SECTORS = {"East Siberian": (140, 180), "Chukchi": (180, 205), "Beaufort": (205, 240)}


def toa_monthly(lat_deg):
    """Monthly-mean top-of-atmosphere insolation (W m-2), (12, ...) for each latitude."""
    lat = np.deg2rad(lat_deg)
    out = []
    for m in range(1, 13):
        days = pd.date_range(f"2001-{m:02d}-01", periods=pd.Timestamp(f"2001-{m:02d}-01").days_in_month).dayofyear.values
        q = []
        for d in days:
            dec = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + d) / 365)
            h0 = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))
            q.append(1361 / np.pi * (1 + 0.033 * np.cos(2 * np.pi * d / 365))
                     * (h0 * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(h0)))
        out.append(np.mean(q, 0))
    return np.stack(out)


aux = xr.open_dataset(paths.AUX).isel(y=slice(None, None, STEP), x=slice(None, None, STEP))
ocean = aux.mask.values.astype(bool)
lat, lon = aux.latitude.values, (aux.longitude.values + 360) % 360
files = sorted(glob.glob(f"{paths.DATACUBE_SICQM}/*_datacube_*.nc"))
cube = xr.open_mfdataset(files, combine="by_coords")["datacube"].isel(y=slice(None, None, STEP), x=slice(None, None, STEP))
names = STATES + FORCINGS + DEGREE
data = cube.sel(var_names=names).transpose("time", "var_names", "y", "x").load().values.astype(np.float32)
times = pd.DatetimeIndex(cube.time.values)
data = np.nan_to_num(data)
V = {n: i for i, n in enumerate(names)}
sic = data[:, V["sic"]]
S = toa_monthly(lat)                                                   # (12, y, x)

# Memory: absorbed-by-open-water sunlight since April of the same year, through month t.
mem = np.zeros_like(sic)
for i, t in enumerate(times):
    if t.month >= 4:
        mem[i] = (mem[i - 1] if t.month > 4 else 0) + (1 - sic[i]) * S[t.month - 1]

# Seasonal ice zone: ice in March, not full in September (1995-2010 climatology).
clim = lambda m: sic[(times.month == m) & (times.year <= 2010)].mean(0)
zone = ocean & (clim(3) > 0.15) & (clim(9) < 0.85)
sector = np.full(lat.shape, "", object)
for n, (a, b) in SECTORS.items():
    sector[((lon >= a) & (lon < b)) & zone] = n
cells = np.flatnonzero(zone.ravel())
xy = np.stack([aux.x_coord.values.ravel()[cells], aux.y_coord.values.ravel()[cells]], 1) / 1e6

rows = {k: [] for k in ("base", "memory", "history", "y", "year", "month", "sector")}
for i, t in enumerate(times[:-1]):
    if t.month not in T_MONTHS or i < 2:
        continue
    st = data[i, [V[n] for n in STATES]].reshape(len(STATES), -1)[:, cells].T
    f0 = data[i, [V[n] for n in FORCINGS]].reshape(len(FORCINGS), -1)[:, cells].T
    f1 = data[i + 1, [V[n] for n in FORCINGS]].reshape(len(FORCINGS), -1)[:, cells].T
    dd = data[i, [V[n] for n in DEGREE]].reshape(len(DEGREE), -1)[:, cells].T
    mon = np.tile(np.eye(4)[T_MONTHS.index(t.month)], (len(cells), 1))
    rows["base"].append(np.hstack([st, f0, f1, dd, mon, xy]))
    rows["memory"].append(mem[i].ravel()[cells, None])
    rows["history"].append(np.stack([sic[i - 1].ravel()[cells], sic[i - 2].ravel()[cells]], 1))
    rows["y"].append(sic[i + 1].ravel()[cells])
    rows["year"].append(np.full(len(cells), t.year)); rows["month"].append(np.full(len(cells), t.month + 1))
    rows["sector"].append(sector.ravel()[cells])
R = {k: np.concatenate(v) for k, v in rows.items()}
print(f"{len(R['y'])} rows, {len(cells)} seasonal-ice-zone cells (every {STEP}th), {len(R['base'][0])} baseline inputs")


def fit(X, y, train, val, seed):
    torch.manual_seed(seed)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 128),
                              torch.nn.ReLU(), torch.nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    tr, va = np.flatnonzero(train), np.flatnonzero(val)
    best, best_state, wait = np.inf, None, 0
    g = np.random.default_rng(seed)
    for epoch in range(200):
        g.shuffle(tr)
        for b in range(0, len(tr), 1024):
            j = tr[b:b + 1024]
            loss = ((net(Xt[j]).squeeze(1) - yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(((net(Xt[va]).squeeze(1) - yt[va]) ** 2).mean())
        if v < best - 1e-6:
            best, best_state, wait = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 8:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(Xt).squeeze(1).numpy()


train, val, test = R["year"] <= 2006, (R["year"] >= 2007) & (R["year"] <= 2010), R["year"] >= 2011
variants = {"baseline": R["base"], "+ memory": np.hstack([R["base"], R["memory"]]),
            "+ last two months of sic": np.hstack([R["base"], R["history"]])}
res = []
for name, X in variants.items():
    for seed in SEEDS:
        p = np.clip(fit(X, R["y"], train, val, seed), 0, 1)
        err = (p - R["y"]) ** 2
        r = {"variant": name, "seed": seed, "all": np.sqrt(err[test].mean())}
        for m in (7, 8, 9, 10):
            r[f"month {m}"] = np.sqrt(err[test & (R["month"] == m)].mean())
        for s in SECTORS:
            r[s] = np.sqrt(err[test & (R["sector"] == s)].mean())
        res.append(r)
        print(f"  {name:26s} seed {seed}: test RMSE {r['all']:.4f}", flush=True)
df = pd.DataFrame(res)
cols = ["all", "month 7", "month 8", "month 9", "month 10"] + list(SECTORS)
mean, spread = df.groupby("variant")[cols].mean(), df.groupby("variant")[cols].agg(lambda x: x.max() - x.min())
pd.set_option("display.width", 200)
print("\nTest RMSE of next-month concentration, 2011-2018 (mean of 3 seeds; lower is better):")
print(mean.loc[list(variants)].round(4).to_string())
print("\nSpread across seeds (max - min):")
print(spread.loc[list(variants)].round(4).to_string())
print("\nChange vs baseline (%):")
print((100 * (mean.loc[list(variants)] / mean.loc["baseline"] - 1)).round(1).to_string())
