"""Per sector, across the months of 2015-2018: is Frazil's one-step extent error a smooth
nonlinear (polynomial) function of where the sector starts, or of the time of year?

From the saved one-month predictions (each from the target's previous month), per sector
(scorecard.SECTORS) and predicted month:
  y      one-step extent error (Frazil, mean over members, minus target), as a share of the sector's ocean area
  cover  the target's extent at the start of the step, same units (what the model sees)
  doy    time of year (month 1-12 scaled to 0-1)
Polynomials of degree 0-4 in each, fitted separately for the melt half (predicted months
May-September) and the freeze half (October-April), since the same cover means different
things on the way down and on the way up. Scored by leave-one-year-out cross-validation
(fit on three years, predict the fourth): with 4 years and 5-7 months per half, a
higher degree only counts if it predicts a year it hasn't seen better.

Run from the repo root:  python experiments/2026-10-30_summer_melt/sector_poly.py [--fields DIR]
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

import frazil.ledger as RL
from frazil import paths

ap = argparse.ArgumentParser()
ap.add_argument("--fields", default=os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw", "one_month_fields"))
ap.add_argument("--datacube", default=os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"))
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
pd.set_option("display.width", 220)

aux = xr.open_dataset(paths.AUX)
ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
size = {s: (area * m).sum() for s, m in sec.items()}
cube = xr.open_mfdataset(sorted(glob.glob(f"{args.datacube}/*_datacube_*.nc")), combine="by_coords")["datacube"]
ct = pd.DatetimeIndex(cube.time.values)
ext = lambda sic, m: (area * m * (np.nan_to_num(sic) > 0.15)).sum()

rows = []
for f in sorted(glob.glob(f"{args.fields}/pred_??????.nc")):
    p = xr.open_dataset(f); t = pd.Timestamp(p.time.values)
    i = int(np.where((ct.year == t.year) & (ct.month == t.month))[0][0])
    prev, tgt = cube.isel(time=i - 1).sel(var_names="sic").values, cube.isel(time=i).sel(var_names="sic").values
    for s, m in sec.items():
        fz = np.mean([ext(x, m) for x in p.sic_members.values])
        rows.append(dict(sector=s, year=t.year, month=t.month, cover=ext(prev, m) / size[s],
                         y=(fz - ext(tgt, m)) / size[s], y_km=(fz - ext(tgt, m)) / 1e6))
D = pd.DataFrame(rows)
D["doy"] = (D.month - 1) / 11
D["half"] = np.where(D.month.isin([5, 6, 7, 8, 9]), "melt May-Sep", "freeze Oct-Apr")


def loyo(g, x, deg):
    """Leave-one-year-out RMSE and in-sample RMSE of a degree-`deg` polynomial y(x)."""
    errs = []
    for yr in g.year.unique():
        tr, te = g[g.year != yr], g[g.year == yr]
        c = np.polyfit(tr[x], tr.y, deg)
        errs.append(te.y - np.polyval(c, te[x]))
    c = np.polyfit(g[x], g.y, deg)
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))), float(np.sqrt(np.mean((g.y - np.polyval(c, g[x])) ** 2))), c


print(f"{len(D) // len(sec)} one-step months, {D.year.min()}-{D.year.max()}, {len(sec)} sectors. "
      "Errors as a share of the sector's ocean area (x100 = percent of the sector).\n")
print("=== Mean one-step extent error by month (M km^2)")
print(D.pivot_table(index="sector", columns="month", values="y_km").round(3).to_string(), "\n")

best = []
for half in ("melt May-Sep", "freeze Oct-Apr"):
    for x in ("cover", "doy"):
        print(f"=== {half}: error as a polynomial in {'starting cover' if x == 'cover' else 'time of year'}; "
              "leave-one-year-out RMSE x100 by degree (in-sample in brackets); * = best out of sample")
        out = []
        for s, g in D[D.half == half].groupby("sector", sort=False):
            r = {"sector": s, "n": len(g), "error sd": 100 * g.y.std()}
            cv = {}
            for deg in range(5):
                if deg >= g.month.nunique() - 1:
                    continue
                cv[deg], ins, c = loyo(g, x, deg)
                r[f"deg {deg}"] = f"{100 * cv[deg]:.2f} ({100 * ins:.2f})"
            k = min(cv, key=cv.get); r[f"deg {k}"] += " *"
            r["best"] = k
            out.append(r)
            best.append(dict(half=half, x=x, sector=s, best=k, cv_best=100 * cv[k], cv_0=100 * cv[0], cv_1=100 * cv.get(1, np.nan)))
        print(pd.DataFrame(out).set_index("sector").round(2).to_string(), "\n")

B = pd.DataFrame(best)
print("=== Summary: best degree out of sample, and how much it beats a constant (deg 0) and a line (deg 1), RMSE x100")
B["gain vs const"] = B.cv_0 - B.cv_best
B["gain vs line"] = B.cv_1 - B.cv_best
print(B.set_index(["half", "x", "sector"]).round(2).to_string())

print("\n=== Melt half, by year: mean error (M km^2) per sector - does the relationship move from year to year?")
print(D[D.half == "melt May-Sep"].pivot_table(index="sector", columns="year", values="y_km").round(3).to_string())

print("\n=== Melt half, straight line in starting cover: error x100 = a + b x cover (b < 0: error grows as the sector opens up); share of variance explained")
for s, g in D[D.half == "melt May-Sep"].groupby("sector", sort=False):
    b, a = np.polyfit(g.cover, g.y, 1)
    r2 = 1 - np.var(g.y - (a + b * g.cover)) / np.var(g.y)
    print(f"  {s:24s} a {100 * a:+.2f}  b {100 * b:+.2f}  R^2 {r2:.2f}   cover range {g.cover.min():.2f}-{g.cover.max():.2f}")

print("\n=== Straight lines per sector, both halves: error x100 = a + b x cover. Slope +- standard error, the range of b over")
print("    leave-one-year-out fits, R^2, and out-of-sample RMSE x100 of the line vs a constant (a per-sector mean)")
rows = []
for half in ("melt May-Sep", "freeze Oct-Apr"):
    for s, g in D[D.half == half].groupby("sector", sort=False):
        (b, a), cov = np.polyfit(g.cover, g.y, 1, cov=True)
        bs = [np.polyfit(g[g.year != yr].cover, g[g.year != yr].y, 1)[0] for yr in g.year.unique()]
        rows.append({"half": half, "sector": s, "a": 100 * a, "b": 100 * b, "b se": 100 * np.sqrt(cov[0, 0]),
                     "b, leave-one-year-out": f"{100 * min(bs):+.1f} to {100 * max(bs):+.1f}",
                     "R^2": 1 - np.var(g.y - (a + b * g.cover)) / np.var(g.y),
                     "RMSE line": 100 * loyo(g, "cover", 1)[0], "RMSE const": 100 * loyo(g, "cover", 0)[0]})
L = pd.DataFrame(rows).set_index(["half", "sector"])
L["line wins"] = L["RMSE line"] < L["RMSE const"]
print(L.round(2).to_string())

print("\n=== One shared line, or one per sector? Out-of-sample RMSE x100 over the sectors in each group (leave one year out)")
for half in ("melt May-Sep", "freeze Oct-Apr"):
    for name, group in (("Pacific shelves (E. Siberian, Chukchi, Beaufort)", ["East Siberian", "Chukchi", "Beaufort"]), ("all 7 sectors", list(sec))):
        g = D[(D.half == half) & D.sector.isin(group)]
        e_shared, e_per = [], []
        for yr in g.year.unique():
            tr, te = g[g.year != yr], g[g.year == yr]
            c = np.polyfit(tr.cover, tr.y, 1); e_shared.append(te.y - np.polyval(c, te.cover))
            for s in group:
                c = np.polyfit(tr[tr.sector == s].cover, tr[tr.sector == s].y, 1)
                e_per.append(te[te.sector == s].y - np.polyval(c, te[te.sector == s].cover))
        r = lambda e: 100 * np.sqrt(np.mean(np.concatenate(e) ** 2))
        c = np.polyfit(g.cover, g.y, 1)
        print(f"  {half:15s} {name:50s} shared line {r(e_shared):.2f} (a {100 * c[1]:+.1f}, b {100 * c[0]:+.1f})   per-sector lines {r(e_per):.2f}")
