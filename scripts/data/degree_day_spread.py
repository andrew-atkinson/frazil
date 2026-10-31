#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The day-to-day spread of daily 2 m temperature, for degree-days without daily data.

Degree-days from a monthly mean miss the thaw days of a month that averages below
freezing (ISS-007). With the month's typical spread of daily temperatures (one
standard deviation per grid cell and calendar month), frazil.preprocess.
degree_days_gaussian() recovers them from the monthly mean alone, which works for
ERA5 and CMIP alike.

Source: ERA5 6-hourly 2 m temperature at 1.5 deg from WeatherBench2 (public Google
Cloud, no account, 1959 to Jan 2023), averaged to daily means north of 39 N and
cached in data/train_data/wb2_daily_t2m.nc (~0.35 GB; streaming 1995-2022 is ~3 GB).

  --check   accuracy against real daily degree-days in years outside the spread's
            climatology (default 2015-2022): per calendar month, the error of the
            monthly-mean method and of the formula, with the climatological spread
            and with each month's own spread (which isolates the normal-shape error)
  default   write the spread on the model grid to data/auxiliary/t2m_daily_spread.nc

Run:  python scripts/data/degree_day_spread.py --selfcheck
      python scripts/data/degree_day_spread.py --check
      python scripts/data/degree_day_spread.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

import frazil.preprocess as P
from frazil import paths

WB2 = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
LAT_MIN = 39.0


def fetch_daily(years, out):
    """Daily-mean 2 m temperature (time, latitude, longitude) north of LAT_MIN, cached to `out`."""
    if os.path.exists(out):
        d = xr.open_dataarray(out)
        have = set(pd.DatetimeIndex(d.time.values).year)
        if set(years) <= have:
            return d.sel(time=d.time.dt.year.isin(list(years)))
    import gcsfs
    t = xr.open_zarr(gcsfs.GCSFileSystem(token="anon").get_mapper(WB2), consolidated=True)["2m_temperature"]
    t = t.sel(latitude=slice(LAT_MIN, 90)).transpose("time", "latitude", "longitude")
    parts = []
    for y in years:
        t0 = time.time()
        parts.append(t.sel(time=str(y)).resample(time="1D").mean().astype("float32").load())
        print(f"  {y}: {len(parts[-1].time)} days  {time.time() - t0:.0f}s", flush=True)
    d = xr.concat(parts, "time").rename("t2m")
    d.to_netcdf(out, encoding={"t2m": {"zlib": True, "complevel": 4}})
    return d


def spread_by_month(daily):
    """(12, lat, lon) pooled within-month standard deviation of daily temperature."""
    t = pd.DatetimeIndex(daily.time.values)
    x = daily.values
    out = np.zeros((12,) + x.shape[1:])
    for m in range(1, 13):
        v = [x[(t.year == y) & (t.month == m)].var(0, ddof=1) for y in sorted(set(t.year[t.month == m]))]
        out[m - 1] = np.sqrt(np.mean(v, 0))
    return out


def check(daily, sigma, cells, years):
    """Per calendar month, error of each method against degree-days from the real daily values."""
    t = pd.DatetimeIndex(daily.time.values)
    x = daily.values.reshape(len(t), -1)[:, cells]
    rows = []
    for y in years:
        for m in range(1, 13):
            d = x[(t.year == y) & (t.month == m)]
            if len(d) == 0:
                continue
            mu, own = d.mean(0), d.std(0, ddof=1)
            true_p = np.clip(d - P.T2M_THRESHOLD, 0, None).mean(0)
            true_f = np.clip(P.T2M_THRESHOLD - d, 0, None).mean(0)
            old_p = np.clip(mu - P.T2M_THRESHOLD, 0, None)
            clim_p, clim_f = P.expected_degrees(mu, sigma.reshape(12, -1)[m - 1, cells])
            own_p, _ = P.expected_degrees(mu, own)
            rows.append(dict(month=m, true_pdd=true_p.mean(),
                             bias_monthly_mean=(old_p - true_p).mean(), rmse_monthly_mean=np.sqrt(((old_p - true_p) ** 2).mean()),
                             bias_spread=(clim_p - true_p).mean(), rmse_spread=np.sqrt(((clim_p - true_p) ** 2).mean()),
                             rmse_own_spread=np.sqrt(((own_p - true_p) ** 2).mean()),
                             fdd_rmse_monthly_mean=np.sqrt(((np.clip(P.T2M_THRESHOLD - mu, 0, None) - true_f) ** 2).mean()),
                             fdd_rmse_spread=np.sqrt(((clim_f - true_f) ** 2).mean())))
    return pd.DataFrame(rows).groupby("month").mean()


def _selfcheck():
    rng = np.random.default_rng(0)
    days = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    sd = np.where(days.month.values < 7, 2.0, 5.0)[:, None, None]
    x = 271.35 + sd * rng.standard_normal((len(days), 1, 4))
    daily = xr.DataArray(x.astype("float32"), dims=("time", "latitude", "longitude"), coords={"time": days})
    s = spread_by_month(daily)
    assert np.allclose(s[:6], 2.0, atol=0.2) and np.allclose(s[6:], 5.0, atol=0.5), s[:, 0, 0]   # ~300 days per month
    tab = check(daily, s, np.arange(4), [2002, 2003])
    assert (tab.rmse_spread < 0.25 * tab.rmse_monthly_mean).all(), tab   # normal days: the formula wins (30-day samples are noisy)
    print("[selfcheck] spread recovered from synthetic daily data; the formula matches normal days")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clim-years", default="1995-2014", help="years the spread climatology is computed from")
    ap.add_argument("--check-years", default="2015-2022", help="years for --check (WeatherBench2 ends Jan 2023)")
    ap.add_argument("--check", action="store_true", help="accuracy report instead of writing the spread")
    ap.add_argument("--daily", default=paths.WB2_DAILY_T2M, help="cache of the daily temperatures")
    ap.add_argument("--aux", default=paths.AUX)
    ap.add_argument("--out", default=paths.T2M_SPREAD)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.selfcheck:
        _selfcheck(); return
    span = lambda s: range(int(s[:4]), int(s[5:]) + 1)
    clim, chk = span(args.clim_years), span(args.check_years)
    years = sorted(set(clim) | (set(chk) if args.check else set()))
    print(f"[wb2] daily 2 m temperature {years[0]}-{years[-1]} -> {args.daily}")
    daily = fetch_daily(years, args.daily)
    sigma = spread_by_month(daily.sel(time=daily.time.dt.year.isin(list(clim))))

    aux = xr.open_dataset(args.aux)
    ocean = aux["mask"].values.astype(bool)
    lon, lat = np.meshgrid(daily.longitude.values, daily.latitude.values)
    idx = P.build_nearest_index(lon, lat, aux["longitude"].values, aux["latitude"].values)   # model cell -> source cell
    if args.check:
        tab = check(daily.sel(time=daily.time.dt.year.isin(list(chk))), sigma, idx.reshape(ocean.shape)[ocean], chk)
        pd.set_option("display.width", 200)
        print(f"\nDaily positive degrees (K per day) over the model's ocean, {args.check_years}, against the real daily "
              f"values; spread from {args.clim_years}:\n{tab.round(3).to_string()}")
        tot = tab.true_pdd.sum()
        print(f"\nannual melt: true {tot:.2f} K day/day-month sum; monthly-mean method misses "
              f"{-tab.bias_monthly_mean.sum() / tot:.0%}; the formula is off by {tab.bias_spread.sum() / tot:+.0%}")
        return
    field = sigma.reshape(12, -1)[:, idx].reshape((12,) + ocean.shape).astype("float32")
    xr.Dataset({"t2m_daily_std": (("month", "y", "x"), field)},
               coords={"month": np.arange(1, 13)},
               attrs={"source": f"WeatherBench2 ERA5 6-hourly 1.5 deg, daily means, {args.clim_years}",
                      "description": "pooled within-month standard deviation of daily-mean 2 m temperature (K)"}
               ).to_netcdf(args.out)
    print(f"[spread] {args.out}  (K, by calendar month; ocean mean "
          + " ".join(f"{m + 1}:{field[m][ocean].mean():.1f}" for m in range(12)) + ")")


if __name__ == "__main__":
    main()
