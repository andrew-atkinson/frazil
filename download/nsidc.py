#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch NSIDC Sea Ice Index monthly extent CSVs (NOAA@NSIDC G02135, v4.0) as an
OBSERVATIONAL baseline for validating GenSIM past the 2018 neXtSIM window.

Monthly extent = ocean area with sea-ice concentration >= 15%, in 10^6 km^2,
Nov-1978 to present. One CSV per calendar month (e.g. September = month 9). Open
access, no login.

Source: https://noaadata.apps.nsidc.org/NOAA/G02135/   (dataset g02135 v4)

Run:  python download/nsidc.py                 # September, North -> data/obs/nsidc/
      python download/nsidc.py --month 0       # all 12 months
"""
from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request

BASE = ("https://noaadata.apps.nsidc.org/NOAA/G02135/"
        "{hem}/monthly/data/{H}_{mm:02d}_extent_v4.0.csv")


def fetch(month, hem, out_dir):
    H = "N" if hem == "north" else "S"
    url = BASE.format(hem=hem, H=H, mm=month)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{H}_{month:02d}_extent_v4.0.csv")
    try:
        urllib.request.urlretrieve(url, out)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"NSIDC {url} -> HTTP {e.code}. Check month/version at "
                         "https://nsidc.org/data/g02135/versions/4")
    except urllib.error.URLError as e:
        raise SystemExit(f"NSIDC fetch failed ({e.reason}). Network/URL issue.")
    n = max(0, sum(1 for _ in open(out)) - 1)
    print(f"  {H}_{month:02d}: {n} years -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", type=int, default=9, help="1-12, or 0 for all 12")
    ap.add_argument("--hemisphere", choices=["north", "south"], default="north")
    ap.add_argument("--out-dir", default="data/obs/nsidc")
    args = ap.parse_args(argv)

    months = range(1, 13) if args.month == 0 else [args.month]
    print(f"[nsidc] Sea Ice Index v4.0 monthly extent, {args.hemisphere}")
    for m in months:
        fetch(m, args.hemisphere, args.out_dir)


if __name__ == "__main__":
    main()
