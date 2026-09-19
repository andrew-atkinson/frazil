#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch gridded observed sea-ice concentration (NSIDC-0051 v2) via NASA Earthdata,
to bias-correct GenSIM's neXtSIM `sic` training targets against observations.

NSIDC-0051 = passive-microwave SIC on the 25 km NH polar-stereographic grid,
1978-2025, daily and monthly. We pull the MONTHLY fields to match the monthly
model, then a later step regrids them and computes the neXtSIM-vs-obs bias.

CREDENTIALS (never put these in the repo or on the command line -- they are live
secrets that would be committed / leak into shell history + `ps`). earthaccess
auto-discovers, in order:
  1. env var  EARTHDATA_TOKEN        (a bearer token; expires ~60-90 days)
  2. env vars EARTHDATA_USERNAME + EARTHDATA_PASSWORD
  3. ~/.netrc  ->  machine urs.earthdata.nasa.gov login <user> password <pass>
                   (chmod 600; does not expire -- best for a pipeline)
  4. interactive prompt
Set ONE of them outside the repo (e.g. `export EARTHDATA_TOKEN=...` in ~/.zshrc
or a chmod-600 ~/.edl you `source`). No EULA is required for NSIDC-0051, only the
login; the first download may show a one-click app-authorization.

Run:  python download/nsidc0051.py --list                 # search + show granule names
      python download/nsidc0051.py --start-year 1995 --end-year 2018
"""
from __future__ import annotations

import argparse
import os
import re
import sys

SHORT_NAME = "NSIDC-0051"
VERSION = "2"
# Monthly granule dates are 6 digits (YYYYMM) before the version; daily are 8.
DATE_TOK = re.compile(r"_(\d{6,8})_v", re.IGNORECASE)


def granule_name(g):
    """Best-effort filename for a granule search result."""
    try:
        for link in g.data_links():
            return link.rsplit("/", 1)[-1]
    except Exception:
        pass
    return str(g)


def is_monthly(name):
    m = DATE_TOK.search(name)
    return bool(m) and len(m.group(1)) == 6


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=1995,
                    help="neXtSIM starts 1995 (the target-correction overlap needs 1995-2018)")
    ap.add_argument("--end-year", type=int, default=2025,
                    help="NSIDC-0051 runs to 2025; 2019-2025 is obs for validating the projection")
    ap.add_argument("--out-dir", default="data/obs/nsidc0051")
    ap.add_argument("--daily", action="store_true", help="fetch daily instead of monthly")
    ap.add_argument("--hemisphere", choices=["north", "south", "both"], default="north",
                    help="NSIDC-0051 ships both hemispheres; Arctic work wants north")
    ap.add_argument("--list", action="store_true", help="search + print granule names, don't download")
    args = ap.parse_args(argv)

    try:
        import earthaccess
    except ImportError:
        raise SystemExit("earthaccess not installed -- run: pip install earthaccess")

    auth = earthaccess.login()  # tries EARTHDATA_TOKEN, env user/pass, ~/.netrc, interactive
    if not getattr(auth, "authenticated", False):
        raise SystemExit(
            "Earthdata login failed. Set EARTHDATA_TOKEN (or EARTHDATA_USERNAME/"
            "PASSWORD, or ~/.netrc) -- see this file's header. Never hardcode it.")

    print(f"[nsidc0051] searching {SHORT_NAME} v{VERSION} "
          f"{args.start_year}-01..{args.end_year}-12", flush=True)
    results = earthaccess.search_data(
        short_name=SHORT_NAME, version=VERSION,
        temporal=(f"{args.start_year}-01-01", f"{args.end_year}-12-31"))
    if not results:
        raise SystemExit("no granules found -- check the year range / dataset availability")

    want_monthly = not args.daily
    hem_tok = {"north": "_N25km_", "south": "_S25km_"}.get(args.hemisphere)
    keep = [g for g in results
            if is_monthly(granule_name(g)) == want_monthly
            and (hem_tok is None or hem_tok in granule_name(g))]
    kind = f"{args.hemisphere} {'monthly' if want_monthly else 'daily'}"
    print(f"[nsidc0051] {len(results)} granules found, {len(keep)} {kind}")

    if args.list or not keep:
        for g in keep[:12]:
            print("   ", granule_name(g))
        if len(keep) > 12:
            print(f"    ... +{len(keep)-12} more")
        if not keep:
            print("(no granules matched the monthly/daily filter -- run --list and check names)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    files = earthaccess.download(keep, args.out_dir)
    print(f"[nsidc0051] downloaded {len(files)} {kind} files -> {args.out_dir}/")


if __name__ == "__main__":
    main()
