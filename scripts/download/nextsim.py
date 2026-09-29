#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download monthly-mean neXtSIM-OPA sea-ice fields (the 6 GenSIM target
variables) from the SASIP THREDDS/OPeNDAP server at IGE-MEOM Grenoble.

This is the simulation GenSIM was trained on:
    OPA-neXtSIM_CREG025-ILBOXE140   (coupled neXtSIM sea-ice + OPA ocean,
    ERA5-forced).  The data are served as per-file NetCDF over OPeNDAP -- NOT
    as a Zarr store -- so we open each monthly file with xarray and pull only
    the variables we need.

Key facts (verified against the server, Aug 2026):
  * Temporal coverage: 1995-2018 only. Earlier/later years do not exist.
  * Monthly means live under .../OPA-neXtSIM_CREG025-ILBOXE140-MEAN/1m/<year>/
  * All 6 GenSIM targets are in the "*_icemod.nc" files:
        sit, sic, damage(->sid), siu, siv, snt
    (the "*_simba.nc" thermodynamic-tendency files are NOT needed).
  * Grid: native CREG025 curvilinear, 603 x 528, with 2-D latitude/longitude
    over (y, x). It is therefore NOT lon/lat-sliceable, and it is a DIFFERENT
    grid from GenSIM's 512x512 North-Polar-Stereographic grid. Regridding +
    wind rotation onto the GenSIM grid is a separate preprocessing step (see
    notebooks/data/data_02_nextsim_to_zarr.ipynb in the gensim repo); this
    script only fetches the raw native-grid targets.

Output: one NetCDF per year (resumable -- existing years are skipped), each
holding the 6 renamed target variables with a monthly time axis.

Examples
--------
Full 1995-2018 download::

    python download.py

A quick two-year test::

    python download.py --start-year 2015 --end-year 2016

Inspect one file without downloading::

    python download.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from frazil import paths

# --------------------------------------------------------------------------- #
# Server / dataset layout
# --------------------------------------------------------------------------- #
SERVER = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/dodsC"
RUN_ROOT = (
    "meomopendap/extract/SASIP/model-outputs/OPA-neXtSIM_CREG025/"
    "OPA-neXtSIM_CREG025-ILBOXE140-MEAN/1m"
)
RUN_PREFIX = "OPA-neXtSIM_CREG025-ILBOXE140"

# Data coverage on the server (hard limits -- nothing exists outside this).
DATA_FIRST_YEAR = 1995
DATA_LAST_YEAR = 2018

# GenSIM's 6 target state variables, in the order the model expects them,
# mapped from the source variable name in the icemod files.
TARGET_MAP = {
    "sit": "sit",      # Sea Ice Thickness [m]
    "sic": "sic",      # Sea Ice Concentration [1]
    "sid": "damage",   # Sea Ice Damage [1]  (renamed from "damage")
    "siu": "siu",      # Sea Ice X Velocity [m/s]
    "siv": "siv",      # Sea Ice Y Velocity [m/s]
    "snt": "snt",      # Surface Snow Thickness [m]
}


def opendap_url(year: int, month: int, kind: str = "icemod") -> str:
    """Build the OPeNDAP URL for one monthly file."""
    fname = f"{RUN_PREFIX}_y{year:04d}m{month:02d}.1m_{kind}.nc"
    return f"{SERVER}/{RUN_ROOT}/{year:04d}/{fname}"


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def open_month(year: int, month: int) -> xr.Dataset | None:
    """Open one monthly icemod file over OPeNDAP and return the 6 targets.

    Returns None (with a warning) if the file cannot be opened, so a single
    missing/late month does not abort the whole year.
    """
    url = opendap_url(year, month, "icemod")
    try:
        ds = xr.open_dataset(url)
    except Exception as exc:  # noqa: BLE001 - network/format errors vary
        print(f"    ! {year}-{month:02d}: could not open "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)
        return None

    missing = [src for src in TARGET_MAP.values() if src not in ds.variables]
    if missing:
        print(f"    ! {year}-{month:02d}: missing variables {missing} -- skipped",
              file=sys.stderr)
        ds.close()
        return None

    # Select the 6 source vars (this is what limits the OPeNDAP transfer),
    # then rename to GenSIM's names and keep the 2-D lat/lon as coordinates.
    src_vars = list(TARGET_MAP.values())
    sub = ds[src_vars].rename({v: k for k, v in TARGET_MAP.items()})
    for coord in ("latitude", "longitude"):
        if coord in ds.variables:
            sub = sub.assign_coords({coord: ds[coord]})
    sub.load()  # pull data now so we can close the remote handle
    ds.close()
    return sub


def download_year(year: int, out_dir: Path, overwrite: bool) -> Path | None:
    out_path = out_dir / f"nextsim_opa_icemod_{year:04d}_monthly.nc"
    if out_path.exists() and not overwrite:
        print(f"[{year}] already present -> {out_path.name} (skip)")
        return out_path

    print(f"[{year}] fetching 12 monthly files over OPeNDAP ...")
    months = []
    for m in range(1, 13):
        ds_m = open_month(year, m)
        if ds_m is not None:
            months.append(ds_m)
            print(f"    ok {year}-{m:02d}")

    if not months:
        print(f"[{year}] no data retrieved -- nothing written", file=sys.stderr)
        return None

    year_ds = xr.concat(months, dim="time").sortby("time")

    # Drop any carried-over bounds variable and reset inherited metadata: the
    # source files include reserved/duplicate global attributes (and a
    # time-bounds reference) that the netCDF4 writer rejects with
    # "String match to name in use". Start clean with only our own attrs and
    # no stale encoding.
    year_ds = year_ds.drop_vars(
        [v for v in ("time_bnds", "nv") if v in year_ds.variables]
    )
    for v in list(year_ds.variables):
        year_ds[v].encoding.clear()
        year_ds[v].attrs.pop("bounds", None)
    year_ds.attrs = {
        "title": "neXtSIM-OPA CREG025-ILBOXE140 monthly-mean sea-ice targets",
        "source": f"{SERVER}/{RUN_ROOT}",
        "note": ("Native curvilinear CREG025 grid (603x528). Regrid + wind "
                 "rotation onto the GenSIM 512x512 NPS grid is a separate step."),
        "variables": "sit, sic, sid(=damage), siu, siv, snt",
    }
    encoding = {v: {"zlib": True, "complevel": 4} for v in TARGET_MAP}
    year_ds.to_netcdf(out_path, encoding=encoding)
    print(f"[{year}] wrote {len(months)} months -> {out_path}")
    return out_path


def preflight() -> None:
    """Fail fast with a clear message if no OPeNDAP-capable backend is present.

    Opening the remote NetCDF files requires either netCDF4 (built with DAP
    support, as in the `gensim` conda env) or pydap. A bare xarray install
    (e.g. a fresh venv) has neither and produces a confusing per-file
    "dependencies may not be installed" error for every month.
    """
    import importlib.util

    have_netcdf4 = importlib.util.find_spec("netCDF4") is not None
    have_pydap = importlib.util.find_spec("pydap") is not None
    if have_netcdf4 or have_pydap:
        return
    sys.exit(
        "No OPeNDAP-capable xarray backend found (neither 'netCDF4' nor "
        "'pydap' is importable in this Python).\n"
        "Fix it one of these ways:\n"
        "  * run inside the gensim conda env:  conda activate gensim\n"
        f"  * or install a backend here:  {sys.executable} -m pip install netCDF4 pydap"
    )


def dry_run() -> None:
    """Open a single known-good file and print its structure."""
    url = opendap_url(2015, 12, "icemod")
    print("Dry run -- opening:", url)
    ds = xr.open_dataset(url)
    print("dims:", dict(ds.sizes))
    print("\nTarget variable mapping (gensim <- source):")
    for dst, src in TARGET_MAP.items():
        if src in ds.variables:
            a = ds[src]
            print(f"  {dst:4s} <- {src:8s}  '{a.attrs.get('long_name','')}'"
                  f"  [{a.attrs.get('units','')}]")
        else:
            print(f"  {dst:4s} <- {src:8s}  MISSING")
    ds.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=DATA_FIRST_YEAR)
    p.add_argument("--end-year", type=int, default=DATA_LAST_YEAR)
    p.add_argument("--out-dir", default=paths.NEXTSIM_RAW)
    p.add_argument("--overwrite", action="store_true",
                   help="re-download years even if the output file exists")
    p.add_argument("--dry-run", action="store_true",
                   help="open one file and print its variables, then exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    preflight()
    if args.dry_run:
        dry_run()
        return

    lo = max(args.start_year, DATA_FIRST_YEAR)
    hi = min(args.end_year, DATA_LAST_YEAR)
    if lo > hi:
        sys.exit(f"Requested {args.start_year}-{args.end_year} is entirely "
                 f"outside the available range {DATA_FIRST_YEAR}-{DATA_LAST_YEAR}.")
    if (args.start_year, args.end_year) != (lo, hi):
        print(f"Note: clamped request to available coverage {lo}-{hi} "
              f"(server has {DATA_FIRST_YEAR}-{DATA_LAST_YEAR}).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading neXtSIM-OPA monthly sea-ice targets {lo}-{hi} "
          f"-> {out_dir}/\n")

    written = []
    for year in range(lo, hi + 1):
        path = download_year(year, out_dir, args.overwrite)
        if path is not None:
            written.append(path)

    print(f"\nDone. {len(written)} yearly file(s) in {out_dir}/")
    print("Open them all with:  xarray.open_mfdataset("
          f"'{out_dir}/nextsim_opa_icemod_*_monthly.nc')")


if __name__ == "__main__":
    main()
