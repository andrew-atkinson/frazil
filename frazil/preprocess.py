"""Shared preprocessing: nearest-neighbour regridding onto the model grid, vector
rotation into the grid frame, humidity (Magnus) and degree-day features.
Used by the ERA5, CMIP, NSIDC and extra-channel preprocessing scripts.

Degree-day windows are in MONTHS (AVG_LEN): pdd_month/fdd_month are the current
month's rate and pdd_year/fdd_year a trailing 12-month mean -- a monthly analogue
of GenSIM's daily degree-days (see docs/ISSUES.md ISS-007).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

AVG_LEN = {"month": 1, "year": 12}
T2M_THRESHOLD = 271.35  # K, ~ -1.8 C sea-ice freezing point (same as data_02)

# Magnus / humidity constants (verbatim from GenSIM's data_02).
RD, RV = 287.0597, 461.51
EPSILON = RD / RV
MAGNUS = dict(a1_w=611.21, a3_w=17.502, a4_w=32.19,
              a1_i=611.21, a3_i=22.587, a4_i=-0.7,
              t0=273.16, tice=250.16)


def lonlat_to_xyz(lon_deg, lat_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64).ravel())
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64).ravel())
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def build_nearest_index(src_lon, src_lat, tgt_lon, tgt_lat):
    """Index into the flattened source grid giving the nearest cell per target."""
    tree = cKDTree(lonlat_to_xyz(src_lon, src_lat))
    _, idx = tree.query(lonlat_to_xyz(tgt_lon, tgt_lat), workers=-1)
    return idx


def regrid_time(field_tyx, idx, tgt_shape):
    """Regrid a (time, ys, xs) array to (time, YT, XT) by nearest-neighbour idx."""
    t = field_tyx.shape[0]
    flat = np.asarray(field_tyx, dtype=np.float32).reshape(t, -1)
    return flat[:, idx].reshape(t, *tgt_shape)


def _normalize(ex, ey):
    mag = np.hypot(ex, ey)
    mag = np.where(mag == 0, 1.0, mag)
    return ex / mag, ey / mag


def local_grid_basis(lon2d, lat2d):
    """Orthonormal local i/j index directions of a grid, in (east, north) units.

    Returns (iE, iN, jE, jN) unit vectors shaped like lon2d. i is the +x-index
    direction; j is orthonormalised (Gram-Schmidt) to be exactly perpendicular
    to i, oriented to match the raw +y-index direction. These grids are
    conformal (orthogonal) to ~0.3% of cells, so this is exact almost
    everywhere and stable -- no near-singular blow-ups -- at the pole/seam.
    """
    lon = np.asarray(lon2d, dtype=np.float64)
    lat = np.asarray(lat2d, dtype=np.float64)
    latr = np.deg2rad(lat)

    def dwrap(d):  # wrap longitude differences into [-180, 180]
        return (d + 180.0) % 360.0 - 180.0

    dlon_di = dwrap(np.gradient(lon, axis=1))
    dlat_di = np.gradient(lat, axis=1)
    dlon_dj = dwrap(np.gradient(lon, axis=0))
    dlat_dj = np.gradient(lat, axis=0)

    iE, iN = _normalize(dlon_di * np.cos(latr), dlat_di)
    jE_raw, jN_raw = dlon_dj * np.cos(latr), dlat_dj

    # Gram-Schmidt: j perpendicular to i, sign matched to the raw j direction.
    jE_perp, jN_perp = -iN, iE
    sign = np.sign(jE_perp * jE_raw + jN_perp * jN_raw)
    sign = np.where(sign == 0, 1.0, sign)
    jE, jN = jE_perp * sign, jN_perp * sign
    return iE, iN, jE, jN


def rotate_to_target(a, b, src_basis, tgt_basis, *, label=""):
    """Rotate a vector given in a source grid frame into a target grid frame.

    a, b        : components along the source grid's i, j index directions.
    src_basis   : (iE, iN, jE, jN) orthonormal source basis (None => geographic,
                  i.e. a=east, b=north, as for ERA5 lat-lon winds).
    tgt_basis   : (iE, iN, jE, jN) orthonormal target basis.
    Both bases are orthonormal (see local_grid_basis), so this is a pure
    rotation: forward-combine into geographic east/north, then project onto the
    target axes by dot product. Magnitude is preserved.
    """
    if src_basis is None:
        east, north = a, b
    else:
        siE, siN, sjE, sjN = src_basis
        east = a * siE + b * sjE
        north = a * siN + b * sjN

    tiE, tiN, tjE, tjN = tgt_basis
    ua = east * tiE + north * tiN
    va = east * tjE + north * tjN

    # Sanity: a pure rotation preserves vector magnitude (report median + max).
    src_mag = np.hypot(east, north)
    tgt_mag = np.hypot(ua, va)
    good = np.isfinite(src_mag) & np.isfinite(tgt_mag) & (src_mag > 1e-6)
    if good.any():
        rel = np.abs(tgt_mag[good] - src_mag[good]) / src_mag[good]
        print(f"    rotation[{label}]: median|Δ|mag = {np.nanmedian(rel)*100:.4f}% "
              f"(max {np.nanmax(rel)*100:.3f}%)")
    return ua.astype(np.float32), va.astype(np.float32)


def _sat(temp, a1, a3, a4):
    return a1 * np.exp(a3 * ((temp - MAGNUS["t0"]) / (temp - a4)))


def interp_sat(temp):
    sat_w = _sat(temp, MAGNUS["a1_w"], MAGNUS["a3_w"], MAGNUS["a4_w"])
    sat_i = _sat(temp, MAGNUS["a1_i"], MAGNUS["a3_i"], MAGNUS["a4_i"])
    alpha = ((temp - MAGNUS["tice"]) / (MAGNUS["t0"] - MAGNUS["tice"])) ** 2
    alpha = np.clip(alpha, 0, 1)
    return alpha * sat_w + (1 - alpha) * sat_i


def specific_humidity(d2m, pressure):
    sat_d2m = interp_sat(d2m)
    return sat_d2m * EPSILON / (pressure + sat_d2m * (EPSILON - 1))


def relative_humidity(t2m, d2m):
    return interp_sat(d2m) / interp_sat(t2m) * 100.0


def degree_days(t2m_tyx, times, avg_len=AVG_LEN, threshold=T2M_THRESHOLD):
    """Rolling positive/freezing degree-day features (monthly windows)."""
    da = xr.DataArray(t2m_tyx, dims=("time", "y", "x"),
                      coords={"time": times})
    diff = da - threshold
    pdd = diff.clip(min=0)
    fdd = (-diff).clip(min=0)
    out = {}
    for suffix, win in avg_len.items():
        out[f"pdd_{suffix}"] = pdd.rolling(
            time=win, min_periods=win, center=False).mean("time")
        out[f"fdd_{suffix}"] = fdd.rolling(
            time=win, min_periods=win, center=False).mean("time")
    return out


def ym_key(times):
    idx = pd.DatetimeIndex(np.asarray(times))
    return list(zip(idx.year.tolist(), idx.month.tolist()))
