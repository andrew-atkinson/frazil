"""Unit tests for frazil.preprocess, frazil.io, frazil.ledger and frazil.models on
small synthetic inputs (no data needed). The slow tier checks the same code on the
real data end to end."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import frazil.ledger as ledger
from frazil import STATES
from frazil.io import load_nsidc, load_series, load_sic, save_state, snapshot_date
from frazil.preprocess import (build_nearest_index, degree_days, local_grid_basis, regrid_time,
                               relative_humidity, rotate_to_target, specific_humidity, ym_key)


# ---- preprocess -------------------------------------------------------------

def test_regrid_picks_the_nearest_source_cell():
    lon1d, lat1d = np.arange(0, 40, 10.0), np.array([60.0, 70.0, 80.0])
    src_lon, src_lat = np.meshgrid(lon1d, lat1d)
    tgt_lon, tgt_lat = np.array([[10.4, 29.6]]), np.array([[70.3, 79.8]])   # next to (10,70), (30,80)
    idx = build_nearest_index(src_lon, src_lat, tgt_lon, tgt_lat)
    field = np.arange(src_lon.size, dtype=float).reshape(1, *src_lon.shape)
    out = regrid_time(field, idx, tgt_lon.shape)
    assert out[0, 0, 0] == field[0, 1, 1] and out[0, 0, 1] == field[0, 2, 3]


def test_wind_rotation_preserves_speed():
    lon, lat = np.meshgrid(np.linspace(-170, 170, 12), np.linspace(60, 85, 8))
    basis = local_grid_basis(lon, lat)
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=lon.shape), rng.normal(size=lon.shape)
    ua, va = rotate_to_target(u, v, src_basis=None, tgt_basis=basis, label="test")
    assert np.allclose(np.hypot(ua, va), np.hypot(u, v), rtol=1e-4)


def test_humidity():
    assert relative_humidity(np.array(270.0), np.array(270.0)) == pytest.approx(100.0)
    q = specific_humidity(np.array([250.0, 270.0]), np.array(101325.0))
    assert 0 < q[0] < q[1] < 0.01                      # more moisture at a higher dewpoint, sane magnitude


def test_degree_days_monthly_rate_and_trailing_year():
    times = pd.date_range("2000-01-15", periods=14, freq="MS")
    thr = 271.35
    t2m = np.full((14, 1, 1), thr - 10.0); t2m[5:8] = thr + 4.0          # three warm months
    dd = degree_days(t2m, times)
    pdd_m, fdd_m = dd["pdd_month"].values[:, 0, 0], dd["fdd_month"].values[:, 0, 0]
    assert np.allclose(pdd_m[5:8], 4.0) and np.allclose(pdd_m[:5], 0) and np.allclose(fdd_m[:5], 10.0)
    pdd_y = dd["pdd_year"].values[:, 0, 0]
    assert np.isnan(pdd_y[:11]).all()                                     # needs 12 months of spin-up
    assert pdd_y[11] == pytest.approx(3 * 4.0 / 12)


def test_ym_key():
    assert ym_key(pd.to_datetime(["2019-01-15", "2020-12-15"])) == [(2019, 1), (2020, 12)]


# ---- io ---------------------------------------------------------------------

def test_save_state_roundtrip_and_snapshot_date(tmp_path):
    sm = np.random.default_rng(0).random((len(STATES), 4, 5)).astype(np.float32)
    path = tmp_path / "state_202509.nc"
    save_state(sm, pd.Timestamp("2025-09-01"), str(path))
    st = xr.open_dataset(path)["state"]
    assert list(st.var_names.values) == STATES and np.array_equal(st.isel(time=0).values, sm)
    assert snapshot_date(str(path)) == pd.Timestamp("2025-09-15")
    assert snapshot_date("members_202509.nc") is None


def _cube(path, year, names, value):
    t = pd.date_range(f"{year}-01-15", periods=12, freq="MS") + pd.Timedelta(days=14)
    data = np.full((12, len(names), 3, 3), value, np.float32); data[0, 0, 0, 0] = np.nan
    xr.DataArray(data, dims=("time", "var_names", "y", "x"), coords={"time": t, "var_names": names},
                 name="datacube").to_dataset().to_netcdf(path)


def test_load_series_concatenates_years_in_order(tmp_path):
    names = ["sic", "tus"]
    _cube(tmp_path / "monthly_datacube_2001.nc", 2001, names, 2.0)
    _cube(tmp_path / "monthly_datacube_2000.nc", 2000, names, 1.0)
    arr, times = load_series(str(tmp_path), ["tus"])
    assert arr.shape == (24, 1, 3, 3) and times.is_monotonic_increasing
    assert arr[0, 0, 1, 1] == 1.0 and arr[-1, 0, 1, 1] == 2.0 and np.isfinite(arr).all()   # NaN -> 0
    sic, t = load_sic(str(tmp_path))
    assert sic.shape == (24, 3, 3) and t[0].year == 2000


def test_load_nsidc_reads_the_requested_column(tmp_path):
    (tmp_path / "N_09_extent_v4.0.csv").write_text(
        "year, mo, source, region, extent, area\n2024, 9, X, N, 4.35, 2.91\n1988, 9, X, N, -9999, -9999\n")
    assert load_nsidc(str(tmp_path), 9) == {2024: 4.35e6}
    assert load_nsidc(str(tmp_path), 9, "area") == {2024: 2.91e6}


# ---- ledger -----------------------------------------------------------------

def test_ledger_writes_deterministic_records(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", str(tmp_path / "experiments.jsonl"))
    monkeypatch.setattr(ledger, "git_state", lambda archive=True: {"commit": "abc", "dirty": False})
    rec = lambda: {"type": "result", "experiment": "t", "params": {"x": 0.123456789},
                   "metrics": {}, "timestamp": "2026-01-01T00:00:00Z"}
    a, b = ledger.append(rec()), ledger.append(rec())
    lines = (tmp_path / "experiments.jsonl").read_text().splitlines()
    assert a == b and len(lines) == 2 and json.loads(lines[0])["run_id"] == a      # same record, same id
    assert ledger._round({"v": [0.123456789, 2]}) == {"v": [0.1235, 2]}


# ---- models -----------------------------------------------------------------

def test_default_ckpt_prefers_monthly_then_newest_last(tmp_path):
    from frazil.models import default_ckpt
    (tmp_path / "last.ckpt").write_text("a"); os.utime(tmp_path / "last.ckpt", (1, 1))
    (tmp_path / "last-v1.ckpt").write_text("b")
    assert default_ckpt(str(tmp_path)).endswith("last-v1.ckpt")
    (tmp_path / "monthly.ckpt").write_text("c")
    assert default_ckpt(str(tmp_path)).endswith("monthly.ckpt")
