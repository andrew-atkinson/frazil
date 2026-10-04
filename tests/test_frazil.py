"""Unit tests for the frazil package (no data needed)."""
import os

import numpy as np
import pytest
import torch

from conftest import ROOT
from frazil.diagnostics import ice_area_km2, small_scale_energy
from frazil.ensemble import lpmm, metrics, pmm
from frazil.ledger import LEDGER, ROOT as LEDGER_ROOT, model_tag
from gensim.utils import expand_forcing_channels


@pytest.mark.parametrize("ckpt,tag", [
    ("data/models/monthly/monthly.ckpt", "monthly"),
    ("data/models/monthly_pf/last.ckpt", "monthly_pf"),
    ("data/models/monthly_pf/last-v1.ckpt", "monthly_pf"),
    ("data/models/monthly_pf_qm/last.ckpt", "monthly_pf_qm"),
    ("data/models/monthly_pf_qm_sst/last.ckpt", "monthly_pf_qm_sst"),
    ("data/models/monthly_qm/last.ckpt", "monthly_qm"),
    ("data/models/monthly_pf/step_20000.ckpt", "monthly_pf_step_20000"),
])
def test_model_tag(ckpt, tag):
    assert model_tag(ckpt) == tag


def test_ledger_lives_in_repo_results():
    assert os.path.samefile(LEDGER_ROOT, ROOT)
    assert LEDGER == os.path.join(ROOT, "results", "experiments.jsonl")


def _edge_members(n=8, size=40, seed=0):
    """n members with a sharp 0/1 ice edge at slightly different columns."""
    rng = np.random.default_rng(seed)
    cols = np.arange(size)[None, :]
    return np.stack([(cols < size // 2 + s) * np.ones((size, 1)) for s in rng.integers(-4, 5, n)]).astype(float)


def test_pmm_keeps_mean_ranking_and_member_values():
    m = _edge_members(); mask = np.ones(m.shape[1:], bool)
    p = pmm(m, mask)
    order = np.argsort(m.mean(0)[mask], kind="stable")
    assert np.all(np.diff(p[mask][order]) >= 0)                                   # the mean's pattern
    assert np.allclose(np.sort(p[mask]), np.sort(m[:, mask].ravel())[len(m) // 2::len(m)])  # members' values
    band = lambda f: ((f > 0.15) & (f < 0.8)).mean()
    assert band(p) < band(m.mean(0))                                              # no smeared edge


def test_lpmm_shape_and_bounds():
    m = _edge_members(size=48); mask = np.ones(m.shape[1:], bool)
    out = lpmm(m, mask, tile=16, halo=8, min_cells=10)
    assert out.shape == mask.shape and out.min() >= 0 and out.max() <= 1


def test_metrics_perfect_field_scores_zero():
    o = _edge_members()[0]; ocean = np.ones(o.shape, bool); ca = np.ones(o.shape)
    s = metrics(o, o, ocean, ca)
    assert s["iiee"] == 0 and s["ext"] == 0 and s["rmse"] == 0 and s["sharp"] == pytest.approx(1)


def test_ice_area_extent_vs_area():
    ocean = np.array([[1, 1], [1, 0]], bool); ca = np.full((2, 2), 10.0)
    sic = np.array([[[0.5, 0.1], [0.9, 1.0]]])
    assert ice_area_km2(sic, ocean, ca, "extent", 0.15)[0] == 20.0
    assert ice_area_km2(sic, ocean, ca, "area", 0.15)[0] == pytest.approx(15.0)


def test_small_scale_energy_smooth_vs_noisy():
    rng = np.random.default_rng(0); ocean = np.ones((64, 64), bool)
    smooth = np.outer(np.linspace(0, 1, 64), np.ones(64))
    assert small_scale_energy(smooth + 0.3 * rng.standard_normal((64, 64)), ocean) > small_scale_energy(smooth, ocean)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_expand_forcing_channels_preserves_output(k):
    """The expanded input layer must give the same output for any values on the new
    channels. Checked against the channel layout as documented, independently of the
    function's index arithmetic: [noisy 6][states 6][forcings(t) f][forcings(t+1) f][degree days 4][ones]."""
    torch.manual_seed(0)
    p2, F, f_old = 4, 16, 4
    f_new = f_old + k
    c_old, c_new = 12 + 2 * f_old + 4 + 1, 12 + 2 * f_new + 4 + 1
    key = "network.tokenizer.in_encoder.weight"
    w_old = torch.randn(F, c_old * p2)
    out = expand_forcing_channels({key: w_old, "encoder.mean": torch.zeros(18, 1, 1)},
                                  {key: torch.empty(F, c_new * p2), "encoder.mean": torch.zeros(18 + 2 * k, 1, 1)},
                                  patch_size=2)
    assert "encoder.mean" not in out                       # new stats come from the config
    x_old = torch.randn(c_old, p2)
    x_new = torch.randn(c_new, p2)                         # new channels: arbitrary values
    x_new[:12 + f_old] = x_old[:12 + f_old]                                 # states + forcings(t)
    x_new[12 + f_new:12 + f_new + f_old] = x_old[12 + f_old:12 + 2 * f_old]  # forcings(t+1)
    x_new[12 + 2 * f_new:] = x_old[12 + 2 * f_old:]                         # degree days + ones
    y_old = w_old @ x_old.reshape(-1)
    y_new = out[key] @ x_new.reshape(-1)
    assert torch.allclose(y_old, y_new, atol=1e-5)


def test_expand_forcing_channels_rejects_other_shape_changes():
    with pytest.raises(ValueError):
        expand_forcing_channels({"network.head.weight": torch.zeros(3, 3)},
                                {"network.head.weight": torch.zeros(4, 3)}, patch_size=2)
