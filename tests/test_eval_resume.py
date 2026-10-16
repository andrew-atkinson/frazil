"""eval_monthly's one-month skill resumes after an interruption with exactly the uninterrupted result."""
import importlib.util
import os

import numpy as np
import pytest
import torch
import xarray as xr

from conftest import ROOT
from frazil import STATES, DEGREE

spec = importlib.util.spec_from_file_location("eval_monthly", os.path.join(ROOT, "scripts", "evaluate", "eval_monthly.py"))
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


class FakeModel:
    """Persistence plus random noise, so seeding matters; can fail after a number of calls."""
    forcing_names = ["tus"]

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def __call__(self, states, forc, resolution, mesh, mask, degree_days):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt
        return states[:, 0] + 0.1 * torch.randn_like(states[:, 0])


def data(T=6, H=8, W=8):
    rng = np.random.default_rng(0)
    names = STATES + ["tus"] + DEGREE
    cube = xr.DataArray(rng.random((T, len(names), H, W)), dims=("time", "var_names", "y", "x"),
                        coords={"var_names": names})
    aux = xr.Dataset({"mask": (("y", "x"), np.ones((H, W), int)),
                      "x_coord": (("y", "x"), rng.random((H, W))), "y_coord": (("y", "x"), rng.random((H, W)))})
    return cube, aux


def test_resume_matches_uninterrupted_run(tmp_path):
    cube, aux = data()
    cpu = torch.device("cpu")
    full = em.eval_skill(FakeModel(), cube, aux, cpu, progress=False, n_ens=2)
    state = str(tmp_path / "progress.json")
    with pytest.raises(KeyboardInterrupt):
        em.eval_skill(FakeModel(fail_after=2), cube, aux, cpu, progress=False, n_ens=2, state_file=state)
    assert os.path.exists(state)
    model = FakeModel()
    resumed = em.eval_skill(model, cube, aux, cpu, progress=False, n_ens=2, state_file=state)
    assert model.calls == 3                      # 5 month pairs, 2 already done
    assert resumed == full
    assert not os.path.exists(state)             # cleaned up when complete
