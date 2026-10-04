"""End-to-end checks that need data and trained models (pytest --slow).
These cover the paths the fast tier can't: a real model, real data, real output."""
import glob
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import ROOT, run

pytestmark = pytest.mark.slow
MODEL = os.path.join(ROOT, "data/models/monthly_pf_qm/last.ckpt")
HINDCAST = os.path.join(ROOT, "plots/freerun/snapshots/monthly_pf_qm_era5_2018")


def need(*paths):
    missing = [p for p in paths if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        pytest.skip(f"missing {missing}")


def test_freerun_with_map_and_members(tmp_path):
    """A one-year free-run with every output on: diagnostics, final map, snapshots, members."""
    need(MODEL, "data/train_data/monthly_datacube")
    run(["scripts/project/freerun_monthly.py", "--ckpt", MODEL, "--forcing", "era5", "--start", "2016-01",
         "--years", "1", "--n-ens", "2", "--fast", "--snapshot-every", "12", "--save-members", "sic", "sit",
         "--no-log", "--out-dir", str(tmp_path)])
    assert glob.glob(f"{tmp_path}/freerun_*_map.png"), "final-state map not drawn"
    csv = pd.read_csv(glob.glob(f"{tmp_path}/freerun_*.csv")[0])
    assert (csv.n_nonfinite == 0).all() and len(csv) == 13
    mem = xr.open_dataset(glob.glob(f"{tmp_path}/snapshots/*/members_*.nc")[0])
    assert set(mem.data_vars) == {"sic", "sit"} and mem.sizes["member"] == 2


def test_forcing_rebuild_is_bit_identical(tmp_path):
    """Preprocessing through frazil.preprocess reproduces the training forcing exactly."""
    need("data/train_data/data_stream-moda_stepType-avgua.nc", "data/train_data/monthly_datacube/monthly_datacube_2018.nc")
    run(["scripts/data/preprocess_monthly.py", "--forcings-only", "--years", "2018", "--out-dir", str(tmp_path), "--overwrite"])
    a = xr.open_dataset(f"{tmp_path}/era5_datacube_2018.nc")["datacube"]
    b = xr.open_dataset(os.path.join(ROOT, "data/train_data/monthly_datacube/monthly_datacube_2018.nc"))["datacube"]
    for v in [str(x) for x in a.var_names.values]:
        assert np.array_equal(a.sel(var_names=v).values, b.sel(var_names=v).values, equal_nan=True), v


def test_zero_init_expansion_reproduces_the_seed_model():
    """A model widened by one forcing channel gives exactly the seed model's output."""
    need(MODEL, "data/train_data/monthly_datacube/monthly_datacube_2016.nc")
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import frazil.models as E
    from frazil import paths
    from gensim.utils import expand_forcing_channels
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    A = E.load_model(MODEL, os.path.join(ROOT, paths.CONFIG_FORECAST), dev, os.path.join(ROOT, paths.CONFIG_TRAIN))
    cfg = OmegaConf.load(os.path.join(ROOT, paths.CONFIG_FORECAST))
    cfg.surrogate.network = OmegaConf.load(os.path.join(os.path.dirname(MODEL), ".hydra", "config.yaml")).surrogate.network
    cfg.surrogate.network.n_input = 26
    sd = torch.load(MODEL, map_location="cpu", weights_only=False)["state_dict"]
    m, s = sd["encoder.mean"].flatten().tolist(), sd["encoder.std"].flatten().tolist()
    cfg.surrogate.encoder.mean = m[:10] + [60.0] + m[10:14] + [60.0] + m[14:]
    cfg.surrogate.encoder.std = s[:10] + [20.0] + s[10:14] + [20.0] + s[14:]
    B = instantiate(cfg.surrogate)
    net = {k[len("ema_model.module."):]: v for k, v in sd.items() if k.startswith("ema_model.module.")}
    B.network.load_state_dict(expand_forcing_channels(net, B.network.state_dict(), patch_size=2), strict=True)
    B = B.to(dev); B.set_inference_model(compile_model=False); B.eval()
    cube = xr.open_dataset(os.path.join(ROOT, "data/train_data/monthly_datacube/monthly_datacube_2016.nc"))["datacube"]
    aux = xr.open_dataset(os.path.join(ROOT, paths.AUX))
    a = lambda t, names: np.nan_to_num(cube.isel(time=t).sel(var_names=names).values)
    T = lambda x: torch.as_tensor(x[None], dtype=torch.float32, device=dev)
    kw = dict(resolution=torch.full((1, 1), 12.5, device=dev), mesh=T(aux[["x_coord", "y_coord"]].to_dataarray("c").values / 1000),
              mask=T(aux["mask"].values[None]), degree_days=T(a(5, E.DEGREE)))
    st = T(a(5, E.STATES))[:, None]
    def go(model, names):
        torch.manual_seed(0)
        with torch.no_grad():
            return model(st, T(np.stack([a(5, names), a(6, names)])), **kw).cpu().numpy()
    assert np.array_equal(go(A, E.FORCINGS), go(B, E.FORCINGS + ["rhus"]))


@pytest.mark.parametrize("summary", ["mean", "pmm", "member"])
def test_animation_frame_types(summary, tmp_path):
    need(HINDCAST)
    run(["scripts/project/animate_snapshots.py", "--snap-dir", HINDCAST, "--var", "sic", "--month", "9",
         "--summary", summary, "--out", str(tmp_path / f"{summary}.gif")])
    assert (tmp_path / f"{summary}.gif").stat().st_size > 0


def test_report_from_another_folder(tmp_path):
    need("results/experiments.jsonl")
    run([os.path.join(ROOT, "scripts/evaluate/results_report.py")], cwd=tmp_path)
