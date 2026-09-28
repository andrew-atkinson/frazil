"""Load trained models: architecture and forcing list from the run's saved config,
input normalization from the checkpoint. Also re-exports the variable lists and
model_tag, so scripts can `import frazil.models as E` and use E.STATES, E.load_model...
"""
from __future__ import annotations

import glob
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import xarray as xr
from omegaconf import OmegaConf
from hydra.utils import instantiate

import gensim
import gensim.network
from frazil import STATES, FORCINGS, DEGREE, UNITS  # noqa: F401  (re-exported)
from frazil.ledger import model_tag  # noqa: F401  (re-exported)


def load_model(ckpt_path, config, device, train_config):
    cfg = OmegaConf.load(config)
    # Architecture and forcing variables come from the run's own saved config
    # (data/models/<run>/.hydra/config.yaml) when present, else --train-config.
    run_cfg = os.path.join(os.path.dirname(ckpt_path), ".hydra", "config.yaml")
    tcfg = OmegaConf.load(run_cfg if os.path.exists(run_cfg) else train_config)
    cfg.surrogate.network = tcfg.surrogate.network
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    if "encoder.mean" in sd:  # input normalization exactly as trained
        cfg.surrogate.encoder.mean = sd["encoder.mean"].flatten().tolist()
        cfg.surrogate.encoder.std = sd["encoder.std"].flatten().tolist()
    model = instantiate(cfg.surrogate)  # GenSIMForecastModule (recursive)
    model.forcing_names = list(tcfg.data.get("forcing_variables", FORCINGS))
    gensim.network.USE_FLASH_ATTN = (
        gensim.network.USE_FLASH_ATTN and device.type == "cuda")

    # Prefer the EMA weights (the inference-quality copy); fall back to network.*
    prefix = "ema_model.module." if any(k.startswith("ema_model.module.")
                                        for k in sd) else "network."
    net_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.network.load_state_dict(net_sd, strict=False)
    print(f"[ckpt] loaded {len(net_sd)} tensors from '{prefix}*' "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")

    model = model.to(device)
    model.set_inference_model(compile_model=device.type == "cuda")
    model.eval()
    return model


def default_ckpt(ckpt_dir="data/models/monthly"):
    """The canonical monthly.ckpt if present, else the newest last*.ckpt.
    (Lightning's -v rotation makes plain last.ckpt an unreliable name, so we
    consolidate the trained model to monthly.ckpt and prefer that.)"""
    canonical = f"{ckpt_dir}/monthly.ckpt"
    if os.path.exists(canonical):
        return canonical
    cks = glob.glob(f"{ckpt_dir}/last*.ckpt")
    if not cks:
        raise SystemExit(f"no monthly.ckpt or last*.ckpt in {ckpt_dir}")
    return max(cks, key=os.path.getmtime)


def load_val_cube(datacube_dir, val_start_year):
    files = sorted(f for f in glob.glob(f"{datacube_dir}/monthly_datacube_*.nc")
                   if int(f.split("_")[-1].split(".")[0]) >= val_start_year)
    return xr.open_mfdataset(files, combine="by_coords")["datacube"].load()
