#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training smoke test for the monthly GenSIM config.

Instantiates GenSIMTrainModule from config_train_monthly.yaml and pushes one
synthetic batch through the core loss path (estimate_loss + backward). The goal
is to catch config / shape / normalization-dimension errors *before* committing
to a real training run -- not to train anything.

Why synthetic + small grid:
  * The real NeXtSIMDataset expects a single zarr `datacube` store; our monthly
    data is per-year NetCDF (converting it is a separate step).
  * The full 512x512 grid would blow up self-attention (O(N^2) tokens); training
    normally patches to 64x64. estimate_loss runs the network at the batch's
    spatial size, so a 64x64 synthetic grid exercises the same code cheaply.

What this verifies:
  * config_train_monthly.yaml instantiates (network, encoder w/ 18-dim stats,
    decoder w/ 6-dim stats, sampler, log-scale model).
  * The 18-channel encoder input and 6-channel decoder tendency line up.
  * A forward pass, flow-matching loss, and backward all produce finite values.

Run:  python experiments/smoke_train_monthly.py
"""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from gensim.utils import get_empty_labels


def build_synthetic_batch(B=1, T=2, H=64, W=64):
    def rnd(*s, lo=0.0, hi=1.0):
        return (torch.rand(*s) * (hi - lo) + lo).float()

    # states: (B, T, 6, H, W) in roughly physical ranges (sit, sic, sid, siu, siv, snt)
    states = torch.cat([
        rnd(B, T, 1, H, W, lo=0, hi=3),
        rnd(B, T, 1, H, W, lo=0, hi=1),
        rnd(B, T, 1, H, W, lo=0, hi=1),
        rnd(B, T, 1, H, W, lo=-.3, hi=.3),
        rnd(B, T, 1, H, W, lo=-.3, hi=.3),
        rnd(B, T, 1, H, W, lo=0, hi=.5),
    ], dim=2)
    forcings = rnd(B, T, 4, H, W, lo=-2, hi=2)     # tus, huss, uas, vas (t and t+1)
    degree_days = rnd(B, 4, H, W, lo=0, hi=10)      # pdd/fdd month/year
    yy, xx = torch.meshgrid(torch.linspace(-3000, 3000, H),
                            torch.linspace(-3000, 3000, W), indexing="ij")
    mesh = torch.stack([xx, yy], 0)[None].float()   # (B, 2, H, W) in km
    mask = torch.ones(B, 1, H, W).float()
    resolution = torch.full((B, 1), 12.5)
    return dict(states=states, forcings=forcings, degree_days=degree_days,
                mesh=mesh, mask=mask), resolution


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_train_monthly.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    # patch_generator/train_augmentation are only used by training_step (not
    # estimate_loss); patch_generator also references surrogate.patch_size, which
    # is normally supplied by Hydra config-group composition. Null them here.
    cfg.surrogate.patch_generator = None
    cfg.surrogate.train_augmentation = None
    OmegaConf.resolve(cfg)

    print(f"[1] instantiating from {args.config} ...")
    module = instantiate(cfg.surrogate, _recursive_=False).train()
    n_params = sum(p.numel() for p in module.network.parameters())
    print(f"    OK. encoder.mean={module.encoder.mean.shape[0]} "
          f"decoder.std={module.decoder.std.shape[0]} "
          f"n_input={cfg.surrogate.network.n_input} params={n_params/1e6:.1f}M")

    batch, resolution = build_synthetic_batch()
    labels = get_empty_labels(batch["states"], module._LABELS_DIMS)
    print("[2] batch:", {k: tuple(v.shape) for k, v in batch.items()})

    print("[3] forward + loss ...")
    out = module.estimate_loss(batch, resolution, labels, prefix="train")
    loss = out["loss"]
    print(f"    loss = {float(loss):.4f}  finite={bool(torch.isfinite(loss))}")

    print("[4] backward ...")
    loss.backward()
    grads = [p.grad for p in module.network.parameters() if p.grad is not None]
    gnorm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    print(f"    params w/ grad: {len(grads)}  grad-norm: {float(gnorm):.4g}  "
          f"finite={bool(torch.isfinite(gnorm))}")

    ok = bool(torch.isfinite(loss) and torch.isfinite(gnorm))
    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
