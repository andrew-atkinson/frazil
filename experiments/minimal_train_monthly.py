#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal real training run for the monthly GenSIM config.

Unlike smoke_train_monthly.py (synthetic 64x64 batch through estimate_loss),
this exercises the FULL path exactly as train.py would, just briefly:

    config_train_monthly.yaml
      -> SurrogateDataModule -> NeXtSIMDataset -> DataLoader  (real monthly zarr)
      -> GenSIMTrainModule.training_step
           -> PatchGenerator (512x512 -> n_patches x 64x64 patches)
           -> estimate_loss (flow-matching) -> manual backward -> optimizer step

It uses a bare Lightning Trainer with fast_dev_run so nothing is checkpointed or
logged to wandb. The point is to confirm the patched full-grid training_step
runs end to end before committing to a real job -- not to train anything.

Run:  python experiments/minimal_train_monthly.py            # MPS if available
      python experiments/minimal_train_monthly.py --cpu
"""
from __future__ import annotations

import argparse
import os

# Some ops used in the censoring loss (e.g. special.log_ndtr) are not yet
# implemented for MPS; fall back to CPU for just those ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_train_monthly.yaml")
    ap.add_argument("--steps", type=int, default=2, help="fast_dev_run batches")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--cpu", action="store_true", help="force CPU (default: MPS if available)")
    args = ap.parse_args(argv)

    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    cfg = OmegaConf.load(args.config)
    cfg.data.batch_size = args.batch_size
    cfg.data.n_workers = 0
    cfg.compile = False  # torch.compile is unreliable on MPS

    print(f"[1] data module <{cfg.data._target_}> ...")
    dm = instantiate(cfg.data)

    print(f"[2] model <{cfg.surrogate._target_}> "
          f"(patch_size={list(cfg.surrogate.patch_generator.patch_size)}) ...")
    model = instantiate(cfg.surrogate, _recursive_=False)

    accelerator = "cpu" if args.cpu or not torch.backends.mps.is_available() else "mps"
    print(f"[3] Trainer(fast_dev_run={args.steps}, accelerator={accelerator!r}) ...")
    trainer = pl.Trainer(
        fast_dev_run=args.steps,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    print("[4] trainer.fit ...")
    trainer.fit(model, datamodule=dm)
    print("\nMINIMAL TRAINING RUN OK")


if __name__ == "__main__":
    main()
