#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Short real training run on the monthly data to watch the loss descend.

Runs a few hundred real training_step iterations (patched full-grid flow
matching) and records the per-step train loss, printing a moving average and
saving a loss curve. This is a sanity check that the monthly setup actually
learns -- not a full training job.

Notes / caveats:
  * The flow-matching loss is high-variance (fresh noise + random pseudo-time
    each step), so only the MOVING AVERAGE is meaningful, not single steps.
  * GenSIMTrainModule.training_step recreates its optimizer every step (Adam
    momentum resets each step; effectively sign-scaled steps at lr). Descent is
    therefore slower/noisier than a normal Adam run -- expected, not a bug here.
  * Run bounded by max_epochs * limit_train_batches (manual optimization makes
    global_step/max_steps unreliable).

Run:  python experiments/train_watch_monthly.py --steps 300 --batch-size 2
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from omegaconf import OmegaConf
from hydra.utils import instantiate

import gensim.augmentation as _aug


def _extract_patches_clamped(in_tensor, sampled_top, sampled_left, sampled_levels,
                             size_y, size_x, pad_y, pad_x, n_patches):
    """Drop-in for gensim.augmentation.extract_patches that clamps patch starts.

    The repo's patch sampler occasionally returns a start position within one
    patch-width of the (padded) boundary, so the slice is clipped and the
    assignment raises a size-mismatch. Clamping each start to
    [0, padded_dim - size*level] guarantees a full-size patch. Only edge patches
    (a tiny fraction) shift inward by a few pixels.
    """
    nb, nc, _, _ = in_tensor.shape
    in_tensor = F.pad(in_tensor, (pad_y, pad_y, pad_x, pad_x), mode="replicate")
    ph, pw = in_tensor.shape[-2], in_tensor.shape[-1]
    out_tensor = torch.empty(nb * n_patches, nc, size_y, size_x,
                             device=in_tensor.device, dtype=in_tensor.dtype)
    for k, (start_top, start_left, level) in enumerate(
            zip(sampled_top, sampled_left, sampled_levels)):
        level = int(level)
        st = min(max(int(start_top), 0), ph - size_y * level)
        sl = min(max(int(start_left), 0), pw - size_x * level)
        patch = in_tensor[k % nb, :, st:st + size_y * level, sl:sl + size_x * level]
        if level > 1:
            patch = patch.view(nc, size_y, level, size_x, level).mean(dim=(-3, -1))
        out_tensor[k] = patch
    return out_tensor


_aug.extract_patches = _extract_patches_clamped  # patch the repo's buggy edge case


class LossRecorder(pl.Callback):
    def __init__(self, report_every=25):
        self.losses = []
        self.report_every = report_every

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = float(outputs["loss"] if isinstance(outputs, dict) else outputs)
        self.losses.append(loss)
        n = len(self.losses)
        if n % self.report_every == 0:
            recent = np.mean(self.losses[-self.report_every:])
            print(f"  step {n:4d}  loss(last)={loss:7.4f}  "
                  f"loss(avg{self.report_every})={recent:7.4f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_train_monthly.yaml")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-png", default="experiments/train_loss_monthly.png")
    args = ap.parse_args(argv)

    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    cfg = OmegaConf.load(args.config)
    cfg.data.batch_size = args.batch_size
    cfg.data.n_workers = 0
    cfg.compile = False
    # The repo's multi-scale patcher (coarse_levels [1,2,4,8]) raises a patch-size
    # mismatch at grid boundaries for some random samples. Use single-scale
    # patches for this stability-focused descent run.
    cfg.surrogate.patch_generator.coarse_levels = [1]
    cfg.surrogate.patch_generator.coarse_probs = [1.0]

    dm = instantiate(cfg.data)
    dm.setup("fit")
    per_epoch = min(len(dm._train_dataset) // args.batch_size, args.steps)
    max_epochs = int(np.ceil(args.steps / per_epoch))
    print(f"[run] ~{args.steps} steps  (batch_size={args.batch_size}, "
          f"{per_epoch} batches/epoch x {max_epochs} epochs)")

    model = instantiate(cfg.surrogate, _recursive_=False)
    accelerator = "cpu" if args.cpu or not torch.backends.mps.is_available() else "mps"
    recorder = LossRecorder()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=per_epoch,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator=accelerator, devices=1,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, callbacks=[recorder],
    )
    print(f"[run] accelerator={accelerator}\n")
    trainer.fit(model, datamodule=dm)

    losses = np.array(recorder.losses)
    w = max(5, len(losses) // 10)
    first = losses[:w].mean()
    last = losses[-w:].mean()
    print(f"\n[result] {len(losses)} steps | "
          f"first-{w} avg = {first:.4f} | last-{w} avg = {last:.4f} | "
          f"Δ = {last - first:+.4f} ({100*(last-first)/first:+.1f}%)")

    # Loss curve with moving average.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ma = np.convolve(losses, np.ones(w) / w, mode="valid")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(losses, color="0.7", lw=0.8, label="per-step loss")
        ax.plot(np.arange(w - 1, len(losses)), ma, color="#E65007", lw=2,
                label=f"moving avg ({w})")
        ax.set_xlabel("training step"); ax.set_ylabel("flow-matching loss")
        ax.set_title("Monthly GenSIM — training loss"); ax.legend()
        fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
        print(f"[plot] -> {args.out_png}")
    except Exception as e:
        print("[plot] skipped:", e)

    print("DESCENT" if last < first else "NO CLEAR DESCENT")


if __name__ == "__main__":
    main()
