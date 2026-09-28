#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A/B test of the GenSIM optimizer bug fix.

BASELINE  = the repo's current training_step, which rebuilds a fresh AdamW every
            step (Adam moments reset each step; the LR scheduler is created after
            the step and discarded, so LR is pinned at self.lr).
FIXED     = a training_step that uses the persistent optimizers/scheduler from
            configure_optimizers via self.optimizers() / self.lr_schedulers()
            (Adam momentum accumulates; warmup+cosine schedule is live).

Both runs share seed, data order, and settings. Warmup is shortened (lr_warmup=10)
so the FIXED run reaches ~the same LR quickly -- isolating the momentum effect
rather than confounding it with a long 0->lr ramp. Single-scale patches +
clamped extract_patches (repo boundary bug) keep both runs from crashing.

Run:  python experiments/ab_optimizer_monthly.py --steps 200 --batch-size 2
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
from gensim.train_module import GenSIMTrainModule
from gensim.utils import get_empty_labels


# --- clamped patch extraction (works around the repo's boundary bug) --------- #
def _extract_patches_clamped(in_tensor, sampled_top, sampled_left, sampled_levels,
                             size_y, size_x, pad_y, pad_x, n_patches):
    nb, nc, _, _ = in_tensor.shape
    in_tensor = F.pad(in_tensor, (pad_y, pad_y, pad_x, pad_x), mode="replicate")
    ph, pw = in_tensor.shape[-2], in_tensor.shape[-1]
    out = torch.empty(nb * n_patches, nc, size_y, size_x,
                      device=in_tensor.device, dtype=in_tensor.dtype)
    for k, (st, sl, level) in enumerate(zip(sampled_top, sampled_left, sampled_levels)):
        level = int(level)
        st = min(max(int(st), 0), ph - size_y * level)
        sl = min(max(int(sl), 0), pw - size_x * level)
        patch = in_tensor[k % nb, :, st:st + size_y * level, sl:sl + size_x * level]
        if level > 1:
            patch = patch.view(nc, size_y, level, size_x, level).mean(dim=(-3, -1))
        out[k] = patch
    return out


_aug.extract_patches = _extract_patches_clamped
_ORIG_STEP = GenSIMTrainModule.training_step


# --- the FIXED training_step: use persistent optimizers + live scheduler ------ #
def fixed_training_step(self, batch, batch_idx):
    resolution = batch.pop("resolution", 12.)
    if self.patch_generator is not None:
        batch, resolution = self.patch_generator(batch, resolution)
    if self.train_augmentation is not None:
        batch, labels = self.train_augmentation(batch)
    else:
        labels = get_empty_labels(batch["states"], self._LABELS_DIMS)

    opt_net, opt_scale = self.optimizers()
    scheduler = self.lr_schedulers()

    opt_net.zero_grad()
    opt_scale.zero_grad()
    outputs = self.estimate_loss(batch, resolution, labels, prefix="train")
    self.manual_backward(outputs["loss"])
    self.clip_gradients(opt_net, gradient_clip_val=1.,
                        gradient_clip_algorithm="norm")
    opt_net.step()
    if self.optimize_scale:
        opt_scale.step()
    scheduler.step()
    return outputs


class Recorder(pl.Callback):
    def __init__(self):
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(float(outputs["loss"] if isinstance(outputs, dict) else outputs))


def run(step_fn, tag, steps, batch_size, accelerator):
    pl.seed_everything(0, workers=True)
    cfg = OmegaConf.load("config_train_monthly.yaml")
    cfg.data.batch_size = batch_size
    cfg.data.n_workers = 0
    cfg.compile = False
    cfg.surrogate.patch_generator.coarse_levels = [1]
    cfg.surrogate.patch_generator.coarse_probs = [1.0]
    cfg.surrogate.lr_warmup = 10  # short warmup so FIXED reaches ~lr quickly

    GenSIMTrainModule.training_step = step_fn
    dm = instantiate(cfg.data)
    dm.setup("fit")
    per_epoch = min(len(dm._train_dataset) // batch_size, steps)
    epochs = int(np.ceil(steps / per_epoch))
    model = instantiate(cfg.surrogate, _recursive_=False)
    rec = Recorder()
    trainer = pl.Trainer(
        max_epochs=epochs, limit_train_batches=per_epoch, limit_val_batches=0,
        num_sanity_val_steps=0, accelerator=accelerator, devices=1,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        enable_progress_bar=False, callbacks=[rec],
    )
    print(f"[{tag}] running {steps} steps ...", flush=True)
    trainer.fit(model, datamodule=dm)
    losses = np.array(rec.losses)
    w = max(5, len(losses) // 10)
    print(f"[{tag}] first-{w}={losses[:w].mean():.4f}  last-{w}={losses[-w:].mean():.4f}  "
          f"Δ={losses[-w:].mean()-losses[:w].mean():+.4f}", flush=True)
    return losses


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-png", default="experiments/ab_optimizer.png")
    args = ap.parse_args(argv)

    torch.set_float32_matmul_precision("medium")
    acc = "cpu" if args.cpu or not torch.backends.mps.is_available() else "mps"

    base = run(_ORIG_STEP, "baseline", args.steps, args.batch_size, acc)
    GenSIMTrainModule.training_step = _ORIG_STEP  # restore before re-patching
    fixed = run(fixed_training_step, "fixed", args.steps, args.batch_size, acc)

    def ma(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    w = max(5, min(len(base), len(fixed)) // 10)
    print("\n================ A/B RESULT ================")
    print(f"baseline  last-{w} avg = {base[-w:].mean():.4f}")
    print(f"fixed     last-{w} avg = {fixed[-w:].mean():.4f}")
    better = fixed[-w:].mean() < base[-w:].mean()
    print(f"=> FIXED trains {'BETTER' if better else 'NOT better'} "
          f"(Δ={fixed[-w:].mean()-base[-w:].mean():+.4f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(base, color="#bbbbbb", lw=0.7)
        ax.plot(fixed, color="#ffc09f", lw=0.7)
        ax.plot(np.arange(w - 1, len(base)), ma(base, w),
                color="#666666", lw=2.2, label=f"baseline (per-step optimizer), ma{w}")
        ax.plot(np.arange(w - 1, len(fixed)), ma(fixed, w),
                color="#E65007", lw=2.2, label=f"fixed (persistent Adam), ma{w}")
        ax.set_xlabel("training step"); ax.set_ylabel("flow-matching loss")
        ax.set_title("Optimizer A/B — monthly GenSIM"); ax.legend()
        fig.tight_layout(); fig.savefig(args.out_png, dpi=130)
        print(f"[plot] -> {args.out_png}")
    except Exception as e:
        print("[plot] skipped:", e)


if __name__ == "__main__":
    main()
