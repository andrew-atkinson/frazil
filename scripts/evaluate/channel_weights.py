#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
How much does a model use each input channel? Prints the norm of the first
layer's weights per input channel (inputs are standardised, so norms compare),
for the training weights and the EMA weights used at inference, as a % of the
median original forcing (tus huss uas vas at t and t+1).

Use it mid-training on a step_*.ckpt to see whether new forcing channels are
being learned, before committing to a full run. A run can feed a channel in
scaled up (config `input_scale: {name: factor}`, encoder std divided by it) so
it learns faster; norms here are multiplied by that factor to stay comparable.

Run:  python scripts/evaluate/channel_weights.py data/models/monthly_pf_qm_rad/step_10000.ckpt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

ORIGINAL = ["tus", "huss", "uas", "vas"]


def channel_norms(ckpt):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), ".hydra", "config.yaml"))
    forc = list(cfg.data.forcing_variables)
    scale = dict(cfg.get("input_scale") or {})
    p2 = cfg.surrogate.network.patch_size ** 2
    names = ([f"noisy_{i}" for i in range(6)] + list(cfg.data.state_variables)
             + [f"{v}(t)" for v in forc] + [f"{v}(t+1)" for v in forc]
             + ["pdd_month", "fdd_month", "pdd_year", "fdd_year", "ones"])
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    rows = {}
    for label, prefix in [("train", "network."), ("ema", "ema_model.module.")]:
        w = sd[prefix + "tokenizer.in_encoder.weight"]
        n = w.reshape(w.shape[0], -1, p2).pow(2).sum((0, 2)).sqrt().numpy()
        n = n * np.array([scale.get(x.split("(")[0], 1.0) for x in names])
        ref = np.median([n[i] for i, x in enumerate(names) if x.split("(")[0] in ORIGINAL])
        rows[label] = {x: 100 * n[i] / ref for i, x in enumerate(names) if "(" in x}
    return forc, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    args = ap.parse_args(argv)
    forc, rows = channel_norms(args.ckpt)
    print(f"input-weight norm, % of the median original forcing  ({args.ckpt})")
    print(f"  {'channel':12s} {'train':>7s} {'EMA':>7s}")
    for x in rows["train"]:
        flag = "   <- new" if x.split("(")[0] not in ORIGINAL else ""
        print(f"  {x:12s} {rows['train'][x]:6.0f}% {rows['ema'][x]:6.0f}%{flag}")


if __name__ == "__main__":
    main()
