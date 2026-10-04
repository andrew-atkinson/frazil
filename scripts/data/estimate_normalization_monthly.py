#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate normalization statistics for the monthly GenSIM datacube and wire them
into monthly copies of the Hydra configs.

Background (verified against the code)
--------------------------------------
GenSIM's GaussianEncoder standardizes an 18-channel input tensor:

    channels = [ 6 states, 4 forcings(t), 4 forcings(t+1), 4 degree-days ]
             = [ sit sic sid siu siv snt,
                 tus huss uas vas,   (t)
                 tus huss uas vas,   (t+1, identical stats),
                 pdd_month fdd_month pdd_year fdd_year ]

so encoder.mean / encoder.std each have 18 entries (the forcing block appears
twice, with identical statistics; rhus is computed but NOT fed to the network).

The GaussianDecoder predicts the *tendency* (next state minus current state):
    prediction = first_guess + net_out * std + mean
so decoder.std (6) is the std of  state[t+step] - state[t]  and decoder.mean (6)
is that tendency's mean (~0). Physical lower/upper bounds are unchanged.

Monthly design choice
----------------------
data_04 used `shift(time=-2)` because the 6-hourly data made a 12 h model step
two rows apart. Monthly data has one row per step, so the tendency uses
`shift(time=-1)` (i.e. STEP_MONTHS below). Everything else mirrors data_04:
mean/std over time and ocean cells, ddof=1.

Outputs
-------
  * prints the stat vectors,
  * writes results/normalization_monthly.json,
  * writes configs/config_train_monthly.yaml and configs/config_forecast_monthly.yaml with the
    encoder/decoder mean/std replaced (architecture and bounds untouched).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from frazil import paths

# Datacube variable layout (from preprocess_monthly.py).
STATES = ["sit", "sic", "sid", "siu", "siv", "snt"]
FORCINGS = ["tus", "huss", "uas", "vas"]          # network forcings (rhus excluded)
DEGREE = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]

STEP_MONTHS = 1  # model step = 1 month  (data_04 used 2 six-hourly rows)


def compute_stats(datacube_glob):
    files = sorted(glob.glob(datacube_glob))
    if not files:
        raise SystemExit(f"no datacube files matched: {datacube_glob}")
    ds = xr.open_mfdataset(files)["datacube"].load()
    print(f"[stats] {len(files)} files, dims={dict(ds.sizes)}")

    def pick(v):
        return ds.sel(var_names=v)

    # Per-variable mean/std over time + ocean cells (land is NaN -> skipped).
    reduce_dims = ["time", "y", "x"]
    mean = {v: float(pick(v).mean(reduce_dims, skipna=True)) for v in
            STATES + FORCINGS + DEGREE}
    std = {v: float(pick(v).std(reduce_dims, ddof=1, skipna=True)) for v in
           STATES + FORCINGS + DEGREE}

    # Tendency (state[t+step] - state[t]) statistics for the 6 states.
    tend_mean, tend_std = {}, {}
    for v in STATES:
        s = pick(v)
        d = s.shift(time=-STEP_MONTHS) - s
        tend_mean[v] = float(d.mean(reduce_dims, skipna=True))
        tend_std[v] = float(d.std(reduce_dims, ddof=1, skipna=True))

    # Assemble encoder (18) and decoder (6) vectors in channel order.
    enc_mean = ([mean[v] for v in STATES]
                + [mean[v] for v in FORCINGS]
                + [mean[v] for v in FORCINGS]
                + [mean[v] for v in DEGREE])
    enc_std = ([std[v] for v in STATES]
               + [std[v] for v in FORCINGS]
               + [std[v] for v in FORCINGS]
               + [std[v] for v in DEGREE])
    dec_mean = [tend_mean[v] for v in STATES]
    dec_std = [tend_std[v] for v in STATES]

    return dict(
        per_var_mean=mean, per_var_std=std,
        tendency_mean=tend_mean, tendency_std=tend_std,
        encoder_mean=enc_mean, encoder_std=enc_std,
        decoder_mean=dec_mean, decoder_std=dec_std,
    )


def rnd(x, n=4):
    return [round(float(v), n) for v in x]


def wire_config(src_path, out_path, stats):
    """Load a Hydra config, replace encoder/decoder mean+std, save a monthly copy."""
    if not Path(src_path).exists():
        print(f"[config] {src_path} not found -- skipped")
        return
    cfg = OmegaConf.load(src_path)
    cfg.surrogate.encoder.mean = rnd(stats["encoder_mean"])
    cfg.surrogate.encoder.std = rnd(stats["encoder_std"])
    cfg.surrogate.decoder.mean = rnd(stats["decoder_mean"])
    cfg.surrogate.decoder.std = rnd(stats["decoder_std"])
    OmegaConf.save(cfg, out_path)
    print(f"[config] wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacube",
                    default=f"{paths.DATACUBE}/monthly_datacube_*.nc")
    ap.add_argument("--out-json", default="results/normalization_monthly.json")
    ap.add_argument("--no-configs", action="store_true",
                    help="only compute/print stats, don't write monthly configs")
    args = ap.parse_args(argv)

    stats = compute_stats(args.datacube)

    order = STATES + FORCINGS + DEGREE
    print("\nvariable        mean         std")
    for v in order:
        print(f"  {v:10s} {stats['per_var_mean'][v]:>11.4f}  "
              f"{stats['per_var_std'][v]:>10.4f}")
    print("\ntendency (state[t+1]-state[t])   mean         std")
    for v in STATES:
        print(f"  {v:10s} {stats['tendency_mean'][v]:>11.5f}  "
              f"{stats['tendency_std'][v]:>10.5f}")

    print("\nencoder.mean (18):", rnd(stats["encoder_mean"]))
    print("encoder.std  (18):", rnd(stats["encoder_std"]))
    print("decoder.mean  (6):", rnd(stats["decoder_mean"], 5))
    print("decoder.std   (6):", rnd(stats["decoder_std"], 5))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump({k: (rnd(v, 6) if isinstance(v, list) else v)
                   for k, v in stats.items()}, fh, indent=2)
    print(f"\n[json] wrote {args.out_json}")

    if not args.no_configs:
        wire_config("configs/config_train.yaml", "configs/config_train_monthly.yaml", stats)
        wire_config("configs/config_forecast.yaml", paths.CONFIG_FORECAST, stats)


if __name__ == "__main__":
    main()
