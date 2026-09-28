#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert the per-year monthly datacube NetCDFs into the consolidated zarr stores
that gensim's data pipeline (NeXtSIMDataset / SurrogateDataModule) expects.

The data module loads, from its `data_path`:
    <data_path>/train<suffix>.zarr        (fit stage)
    <data_path>/validation<suffix>.zarr   (fit/validate stages)
each a single zarr *group* with:
    - "datacube": array of shape (time, var_names, y, x)
    - "var_names": 1-D array of the variable-name strings
NeXtSIMDataset then draws sliding windows of `n_cycles` frames spaced `delta_t`
apart, so the time axis must be a single continuous monthly sequence.

  *** MONTHLY: delta_t = 1 ***  With delta_t=1 and n_cycles=2 the dataset yields
  consecutive-month pairs (t, t+1) -- one-month-ahead prediction. (The original
  6-hourly setup used delta_t=2 to make a 12 h step.) This matches the
  `shift(time=-1)` used for the decoder tendency stats. Set delta_t=1 in the
  data-module config (done in configs/config_train_monthly.yaml by this repo's setup).

Key format requirements (verified against the code):
  * zarr **v2** -- NeXtSIMDataset reads the datacube through tensorstore with
    `metadata_key='.zarray'`, which is the zarr-v2 layout. zarr-python 3.x
    defaults to v3, so we pass zarr_format=2 explicitly.
  * land cells are filled with 0 (the dataset nan_to_num's anyway; we make it
    explicit and deterministic).

Split: train = 1995..(val_start-1), validation = val_start..end. Default
val_start=2015 gives train 1995-2014 (the paper's 20-year training period) and
validation 2015-2018.

Usage
-----
    python scripts/data/build_zarr_monthly.py
    python scripts/data/build_zarr_monthly.py --val-start-year 2016
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

STATES = ["sit", "sic", "sid", "siu", "siv", "snt"]
FORCINGS = ["tus", "huss", "uas", "vas"]
DEGREE = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
VAR_ORDER = STATES + FORCINGS + DEGREE  # 14 channels used by the model


def year_of(path):
    # monthly_datacube_YYYY.nc
    return int(Path(path).stem.split("_")[-1])


def build_split(files, name, out_dir, overwrite, ocean_flat, var_order=VAR_ORDER):
    """Write a masked datacube: (time, var_names, grid) over ocean points only.

    The pipeline stores states/forcings/degree_days as flattened ocean points
    and scatters them back to (y, x) via unmask_tensor at batch transfer (using
    the aux mask, C-order). So the last dim here must be the ocean cells in
    C-order of the (y, x) grid -- exactly `mask.reshape(-1)` True positions.
    """
    out_path = out_dir / f"{name}.zarr"
    if out_path.exists():
        if not overwrite:
            print(f"[{name}] exists -> {out_path.name} (skip)")
            return out_path
        shutil.rmtree(out_path)

    da = xr.open_mfdataset(sorted(files), combine="by_coords")["datacube"]
    da = da.sel(var_names=var_order).fillna(0.0).astype("float32")
    da = da.transpose("time", "var_names", "y", "x")

    cube = da.values                                  # (time, var, y, x)
    T, V, Y, X = cube.shape
    masked = cube.reshape(T, V, Y * X)[..., ocean_flat]   # (time, var, n_ocean)

    ds = xr.Dataset(
        {"datacube": (("time", "var_names", "grid"), masked)},
        coords={"time": da["time"].values,
                "var_names": np.array(var_order, dtype="<U12")},
    ).chunk({"time": 1})

    print(f"[{name}] writing {T} months, {V} vars, {masked.shape[-1]} ocean pts "
          f"-> {out_path} (zarr v2, masked)")
    ds.to_zarr(out_path, mode="w", zarr_format=2, consolidated=True)
    return out_path


def verify(zarr_path, aux_path):
    """Exercise the full path: NeXtSIMDataset -> collate -> unmask hook."""
    import torch
    from gensim.dataset import NeXtSIMDataset
    from gensim.data_module import SurrogateDataModule
    from gensim.utils import unmask_tensor

    d = NeXtSIMDataset(
        str(zarr_path), aux_path=aux_path, delta_t=1, n_cycles=2,
        state_variables=STATES, forcing_variables=FORCINGS,
    )
    sample = d[0]
    masked_shapes = {k: tuple(np.asarray(v).shape) for k, v in sample.items()}
    # states/forcings/degree_days are stored masked (last dim = ocean points)
    assert masked_shapes["states"][-1] == int(d.mask.sum()), \
        "stored states should be flattened ocean points"

    # Batch + apply the real unmask hook, then check the full (y, x) shapes.
    batch = {k: torch.as_tensor(v)[None] for k, v in sample.items()}
    for k in SurrogateDataModule._UNMASK_KEYS:
        batch[k] = unmask_tensor(batch[k], batch["mask"])
    full = {k: tuple(v.shape) for k, v in batch.items()}
    print(f"    OK: len={len(d)}  masked states={masked_shapes['states']} "
          f"-> unmasked {full['states']}")
    assert full["states"][1] == 2 and full["states"][2] == 6
    assert full["states"][-2:] == d.mask.shape, "unmasked should be full (y, x)"
    return len(d)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datacube-dir", default="data/train_data/monthly_datacube")
    ap.add_argument("--out-dir", default="data/train_data")
    ap.add_argument("--suffix", default="_monthly",
                    help="written as train<suffix>.zarr / validation<suffix>.zarr")
    ap.add_argument("--val-start-year", type=int, default=2015)
    ap.add_argument("--aux-path", default="data/auxiliary/ds_auxiliary.nc")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--extra-forcings", nargs="*", default=[],
                    help="extra forcing channels to include (e.g. ssrd strd sst); "
                         "must exist in the datacube -- see add_forcing_channels.py")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(f"{args.datacube_dir}/monthly_datacube_*.nc"))
    if not files:
        raise SystemExit(f"no datacube files in {args.datacube_dir}")
    train_files = [f for f in files if year_of(f) < args.val_start_year]
    val_files = [f for f in files if year_of(f) >= args.val_start_year]
    print(f"train years: {sorted(year_of(f) for f in train_files)}")
    print(f"val   years: {sorted(year_of(f) for f in val_files)}\n")

    # Ocean-point selector: mask flattened in C-order (matches unmask_tensor).
    ocean_flat = xr.open_dataset(args.aux_path)["mask"].values.astype(bool).reshape(-1)
    print(f"ocean points: {int(ocean_flat.sum())} of {ocean_flat.size}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var_order = STATES + FORCINGS + list(args.extra_forcings) + DEGREE
    train_path = build_split(train_files, f"train{args.suffix}", out_dir,
                             args.overwrite, ocean_flat, var_order)
    val_path = build_split(val_files, f"validation{args.suffix}", out_dir,
                           args.overwrite, ocean_flat, var_order)

    if not args.no_verify:
        print("\n[verify]")
        for p in (train_path, val_path):
            verify(p, args.aux_path)

    print(f"\nDone. Set the data-module config to:\n"
          f"  data_path: {args.out_dir}\n  suffix: '{args.suffix}'\n  delta_t: 1")


if __name__ == "__main__":
    main()
