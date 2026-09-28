# Quantile-mapped `sic` targets

**Date:** 2026-09-21 (`monthly_qm`), 2026-09-23 (`monthly_pf_qm`) · **Status:** done

## Question
neXtSIM, the training target, is too icy in September compared with observations. If the targets' concentration is mapped onto NSIDC's distribution before training, does the model inherit less of that bias?

## Setup
- `scripts/data/correct_sic_targets_qm.py`: per grid cell and calendar month, map neXtSIM `sic` onto NSIDC-0051's distribution (1995–2018). Only `sic` changes. It moves the 1995–2018 September extent 5.97 → 5.59 M km² (observed 5.60).
- `monthly_qm`: base model + 20k steps on these targets (`configs/experiment/sicqm.yaml`).
- `monthly_pf_qm`: `monthly_pf` (55k) + 40k pushforward steps on these targets (`pushforward` recipe with `data.suffix=_sicqm`).

## Commands
```bash
python scripts/data/correct_sic_targets_qm.py
python scripts/data/build_zarr_monthly.py --datacube-dir data/train_data/monthly_datacube_sicqm --suffix _sicqm
python train.py +experiment=pushforward exp_name=monthly_pf_qm init_from=data/models/monthly_pf/last.ckpt data.suffix=_sicqm surrogate.total_steps=40000 trainer.max_steps=40000 surrogate.pushforward_prob=0.8 surrogate.pushforward_substeps=10
```

## Runs
| what | ledger run_id |
| --- | --- |
| `monthly_pf_qm`, 2019–25 hindcast, members saved | `f3d9c5bc56e2` |
| `monthly_pf_qm`, CMIP free-run 2015–2100 (animation) | `d556f1e3105f` |

## Results
`monthly_qm` alone barely moved: one-step September 5.06 → 5.03 M km² against a corrected target of ~4.95. Its averaged weights still held ~61% of the seed model (ISS-005).

`monthly_pf_qm` against `monthly_pf`, 2019–25 out of sample, per member:

| | September bias | July | August | summer ice-edge error | East Siberian | Chukchi |
| --- | --- | --- | --- | --- | --- | --- |
| `monthly_pf` | +0.47 | +0.60 | +0.45 | 1.99 | +0.25 | +0.19 |
| `monthly_pf_qm` | **+0.37** | **+0.28** | **+0.23** | **1.83** | **+0.20** | **+0.16** |

## Conclusion
With pushforward, QM targets give the best model so far on years no model or correction saw (QM was fitted on 1995–2018). It fixes the inherited (Chukchi, Laptev) part, not the emulator's own Pacific-shelf excess. `monthly_pf_qm` is the model for the animation.
