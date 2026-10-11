# Downward solar and thermal radiation as inputs

**Date:** 2026-09-27 (trained) · **Status:** done (scored 2026-10-09, ISS-004) · **Model:** `monthly_pf_qm_rad`

## Question
Summer melt is driven mostly by absorbed sunlight, and monthly air temperature over melting ice sits near 0 °C whether the month was sunny or cloudy. Does giving the model radiation fix the Pacific-shelf melt deficit?

## Setup
- Recipe `configs/experiment/pf_qm_rad.yaml`: seeded from `monthly_pf_qm`, QM targets and pushforward, plus `ssrd` and `strd` at *t* and *t+1* (28 inputs, zero-initialised), 20k steps.
- Two changes so new inputs can be learned: EMA rate 0.999 (was 0.9999), and the new channels fed in 10× larger (encoder std / 10) so they learn ~10× faster under Adam.

## Commands
```bash
python train.py +experiment=pf_qm_rad
python scripts/evaluate/channel_weights.py data/models/monthly_pf_qm_rad/step_10000.ckpt
python scripts/project/freerun_monthly.py --ckpt data/models/monthly_pf_qm_rad/last.ckpt --forcing era5 --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members --datacube data/train_data/monthly_datacube_era5x --era5-ext data/train_data/era5_forcing_datacube_era5x [--climatology-forcing ssrd strd]
```

## Results so far
Input-weight norm relative to the other forcings, inference weights: `ssrd` 4–5% at 10k steps and 6–8% at 15.5k; `strd` 3–4% and 5–6%; at 20k (final) `ssrd` 8–10% and `strd` 6–7%. Growing slowly, the same pattern as SST.

## Runs
| what | ledger run_id |
| --- | --- |
| real radiation | `0c02ed5c4fc5` |
| climatological radiation | `6c603ed531ac` |
| A/B partner `monthly_pf_qm` | `f3d9c5bc56e2` |
| control `monthly_pf_qm_ema` (same recipe, no radiation) | `9e6ceab8ef11` |

## Results
Per member, 8 members, against NSIDC-0051, 2019–25 (development years 2019–22 in brackets). M km² except r. The `monthly_pf_qm` column reproduces the v0.2.0 baseline in the roadmap exactly.

| measure | `monthly_pf_qm` | + radiation | + climatological radiation |
| --- | --- | --- | --- |
| September bias | +0.37 (+0.27) | **+0.20 (+0.12)** | +0.22 (+0.14) |
| June / July / August bias | +0.52 / +0.28 / +0.23 | **+0.65** / +0.26 / +0.28 | +0.67 / +0.29 / +0.32 |
| ice-edge error Sep / Jul–Sep / Dec–Mar | 1.53 / 1.83 / 0.93 | **1.21 / 1.45 / 0.82** | 1.22 / 1.46 / 0.82 |
| September r | 0.86 | 0.90 | 0.89 |
| drift growth (2023–25 minus 2019–21) | +0.34 | +0.30 | +0.29 |
| Sep bias E. Siberian / Beaufort / Chukchi / Laptev | +0.20 / +0.21 / +0.16 / −0.11 | +0.15 / +0.14 / +0.09 / −0.07 | +0.16 / +0.14 / +0.10 / −0.07 |

The model is much better than `monthly_pf_qm` (September bias nearly halved, summer ice-edge error down a fifth), **but replacing the radiation with its climatology changes almost nothing** (≤ 0.04 on every measure). So the gain is not from radiation. The new inputs reach only 6–10% of the other forcings' weight (above), consistent with that.

What else differs from `monthly_pf_qm`: 20k more steps of the same training (pushforward on QM targets) with **EMA 0.999 instead of 0.9999**. `monthly_pf_qm`'s inference weights still held 37% of its seed `monthly_pf`, which was trained on the icy original targets (ISS-005); here they follow the QM training. The likely explanation is that `monthly_pf_qm` was never fully de-iced, and this run finished the job.

June got worse (+0.52 → +0.65), the one measure that did.

## Control (2026-10-10)
`configs/experiment/pf_qm_ema.yaml`: the same fine-tune without the radiation channels (24 inputs).

| measure | `monthly_pf_qm` | + radiation | control, no radiation |
| --- | --- | --- | --- |
| September bias | +0.37 (+0.27) | +0.20 (+0.12) | +0.22 (+0.12) |
| June / July / August bias | +0.52 / +0.28 / +0.23 | +0.65 / +0.26 / +0.28 | +0.61 / +0.19 / +0.29 |
| ice-edge error Sep / Jul–Sep / Dec–Mar | 1.53 / 1.83 / 0.93 | 1.21 / 1.45 / 0.82 | 1.22 / 1.46 / 0.82 |
| September r | 0.86 | 0.90 | 0.86 |
| drift growth | +0.34 | +0.30 | +0.32 |
| Sep bias E. Siberian / Beaufort / Chukchi / Laptev | +0.20 / +0.21 / +0.16 / −0.11 | +0.15 / +0.14 / +0.09 / −0.07 | +0.15 / +0.14 / +0.10 / −0.08 |

The control matches the radiation model within 0.07 on every measure (within 0.03 on most), so the gain is the fine-tune (EMA 0.999, 20k more steps), confirmed.

## Conclusion
Radiation as an input: no measurable effect at this training length (and no leak). The improvement comes from the fine-tune protocol, confirmed by the control. `monthly_pf_qm_ema` needs no new inputs, so it can run CMIP projections; it is the v0.4.0 candidate. On the development years (2019–22) it meets the September bias target (+0.12 ≤ +0.20) and July–September ice-edge error (1.42, past the 1.50 stretch goal); not June (+0.64), East Siberian September (+0.19) or drift growth (+0.32).
