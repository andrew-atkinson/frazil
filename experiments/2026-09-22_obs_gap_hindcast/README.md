# Observation gap and the out-of-sample hindcast

**Date:** 2026-09-22 (in-sample hindcast) and 2026-09-24 (2019–25) · **Status:** done · **Detail:** [docs/OBS_GAP.md](../../docs/OBS_GAP.md)

## Question
The model's September extent sits well above NSIDC. How much of that is the model, the targets, the forcing, drift or measurement, and does the model hold up on years it never saw?

## Setup
- In-sample: ERA5-forced free-run from January 2013 truth, compared with neXtSIM and NSIDC (2013–17).
- Out-of-sample: `monthly_pf` from the December 2018 state under real ERA5 weather to 2025 (`--forcing era5` with the forcing-only extension cube), scored against NSIDC-0051.

## Commands
```bash
python scripts/project/freerun_monthly.py --forcing era5 --ckpt data/models/monthly_pf/last.ckpt --start 2018-12 --years 7 --n-ens 8 --fast --snapshot-every 1 --save-members
```

## Runs
| what | ledger run_id |
| --- | --- |
| `monthly_pf`, 2019–25, members saved | `e0e924e7278d` |

## Results
- Year-to-year September variations are captured: correlation 0.82 over 2019–25 (2020 low, 2021–22 high).
- September extent bias +0.47 M km² per member, growing over the run (+0.27 → +0.63).
- The error is almost all melt season (June peak), and it is *placed*: ~1.1 M km² of extra ice and ~0.6 M km² missing.
- By sector, the emulator's own error is on the Pacific shelf (East Siberian, Beaufort ~+0.2 each); Chukchi and Laptev errors are inherited from neXtSIM.
- Ruled out: thickness build-up (the model has slightly *less* volume than neXtSIM), a large forcing bias (CMIP vs ERA5 ~+0.13) and a metric artefact (~0.1).

## Conclusion
The model responds realistically to real weather; the persistent problem is too little summer melt on the Pacific shelf. See ISS-002, and the SST and radiation experiments.
