# Ensemble averaging and how to summarise an ensemble

**Date:** 2026-09-25 · **Status:** done · **Tool:** `scripts/evaluate/ensemble_summaries.py`

## Question
Snapshots stored the average of 8 members. Does averaging distort the ice edge, and which way of turning 8 members into one field is best, for science and for the animation?

## Setup
`monthly_pf`, 2019–25 hindcast with all members saved. Compared per month against NSIDC-0051: extent computed per member vs from the averaged field, and six summaries (mean, single member, median, consensus edge, probability-matched mean PMM, local PMM).

## Commands
```bash
python scripts/evaluate/ensemble_summaries.py plots/freerun/snapshots/monthly_pf_era5_2018
```

## Results
Averaging inflates September extent by ~+0.19 M km² (averaged field +0.66 vs per-member +0.47): members disagree where the edge is, and the average invents a band of partial ice. July–September, vs obs:

| method | extent bias | ice-edge error | partial-ice band bias | RMSE | sharpness (obs = 1) |
| --- | --- | --- | --- | --- | --- |
| mean | +0.69 | 1.92 | +0.78 | **0.173** | 0.44 |
| single member | +0.51 | 1.94 | +0.38 | 0.181 | **0.56** |
| PMM | +0.51 | **1.87** | +0.41 | 0.176 | 0.46 |
| local PMM | **+0.49** | 1.89 | +0.41 | 0.177 | 0.46 (tile seams) |

## Conclusion
Score ensembles per member or with PMM, never by the averaged field's extent. For the animation: PMM for faithful frames, a single member for texture (`animate_snapshots.py --summary pmm|member`). How this depends on ensemble size: [ensemble_size](../2026-09-28_ensemble_size/).
