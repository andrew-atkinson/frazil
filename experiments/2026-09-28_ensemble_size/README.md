# Averaging vs ensemble size

**Date:** 2026-09-28 · **Status:** done · **Tool:** `scripts/evaluate/ensemble_size.py`

## Question
Averaging members cancels their random differences but smears the ice edge (see [ensemble_averaging](../2026-09-25_ensemble_averaging/)). How do the benefit and the cost change with the number of members, and how many are worth paying for? Run time grows linearly with ensemble size (about 5 s per step for 1 member, 46 s for 8, with `--fast`).

## Setup
`monthly_pf_qm`, 2019–25 hindcast with 8 members saved. Members evolve independently, so any *k* of them stands in for a *k*-member run. For *k* = 1…8, up to 10 subsets per month, summer (Jul–Sep) and winter (Jan–Mar) months, 21 of each; each subset summarised by the plain mean and by PMM, scored against NSIDC-0051. Then a fit of the ensemble-averaging law RMSE² = *a* + *b*/*k*: *a* is the error no averaging can remove, *b* the member-to-member noise.

No new model runs: it reuses the saved members.

## Commands
```bash
python scripts/evaluate/ensemble_size.py plots/freerun/snapshots/monthly_pf_qm_era5_2018
```

## Results
**Summer, plain mean of k members** (extent, band and edge errors in M km² vs NSIDC; sharpness: observations = 1):

| k | RMSE | ice-edge error | extent bias | partial-ice band bias | sharpness |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.174 | 1.83 | +0.29 | +0.24 | 0.57 |
| 2 | 0.168 | 1.77 | +0.40 | +0.44 | 0.50 |
| 4 | 0.164 | 1.74 | +0.44 | +0.52 | 0.46 |
| 8 | 0.163 | 1.73 | +0.45 | +0.56 | 0.44 |

**Summer, PMM of k members:** extent bias stays at +0.29 to +0.30 and the partial-ice band at +0.24 to +0.26 for every *k*; ice-edge error 1.76 (k = 2), 1.72 (k = 4), 1.70 (k = 8), the best of any summary; sharpness 0.51 → 0.46.

**Averaging-law fit:**

| season | noise share of a 1-member error | RMSE 1 / 4 / 8 / 16 / ∞ | share of the possible gain at k = 4 / 8 |
| --- | --- | --- | --- |
| summer | 15% | 0.174 / 0.164 / 0.163 / 0.162 / 0.161 | 74% / 87% |
| winter | 6% | 0.160 / 0.156 / 0.155 / 0.155 / 0.155 | 75% / 87% |

## Conclusion
- **Averaging can only remove the small random part of the error** (15% in summer, 6% in winter). The rest is the model's systematic error, mainly misplaced Pacific-shelf ice (ISS-002), which no ensemble size touches: infinitely many members would take summer RMSE only from 0.174 to 0.161.
- **Four members give ~85% of the 8-member gain for half the run time;** 16 would add almost nothing (RMSE 0.1625 → 0.1617) for twice the cost.
- **The plain mean gets worse at the ice edge as members are added:** its extent bias rises by half and its partial-ice band doubles by *k* = 8. PMM keeps the gain without that cost at every size.
- **Practice:** 4 members + PMM for the animation (the 85-year CMIP run drops from ~13 h to ~6.5 h), or 1 member for texture (~1.5 h). 8 members for scoring, always per member or with PMM.

Caveats: one model, one 7-year period, 21 months per season; sizes above 8 are extrapolated from the fit.
