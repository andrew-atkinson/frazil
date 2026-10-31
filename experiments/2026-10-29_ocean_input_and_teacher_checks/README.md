# Ocean input and teacher checks

**Date:** 2026-10-29 · **Branch:** `exp/ocean-input-and-teacher-checks` · **Status:** done · **Issues:** ISS-002, ISS-022, ISS-028

## Question
[pacific_shelf_ocean_heat](../2026-10-29_pacific_shelf_ocean_heat/) found that Frazil's own Pacific-shelf excess follows the training simulation's subsurface ocean heat, which Frazil can't see. Two ways to give it that information:

- **A.** Keep the neXtSIM-OPA teacher (1995–2014 training) and add ocean heat from the TOPAZ4b ocean reanalysis (Copernicus `ARCTIC_MULTIYEAR_PHY_002_003`, 1991 to mid-2026) as an input.
- **B.** Switch the teacher to the Copernicus neXtSIM sea-ice reanalysis (`ARCTIC_MULTIYEAR_PHY_ICE_002_016`, ISS-022): stand-alone neXtSIM forced by ERA5 and TOPAZ4b, assimilating satellite concentration and, from October 2010, CS2SMOS winter thickness; 1993 to mid-2026. Same TOPAZ input, but targets and input share one ocean, and training can run to 2022.

The [neXtSIM v2 paper](https://doi.org/10.5194/gmd-19-6467-2026) (`literature/gmd-19-6467-2026.pdf`) reports that neXtSIM overestimates the September minimum and makes first-year ice too thick and old ice too thin. Two cheap checks before choosing:

1. **Ocean:** does TOPAZ4b show the same Pacific-shelf upper-ocean heat as neXtSIM-OPA's own ocean over 1995–2018 (level, trend, year-to-year)? If not, route A gives Frazil an ocean its targets never felt.
2. **Teacher:** on the Pacific shelves, how do the reanalysis, neXtSIM-OPA (raw and the v0.4.0 QM targets) and Frazil v0.4.0 compare with observed September extent (NSIDC-0051); and is neXtSIM-OPA's winter thickness (which the QM correction leaves untouched) too thick against the reanalysis, which assimilates CS2SMOS thickness?

## Setup
- `scripts/download/topaz_ocean_heat.py`: TOPAZ4b monthly temperature (16 levels to 100 m) and mixed-layer depth for the same box as the neXtSIM-OPA heat (north of 64° N, 90° E–110° W), heat content above freezing in 0–20 m and 20–100 m, SST and mixed-layer depth, on the model grid, 1995–2025: `data/train_data/topaz_ocean_heat/`.
- `scripts/download/nextsim_reanalysis.py --variables sithick`: the reanalysis thickness on the model grid (as the concentration was for ISS-022): `data/train_data/nextsim_reanalysis_sit_monthly.nc`.
- `analysis.py` (this folder), per sector (East Siberian, Chukchi, Beaufort; Laptev as a contrast), cells north of 64° N covered by every source. After 2018 only the development years 2019–22; the 2023–25 holdout is left alone.

## Commands
```bash
python scripts/download/topaz_ocean_heat.py --years 1995-2025
python scripts/download/nextsim_reanalysis.py --variables sithick
python experiments/2026-10-29_ocean_input_and_teacher_checks/analysis.py
```

## Results
Full output: `output.txt`. Sectors are cells north of 64° N.

**1. TOPAZ4b agrees with neXtSIM-OPA's ocean in the mixed layer, not below it.**

| sector | 0–20 m trend per decade, OPA / TOPAZ (MJ m⁻²) | r year to year (detrended) | 20–100 m trend per decade, OPA / TOPAZ | r year to year (detrended) |
| --- | --- | --- | --- | --- |
| East Siberian | +11 / +8 | 0.90 | +15 / +23 | 0.17 |
| Chukchi | +10 / +13 | 0.94 | +26 / +26 | 0.02 |
| Beaufort | +6 / +3 | 0.92 | **+39 / +2** | −0.28 |
| Laptev | +8 / +10 | 0.84 | **−3 / +63** | −0.48 |

In the mixed layer the two oceans have similar levels and trends and track each other year by year. Below 20 m they don't: TOPAZ holds 15–50% more heat, has no Beaufort rise (where neXtSIM-OPA's subsurface heat matched Frazil's error) and a strong Laptev rise (where Frazil is not too icy), and the year-to-year agreement is nil. Both are plausible readings of the literature (Beaufort Gyre heat gain; Atlantic-water heat reaching the eastern Eurasian shelves), so this says nothing about which is right, only that they are different oceans below 20 m. Caveat: TOPAZ is interpolated to standard depths, and its bathymetry on the shallow shelves may differ from NEMO's, which would affect the levels more than the trends.

**2a. September extent minus NSIDC-0051 (M km²): the reanalysis is too icy on the Pacific shelves too.**

| | East Siberian | Chukchi | Beaufort | Laptev |
| --- | --- | --- | --- | --- |
| reanalysis, 1995–2018 | +0.14 | +0.08 | +0.11 | −0.16 |
| neXtSIM-OPA raw, 1995–2018 | +0.05 | +0.12 | +0.06 | −0.06 |
| v0.4.0 targets (QM), 1995–2018 | 0.00 | 0.00 | 0.00 | 0.00 |
| reanalysis, 2019–22 | +0.12 | +0.06 | +0.20 | −0.12 |
| Frazil v0.4.0, 2019–22 | +0.16 | +0.09 | +0.01 | −0.07 |

Despite assimilating concentration, the reanalysis has as much or more September excess on the Pacific shelves as the current teacher (its better pan-Arctic September in ISS-022 is partly too little ice in the Laptev cancelling too much on the Pacific side). As a teacher it would need the same QM correction, and wouldn't by itself fix the shelves.

**2b. Winter thickness: the teacher's first-year ice on the Pacific shelves is too thick, and Frazil inherits it.** Model minus reanalysis (which assimilates CS2SMOS winter thickness), metres, sector area mean:

| | East Siberian | Chukchi | Beaufort | Laptev |
| --- | --- | --- | --- | --- |
| neXtSIM-OPA (= the v0.4.0 thickness target), Nov–Apr, winters 2011–18 | +0.15 | +0.15 | +0.08 | +0.11 |
| neXtSIM-OPA, April | **+0.26** | **+0.29** | +0.14 | +0.14 |
| Frazil v0.4.0, Nov–Apr, winters 2020–22 | +0.14 | +0.17 | +0.10 | +0.07 |
| Frazil v0.4.0, April | **+0.29** | **+0.33** | +0.22 | +0.14 |

The excess grows through the winter and by April is about twice as large on the East Siberian and Chukchi shelves as in the Laptev, matching the v2 paper's description of neXtSIM (first-year ice too thick, more growth than observed by April). The QM correction changed only concentration, so the thickness targets still carry it, and Frazil reproduces it almost exactly, adding a little of its own in the Beaufort (+0.08 m in April). The reanalysis is not truth (CS2SMOS is itself uncertain for thin ice, and it shares neXtSIM's physics), but it is the observation-constrained reference here.

## Conclusion
- **Route A as planned (TOPAZ subsurface heat as an input) is out.** TOPAZ's 20–100 m ocean is not the one the targets felt, and it lacks the Beaufort rise that matched Frazil's error. TOPAZ's mixed layer does match, but the mixed layer doesn't separate the sectors (here and in step 1), so it is a weak signal for this problem.
- **Route B loses its main appeal.** The reanalysis is too icy on the Pacific shelves in September as well, so it would need the same correction; its longer record and constrained thickness remain useful, mainly as a reference.
- **New and actionable: the thickness targets.** The teacher's winter ice on the Pacific shelves is 0.26–0.29 m too thick by April, about twice the Laptev's, and the concentration-only correction left that in Frazil's targets. Thicker April ice survives longer into the summer, which fits a September excess on exactly these shelves. Next: correct the thickness targets against the reanalysis over the CryoSat era (winters 2011–18, per cell and month, like the concentration correction; summer months need a choice, since CS2SMOS has no summer), fine-tune, and score the shelves.
- **The ocean-heat idea isn't dead,** but its source has to be the teacher's own ocean: neXtSIM-OPA's heat over the whole domain as an extra state Frazil predicts (~9 GB to stream), not an outside reanalysis.
