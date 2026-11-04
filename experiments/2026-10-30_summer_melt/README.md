# Frazil's summer melt against its targets

**Date:** 2026-10-30 · **Branch:** `exp/sit-target-correction` · **Status:** done (no training) · **Issues:** ISS-025, ISS-028

## Question
The thickness sensitivity test ([sit_target_correction](../2026-10-29_sit_target_correction/)) showed Frazil learns the summer melt of its targets. Does v0.4.0 melt as much ice as its own targets do, and if so, where does its Pacific-shelf September excess come from?

## Setup
- **Free runs already on disk:** the drift runs (`drift.py --run --fields`), 2 members, started every December 2008–2014 and run to December 2018, for v0.4.0 (`monthly_pf_qm_ema_qmw`) and the thickness test (`monthly_pf_qm_ema_qmw_sit`). Each is compared month by month with the targets it was trained on. Every summer counts once per start (lead year 1 = the first summer after the start).
- **Pan-Arctic:** the runs' monthly mean thickness and concentration (unweighted mean over ocean cells); the target computed the same way. Start-state check: 0.0000 m.
- **By sector, June–September** (the months the drift runs saved as fields): area-mean thickness from the ensemble-mean state, and from each member's concentration, averaged over members as the scorecard does: extent (>15%), ice area, and the extent of loose ice (15–80%).
- `analysis.py`, `output.txt`.

## Results
**Frazil melts as much thickness as its targets.** April-to-September loss, pan-Arctic mean (m):

| | Frazil | target | ratio |
| --- | --- | --- | --- |
| v0.4.0, first summer | 0.85 | 0.83 | 1.02 |
| v0.4.0, fourth summer on | 0.87 | 0.86 | 1.02 |
| thickness test, all summers | 0.71–0.74 | 0.69–0.72 | 1.02–1.04 |

Its monthly mean thickness stays within 0.02 m of the target in every month, and its concentration within 0.01. From June to September it loses 2–8% *more* thickness than the target in every Pacific and Laptev sector. The thickness test learned its targets' weaker melt just as faithfully, which is why its September got worse.

**On the Pacific shelves the ice cover doesn't retreat in August, although the thickness does.** v0.4.0 minus its target, all summers:

| | June | July | August | September |
| --- | --- | --- | --- | --- |
| extent, East Siberian / Chukchi / Beaufort (M km²) | +0.00 / +0.03 / +0.01 | −0.04 / −0.00 / −0.01 | +0.08 / +0.12 / +0.08 | +0.09 / +0.12 / +0.11 |
| thickness, same sectors (m) | +0.07 / +0.01 / +0.09 | +0.08 / +0.02 / +0.10 | +0.05 / +0.00 / +0.10 | +0.00 / −0.01 / +0.06 |

The extent is right in July and 0.1 M km² too large in each sector by August. In the Chukchi the thickness is right all summer, so the excess there is ice cover alone.

**Two parts: one there from the first summer, one that builds up.** August–September, the three Pacific sectors together, v0.4.0 minus its target:

| summer after the start | extent (M km²) | ice area (M km²) | loose ice, 15–80% (M km²) | thickness (m) |
| --- | --- | --- | --- | --- |
| 1st | +0.15 | +0.18 | −0.07 | −0.02 |
| 2nd | +0.27 | +0.19 | +0.17 | +0.00 |
| 3rd | +0.27 | +0.26 | +0.08 | +0.02 |
| 4th on | +0.35 | +0.24 | +0.28 | +0.06 |

- **The first summer:** 0.15 M km² too much ice, at the right thickness and as compact ice, not loose. Eight months after starting from the target's own state, Frazil melts the thickness but keeps the concentration up in August.
- **Later summers:** the excess doubles, mostly as loose ice at the edge, while the thickness slowly creeps up (+0.06 m). That is the regional build-up found in [regional_model_vs_target](../2026-10-25_regional_model_vs_target/).
- The thickness test has the same shelf pattern (+0.17 in the first summer, +0.37 from the fourth). Its larger September excess sits outside these sectors.

## Conclusion
- **Summer thickness melt is not Frazil's error.** It matches whatever melt its targets have, to within a few percent.
- **The shelf excess is in how concentration follows the melt.** In August Frazil keeps too much ice cover for the thickness it has, from the first summer on. That suggests the model hasn't learned the late-summer link between thinning and opening up on the shelves (the teacher's lateral melt and open-water feedback), or that the information it would need, such as the ocean's stored heat, isn't in its inputs.
- **Next:** the same check one month at a time from the target's own state (below).

## One month at a time
If August's ice cover is too high even when each month starts from the target's state, the error is in the learned one-month step. If not, it builds up between May and August. `eval_monthly.py --save-fields` now saves each one-month prediction (the ensemble-mean state and each member's concentration); `one_month.py` compares them with the target by sector. Validation years 2015–2018, 8 members.

```bash
python scripts/evaluate/eval_monthly.py --ckpt data/models/monthly_pf_qm_ema_qmw/last.ckpt --datacube data/train_data/monthly_datacube_sicqmw9 --n-ens 8 --save-fields data/models/monthly_pf_qm_ema_qmw/one_month_fields --note "pf_qm_ema_qmw one-month skill, fields saved (summer_melt)"
python experiments/2026-10-30_summer_melt/one_month.py
```
Ledger run `63c673161c0b` (one-month skill identical to v0.4.0's scorecard: concentration 0.58, thickness 0.51); `one_month_output.txt`.

**The error is in the one-month step.** Started from the target's own July, Frazil's August shelf ice cover is already too high, and by more than in the free runs' first summer (+0.15). One step minus target, East Siberian + Chukchi + Beaufort:

| predicted month | extent (M km²) | ice area (M km²) | loose ice, 15–80% (M km²) | thickness (m) |
| --- | --- | --- | --- | --- |
| July | −0.06 | +0.07 | −0.21 | +0.05 |
| August | **+0.22** | +0.10 | +0.38 | +0.01 |
| September | **+0.23** | +0.22 | −0.03 | −0.01 |
| October | **−0.23** | −0.07 | −0.26 | −0.00 |

Every year has it: August +0.19, +0.48, +0.07, +0.15 and September +0.29, +0.31, +0.10, +0.21 for 2015–18. The thickness is right in both months.

**Step size as a share of the target's** (superseded below: a single ratio assumes the error is proportional to the change, and September contradicts that):

| month | shelves: target's change (M km²) | Frazil / target | pan-Arctic: target's change | Frazil / target |
| --- | --- | --- | --- | --- |
| June | −0.25 | 1.00 | −1.46 | 0.87 |
| July | −0.44 | 1.14 | −2.61 | 0.99 |
| August | −1.25 | **0.82** | −3.04 | 0.92 |
| September | −0.58 | **0.61** | −0.93 | **0.79** |
| October | +1.27 | **0.82** | +2.61 | 0.94 |
| November | +0.93 | 0.97 | +2.78 | 0.97 |

September has a smaller change than August but the worst ratio, and July overshoots, so the error is not proportional to the size of the change.

### What shape is the error? (`step_shape.py`, `step_shape_output.txt`)
Cell by cell, June–November 2015–18:
- **Against the size of the change: no usable shape.** Polynomials of degree 2 and 3 in the target's change fit Frazil's change no better than a line (shelves, summer: RMSE 0.1127, 0.1127, 0.1126). The binned curve does look damped (large changes 0.7–0.85 of the target's), but binning on the true change produces that for *any* imperfect forecast (regression dilution), so it says nothing about a bias. The error has to be conditioned on what the model sees, not on the answer.
- **Against the starting state: curved.** Shelves, error in the month's concentration change by the cell's starting concentration (positive = too little melt):

| starting concentration | June | July | August | September | share of shelf ice, August |
| --- | --- | --- | --- | --- | --- |
| 15–50% | −0.02 | −0.02 | +0.02 | **+0.10** | 10% |
| 50–80% | −0.00 | +0.03 | **+0.07** | **+0.08** | 55% |
| 80–95% | −0.03 | +0.03 | −0.05 | +0.01 | 33% |
| over 95% | −0.02 | −0.01 | **−0.12** | −0.04 | 1% |

In August Frazil melts compact ice *too fast* and loose ice too slowly. It pulls concentrations towards the middle, while in the target loose ice opens up quickly. Real melt-out is a threshold process: once the pack opens, the open water takes up heat and the rest goes fast (lateral melt, ice–albedo feedback). Frazil treats it as gradual, and most of the shelf ice is in the 50–80% range in August, so the net is too much ice cover. That is the nonlinearity: in the state (concentration, season), not in the size of the step. Four summers only.

### Per sector, across time: polynomial or not? (`sector_poly.py`, `sector_poly_output.txt`)
Per sector (the scorecard's seven longitude bands), each one-step month's extent error as a share of the sector, fitted as a polynomial of degree 0–4 in either the sector's starting ice cover (an input the model sees) or the time of year. Melt (May–September) and freeze (October–April) are fitted separately, since the same cover means different things on the way down and up. Each degree is scored on years it wasn't fitted on (leave one year out), so a higher degree only counts if it predicts an unseen year better.

- **In starting cover, nothing beyond a straight line holds up.** In the melt half a line beats a constant for East Siberian, Chukchi, Beaufort and Barents/Kara. Degrees 2–3 fit the fitted years better but predict the unseen year worse (Beaufort, degree 3: 6.9 vs 3.9 for the line). In the freeze half, Chukchi and Beaufort take a quadratic, gaining little over the line (0.1 points).
- **The line: on the Pacific shelves the error grows as the sector opens up.** Error (% of the sector) ≈ a + b × cover, melt half:

| sector | a | b | error at half cover | R² |
| --- | --- | --- | --- | --- |
| Beaufort | +17.3 | −16.8 | +8.9% | 0.32 |
| Chukchi | +14.3 | −13.2 | +7.7% | 0.27 |
| East Siberian | +6.7 | −7.7 | +2.8% | 0.27 |
| Laptev | +5.3 | −5.5 | +2.5% (the line doesn't beat a constant) | 0.16 |
| Barents/Kara | −0.9 | +4.4 | +1.3% (opposite: too slow early in the melt, June) | 0.19 |

  Near zero at full cover, rising as the pack opens: the sector-scale form of the cell-level result (loose ice doesn't open up fast enough). It explains about a third of the variance at most.
- **In time of year, the higher degrees are not laws.** Beaufort and Barents/Kara pick a cubic, and the East Siberian a quadratic, but each half has only 5–7 distinct months. A cubic through five month values is a table of monthly means, not a smooth relationship.
- **Across years:** the melt-half mean error is positive every year on the Chukchi (+0.01 to +0.09 M km²) and, except 2017, on the Beaufort; 2017 is small everywhere. Four years can't support a polynomial in time.
- **Straight lines per sector, both halves** (error as % of the sector = a + b × starting cover):

| sector | melt: b (range leaving out one year) | melt R² | freeze: b (range) | freeze R² |
| --- | --- | --- | --- | --- |
| East Siberian | −7.7 (−8.6 to −5.5) | 0.27 | +11.4 (+5.2 to +14.7) | 0.50 |
| Chukchi | −13.2 (−17.0 to −7.8) | 0.27 | +7.9 (+6.0 to +9.9) | 0.53 |
| Beaufort | −16.8 (−21.3 to −12.9) | 0.32 | +4.1 (+3.4 to +4.9) | 0.62 |
| Laptev | −5.5 (−9.4 to −2.4), not better than a constant | 0.16 | none | 0.00 |
| Barents/Kara | +4.4 (+3.5 to +5.0) | 0.19 | none | 0.01 |
| Canadian/Baffin/Hudson | +2.0 | 0.15 | −1.5 | 0.12 |
| Greenland | +2.6 | 0.12 | none | 0.00 |

  - **On the three Pacific shelves both lines pass through zero at full cover (a + b within ±1%), so the error is proportional to the open part of the sector:** in the melt too much ice, in the freeze too little. With the sector partly open, Frazil's step stays too close to the current state in both directions. The slope keeps its sign in every leave-one-year-out fit. This is conditioned on the model's input (starting cover), so unlike the step-size ratio above it is not a regression-dilution artefact.
  - **One line serves all three Pacific shelves:** out of sample a shared line does as well as one per sector (melt 4.02 vs 3.97, freeze 2.27 vs 2.37). Melt: error ≈ +13% × open share (a +13.3, b −13.2); freeze: ≈ −8% × open share (a −8.0, b +8.3). Over all seven sectors per-sector lines are better (2.91 vs 3.35): the Atlantic side (Barents/Kara, Greenland, Canadian) has small slopes of the opposite sign in the melt.
- **Ceiling:** 20 melt months per sector. If a higher-order shape exists it is below what four summers can show; the 2011–14 one-month run would double the sample.

### Proximity to open water (`proximity.py`, `proximity_output.txt`)
Cell by cell: the error in the month's concentration change against the cell's distance, at the start of the step, to open water (melt half, ice cells) or to the ice (freeze half, open cells). Straight lines in distance, starting concentration and the sector's open share, scored leaving one year out.

- **On the Pacific shelves the error barely changes with distance.** Melt: +0.02 to +0.04 too little melt from the edge out to 400 km, falling only deep in the pack (+0.01 beyond 400 km). Freeze: −0.02 to −0.03 too little freezing in open water at every distance up to 400 km. Elsewhere the errors are small and near the edge of the opposite sign.
- **Distance adds nothing out of sample.** For the shared Pacific line it scores the same as a constant (melt 11.84 vs 11.84), and its slope is about zero (−0.1 per 100 km). The sector's open share is the best of the single predictors (11.51), and the cell's own starting concentration comes next (11.61).
- **So it is not an edge effect.** The whole partly open Pacific shelf is biased together, scaling with how open the sector is. That points to something regional, such as the shelf seas' shallow water and its heat, rather than local ice-edge processes.

### Seafloor depth (`depth.py`, `depth_output.txt`)
Depth from TOPAZ4b's own seafloor (`scripts/download/topaz_depth.py`: the deepest wet level in August 2015, on the model grid; 3% of ocean cells north of 64 N have no TOPAZ water within 30 km and are left out). The sectors run from the coast to the pole, so each has shelf and basin.

- **The error is not in the shallow water.** Pacific shelves, melt half, by depth (error in the month's concentration change, positive = too little melt):

| depth | May | June | July | August | September |
| --- | --- | --- | --- | --- | --- |
| under 25 m | +0.00 | −0.04 | +0.01 | +0.00 | −0.02 |
| 25–50 m | +0.02 | −0.06 | −0.00 | −0.01 | +0.01 |
| 50–100 m | +0.01 | −0.02 | +0.02 | −0.02 | +0.07 |
| 100–200 m | +0.01 | −0.01 | +0.04 | +0.04 | +0.09 |
| 200–500 m | +0.02 | −0.00 | +0.02 | +0.05 | +0.08 |
| 500–1000 m | +0.01 | −0.01 | +0.04 | **+0.09** | **+0.10** |
| over 1000 m | +0.01 | −0.01 | +0.02 | +0.03 | +0.08 |

  The shallow shelf (under 50 m) is about right or melts slightly too fast. The excess is over the shelf break and continental slope (100–1000 m), largest at 500–1000 m in August and September. That is where Pacific Water runs along the Chukchi and Beaufort slopes and spreads under the basin ice, so it fits the ocean-heat idea ([pacific_shelf_ocean_heat](../2026-10-29_pacific_shelf_ocean_heat/)). It is also where the ice still is late in the summer, though, and distance to open water doesn't matter (above).
- **Not unique to the Pacific side.** The Laptev and Barents/Kara also have too little melt at 50–500 m (+0.02 to +0.04), but the opposite over the deep basin (−0.02 over 1000 m, against +0.03 on the Pacific side). The Pacific difference is mostly over the slope and basin.
- **As a predictor, depth adds little.** Pacific shelves, shared line, out of sample: constant 11.84, depth 11.78, open share 11.50, open share + depth 11.49. Shared line: error ≈ −2.6 + 1.6 × log10(depth), so −0.3 at 30 m and +2.0 at 1000 m (×100). Depth matters a little, in the deeper-is-worse direction, but the sector's open share carries most of what can be predicted.

## Conclusion, updated
- **The shelf excess comes from Frazil's own one-month step:** from the true state, August and September shelf ice cover is +0.22 and +0.23 M km² too high, at the right thickness. Drift adds to it in later years (above) but isn't the source.
- **The step is wrong in a state-dependent, nonlinear way, not by a fixed fraction:** in August and September loose ice (15–80%) doesn't open up fast enough, while compact ice melts slightly too fast. Frazil smooths a threshold-like melt-out.
- **Two explanations to separate next.**
  - **Generalisation:** 2015–18 is warmer than the training years. If loose ice opens up as fast as the target's on training years (2011–14), Frazil hasn't learned to extrapolate to faster melt-out, and it lacks something that says the retreat should be faster there, such as the ocean's stored heat ([pacific_shelf_ocean_heat](../2026-10-29_pacific_shelf_ocean_heat/)).
  - **Training:** if the same smoothing shows on training years, the model doesn't represent the threshold at all, a matter of the loss, the sampler or capacity.
  - Either way, compare by starting state and season (`step_shape.py --fields ...`), not by a ratio.
  - **The check:** the same one-month run on 2011–14 (`--val-start-year 2011 --max-pairs 47`, about 2 hours). Done: see below. The answer is both.

## The check: training years 2011–14
The same one-month run on 2011–14, years the model trained on (`eval_monthly.py --val-start-year 2011 --max-pairs 47 --save-fields data/models/monthly_pf_qm_ema_qmw/one_month_fields_2011`, ledger `adcb03c22dbd`), then the same five scripts with `--fields` pointing there. Outputs: `*_output_2011.txt`.

| Pacific shelves, one step from the true state | 2011–14 (trained on) | 2015–18 (held out) |
| --- | --- | --- |
| August extent error, M km² | +0.13 | +0.22 |
| September extent error, M km² | +0.07 | +0.23 |
| October extent error, M km² | −0.10 | −0.23 |
| worst year, August | 2012 +0.27 | 2016 +0.48 |
| August error, 50–80% starting ice | +0.05 | +0.07 |
| September error, 15–50% starting ice | +0.04 | +0.10 |
| August error, >95% starting ice | +0.02 | −0.12 |
| shared melt line in open share (×100) | +5.0 − 4.8 × open share | +13.3 − 13.2 × open share |
| September error at 100–200 m depth | +0.08 | +0.09 |
| August / September error at 500–1000 m | +0.04 / +0.03 | +0.09 / +0.10 |
| one-month sic skill vs persistence | 62% | 58% |

- **Part of the error is there on training years.** Same sign, same months, same place: loose ice doesn't open up fast enough in August, the error depends on the open share, and the worst year is the fastest melt-out (2012, as 2016 later). So Frazil doesn't fully reproduce the threshold-like melt-out even on summers it was trained on. That part is a matter of training: the loss, the sampler or capacity.
- **Most of it grows out of sample.** In 2015–18 the August error is 1.7 times as large, September's 3 times, the October freeze-up error twice, and the open-share slope nearly 3 times as steep. The growth is where it was absent before: compact ice melting too fast (−0.12, against +0.02), and the deeper slope and basin (500–1000 m: about +0.04 rising to +0.09/+0.10). The shelf break (100–200 m) is about the same in both periods (September +0.08 and +0.09).
- **Reading.** The shelf-break part is something Frazil can't fit even in training, and the slope-and-basin part appears only in the warmer held-out years. Both are where Pacific Water carries heat, which Frazil's inputs don't include. That fits a missing ocean input better than a training fault, but this test can't prove it: four years each side, and 2015–18 also differs in ice state.
- **Next test:** the teacher's own ocean heat as an input (below). If it fixes the slope-and-basin part out of sample, the ocean is the lever.

## Ocean heat as an input (training)
Not TOPAZ: its 20–100 m ocean isn't the one the targets felt ([ocean_input_and_teacher_checks](../2026-10-29_ocean_input_and_teacher_checks/)). Instead the training simulation's own heat, neXtSIM-OPA at 0–20 m and 20–100 m (`data/train_data/nextsim_ocean_heat/`, already on disk; nothing streamed).

- **Channels:** `scripts/data/add_ocean_heat_channels.py` puts the heat on the model grid inside the Pacific/Siberian box (north of 64° N, 90° E–110° W), the in-box mean elsewhere, appended to the v0.4.0 targets: `monthly_datacube_sicqmw9_ohc/`, `train_sicqmw9_ohc.zarr`, `validation_sicqmw9_ohc.zarr`.
- **Lagged one month.** Forcings go in at months t and t+1, and heat at t+1 is near zero wherever the t+1 ice is, which would leak the answer (as SST did). The channel at month k holds month k−1's heat, so the model sees months t−1 and t.
- **Model:** `pf_qm_ema_qmw_ohc`: exactly v0.4.0's recipe (from `monthly_pf_qm`, 20k steps, EMA 0.999, seed 1), plus the two channels, whose input weights start at zero. Diagnostic only: neXtSIM-OPA ends in 2018, so no 2019–25 hindcasts.
- **Scoring:** the one-month check on 2015–18 (`eval_monthly.py --datacube data/train_data/monthly_datacube_sicqmw9_ohc --n-ens 8 --max-pairs 47 --save-fields data/models/monthly_pf_qm_ema_qmw_ohc/one_month_fields`), then `one_month.py`, `step_shape.py`, `sector_poly.py` and `depth.py` on it, against v0.4.0's numbers above.
- **Status:** finished, 20,000 steps (the first run stopped at step 12,744; ledger run `cf293c5437fe` scored that partial checkpoint and isn't this model's result). The result below is ledger run `c4a0cd9680e3`.

### Result: it helps in September, and isn't the fix (`compare_ohc.py`, `compare_ohc_output.txt`, `*_output_ohc.txt`)
Paired against v0.4.0 on 2015–18, same targets, same seed.

| Pacific shelves, one step from the true state | v0.4.0 | with ocean heat |
| --- | --- | --- |
| August extent error, M km² | +0.223 | +0.209 |
| September | +0.228 | +0.167 |
| October | −0.225 | −0.237 |
| August / September error at 500–1000 m | +0.086 / +0.101 | +0.084 / +0.078 |
| September error at 100–200 m | +0.085 | +0.067 |
| August error, 50–80% starting ice | +0.069 | +0.069 |
| August error, >95% starting ice | −0.119 | −0.116 |
| shared melt line (×100), out-of-sample RMSE | +13.3 − 13.2 × open share, 4.02 | +11.0 − 11.0 × open share, 4.04 |
| melt-half concentration RMSE | 0.1129 | 0.1113 (−1.4%) |

- **September improves by about a quarter, in all four years** (−0.04, −0.10, −0.05, −0.06). East Siberian +0.035 → +0.007, Beaufort +0.102 → +0.083, Chukchi +0.091 → +0.077, and the slope and basin share in it.
- **August barely moves** (−6%, and 2018 gets slightly worse), and October's freeze-up error gets slightly worse. The starting-state pattern is unchanged: loose ice still opens too slowly and compact ice still melts too fast.
- **Elsewhere nothing changes.** Barents/Kara and Greenland, outside the heat box, move by at most 0.004 M km²; melt-half RMSE outside the box +0.4%. Overall skill against persistence is the same (concentration 58.1% against 58.0%, the rest within 0.1 of a point).
- **The excess isn't concentrated where the teacher's ocean is warm.** In v0.4.0 itself, August–September, inside each starting-concentration class: at 0–20 m the warmest third of cells has less excess in loose ice (+0.03 to +0.04, against +0.09 to +0.13), and compact ice there melts far too fast (−0.28). At 20–100 m the relation is weak and inconsistent (50–80% ice +0.06, +0.06, +0.10 from cool to warm; 15–50% ice +0.04, +0.09, +0.08). This only covers the top 100 m, as monthly means.
- **Weakens the reading in the training-years section.** The depth pattern fitted Pacific Water, but giving the model the ocean the targets felt leaves most of the error, and all of August's, in place. Ocean heat is a contributor in September at most, not the lever.

**What this can't settle:**
- **One pair of trainings.** All four years come from the same two models, so four-for-four says nothing about how much a retraining alone would move September. Sectors outside the box barely moved, which helps, but their errors are small to begin with, so they're a weak control.
- **The channels may be underused.** They started from zero and moved as far as the atmospheric channels did in this training (0.14–0.29 against 0.18–0.38), but that leaves them under a tenth of the atmospheric channels' size (2.3–5.5). September went from +0.201 at step 12,744 to +0.167 at step 20,000, so it may not have finished learning.
- **Next test:** run this trained model on July–October 2015–18 with the heat replaced by its 1995–2014 monthly climatology, no retraining. If the September gain goes, it comes from each year's actual heat; if it stays, from the seasonal pattern or the training path.
