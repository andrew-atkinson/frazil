# Review of data and model assumptions

**Date:** 2026-11-14 · **Branch:** `exp/loose-ice` · **Status:** in progress (review done; ISS-030 shown to shape the loose-ice error; teacher-space test next) · **Issues:** ISS-029, ISS-030

## Question

Frazil still differs from the established numbers it is scored against. Those are NSIDC-0051 observations and the neXtSIM-OPA teacher simulation.
- **Out of sample (2019–22), against NSIDC:** June +0.66, August +0.08 and September +0.08 M km². The holdout's September is +0.34.
- **One month at a time, against its own targets:** on the Pacific shelves the error is +0.22 M km² in August. Loose ice doesn't open up enough, and too little of it ends partly open (0–15%).

Most experiments so far have tested missing inputs: SST, radiation, degree-days, ocean memory and ocean heat. All of them had little or no effect. This review asks the other question. Is something in the data or the model's set-up working against the physics?

## What the review found

**1. Augmentation: only the rotation is ever applied (bug, ISS-029).** `DataAugmentation.forward` (`gensim/augmentation.py`) applies each transform to the *original* tensor and keeps only the last result. So the horizontal and vertical flips never happen, but their labels (fed to the network as conditioning) still say they did, half the time each. The code is inherited from GenSIM. Checked directly: with hflip at probability 1 and the others at 0, the output is unflipped and the label says flipped.

**2. Augmentation rotates the fields but not the vectors (scientific, ISS-029).** The 90° rotation moves every channel's array. Drift (`siu`, `siv`) and wind (`uas`, `vas`) keep their components in the old frame. In half of all training samples, the drift and wind therefore point the wrong way relative to the coastlines, the ice edge and each other's spatial pattern. The angle between drift and wind is kept, because neither component is rotated. The rotation label is correct, so the network *can* learn this as a separate case. But it has to learn advection and wind-driven opening of the pack twice, once of them with unphysical geometry. Wind-driven divergence at the edge is how loose ice opens.

**3. The concentration correction leaves the other variables behind (scientific, ISS-030).** QM changes `sic` only. Thickness `sit` is a grid-cell mean (about 0.1 m where `sic` is 5–15%, 2.2 m where it is above 90%), and it is left as the teacher had it. From `qm_consistency.py`, 1995–2014:

| | Pacific, Aug | Pacific, Sep | Barents/Kara, Aug | Barents/Kara, Sep |
| --- | --- | --- | --- | --- |
| area where QM `sic` = 0 but `sit` > 0.1 m | 0.7% | 2.3% | 5.9% | 4.9% |
| area where QM `sic` < 15% but the teacher had > 50% | 0.9% | 3.2% | 4.0% | 4.1% |
| implied thickness per unit of ice (`sit`/`sic`), teacher → QM | 1.98 → 2.07 m | 1.57 → 1.91 m | 1.77 → 1.62 m | 1.36 → 1.48 m |

So the targets the model learns from include open water with ice volume in it, and ice cover whose thickness doesn't match. In the Pacific in September, the per-ice thickness is a fifth higher than the teacher's. The model has to predict both variables from one state, and no physics links them the way the targets do.

**4. The QM maps are fitted on 9 values each.** The 9-year window gives each cell and calendar month 9 samples for 21 quantile knots. That is close to swapping in the observation of the same rank. On 2015–18 the targets are therefore nearly the observations themselves, re-ordered by the teacher. So the validation loss and the one-month skill partly measure how well the model reproduces NSIDC. Each cell is also mapped separately, which adds noise from cell to cell.

**5. Drift and wind are in the same frame (ruled out).** `drift_vs_wind.py`: in winter, ice above 90% concentration with wind above 3 m s⁻¹ drifts 20–31° to the right of the wind at 1.5–2% of its speed, in every 45° longitude band. One exception is 225–270° E, the Canadian Archipelago: 0.26%, as expected for landfast and thick ice. A missed rotation in the regridding would show an angle that changes with longitude. It doesn't.

**6. The members don't simply copy the uncorrected teacher.** `raw_vs_qm.py`: the teacher also puts more loose ice in the partly-open band than the members do, so the gap isn't the model ignoring the correction. But the members' *mean* concentration on the Pacific shelves sits between the QM target and the teacher. In August, from 50–80%: QM 0.32, members 0.39, teacher 0.35 (2015–18). That fits a correction that imprinted on `sic` while the dynamics underneath stayed the teacher's.

**Not checked yet:**
- **The teacher's atmosphere.** The repo doesn't record which atmosphere forced the neXtSIM-OPA run. If it wasn't ERA5, the model's inputs aren't the weather that made its targets, and part of the error can't be learned away.
- **How uncertain the observations are.** NSIDC-0051 (NASA Team) is known to read low in summer, when melt ponds look like open water. It also has no data on about 0.45 M km² of coastal ocean (ISS-027). The NSIDC Sea Ice Index uses the same algorithm, so the two "established" numbers share these biases.
- **Monthly means.** A month that is ice-free from mid-August averages to about 50%. With monthly-mean weather, the model can't tell when in the month the ice went. This is structural, and the weekly step (ISS-014) is the fix, not anything below.

## Follow-up on ISS-029 and ISS-030 (2026-11-14, no model calls)

**7. ISS-030 is the stronger lead: the targets break the link between thickness and opening, and the model's error follows the break.** `qm_error_location.py` groups cells by the model's inputs only.

In the teacher, thin loose ice opens far more often than thick loose ice. In the QM targets that relation is flattened or reversed. Share of cells starting at 50–80% in July that end below 15% in August, 1995–2014, by the start's thickness per unit of ice (`sit`/`sic`):

| | under 1 m | 1–2 m | over 2 m |
| --- | --- | --- | --- |
| Pacific shelves, teacher | 74% | 30% | 10% |
| Pacific shelves, QM targets | 48% | 26% | 20% |
| Barents/Kara, teacher | 79% | 31% | 12% |
| Barents/Kara, QM targets | 17% | 27% | 46% |

- **Why:** QM moved the starting `sic` by more than 15 points in most loose-ice cells, and left `sit` alone. Where it lowered `sic`, the cell holds thick ice at low cover that the observations then remove. Where it raised `sic`, the cell holds thin ice that the observations keep. On the Pacific shelves in August (2015–18), only 20% of cells starting at 50–80% were moved by less than 5 points.
- **v0.4.0 keeps thick ice and melts thin ice, as the teacher's physics would.** Its one-month bias from 50–80% in August, by the input's thickness:

| | under 1 m | 1–2 m | over 2 m |
| --- | --- | --- | --- |
| Pacific shelves, 2011–14 (trained on) | −0.01 | +0.03 | +0.08 |
| Pacific shelves, 2015–18 | +0.02 | +0.07 | +0.09 |
| Barents/Kara, 2011–14 (trained on) | −0.11 | −0.07 | 0.00 |
| Barents/Kara, 2015–18 | −0.21 | −0.09 | +0.07 |

- **On the Pacific shelves in the training years, four fifths of that class's bias comes from the cells over 2 m** (0.040 of 0.050, weighting by area). That is the loose-ice gap: ice the inputs call thick, which the targets open and the model doesn't.
- **The error's sign follows the direction QM moved the start.** Barents/Kara, August, 50–80%, 2015–18: +0.09 where QM lowered the start by more than 15 points, −0.19 where it raised it by more than 15.
- **Not all of it.** On the Pacific shelves in September 2015–18 the bias is +0.07 to +0.13 in every group. That part isn't sorted by thickness or by the QM move.

**8. The QM targets follow the drift half as well as the teacher's** (`vector_frame.py`). Rank correlation between the ice transport implied by the drift, −div(`sic` × drift) over a month, and the actual change of `sic`, over marginal ice: teacher 0.16 (October–March), 0.19 (April–June), 0.21 (July–September); QM targets 0.07, 0.05, 0.10. The drift is the teacher's in both.

**9. ISS-029 confirmed in detail; its cost is still unmeasured.**
- **Same code as upstream GenSIM** (only the patch clamp differs), active at probability 0.5 per transform in every training config. In effect half the samples are rotated and half untouched; the two flip labels are coin tosses. Inference uses all-zero labels.
- **The components do line up with the grid's axes, and a 90° misreading destroys their meaning.** Same transport test with the drift read eight ways: the true reading gives 0.16–0.21, the two 90° readings −0.07 to +0.07, the mirror images within ±0.04. So a rotated training sample shows the network a drift with no usable relation to the ice change, unless it learns to turn the vectors itself from the rotation label.
- **What that costs the unrotated case the model is run in can't be told from data.** It needs C1 or D1.

## Plan

Cheapest and most decisive first. All of A and B are analysis only, which I run. Every model call in C and D is yours to start; I'll give you the line. Training A/Bs repeat v0.4.0's recipe exactly (seeded from `monthly_pf_qm`, 20k steps, EMA 0.999, seed 1), so v0.4.0 is the comparison for each. Run-to-run noise is about 0.01 M km² on September bias and 0.05–0.08 on July and August.

**A. Finish what is running**
- A1. **Done:** `pf_qm_ema_qmw_sic0` doesn't help. It moves members from 0 into 0–15%, and the Pacific one-month extent error rises slightly (August +0.22 → +0.28 M km²); see the loose-ice README.
- A2. **Decided:** skip `pf_qm_ema_qmw_nocens`. Censoring at 0 alone made no difference to opening, so the GPU time goes to C1.

**B. Checks without training**
- B1. **The teacher's forcing.** Read the run's metadata on the SASIP server and the paper describing it. If it's not ERA5, compare the two atmospheres over the Pacific shelves in summer. That difference is a floor on what any input can fix.
- B2. **Done, see finding 7.** Does the error sit where QM moved the targets? For the training years 2011–14, split v0.4.0's one-month concentration error and partly-open shares by how far QM moved the starting cell (|QM − teacher|) and by the inconsistent cells from finding 3, within each starting class. If the error gathers there, the targets are the cause. If it's spread evenly, they're not.
- B3. **How noisy the QM maps are.** Bootstrap the 9-year fits (resample years) and measure how much each cell's mapped value moves. Then compare with the model's one-month error on the shelves. If the map noise is the same size, part of the "error" is noise in the targets.
- B4. **Observational uncertainty.** Score v0.4.0's 2019–22 hindcast against a second product (NSIDC-0079 Bootstrap, or OSI-SAF). Do this for the Pacific shelves in August–September and the Atlantic side in June, to see how much of each gap is inside the spread between products. This needs a download, so I'll ask first with the file names and sizes.

**C. Training A/Bs (about 11 h each, one at a time)**
- C0. **A first look at C2 without training.** `monthly_pf` is an existing model trained on the teacher's own states. Run it one month at a time on the teacher cube for August and September (2015–18 and 2011–14, about 25 minutes each), saving fields. Then, with no model calls: (a) check whether it opens loose ice by thickness as the teacher does; (b) apply the windowed QM maps to its members and score them against the QM target beside v0.4.0. Not a clean A/B (60k steps, no EMA fine-tune), but it shows whether the loose-ice gap exists in teacher space at all.
- C1. **Augmentation fixed** (ISS-029). Chain the transforms, and transform the vector components with the fields: flip x negates `u`, flip y negates `v`, and a 90° rotation swaps and negates them. Do this for drift and wind alike. Add a test that rotating a uniform eastward wind gives a uniform wind along the new x axis. If the fix moves the Pacific August error or the partly-open share by more than the noise, the augmentation was costing physics.
- C2. **Consistent targets** (ISS-030), without a proportional rule. Scaling `sit` by the change in `sic` would be one. Instead, train on the teacher's own, consistent states with the same recipe, and apply the QM to the *predictions*: each member's `sic` after every step, using the windowed maps as now. Two comparisons:
  - with v0.4.0 on the 2019–22 scorecard;
  - with the QM target on the one-month check.

  If it matches or beats v0.4.0, the correction belongs in post-processing and the model learns undistorted physics. Caveat: after 2018 the maps can only use the last window. That is the same limit v0.4.0's targets have.
- C3. **Only after C1 and C2.** Combine whichever helped, and re-run the loose-ice tables and the full scorecard.

**D. Model-call diagnostic (optional, about 2 h)**
- D1. Run v0.4.0 one month at a time on inputs rotated 90°, with the rotation label set as in training, then rotate the output back and score it. If the error on the Pacific shelves is clearly larger than unrotated, the "rotated" physics was learned badly. That would make C1 more likely to pay off before spending 11 hours on it.

**Order (revised after findings 7–9):** C0 → C2 if C0 holds → C1 → B4 when you approve the download. B1 and B3 stay open; B2 is done (finding 7). D1 only if C1 needs a cheaper first look.

## Files

- `qm_consistency.py`, `qm_consistency_output.txt`: finding 3, plus the 0–15% band in the teacher, QM and NSIDC.
- `drift_vs_wind.py`, `drift_vs_wind_output.txt`: finding 5.
- `raw_vs_qm.py`, `raw_vs_qm_output.txt`: finding 6.
- `qm_error_location.py`, `qm_error_location_output.txt`: finding 7.
- `vector_frame.py`, `vector_frame_output.txt`: findings 8 and 9.
