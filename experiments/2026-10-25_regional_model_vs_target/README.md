# Model or target? The summer excess by region

**Date:** 2026-10-25 · **Branch:** `exp/regional-model-vs-target` · **Status:** done · **Issues:** ISS-002, ISS-028

## Question
Two errors survived every change in v0.4.0: the East Siberian Sea is +0.1 to +0.25 M km² too icy every September, and the Beaufort Sea about +0.3 from 2023. For each region, how much of the model's error against observations is its own (model minus its training target) and how much is already in the target (target minus NSIDC)? Roadmap *Next*, analyses 1 and 2.

## Setup
- **Runs:** `drift.py --model monthly_pf_qm_ema_qmw --run --fields`: the drift protocol's free-runs (each December 2008–2014 to December 2018, ERA5 weather, 2 members), now also saving every member's concentration for June–October (`freerun_monthly.py --snapshot-months`). Runs are exactly repeatable, so the drift numbers match the earlier ones.
- **Analysis** (`analysis.py`, this folder): per region and summer month, 2009–18: model − target, target − NSIDC and model − NSIDC; model − target by years since the start; and, from the datacubes alone, target − NSIDC every September 1995–2018 for neXtSIM, the fixed-map QM and the 9-year-window QM targets. Cells where NSIDC has no data are left out of every term (ISS-027).
- No holdout years: everything is before 2019.

**Reading it:** if the East Siberian excess sits in model − target, the model doesn't reproduce its own target there (a modelling problem); if it sits in target − NSIDC, the training data carries it (a data problem). Beaufort 2023–25 is outside this period; the 2009–18 Beaufort split shows whether either part was already there before.

## Commands
```bash
python scripts/evaluate/drift.py --model monthly_pf_qm_ema_qmw --run --fields
python experiments/2026-10-25_regional_model_vs_target/analysis.py
```

## Runs
| what | ledger run_id |
| --- | --- |
| drift runs with fields (drift numbers identical to `71093e5f2af0`) | `79444e8711a1` |

## Results
`monthly_pf_qm_ema_qmw`, 7 starts × 2 members, 2009–18, M km² (± standard error over starts).

**September, by region:**

| | model − target | target − NSIDC | model − NSIDC |
| --- | --- | --- | --- |
| all | +0.17 ± 0.03 | −0.07 | +0.10 |
| East Siberian | **+0.09 ± 0.01** | −0.02 | +0.07 |
| Chukchi | **+0.12 ± 0.00** | −0.01 | +0.12 |
| Beaufort | **+0.11 ± 0.01** | −0.01 | +0.10 |
| Laptev | −0.03 | −0.01 | −0.04 |
| Barents/Kara | −0.04 | +0.01 | −0.04 |
| Canadian/Baffin/Hudson | −0.06 | −0.03 | −0.09 |

**The Pacific-shelf excess is the model's own.** In the East Siberian, Chukchi and Beaufort seas the model sits 0.09–0.12 above its own target, while the target matches NSIDC to within 0.02 (and, over 1995–2018, to within 0.01 in all three: the QM correction did its job). It's a modelling problem, not a data problem.

**When:** it appears in August (East Siberian +0.08, Beaufort +0.09) and September; June and July are on target, and October is below it (East Siberian −0.07): too little late-summer melt, and a later refreeze.

**It builds up over a run.** September model − target by years since the start: East Siberian 0.00 in year 1, 0.05–0.09 in years 2–4, 0.10–0.18 in years 5–8; Beaufort 0.07 → 0.13–0.17. Chukchi is +0.10 from the first year and doesn't grow. Pan-Arctic, the drift protocol saw no drift because these are offset by too little ice elsewhere (Laptev, Barents/Kara, Canadian Arctic) and in other months.

**Through thickness.** The model's June ice thickness, model − target, grows from about 0 in year 1 to +0.15–0.20 m by years 5–8 in the East Siberian and Beaufort seas (about 10% of their 1.7–1.8 m), but not in the Chukchi Sea:

| June thickness, model − target (m), by years since start | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| East Siberian | −0.06 | −0.01 | +0.02 | +0.07 | +0.16 | +0.15 | +0.11 | +0.20 |
| Beaufort | −0.01 | +0.01 | +0.08 | +0.10 | +0.13 | +0.07 | +0.18 | +0.19 |
| Chukchi | −0.00 | −0.03 | −0.01 | +0.09 | −0.00 | +0.01 | −0.02 | +0.06 |

## Conclusion
Two mechanisms, both the model's own. **East Siberian and Beaufort:** the ice thickens slowly beyond its target year after year (regional thickness drift), so more of it survives each summer and the September excess grows over the first 4–5 years. **Chukchi:** too little late-summer melt from the first year, without thickening.

This also bears on the Beaufort 2023–25 excess (ISS-028): in the 2019–25 hindcast those are years 5–7 since the December 2018 start, where the Beaufort's own excess here has grown to +0.11–0.17. Part of the 2023–25 jump is this build-up, not only something specific to those summers.

Next: why the ice thickens there (winter growth too strong, or summer thinning too weak), from runs saving winter thickness; and a fix that targets regional thickness drift.
