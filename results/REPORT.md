# Experiment results report

_generated from results/experiments.jsonl (19 records)_

## Runs

| when (UTC) | experiment | model | note | run_id |
|---|---|---|---|---|
| 2026-09-27T21:06 | freerun_monthly | monthly_pf_qm |  | `d556f1e3105f` |
| 2026-09-27T16:19 | freerun_monthly | monthly_pf_qm |  | `f3d9c5bc56e2` |
| 2026-09-27T14:50 | freerun_monthly | monthly_pf_qm_sst |  | `24a126bd2129` |
| 2026-09-27T05:14 | freerun_monthly | monthly_pf_qm_sst |  | `746d3defa31e` |
| 2026-09-27T01:09 | freerun_monthly | monthly_pf_qm_sst |  | `6075d86da732` |
| 2026-09-26T23:37 | freerun_monthly | monthly_pf_qm_sst |  | `05663de3d51f` |
| 2026-09-25T18:37 | freerun_monthly | monthly_pf | 2019-25 hindcast with per-member extent + member | `e0e924e7278d` |
| 2026-09-25T05:07 | freerun_monthly | monthly_pf_qm |  | `067961ec04ed` |
| 2026-09-25T03:49 | freerun_monthly | monthly_pf |  | `ef352360aafa` |
| 2026-09-23T11:02 | freerun_monthly | monthly_pf | ERA5 hindcast, harder pushforward (55k) | `674662ed06e3` |
| 2026-09-22T13:01 | freerun_monthly | monthly_qm | ERA5-forced hindcast to isolate forcing vs CMIP | `0130863795b9` |
| 2026-09-22T05:53 | rollout_monthly | monthly_qm | sic quantile-mapped | `98c27bccd2ca` |
| 2026-09-22T05:19 | freerun_monthly | monthly_qm |  | `33abca34c6cc` |
| 2026-09-20T01:39 | freerun_monthly | monthly_pf |  | `114f5eacb838` |
| 2026-09-19T22:07 | freerun_monthly | monthly | baseline drift | `b82393f7b3a1` |
| 2026-09-19T16:11 | rollout_monthly | monthly_pf | pushforward 20k | `aa3665d36b0d` |
| 2026-09-19T15:36 | rollout_monthly | monthly | baseline monthly.ckpt | `230e9ac8d153` |
| 2026-09-19T15:00 | project |  | start of results tracking: monthly baseline + pu | `bf9245feb56d` |
| 2026-09-19T14:02 | rollout_monthly | monthly_pf | pushforward 20k [BACK-FILLED from plots/rollout_ | `3afe6e18abd5` |

## rollout_monthly — skill vs climatology (mean of 6 vars, %)

| model | note | 1mo | 3mo | 6mo | 12mo |
|---|---|---|---|---|---|
| monthly | baseline monthly.ckpt | +45 | +26 | +20 | +20 |
| monthly_pf | pushforward 20k | +46 | +28 | +22 | +22 |
| monthly_pf | pushforward 20k [BACK-FI | +46 | +28 | +22 | +22 |
| monthly_qm | sic quantile-mapped | +46 | +28 | +22 | +21 |

## freerun_monthly — drift (higher retention / smaller drift = better)

| model | note | yrs | sharp SIT retention | area drift (Mkm²) | non-finite |
|---|---|---|---|---|---|
| monthly | baseline drift | 15.0 | 0.586 | -0.30 | 0 |
| monthly_pf |  | 15.0 | 0.615 | -0.28 | 0 |
| monthly_qm |  | 30.0 | 0.428 | -0.53 | 0 |
| monthly_qm | ERA5-forced hindcast to  | 5.0 | 0.774 | -0.22 | 0 |
| monthly_pf | ERA5 hindcast, harder pu | 5.0 | 0.802 | -0.14 | 0 |
| monthly_pf |  | 7.0 | 0.526 | -0.61 | 0 |
| monthly_pf_qm |  | 7.0 | 0.543 | -0.54 | 0 |
| monthly_pf | 2019-25 hindcast with pe | 7.0 | 0.526 | -0.61 | 0 |
| monthly_pf_qm_sst |  | 7.0 | 0.540 | -0.56 | 0 |
| monthly_pf_qm_sst |  | 7.0 | 0.541 | -0.53 | 0 |
| monthly_pf_qm_sst |  | 7.0 | 0.542 | -0.54 | 0 |
| monthly_pf_qm_sst |  | 7.0 | 0.543 | -0.49 | 0 |
| monthly_pf_qm |  | 7.0 | 0.543 | -0.54 | 0 |
| monthly_pf_qm |  | 85.0 | 0.307 | -1.77 | 0 |
