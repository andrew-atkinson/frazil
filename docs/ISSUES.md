# Issues

Open problems, known limitations and ideas, with a stable ID so commits, branches and the changelog can refer to them (`fix/iss-001-...`, "closes ISS-001"). When an issue is resolved, move it to **Closed** with the date and the commit or branch. Once the project settles on GitHub, open items can move to GitHub Issues and this file can point there.

Status: **open**, **in progress**, **parked** (deliberately not pursued now), **closed**. Kind: bug, research, feature, chore, limitation.

## Open

| ID | Kind | Status | Summary | Notes |
| --- | --- | --- | --- | --- |
| ISS-001 | feature | open | Results report can't tell runs of the same model apart | Show each run's output tag (e.g. `monthly_pf_qm_sst_era5_2018_clim-sst`) in the Runs table of `results/REPORT.md`. |
| ISS-002 | research | open | Emulator's own summer over-extent on the Pacific shelf | East Siberian and Beaufort ~+0.2 M km² each in September, growing over a free-run; not inherited from neXtSIM. See OBS_GAP.md (2026-09-25). |
| ISS-003 | research | open | New forcing channels barely learned | SST reached 2–5% of the other forcings' input weight (inference copy), radiation 5–8% even when fed in 10× larger. Unclear whether under-training or redundancy with ERA5 air temperature. |
| ISS-004 | research | in progress | Radiation model attribution | Run the 2019–25 hindcast for `monthly_pf_qm_rad` with and without `--climatology-forcing ssrd strd`. |
| ISS-005 | limitation | open | Past fine-tunes are mostly their seed model | With `ema_rate 0.9999` every 4 steps, the inference weights keep 61% of the seed after 20k steps, 37% after 40k. New fine-tunes use 0.999; `monthly_qm`, `monthly_pf_qm`, `monthly_pf_qm_sst` are affected. |
| ISS-006 | limitation | open | ERA5 forcing carries the observed ice | ERA5 prescribes SST (271.46 K under observed ice) and its air temperature is shaped by the same ice, so reanalysis-forced hindcasts are partly informed by observations. |
| ISS-007 | research | open | Degree-days from monthly-mean temperature | Positive degree-days are computed from each month's mean, so a month averaging below freezing counts zero melt even with thaw days. |
| ISS-008 | limitation | open | SST and radiation models can't run CMIP projections | The CMIP cube has no `tos`, `rsds` or `rlds`; add them to `download/cmip.py` and `preprocess_cmip.py` if either model is kept. |
| ISS-009 | research | open | CMIP forcing ~0.36 K colder than ERA5 over 2010–18 | The cube carries no bias-correction stamp; the per-month correction over 1995–2014 doesn't hold for later years. Worth ~+0.13 M km² of September ice. |
| ISS-010 | chore | open | Licence of the SASIP neXtSIM-OPA files unverified | The Zenodo subset is CC BY 4.0; check the SASIP server's terms before redistributing any of its files. |
| ISS-011 | feature | open | Thickness frames for the animation | The finished CMIP free-run saved only `sic` per member; re-run with `--save-members sic sit` for PMM or single-member thickness frames. |
| ISS-012 | chore | open | Offer GenSIM's bug fixes upstream (optional) | Optimizer rebuilt every step, patch-boundary crash, MPS fallback. Check they still exist upstream (`git fetch upstream`) before filing. |
| ISS-016 | chore | in progress | Repository housekeeping | Configs out of the root, scripts vs library code, central paths and defaults, tests, dead code. Branch `chore/housekeeping`. |

## Parked

| ID | Kind | Summary | Why parked |
| --- | --- | --- | --- |
| ISS-013 | research | Harder or from-scratch QM fine-tune | QM targets address the second-smallest error term; see OBS_GAP.md. |
| ISS-014 | research | Daily-data retrain (SASIP `1d/`, ~35× more samples) | Biggest untapped data lever; ~4 GB download and much longer training. |
| ISS-015 | research | Multi-model CMIP ensemble | One trajectory today; several models would give a spread for projections. |

## Closed

| ID | Kind | Summary | Closed |
| --- | --- | --- | --- |
| — | bug | CMIP free-runs ignored `--start` when reading forcing | 2026-09-27, released in 0.1.0 |
| — | bug | Results report labelled every `monthly_pf_*` run as `monthly_pf` | 2026-09-27, released in 0.1.0 |
| — | bug | ERA5 radiation and SST merged on mismatched time stamps | 2026-09-26, released in 0.1.0 |
