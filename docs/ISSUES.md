# Issues

Open problems, known limitations and ideas, with a stable ID so commits, branches and the changelog can refer to them (`fix/iss-001-...`, "closes ISS-001"). When an issue is resolved, move it to **Closed** with the date and the commit or branch. Once the project settles on GitHub, open items can move to GitHub Issues and this file can point there.

Status: **open**, **in progress**, **parked** (deliberately not pursued now), **closed**. Kind: bug, research, feature, chore, limitation. **Target** is the [roadmap](ROADMAP.md) milestone that addresses it (— for a documented limitation with no fix planned). Bugs also carry a severity, S1–S3, and follow the bug-fixing route in [CONTRIBUTING.md](../CONTRIBUTING.md#fixing-bugs).

## Open

| ID | Kind | Status | Target | Summary | Notes |
| --- | --- | --- | --- | --- | --- |
| ISS-001 | feature | open | v0.3.0 | Results report can't tell runs of the same model apart | Show each run's output tag (e.g. `monthly_pf_qm_sst_era5_2018_clim-sst`) in the Runs table of `results/REPORT.md`. |
| ISS-002 | research | open | v0.5.0 | Emulator's own summer over-extent on the Pacific shelf | East Siberian and Beaufort ~+0.2 M km² each in September, growing over a free-run; not inherited from neXtSIM. See OBS_GAP.md (2026-09-25). |
| ISS-003 | research | open | v0.5.0 | New forcing channels barely learned | SST reached 2–5% of the other forcings' input weight (inference copy), radiation 5–8% even when fed in 10× larger. Unclear whether under-training or redundancy with ERA5 air temperature. |
| ISS-004 | research | in progress | v0.5.0 | Radiation model attribution | Run the 2019–25 hindcast for `monthly_pf_qm_rad` with and without `--climatology-forcing ssrd strd`. |
| ISS-005 | limitation | open | v0.5.0 | Past fine-tunes are mostly their seed model | With `ema_rate 0.9999` every 4 steps, the inference weights keep 61% of the seed after 20k steps, 37% after 40k. New fine-tunes use 0.999; `monthly_qm`, `monthly_pf_qm`, `monthly_pf_qm_sst` are affected. |
| ISS-006 | limitation | open | — | ERA5 forcing carries the observed ice | ERA5 prescribes SST (271.46 K under observed ice) and its air temperature is shaped by the same ice, so reanalysis-forced hindcasts are partly informed by observations. |
| ISS-007 | research | open | v0.5.0 | Degree-days from monthly-mean temperature | Positive degree-days are computed from each month's mean, so a month averaging below freezing counts zero melt even with thaw days. |
| ISS-008 | limitation | open | v0.6.0 | SST and radiation models can't run CMIP projections | The CMIP cube has no `tos`, `rsds` or `rlds`; add them to `scripts/download/cmip.py` and `preprocess_cmip.py` if either model is kept. |
| ISS-009 | research | open | v0.6.0 | CMIP forcing ~0.36 K colder than ERA5 over 2010–18 | The cube carries no bias-correction stamp; the per-month correction over 1995–2014 doesn't hold for later years. Worth ~+0.13 M km² of September ice. |
| ISS-010 | chore | open | v0.3.0 | Licence of the SASIP neXtSIM-OPA files unverified | The Zenodo subset is CC BY 4.0; check the SASIP server's terms before redistributing any of its files. |
| ISS-011 | feature | open | v0.4.0 | Thickness frames for the animation | The finished CMIP free-run saved only `sic` per member; re-run with `--save-members sic sit` for PMM or single-member thickness frames. |
| ISS-012 | chore | open | later | Offer GenSIM's bug fixes upstream (optional) | Optimizer rebuilt every step, patch-boundary crash, MPS fallback. Check they still exist upstream (`git fetch upstream`) before filing. |
| ISS-015 | research | open | v0.6.0 | Multi-model CMIP ensemble | Forcing from at least three CMIP6 models, so projections show a range (one model today). Was parked; now planned for v0.6.0. |
| ISS-017 | chore | open | v0.3.0 | Downloaded data: how, where and why | Several files look like leftovers from manual downloads. Map every file in `data/` to the command that produces it, its path, and what uses it, then make each download reproducible by a script with a default in `frazil/paths.py`. Leads: **(1)** the ERA5 training file `data/train_data/data_stream-moda_stepType-avgua.nc` (2026-08-20) came from the original edit-the-script downloader plus a manual unzip; `scripts/download/era5.py --years 1994-2018` now writes `data/train_data/era5_1994_2018/`, so `paths.ERA5_RAW` points where no script writes. **(2)** Download zips are kept next to their extracted files in `era5_2018_2025/` and `era5_extra_1995_2025/` (~550 MB duplicated). **(3)** `data/obs/nsidc0051/` holds 372 Northern and 372 Southern Hemisphere files; only the Northern ones are used. **(4)** The neXtSIM downloader's default folder was wrong until 2026-09-28, so the data in `data/train_data/nextsim_opa_monthly/` was probably placed by hand. **(5)** Leftovers: `data/drift_run/` and `data/drift_run_log.txt` (archived drift experiment), `monthly_datacube_siccorr/` (1.2 GB, parked mean-shift), `data/auxiliary/ds_demo.nc` (GenSIM demo; still used by `inference_demo.ipynb`). **(6)** Old outputs in `plots/` (local only, gitignored): one-off files from early debugging (`ice-area-no-model.png`, `sic.mp4`, `sit_m7.mp4`, `ensemble_extent_quick.png`, `rollout_multistart.*`) and 250 un-namespaced snapshots in `plots/freerun/snapshots/` from before runs got their own folders. Keep any animation rendered on purpose. |

## Parked

| ID | Kind | Summary | Why parked |
| --- | --- | --- | --- |
| ISS-019 | chore | Run the fast tests on GitHub Actions | The pre-push hook covers it locally; CI would need the full PyTorch stack installed on every push. Worth it if others contribute. |
| ISS-013 | research | Harder or from-scratch QM fine-tune | QM targets address the second-smallest error term; see OBS_GAP.md. |
| ISS-014 | research | Daily-data retrain (SASIP `1d/`, ~35× more samples) | Biggest untapped data lever; ~4 GB download and much longer training. |

## Closed

| ID | Kind | Summary | Closed |
| --- | --- | --- | --- |
| ISS-016 | chore | Repository housekeeping | 2026-10-04: configs in `configs/` with experiment recipes; `frazil/` package and `scripts/` by stage; central paths; tests and pre-push hook; `STATUS.md` folded into the README, CLI.md and the lab notebook. |
| ISS-018 | chore | Tests catch too little; regressions slip through | 2026-10-04, `chore/tests`: fast and slow pytest tiers, snapshots, structural rules, pre-push hook; all seven past regressions reintroduced and caught. |
| — | bug | CMIP free-runs ignored `--start` when reading forcing | 2026-09-27, released in 0.1.0 |
| — | bug | Results report labelled every `monthly_pf_*` run as `monthly_pf` | 2026-09-27, released in 0.1.0 |
| — | bug | ERA5 radiation and SST merged on mismatched time stamps | 2026-09-26, released in 0.1.0 |
