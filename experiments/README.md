# Experiments

The lab notebook: one folder per experiment, named `YYYY-MM-DD_short-name`, each with a `README.md` saying what was asked, how it was run, what came out and what it means. Code lives in `scripts/` and `frazil/`, training recipes in `configs/experiment/`, and outputs in `data/models/` and `plots/` (never committed). The results ledger (`results/experiments.jsonl`, summarised in `results/REPORT.md`) links every logged run to the exact code that produced it.

Start a new experiment on an `exp/` branch (see [CONTRIBUTING.md](../CONTRIBUTING.md)): copy [_template.md](_template.md) into a new folder, add a recipe to `configs/experiment/` if it trains a model, pass `--note` on the runs so the report says what they were, and add a line below.

| date | experiment | question | status | result |
| --- | --- | --- | --- | --- |
| 2026-09-19 | [pushforward](2026-09-19_pushforward/) | Does training on the model's own drifted states reduce free-run drift? | done | Modest, consistent gains at long lead and in stability; alone it locks onto the (icy) training targets. |
| 2026-09-22 | [obs_gap_hindcast](2026-09-22_obs_gap_hindcast/) | Why does the model sit above NSIDC, and does it hold out of sample? | done | Tracks real 2019–25 variability (r = 0.82); the gap is misplaced summer ice on the Pacific shelf, growing over a run. |
| 2026-09-25 | [ensemble_averaging](2026-09-25_ensemble_averaging/) | Does averaging members distort the ice edge, and what summarises an ensemble best? | done | Averaging inflates summer extent ~0.2 M km²; PMM keeps the benefit without the smear. |
| 2026-09-27 | [qm_targets](2026-09-27_qm_targets/) | Does quantile-mapping the `sic` targets onto observations fix the inherited bias? | done | With pushforward, the best model out of sample (July bias 0.60 → 0.28 M km²). |
| 2026-09-27 | [sst_channel](2026-09-27_sst_channel/) | Does ERA5 sea-surface temperature as an input help? | done | No effect: the channel was barely learned (2–5% weight at inference). |
| 2026-09-28 | [radiation_channel](2026-09-28_radiation_channel/) | Does downward solar and thermal radiation fix the summer melt deficit? | in progress | Trained; channels at 5–8% weight. Hindcasts pending. |
| 2026-09-28 | [ensemble_size](2026-09-28_ensemble_size/) | How does the benefit of averaging grow with ensemble size? | done | 4 members give ~85% of the 8-member gain; the plain mean's edge gets worse with size, PMM's doesn't. |
