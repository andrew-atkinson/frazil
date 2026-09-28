# Archive

Finished one-off tools, kept for the record. They did their job during development and nothing imports them. They predate the `configs/` layout and later code changes, so they may need updating before they run again. Their full history is in git (`git log --follow archive/<file>`).

| file | what it was | outcome |
| --- | --- | --- |
| `ab_optimizer_monthly.py`, `ab_optimizer.png` | A/B test of GenSIM's optimizer bug (AdamW rebuilt every training step) against the fix | Confirmed the fix; see *Core bug fixes* in the README. |
| `smoke_train_monthly.py` | Instantiate the training module and run a couple of steps to check shapes and config | Superseded by the tests (planned) and real runs. |
| `minimal_train_monthly.py` | A minimal real training run on the monthly data | Superseded by `train.py` with the monthly configs. |
| `train_watch_monthly.py`, `train_loss_monthly.png` | Short training run that plots the loss as it goes | The loss turned out to be a poor progress meter; skill is used instead. |
| `loss_graph.py` | Quick plot of train/validation loss from a run's `metrics.csv` | Same as above. |
| `sweep_skill_monthly.py`, `skill_vs_step.csv` | Skill vs training step, evaluated post hoc on saved step checkpoints | Showed where skill saturates for the base model. |
| `qm_calibrate_poc.py` | Proof of concept: quantile-map model *output* onto NSIDC | Worked on neXtSIM but not on the model; led to quantile-mapping the *targets* instead (`experiments/correct_sic_targets_qm.py`). |
| `correct_sic_targets.py` | Additive mean-shift correction of the `sic` targets | Barely moved extent: a mean shift can't move a sharp ice edge. Superseded by quantile mapping. |
