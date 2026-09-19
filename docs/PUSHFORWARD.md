# Pushforward (rollout) fine-tuning

Teach the monthly model to correct its **own free-running drift** — the ~0.5 M km²
of the September gap that came from the model stepping off-distribution when it
feeds itself, rather than from the neXtSIM training target. This is the "step 3"
fix; it's independent of the (parked) `sic`-target correction.

## Why it's not ordinary rollout training

GenSIM is a **generative flow-matching** model: one forecast step is a full
sampler loop (many network evals), not a single forward pass. So back-propagating
through a multi-step rollout is infeasible on a laptop. Instead we use the
**pushforward** trick (Brandstetter et al.):

1. From the true state *t*, **generate** the model's next state *t+1′* with the
   sampler under `no_grad` (no gradient, no giant graph).
2. Train the normal one-step flow-matching loss **from that drifted state** →
   `t+1′ → true t+2`.

The model learns to pull its own imperfect states back toward truth. Gradients
flow only through the final one-step loss, never through the sampler.

## Run it

```bash
python train.py --config-name config_finetune_pushforward_mac
```

That's the whole command. It:
- starts from `data/models/monthly/monthly.ckpt` (weights only — fresh optimizer),
- writes checkpoints to `data/models/monthly_pf/`,
- trains 20 000 steps by default (`trainer.max_steps`).

## Stop and resume — the important part

**Re-run the exact same command to continue.** On restart, `train.py`
auto-detects `data/models/monthly_pf/last.ckpt` and resumes from it — Lightning
restores the optimizer moments, LR schedule, global step, RNG, and EMA weights,
so it picks up as if it never stopped.

- **Stopping:** `Ctrl-C` (or close the laptop, or let it die) at any time. You
  lose at most the steps since the last checkpoint.
- **Checkpoints are written:** every 500 steps (`model_checkpoint_step`, all kept),
  on every validation (`last.ckpt`, `save_last`), and the best-val one
  (`model_checkpoint_best`). So `last.ckpt` is always current.
- **First launch vs resume is automatic:**
  - No `last.ckpt` yet → initialise weights from `init_from` (monthly.ckpt), fresh optimizer.
  - `last.ckpt` present → resume full training state; `init_from` is ignored.
- **To force a fresh start:** delete `data/models/monthly_pf/` (or set a new
  `exp_name=...`).
- **To resume from a specific checkpoint instead of `last.ckpt`:** pass
  `ckpt_path=data/models/monthly_pf/....ckpt` (takes precedence over auto-resume).

This auto-resume applies to *all* training now, not just the fine-tune — re-running
any `train.py` command continues its run if a `last.ckpt` exists.

## Cost

Every pushforward step *also* runs a sampler generation (`pushforward_substeps`
network evals) on top of the gradient step, so those steps are markedly slower
than plain training — expect this fine-tune to be an **overnight-plus** job on the
laptop. Levers to speed it up (all in the config or via CLI override):

| Knob | Default | Effect |
|---|---|---|
| `surrogate.pushforward_prob` | `0.5` | fraction of steps that do the (costly) generation; lower = faster, weaker |
| `surrogate.pushforward_substeps` | `8` | sampler steps for the no_grad generation; fewer = faster, rougher drifted state |
| `surrogate.pushforward_warmup` | `200` | plain one-step training before pushforward kicks in |
| `surrogate.total_steps` / `trainer.max_steps` | `20000` | length of the fine-tune |
| `surrogate.lr` | `1e-4` | fine-tune learning rate (gentle) |
| `data.n_rollout_steps` | `2` | frames per window (must be ≥2 so a 3rd frame exists) |

Because it's fully resumable, you can run it in short bursts whenever the laptop
is free and it accumulates.

## Check whether it helped

Compare the fine-tuned model against the original on the free-running metrics that
exposed the drift:

```bash
# skill vs lead (should hold up better at long lead)
python experiments/rollout_monthly.py --fast --ckpt data/models/monthly_pf/last.ckpt
# 15-yr cyclic drift (sharpness / area should stay stable at least as well)
python experiments/freerun_monthly.py --years 15 --fast --ckpt data/models/monthly_pf/last.ckpt
```

`train/pushforward` in `data/models/monthly_pf/metrics.csv` logs the fraction of
steps that used the pushforward path, and `train/loss` / `val/loss` track training
as usual (remember flow-matching NLL doesn't go to zero — judge by skill, not loss).

## What changed in the code

- `gensim/dataset.py` — returns per-frame degree-days (so the 2nd transition has
  its forcing). The one-step path slices frame 0, unchanged.
- `gensim/train_module.py` — `pushforward*` params, a stateless `pf_sampler`, a
  no_grad `_generate_next`, and a `_select_batch` that swaps in the generated
  state. Adds nothing to the checkpoint (the sampler holds no parameters).
- `train.py` — auto-resume from `last.ckpt` + `init_from` weight seeding.
- `config_finetune_pushforward_mac.yaml` — inherits the mac config, flips on
  pushforward and the 3-frame window.
