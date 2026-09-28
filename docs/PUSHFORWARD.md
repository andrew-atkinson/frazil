# Pushforward (rollout) fine-tuning

Teach the monthly model to correct its **own free-running drift** — the ~0.5 M km² of the September gap that came from the model stepping off-distribution when it feeds itself, rather than from the neXtSIM training target. This is the "step 3" fix; it's independent of the (parked) `sic`-target correction.

## Why it's not ordinary rollout training

GenSIM is a **generative flow-matching** model: one forecast step is a full sampler loop (many network evals), not a single forward pass. So back-propagating through a multi-step rollout is infeasible on a laptop. Instead we use the **pushforward** trick (Brandstetter et al.):

1. From the true state *t*, **generate** the model's next state *t+1′* with the sampler under `no_grad` (no gradient, no giant graph).
2. Train the normal one-step flow-matching loss **from that drifted state** → `t+1′ → true t+2`.

The model learns to pull its own imperfect states back toward truth. Gradients flow only through the final one-step loss, never through the sampler.

## Run it

```bash
python train.py +experiment=pushforward
```

That's the whole command. It:
- starts from `data/models/monthly/monthly.ckpt` (weights only — fresh optimizer),
- writes checkpoints to `data/models/monthly_pf/`,
- trains 20 000 steps by default (`trainer.max_steps`).

## Stop and resume — the important part

**Re-run the exact same command to continue.** On restart, `train.py` auto-detects `data/models/monthly_pf/last.ckpt` and resumes from it — Lightning restores the optimizer moments, LR schedule, global step, RNG, and EMA weights, so it picks up as if it never stopped.

- **Stopping:** `Ctrl-C` (or close the laptop, or let it die) at any time. You lose at most the steps since the last checkpoint.
- **Checkpoints are written:** every 500 steps (`model_checkpoint_step`, all kept), on every validation (`last.ckpt`, `save_last`), and the best-val one (`model_checkpoint_best`). So `last.ckpt` is always current.
- **First launch vs resume is automatic:**
  - No `last.ckpt` yet → initialise weights from `init_from` (monthly.ckpt), fresh optimizer.
  - `last.ckpt` present → resume full training state; `init_from` is ignored.
- **To force a fresh start:** delete `data/models/monthly_pf/` (or set a new `exp_name=...`).
- **To resume from a specific checkpoint instead of `last.ckpt`:** pass `ckpt_path=data/models/monthly_pf/....ckpt` (takes precedence over auto-resume).

This auto-resume applies to *all* training now, not just the fine-tune — re-running any `train.py` command continues its run if a `last.ckpt` exists.

## Cost

Every pushforward step *also* runs a sampler generation (`pushforward_substeps` network evals) on top of the gradient step, so those steps are markedly slower than plain training — expect this fine-tune to be an **overnight-plus** job on the laptop. Levers to speed it up (all in the config or via CLI override):

| Knob | Default | Effect |
|---|---|---|
| `surrogate.pushforward_prob` | `0.5` | fraction of steps that do the (costly) generation; lower = faster, weaker |
| `surrogate.pushforward_substeps` | `8` | sampler steps for the no_grad generation; fewer = faster, rougher drifted state |
| `surrogate.pushforward_warmup` | `200` | plain one-step training before pushforward kicks in |
| `surrogate.total_steps` / `trainer.max_steps` | `20000` | length of the fine-tune |
| `surrogate.lr` | `1e-4` | fine-tune learning rate (gentle) |
| `data.n_rollout_steps` | `2` | frames per window (must be ≥2 so a 3rd frame exists) |

Because it's fully resumable, you can run it in short bursts whenever the laptop is free and it accumulates.

## Check whether it helped

Compare the fine-tuned model against the original on the free-running metrics that exposed the drift:

```bash
# skill vs lead (should hold up better at long lead)
python scripts/evaluate/rollout_monthly.py --fast --ckpt data/models/monthly_pf/last.ckpt
# 15-yr cyclic drift (sharpness / area should stay stable at least as well)
python scripts/project/freerun_monthly.py --years 15 --fast --ckpt data/models/monthly_pf/last.ckpt
```

`train/pushforward` in `data/models/monthly_pf/metrics.csv` logs the fraction of steps that used the pushforward path, and `train/loss` / `val/loss` track training as usual (remember flow-matching NLL doesn't go to zero — judge by skill, not loss).

## Results — 20k-step fine-tune (Sep 2026)

`monthly.ckpt` → `monthly_pf/last.ckpt` (20 000 steps, `pushforward_prob=0.5`, 8 substeps). Both tests agree: **it works, consistently, modestly** — better on every metric, nothing worse, gains concentrated at long lead / long horizon exactly as intended.

**One-step-ahead skill vs lead** (`rollout_monthly`, skill vs climatology, points gained baseline→pf):

| lead | sit | sic | sid | siu | siv | snt |
|---|---|---|---|---|---|---|
| 1mo | +0 | +1 | +1 | +1 | +1 | +0 |
| 3mo | +1 | +1 | +2 | +2 | +2 | +2 |
| 6mo | +2 | +2 | +4 | +2 | +2 | +2 |
| 12mo | +2 | +2 | +3 | +2 | +2 | +2 |

The gain **grows with lead** (≈0 at 1 mo → +2–4 by 6–12 mo) — the drift-correction signature. `sid` (damage), the noisiest field, benefits most.

**15-yr free-running drift** (`freerun_monthly`, cyclic forcing; stable = good):

| metric | baseline | pushforward |
|---|---|---|
| sharpness SIT retention (final/initial) | 0.586 | **0.615** |
| sharpness SIC retention | 0.910 | **0.921** |
| ice-area drift, final−initial (M km²) | −0.305 | **−0.277** |
| mean SIT drift (m) | −0.068 | **−0.060** |
| years completed / non-finite | 15 / 0 | 15 / 0 |

Better sharpness retention (smooths toward mush less) and slightly less area/mean drift, on every axis. Gains are small here because under *stationary cyclic* forcing the baseline was already stable — the effect is larger at long autoregressive lead (above) and would compound more under trending CMIP forcing.

**Verdict:** `monthly_pf/last.ckpt` is a strictly-better model — worth keeping, not transformative. For more, fine-tune longer (resumable), raise `pushforward_prob`, or go 2-step (`data.n_rollout_steps=3`); returns look diminishing for the "broad monthly evolution" use. Both runs are in `results/experiments.jsonl` (reproducible).

## What changed in the code

- `gensim/dataset.py` — returns per-frame degree-days (so the 2nd transition has its forcing). The one-step path slices frame 0, unchanged.
- `gensim/train_module.py` — `pushforward*` params, a stateless `pf_sampler`, a no_grad `_generate_next`, and a `_select_batch` that swaps in the generated state. Adds nothing to the checkpoint (the sampler holds no parameters).
- `train.py` — auto-resume from `last.ckpt` + `init_from` weight seeding.
- `configs/experiment/pushforward.yaml` — inherits the mac config, flips on pushforward and the 3-frame window.
