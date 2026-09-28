# Pushforward fine-tuning

**Date:** 2026-09-19 (20k steps), extended 2026-09-22 (to 55k) · **Status:** done · **Model:** `monthly_pf`

## Question
Free-runs drift because the model only ever trained on true states. If it also trains from its own generated (drifted) states, does it learn to correct the drift?

## Setup
- Recipe: `configs/experiment/pushforward.yaml`, seeded from the base `monthly` model. With probability `pushforward_prob`, a training step first generates the next state with the sampler (no gradient), then learns the following step from that generated state.
- 20k steps (prob 0.5, 8 sampler substeps), later extended to 55k (prob 0.8, 10 substeps).
- Full method, knobs and cost: [docs/PUSHFORWARD.md](../../docs/PUSHFORWARD.md).

## Commands
```bash
python train.py +experiment=pushforward
python train.py +experiment=pushforward surrogate.total_steps=60000 trainer.max_steps=60000 surrogate.pushforward_prob=0.8 surrogate.pushforward_substeps=10
```

## Results
20k steps, against the base model: skill vs persistence up +1–4 points at every lead (largest at 6–12 months); 15-year cyclic free-run keeps more of its sharpness (thickness 0.586 → 0.615 retained) and drifts less (ice area −0.305 → −0.277 M km²). No instability.

The harder 55k run made the ERA5 hindcast September extent *higher* (5.69 vs 5.48 M km² for the QM model), because pushforward pulls the free-run towards the distribution it trains on, which here is neXtSIM's icy targets.

## Conclusion
Pushforward stabilises free-runs, and `monthly_pf` has the best forecast skill of the original-target models. It cannot remove a bias that is in the targets: combine it with corrected targets (see [qm_targets](../2026-09-27_qm_targets/)).
