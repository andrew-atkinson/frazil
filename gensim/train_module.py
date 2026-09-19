#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Dict, Tuple, Optional, Any

# External modules
import torch
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Internal modules
from .embedding import LogScaleModel
from .sampler import FlowMatchingSampler
from .utils import (
    remove_overlap, get_empty_labels, generate_noise, get_latent_states,
    masked_average, neglogpdf, neglogcdf, split_wd_params,
    sample_uniform_time
)


main_logger = logging.getLogger(__name__)


class GenSIMTrainModule(pl.LightningModule):
    _LABELS_DIMS: int = 3

    def __init__(
            self,
            network: OmegaConf,
            encoder: OmegaConf,
            decoder: OmegaConf,
            lr: float = 1E-4,
            lr_warmup: int = 5000,
            total_steps: int = 250000,
            weight_decay: float = 1E-3,
            ema_rate: float = 0.999,
            ema_update_every: int = 1,
            overlap_size: Tuple[int, int] = (8, 8),
            train_with_overlap: bool = True,
            censoring: bool = True,
            optimize_scale: bool = True,
            epsilon: float = 1E-5,
            patch_generator: Optional[OmegaConf] = None,
            train_augmentation: Optional[OmegaConf] = None,
            pushforward: bool = False,
            pushforward_prob: float = 0.5,
            pushforward_substeps: int = 8,
            pushforward_warmup: int = 0,
    ):
        super().__init__()

        # Neural networks
        self.network = instantiate(network)
        self.encoder = instantiate(encoder)
        self.decoder = instantiate(decoder)

        self.log_scale_model = LogScaleModel(
            n_embedding=self.network.embedder.n_embedding,
            n_time_in=self.network.embedder.n_time_in,
            n_res_in=self.network.embedder.n_res_in,
            n_augment_in=self.network.embedder.n_augment_in,
            n_vars=network.n_output
        )
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.network,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                ema_rate
            ),
            device="cpu"
        )
        self.ema_model.requires_grad_(False)
        self.ema_model = self.ema_model.eval()

        # For sampling
        self.train_with_overlap = train_with_overlap
        self.overlap_slices = (
            slice(overlap_size[0], -overlap_size[0]),
            slice(overlap_size[1], -overlap_size[1]),
        )
        self.censoring = censoring

        # Training parameters
        self.ema_rate = ema_rate
        self.ema_update_every = ema_update_every
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.optimize_scale = optimize_scale

        # Pushforward (rollout) training: with prob `pushforward_prob` (after
        # `pushforward_warmup` steps) generate the next state under no_grad and
        # train the one-step loss FROM that drifted state -> teaches drift
        # correction without backpropagating through the sampler. Needs the data
        # to provide >=3 frames (n_rollout_steps>=2). The sampler is stateless
        # (holds no parameters), so it adds nothing to the checkpoint.
        self.pushforward = pushforward
        self.pushforward_prob = pushforward_prob
        self.pushforward_warmup = pushforward_warmup
        self.pf_sampler = (
            FlowMatchingSampler(model=self.network, n_steps=pushforward_substeps,
                                second_order=False, censoring=censoring)
            if pushforward else None
        )

        self.patch_generator = instantiate(patch_generator)
        self.train_augmentation = instantiate(train_augmentation)
        self.train_time_sampler = sample_uniform_time

        # If needed for divison
        self.epsilon = epsilon

        # To enable the optimization of log scale
        self.automatic_optimization = False

        # To enable training continuation from partial checkpoint
        self.strict_loading = False

        # To save all given parameters
        self.save_hyperparameters()

    def forward(
            self,
            in_tensor: torch.Tensor,
            mesh: torch.Tensor,
            mask: torch.Tensor,
            pseudo_time: torch.Tensor,
            labels: torch.Tensor,
            resolution: torch.Tensor
    ) -> torch.Tensor:
        return self.network(
            in_tensor, mesh=mesh, mask=mask, pseudo_time=pseudo_time,
            labels=labels, resolution=resolution
        )

    def estimate_loss(
            self,
            batch: Dict[str, torch.Tensor],
            resolution: torch.Tensor,
            labels: torch.Tensor,
            prefix: str = "train"
    ) -> Dict[str, torch.Tensor]:
        # Check if scores should be synced
        sync_dist = prefix != "train"

        # Input data
        encoded, latent_mesh, latent_mask = get_latent_states(
            batch["states"][:, :-1], batch["forcings"],
            batch["mesh"], batch["mask"], batch["degree_days"], self.encoder
        )

        # Get linear interpolant
        residual = self.decoder.to_latent(
            batch["states"][:, -1], batch["states"][:, -2],
            batch["mask"]
        )
        noise = generate_noise(residual, latent_mask)
        sampled_time = self.train_time_sampler(residual)
        noised_residual = sampled_time * residual \
            + (1-sampled_time) * noise

        # Get input and target
        in_tensor = torch.cat(
            (noised_residual, encoded), dim=1
        )
        prediction = self.network(
            in_tensor,
            mesh=latent_mesh,
            mask=latent_mask,
            pseudo_time=sampled_time.view(-1, 1),
            labels=labels,
            resolution=resolution
        )

        log_scale = self.log_scale_model(
            pseudo_time=sampled_time.view(-1, 1),
            labels=labels,
            resolution=resolution
        )[:, :, None, None]

        # Estimate loss
        velocity = residual - noise
        error = (velocity - prediction) / (log_scale.exp() + self.epsilon)
        loss = neglogpdf(error, log_scale)

        if self.censoring:
            # Add censoring at lower bound
            loss = torch.where(
                torch.eq(batch["states"][:, -1], self.decoder.lower_bound),
                neglogcdf(error),
                loss
            )
            # Add censoring at upper bound
            loss = torch.where(
                torch.eq(batch["states"][:, -1], self.decoder.upper_bound),
                neglogcdf(-error),
                loss
            )

        loss = masked_average(
            remove_overlap(loss, self.train_with_overlap, self.overlap_slices),
            mask=remove_overlap(
                latent_mask, self.train_with_overlap, self.overlap_slices
            )
        )
        self.log(
            f'{prefix}/loss', loss,
            batch_size=in_tensor.size(0),
            prog_bar=True, sync_dist=sync_dist,
        )
        return {
            "loss": loss,
            "sampled_time": sampled_time,
            "residual": residual,
            "prediction": prediction,
            "velocity": velocity,
            "encoded": encoded,
            "noise": noise,
            "noised_residual": noised_residual,
            "latent_mesh": latent_mesh,
            "latent_mask": latent_mask
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA lives on CPU, so each update forces an MPS->CPU sync of all params.
        # ponytail: update every N steps to amortize that (raise ema_rate to keep
        # the effective window). Default 1 = unchanged.
        if (batch_idx + 1) % self.ema_update_every == 0:
            self.ema_model.update_parameters(self.network)

    def on_train_end(self) -> None:
        torch.optim.swa_utils.update_bn(
            self.trainer.train_dataloader, self.ema_model
        )

    @staticmethod
    def _dd_frame(batch, i):
        """degree_days for frame i, tolerating the legacy 4D (frame-0-only) shape."""
        dd = batch["degree_days"]
        return dd[:, i] if dd.dim() == 5 else dd

    def _pair_batch(self, batch, i, states=None):
        """A 2-frame transition sub-batch (frame i -> i+1) that estimate_loss can
        consume unchanged. `states` overrides the two frames (for pushforward)."""
        return {
            "states": batch["states"][:, i:i + 2] if states is None else states,
            "forcings": batch["forcings"][:, i:i + 2],
            "degree_days": self._dd_frame(batch, i),
            "mesh": batch["mesh"],
            "mask": batch["mask"],
        }

    @torch.no_grad()
    def _generate_next(self, states, forcings, mesh, mask, degree_days,
                       resolution, labels):
        """Sample the next state from `states` (current frame) -- mirrors
        GenSIMForecastModule.forward, on the (already patched) training batch."""
        self.pf_sampler.model = self.network
        encoded, latent_mesh, latent_mask = get_latent_states(
            states, forcings, mesh, mask, degree_days, self.encoder
        )
        first_guess = states[:, -1]
        initial_states = generate_noise(first_guess, mask)
        latent_bounds = self.decoder.get_latent_bounds(first_guess, mask)
        dynamics = self.pf_sampler.sample(
            states=initial_states, encoded=encoded, mesh=latent_mesh,
            mask=latent_mask, labels=labels, resolution=resolution,
            latent_bounds=latent_bounds,
        )
        return self.decoder(dynamics, first_guess=first_guess, mask=mask)

    def _select_batch(self, batch, resolution, labels):
        """One-step batch for estimate_loss. With pushforward active, replace the
        input frame by the model's own generated state so it learns to correct
        its drift; otherwise the plain frame-0 transition."""
        n_frames = batch["states"].size(1)
        use_pf = (
            self.pushforward and self.pf_sampler is not None and n_frames >= 3
            and self.global_step >= self.pushforward_warmup
            and float(torch.rand(1)) < self.pushforward_prob
        )
        if not use_pf:
            return self._pair_batch(batch, 0), False
        # Generate frame-1' from true frame-0, then train frame-1' -> true frame-2.
        gen = self._generate_next(
            batch["states"][:, 0:1], batch["forcings"][:, 0:2],
            batch["mesh"], batch["mask"], self._dd_frame(batch, 0),
            resolution, labels,
        )
        states = torch.stack([gen, batch["states"][:, 2]], dim=1)  # [1', 2]
        return self._pair_batch(batch, 1, states=states), True

    def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        resolution = batch.pop("resolution", 12.)
        if self.patch_generator is not None:
            batch, resolution = self.patch_generator(batch, resolution)
        if self.train_augmentation is not None:
            batch, labels = self.train_augmentation(batch)
        else:
            labels = get_empty_labels(batch["states"], self._LABELS_DIMS)
        
        # Persistent optimizers and scheduler from configure_optimizers. Under
        # manual optimization (automatic_optimization=False) these must be
        # retrieved via self.optimizers()/self.lr_schedulers() so that Adam's
        # moment estimates accumulate across steps and the warmup+cosine LR
        # schedule stays live. (They must not be rebuilt every step.)
        optimizer_net, optimizer_scale = self.optimizers()
        scheduler_net = self.lr_schedulers()

        # Zero gradients
        optimizer_net.zero_grad()
        optimizer_scale.zero_grad()

        # Pick the training pair: the plain frame-0 transition, or -- with
        # pushforward -- a step FROM the model's own generated (drifted) state,
        # so it learns to pull its errors back toward truth.
        pair, used_pf = self._select_batch(batch, resolution, labels)
        outputs = self.estimate_loss(pair, resolution, labels, prefix="train")
        if self.pushforward:
            self.log("train/pushforward", float(used_pf), prog_bar=False)

        # Backward pass
        self.manual_backward(outputs["loss"])

        # Gradient clipping
        self.clip_gradients(
            optimizer_net, gradient_clip_val=1.,
            gradient_clip_algorithm="norm"
        )

        # Optimizer steps
        optimizer_net.step()
        if self.optimize_scale:
            optimizer_scale.step()

        # Scheduler step (per-step interval, see configure_optimizers)
        scheduler_net.step()

        return outputs

    def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        resolution = batch.pop("resolution", 12.)
        if self.patch_generator is not None:
            batch, resolution = self.patch_generator(batch, resolution)
        labels = get_empty_labels(batch["states"], self._LABELS_DIMS)

        # Always the plain one-step (frame 0 -> 1); slice to a 2-frame pair so a
        # >=3-frame rollout batch still validates against truth.
        outputs = self.estimate_loss(
            self._pair_batch(batch, 0), resolution, labels, prefix="val"
        )
        return outputs["loss"]

    def test_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(
            self
    ) -> Any:
        # To get rid of unusual imports when only inference is performed.
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        wd_params, nowd_params = split_wd_params(self.network)
        optimizer_net = torch.optim.AdamW([
            {"params": wd_params, "weight_decay": self.weight_decay},
            {"params": nowd_params, "weight_decay": 0.0}
        ], lr=self.lr, betas=(0.9, 0.99))
        optimizer_scale = torch.optim.Adam(
            self.log_scale_model.parameters(), lr=self.lr, betas=(0.9, 0.99)
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer_net,
            first_cycle_steps=self.total_steps,
            max_lr=self.lr,
            min_lr=1E-6,
            warmup_steps=self.lr_warmup,
        )
        return [
            optimizer_net, optimizer_scale
        ], [{"scheduler": scheduler, "interval": "step"}]
