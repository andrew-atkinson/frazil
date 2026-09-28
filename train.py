#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging

# External modules
import hydra
from omegaconf import DictConfig, OmegaConf

# Internal modules


main_logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path='configs', config_name='config_train_monthly_mac'
)
def train_task(cfg: DictConfig, network_name: str = "surrogate") -> None:
    # Import within main loop to speed up training on Jean Zay
    import wandb
    from hydra.utils import instantiate
    import torch
    import lightning.pytorch as pl

    # PyTorch 2.6 defaults torch.load to weights_only=True, which rejects the
    # OmegaConf config Lightning stores in the checkpoint. Our own checkpoints
    # are trusted, so force full loads. ponytail: global override, fine for a
    # training entrypoint that only loads its own checkpoints.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    from lightning.pytorch.loggers import WandbLogger
    try:
        from wandb.sdk.service.service import ServiceStartTimeoutError
    except ImportError:  # moved/removed across wandb versions
        ServiceStartTimeoutError = Exception

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    torch.use_deterministic_algorithms(mode=False, warn_only=True)
    torch.set_float32_matmul_precision("medium")
    torch._dynamo.config.suppress_errors = True
    
    main_logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: pl.LightningDataModule = instantiate(cfg.data)
    data_module.setup("fit")

    main_logger.info(f"Instantiating model <{cfg[network_name]._target_}")
    model: pl.LightningModule = instantiate(
        cfg[network_name], _recursive_=False
    )
    model.hparams["batch_size"] = cfg.batch_size
    if cfg.compile:
        # Compile for training
        model.network.compile(mode="reduce-overhead")

    if OmegaConf.select(cfg, "callbacks") is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            curr_callback: pl.callbacks.Callback = instantiate(callback_cfg)
            callbacks.append(curr_callback)
    else:
        callbacks = None

    training_logger = None
    if OmegaConf.select(cfg, "logger") is not None:
        try:
            training_logger = instantiate(cfg.logger)
        except ServiceStartTimeoutError:
            # Needed for restart on Jean Zay
            # Set wandb to offline
            training_logger = instantiate(
                cfg.logger, mode="offline", offline=True, log_model=False
            )

    if isinstance(training_logger, WandbLogger):
        main_logger.info("Watch gradients and parameters of model")
        training_logger.watch(model, log="all", log_freq=100)

    # Always add a CSVLogger -> data/models/<exp_name>/metrics.csv for easy
    # offline plotting (pandas), alongside whatever cfg.logger is.
    from lightning.pytorch.loggers import CSVLogger
    csv_logger = CSVLogger("data/models", name=cfg.exp_name, version="")
    loggers = [lg for lg in (training_logger, csv_logger) if lg is not None]

    main_logger.info("Instantiating trainer")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers
    )

    # Resumability. Precedence: an explicit ckpt_path; else auto-resume from this
    # run's own last.ckpt if it exists (so re-running the SAME command picks up
    # exactly where it stopped -- optimizer, scheduler, step, RNG and EMA all
    # restored by Lightning); else, on a fresh fine-tune, initialise weights from
    # cfg.init_from WITHOUT resuming optimizer/step.
    import os
    ckpt_path = cfg.get("ckpt_path")
    last_ckpt = os.path.join("data/models", cfg.exp_name, "last.ckpt")
    if ckpt_path is None and os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        main_logger.info(f"Auto-resuming from {last_ckpt}")
    elif ckpt_path is None and cfg.get("init_from"):
        sd = torch.load(cfg.init_from, map_location="cpu")["state_dict"]
        # More forcing channels than the seed checkpoint -> zero-init the new
        # input weights (identical behaviour at step 0); no-op otherwise.
        from gensim.utils import expand_forcing_channels
        sd = expand_forcing_channels(sd, model.state_dict(), cfg.surrogate.network.patch_size)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        main_logger.info(
            f"Initialised weights from {cfg.init_from} "
            f"(missing {len(missing)}, unexpected {len(unexpected)}); fresh optimizer"
        )

    main_logger.info("Starting training")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)
    main_logger.info("Training finished")
    wandb.finish()


if __name__ == '__main__':
    train_task()
