"""Every base config and training recipe composes, and to exactly the stored config.
A change to a recipe (or to a base config it inherits) shows up here as a diff."""
import glob
import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from conftest import ROOT, check_snapshot

CONFIGS = os.path.join(ROOT, "configs")
BASES = ["config_train_monthly_mac", "config_train_monthly", "config_train"]
RECIPES = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CONFIGS, "experiment", "*.yaml")))


def _compose(name, overrides=()):
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        return OmegaConf.to_container(compose(config_name=name, overrides=list(overrides)))


@pytest.mark.parametrize("base", BASES)
def test_base_config(base, update_snapshots):
    check_snapshot(f"configs/{base}", _compose(base), update_snapshots)


@pytest.mark.parametrize("recipe", RECIPES)
def test_experiment_recipe(recipe, update_snapshots):
    cfg = _compose("config_train_monthly_mac", [f"+experiment={recipe}"])
    assert cfg["exp_name"], recipe
    check_snapshot(f"configs/experiment_{recipe}", cfg, update_snapshots)


def test_train_py_uses_configs_folder():
    src = open(os.path.join(ROOT, "train.py")).read()
    assert "config_path='configs'" in src and "config_name='config_train_monthly_mac'" in src
