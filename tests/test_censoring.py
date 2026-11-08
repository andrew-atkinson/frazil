"""censor_lower switches lower-bound censoring per variable; the default censors all, as before."""
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from conftest import ROOT

def _surrogate(overrides=()):
    with initialize_config_dir(config_dir=f"{ROOT}/configs", version_base=None):
        return compose(config_name="config_train_monthly_mac", overrides=list(overrides)).surrogate


def test_default_censors_every_lower_bound():
    m = instantiate(_surrogate())
    assert m.censor_lower.flatten().tolist() == [True] * 6
    assert "censor_lower" not in m.state_dict()                 # checkpoints unchanged


def test_sic0_recipe_uncensors_sic_only():
    m = instantiate(_surrogate(["+experiment=pf_qm_ema_qmw_sic0"]))
    assert m.censor_lower.flatten().tolist() == [True, False, True, True, True, True]
    states = torch.zeros(1, 6, 2, 2)                            # every variable at its lower bound 0 ...
    at_bound = torch.eq(states, m.decoder.lower_bound) & m.censor_lower
    assert at_bound[0, :, 0, 0].tolist() == [True, False, True, False, False, True]   # ... velocities have -inf


def test_wrong_length_is_refused():
    with pytest.raises(Exception, match="one entry per state variable"):   # hydra wraps it
        instantiate(_surrogate(["+surrogate.censor_lower=[true]"]))
