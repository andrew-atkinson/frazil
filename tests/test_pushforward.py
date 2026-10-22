"""Pushforward picks the right frames: single-step exactly as before, multi-step chains
the model's own states k deep and trains k' -> true k+1."""
from types import SimpleNamespace

import torch

from gensim.train_module import GenSIMTrainModule as M


def fake_module(steps, prob=1.0):
    calls = []
    m = SimpleNamespace(pushforward=True, pf_sampler=object(), pushforward_prob=prob, pushforward_warmup=0,
                        pushforward_steps=steps, global_step=10, _dd_frame=M._dd_frame)
    m._pair_batch = lambda b, i, states=None: M._pair_batch(m, b, i, states)

    def generate(states, forcings, mesh, mask, dd, res, labels):     # "prediction" = last state + 1
        calls.append((states[:, -1].clone(), forcings.clone(), dd.clone()))
        return states[:, -1] + 1
    m._generate_next = generate
    return m, calls


def batch(n_frames):
    f = torch.arange(n_frames, dtype=torch.float32).view(1, n_frames, 1, 1, 1)   # frame i holds value i
    return {"states": f * 10, "forcings": f * 100, "degree_days": f * 1000, "mesh": None, "mask": None}


def test_single_step_matches_the_original():
    m, calls = fake_module(steps=1)
    torch.manual_seed(0)
    pair, k = M._select_batch(m, batch(3), 12.0, None)
    assert k == 1 and len(calls) == 1
    assert float(calls[0][0]) == 0 and calls[0][1].flatten().tolist() == [0, 100]        # from true frame 0
    assert pair["states"][:, :, 0, 0, 0].tolist() == [[1.0, 20.0]]                      # [1', true 2]
    assert pair["forcings"][:, :, 0, 0, 0].tolist() == [[100.0, 200.0]]                 # forcing of frames 1->2


def test_multi_step_chains_its_own_states():
    m, calls = fake_module(steps=3)
    seen = set()
    for seed in range(40):
        calls.clear(); torch.manual_seed(seed)
        pair, k = M._select_batch(m, batch(5), 12.0, None)
        seen.add(k)
        assert len(calls) == k
        assert [float(c[0]) for c in calls] == [0.0] + [float(i) for i in range(1, k)]  # 0, then its own 1', 2'
        assert [c[1].flatten().tolist() for c in calls] == [[100.0 * i, 100.0 * (i + 1)] for i in range(k)]
        assert [float(c[2]) for c in calls] == [1000.0 * i for i in range(k)]           # each frame's degree-days
        assert pair["states"][:, :, 0, 0, 0].tolist() == [[float(k), 10.0 * (k + 1)]]   # [k', true k+1]
    assert seen == {1, 2, 3}


def test_depth_is_capped_by_the_frames_available_and_off_means_plain():
    m, calls = fake_module(steps=5)
    for seed in range(20):
        torch.manual_seed(seed)
        assert M._select_batch(m, batch(4), 12.0, None)[1] <= 2                        # 4 frames: at most 2 deep
    m, calls = fake_module(steps=3, prob=0.0)
    pair, k = M._select_batch(m, batch(5), 12.0, None)
    assert k == 0 and not calls and pair["states"][:, :, 0, 0, 0].tolist() == [[0.0, 10.0]]
