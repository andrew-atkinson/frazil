"""Summarise a generative ensemble into one sea-ice field: mean, member, median,
consensus edge, probability-matched mean (PMM, Ebert 2001) and local PMM (Clark
2017), and score a field against observations. See experiments/ for results."""
from __future__ import annotations

import numpy as np

from frazil.diagnostics import small_scale_energy

THR = 0.15
METHODS = ["mean", "member", "median", "consensus", "pmm", "lpmm"]


def pmm(members, mask):
    """Probability-matched mean over `mask` cells: rank cells by the ensemble
    mean, then give them the pooled member values (every n-th of the sorted pool),
    so the field keeps the mean's pattern but the members' histogram."""
    n = members.shape[0]
    mv = members.mean(0)[mask]
    pooled = np.sort(members[:, mask].ravel())
    vals = pooled.reshape(-1, n)[:, n // 2]
    out = members.mean(0).copy()
    tmp = np.empty_like(mv)
    tmp[np.argsort(mv, kind="stable")] = vals
    out[mask] = tmp
    return out


def lpmm(members, mask, tile=64, halo=32, min_cells=256):
    """Local PMM: PMM computed on each tile plus a halo, written to the tile core
    only, so regions don't borrow values from each other."""
    out = members.mean(0).copy()
    H, W = mask.shape
    for i in range(0, H, tile):
        for j in range(0, W, tile):
            y0, y1 = max(i - halo, 0), min(i + tile + halo, H)
            x0, x1 = max(j - halo, 0), min(j + tile + halo, W)
            m = mask[y0:y1, x0:x1]
            if m.sum() < min_cells:
                continue
            p = pmm(members[:, y0:y1, x0:x1], m)
            cy, cx = i - y0, j - x0
            core = (slice(cy, cy + tile), slice(cx, cx + tile))
            out[i:i + tile, j:j + tile] = np.where(m[core], p[core], out[i:i + tile, j:j + tile])
    return out


def summaries(members, mask):
    mean = members.mean(0)
    prob = (members > THR).mean(0)
    return {"mean": mean, "member": members[0], "median": np.median(members, 0),
            "consensus": np.where(prob >= 0.5, mean, 0.0),
            "pmm": pmm(members, mask), "lpmm": lpmm(members, mask)}


def metrics(f, obs, ocean, ca):
    fi, oi = (f > THR) & ocean, (obs > THR) & ocean
    km = lambda b: float((ca * b).sum()) / 1e6
    return {"ext": km(fi) - km(oi),
            "area": float((ca * f * ocean).sum() - (ca * obs * ocean).sum()) / 1e6,
            "iiee": km(fi & ~oi) + km(~fi & oi),
            "miz": km(fi & (f < 0.8)) - km(oi & (obs < 0.8)),
            "rmse": float(np.sqrt(((f - obs)[ocean] ** 2).mean())),
            "sharp": small_scale_energy(f, ocean) / max(small_scale_energy(obs, ocean), 1e-12)}
