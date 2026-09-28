"""Scalar diagnostics of sea-ice states: extent/area, a small-scale-energy
sharpness proxy, and the per-step summary logged by free-runs."""
from __future__ import annotations

import numpy as np

SIC_THRESHOLD = 0.15  # ice-edge threshold for extent/area


def small_scale_energy(field, ocean):
    """Mean squared gradient magnitude over ocean cells (sharpness proxy).

    Progressive auto-regressive smoothing shows up as this value decaying
    toward zero, so it is a sensitive indicator of spectral collapse.
    """
    gy, gx = np.gradient(field.astype(np.float64))
    grad2 = gx ** 2 + gy ** 2
    return float(grad2[ocean].mean())


def step_diagnostics(state, ocean, cell_area):
    """Compute scalar diagnostics for one ensemble-mean state (6, Y, X)."""
    sit, sic = state[0], state[1]
    siu, siv = state[3], state[4]
    ice = sic > SIC_THRESHOLD
    total_area = float((cell_area * ice * ocean).sum())  # m^2
    speed = np.sqrt(siu ** 2 + siv ** 2)
    return {
        "mean_sit": float(sit[ocean].mean()),
        "mean_sic": float(sic[ocean].mean()),
        "max_sit": float(sit[ocean].max()),
        "ice_area_km2": total_area / 1e6,
        "mean_speed": float(speed[ocean].mean()),
        "frac_sic_saturated": float((sic[ocean] >= 0.999).mean()),
        "frac_sic_zero": float((sic[ocean] <= 1e-4).mean()),
        "frac_sit_zero": float((sit[ocean] <= 1e-4).mean()),
        "sharpness_sit": small_scale_energy(sit, ocean),
        "sharpness_sic": small_scale_energy(sic, ocean),
        "n_nonfinite": int((~np.isfinite(state)).sum()),
    }


def ice_area_km2(sic_thw, ocean, cell_area_km2, metric, threshold):
    """Per-timestep ice area/extent over ocean, in km^2. Pure (testable)."""
    sic = np.nan_to_num(sic_thw)
    if metric == "extent":
        field = (sic > threshold).astype(np.float32)
    else:  # area = concentration-weighted
        field = sic
    return (field * ocean * cell_area_km2).sum(axis=(1, 2))
