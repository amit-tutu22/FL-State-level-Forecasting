"""Differential privacy helpers for FL model updates (clip + Gaussian noise)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def flatten(params: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    shapes = [p.shape for p in params]
    flat = np.concatenate([p.astype(np.float64).ravel() for p in params])
    return flat, shapes


def unflatten(flat: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    out = []
    offset = 0
    for shape in shapes:
        n = int(np.prod(shape))
        out.append(flat[offset : offset + n].reshape(shape).astype(np.float32))
        offset += n
    return out


def clip_l2(vec: np.ndarray, clip_norm: float) -> np.ndarray:
    if clip_norm <= 0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= clip_norm or norm == 0.0:
        return vec
    return vec * (clip_norm / norm)


def apply_dp_to_update(
    global_params: Sequence[np.ndarray],
    local_params: Sequence[np.ndarray],
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Clip the update delta = local - global, add N(0, (noise_multiplier * clip_norm)^2),
    and return global + noisy_delta.
    """
    g_flat, shapes = flatten(global_params)
    l_flat, _ = flatten(local_params)
    delta = clip_l2(l_flat - g_flat, clip_norm)
    if noise_multiplier > 0 and clip_norm > 0:
        sigma = noise_multiplier * clip_norm
        delta = delta + rng.normal(0.0, sigma, size=delta.shape)
    return unflatten(g_flat + delta, shapes)


def approx_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5,
) -> float:
    """
    Rough Gaussian-mechanism composition estimate (research communication only).

    Not a tight moments-accountant bound. Returns +inf when noise_multiplier == 0.
    """
    if noise_multiplier <= 0:
        return float("inf")
    if steps <= 0:
        return 0.0
    # Common tutorial-style bound: eps ≈ sqrt(2 T log(1.25/delta)) / sigma
    return float(np.sqrt(2.0 * steps * np.log(1.25 / delta)) / noise_multiplier)
