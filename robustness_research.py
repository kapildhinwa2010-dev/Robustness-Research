# Robustness-Research - Noise robustness of compression, with tolerance thresholds
# Part of the Kapildhinwa2010-dev Toolkit.

"""RobustnessResearch - noise robustness experiments.

Lineage: robustness-layer repos. Sweeps Gaussian noise levels against
compression methods, reports error vs the clean original per method, and
finds each method noise_tolerance: the largest swept noise level whose
error stays under a threshold.
"""

from __future__ import annotations

import numpy as np

__all__ = ["add_noise", "noise_sweep", "noise_tolerance"]


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def add_noise(matrix: np.ndarray, noise: float, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise scaled by the matrix standard deviation."""
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    if noise < 0.0:
        raise ValueError("noise must be >= 0")
    rng = np.random.default_rng(seed)
    sigma = float(M.std()) or 1.0
    return M + noise * sigma * rng.standard_normal(M.shape)


def _methods(noisy: np.ndarray) -> dict:
    def svd4():
        u, s, vt = np.linalg.svd(noisy, full_matrices=False)
        rank = min(4, s.size)
        return (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    def quant8():
        lo, hi = float(noisy.min()), float(noisy.max())
        levels = 2 ** 8 - 1
        scale = (hi - lo) / levels if hi > lo else 1.0
        codes = np.clip(np.round((noisy - lo) / scale), 0, levels)
        return codes * scale + lo

    def fourier4():
        basis = _fourier_basis(noisy.shape[1], 4)
        coeffs, _, _, _ = np.linalg.lstsq(basis, noisy.T, rcond=None)
        return coeffs.T @ basis.T

    return {"svd_rank4": svd4, "quantize8": quant8, "fourier4": fourier4}


def noise_sweep(matrix: np.ndarray, noise_levels=(0.0, 0.02, 0.05, 0.1, 0.2),
                seed: int = 0) -> list:
    """Error vs the clean original, per method per noise level."""
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    norm = float(np.linalg.norm(M))
    rows = []
    for noise in noise_levels:
        if noise < 0.0:
            raise ValueError("noise levels must be >= 0")
        noisy = add_noise(M, noise, seed=seed)
        for name, func in _methods(noisy).items():
            approx = func()
            rel = float(np.linalg.norm(M - approx)) / norm if norm else 0.0
            rows.append({"method": name, "noise": float(noise), "relative_error": rel})
    return rows


def noise_tolerance(rows: list, method: str, max_error: float = 0.1) -> float:
    """Largest swept noise level whose error stays under max_error.

    Returns -1.0 when the method exceeds the threshold at every level.
    """
    if max_error < 0.0:
        raise ValueError("max_error must be >= 0")
    samples = sorted((r["noise"], r["relative_error"])
                     for r in rows if r["method"] == method)
    if not samples:
        raise ValueError("no rows for method {}".format(method))
    best = -1.0
    for noise, error in samples:
        if error <= max_error:
            best = max(best, noise)
    return best
