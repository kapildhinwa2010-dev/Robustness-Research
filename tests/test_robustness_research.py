"""Tests for Robustness-Research: noise sweeps and tolerance."""

import numpy as np
import pytest

from robustness_research import add_noise, noise_sweep, noise_tolerance


def _matrix():
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    return np.outer(np.sin(t), np.cos(t))


def test_add_noise_zero_identity_and_validation():
    W = _matrix()
    assert np.array_equal(add_noise(W, 0.0), W)
    with pytest.raises(ValueError):
        add_noise(W, -0.1)


def test_noise_sweep_covers_methods_and_levels():
    rows = noise_sweep(_matrix(), noise_levels=(0.0, 0.1), seed=1)
    assert {r["method"] for r in rows} == {"svd_rank4", "quantize8", "fourier4"}
    assert {r["noise"] for r in rows} == {0.0, 0.1}


def test_error_non_decreasing_with_noise_per_method():
    rows = noise_sweep(_matrix(), noise_levels=(0.0, 0.2), seed=1)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append((r["noise"], r["relative_error"]))
    for samples in by_method.values():
        samples.sort()
        assert samples[1][1] >= samples[0][1] - 1e-9


def test_noise_tolerance_finds_largest_safe_level():
    rows = [
        {"method": "m", "noise": 0.0, "relative_error": 0.01},
        {"method": "m", "noise": 0.05, "relative_error": 0.08},
        {"method": "m", "noise": 0.1, "relative_error": 0.2},
    ]
    assert noise_tolerance(rows, "m", max_error=0.1) == pytest.approx(0.05)
    assert noise_tolerance(rows, "m", max_error=0.01) == pytest.approx(0.0)
    assert noise_tolerance(rows, "m", max_error=0.005) == -1.0


def test_noise_tolerance_validates():
    with pytest.raises(ValueError):
        noise_tolerance([], "m")
    with pytest.raises(ValueError):
        noise_tolerance([{"method": "m", "noise": 0.0, "relative_error": 0.0}], "m", max_error=-1)
