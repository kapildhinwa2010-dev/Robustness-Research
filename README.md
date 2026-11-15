# Robustness-Research

[![CI](https://github.com/kapildhinwa2010-dev/Robustness-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Robustness-Research/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Robustness-Research/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Robustness-Research/actions/workflows/codeql.yml)

Noise robustness of compression methods, with per-method tolerance thresholds.

## What it does

Sweeps Gaussian noise levels (scaled by matrix std) against three
compression methods, reports reconstruction error against the clean
original per (method, noise), and computes noise_tolerance: the largest
swept noise level whose error stays under a threshold.

    from robustness_research import noise_sweep, noise_tolerance

    rows = noise_sweep(W, noise_levels=(0.0, 0.05, 0.1, 0.2))
    print(noise_tolerance(rows, "svd_rank4", max_error=0.1))

## API

- add_noise(matrix, noise, seed) - std-scaled Gaussian noise
- noise_sweep(matrix, noise_levels, seed) - error per method per level
- noise_tolerance(rows, method, max_error) - largest safe swept level

## Honest results and limitations

- Expected result: quantization error stays roughly flat as noise grows
  (noise passes through); SVD absorbs some noise but distorts the signal.
  Neither wins in general.
- Tolerance is limited to the swept grid; the true crossing point between
  two levels is not interpolated.
- Gaussian noise only; adversarial or structured corruption is out of scope.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
