"""
Shared fixtures for the slow suite (tests/test_slow_*).

The fast suite (test_00 to test_12) does not use anything here; this file only
defines fixtures, so collecting it has no side effects. Consumers take the
fixtures as arguments rather than importing this module, so the suite survives
--import-mode=importlib.
"""

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO / 'weights'
HOLDOUT_DIR = (REPO.parent / 'Sydney' / 'data' / 'fast'
               / 'L30_N216_grouped' / 'explicit_holdout')


@pytest.fixture(scope='session')
def kernels_L30():
    """Fast-path kernels at the production degree, computed once per session."""
    from starspot_sbi.kernels import precompute_kernels_fast
    kx, ky, kphot = precompute_kernels_fast(30)
    return {'x': kx, 'y': ky, 'phot': kphot}


@pytest.fixture(scope='session')
def prior_surfaces():
    """
    A sampler of surfaces from the documented dataset prior
    (docs/conventions.md section 12): 1 to 11 spots inclusive, sin-uniform
    latitude on (-90, 90), longitude uniform on (0, 360), radius uniform on
    (6, 12) degrees, contrast uniform on (0.5, 0.9), Lanczos taper on.

    Returns a callable (rng, n, L, return_params=False) giving a list of
    complex coefficient vectors, or (surfaces, params) when return_params is
    true, where params holds the list of spot dictionaries per surface.
    """
    from starspot_sbi.surfaces import generate_spotted_surface

    def sample(rng, n, L=30, return_params=False):
        surfaces, params = [], []
        for _ in range(n):
            k = int(rng.integers(1, 12))
            spots = [{'theta': float(np.arccos(rng.uniform(-1.0, 1.0))),
                      'phi': float(np.radians(rng.uniform(0.0, 360.0))),
                      'radius': float(np.radians(rng.uniform(6.0, 12.0))),
                      'contrast': float(rng.uniform(0.5, 0.9))}
                     for _ in range(k)]
            surfaces.append(generate_spotted_surface(L, spots, lanczos=True))
            params.append(spots)
        return (surfaces, params) if return_params else surfaces

    return sample
