"""
Shared fixtures and helpers for the slow suite (tests/test_slow_*).

The fast suite (test_00 to test_11) does not use anything here; this file only
defines fixtures and helper functions, so collecting it has no side effects.
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


def sample_prior_surfaces(rng, n, L=30):
    """
    Surfaces drawn from the documented dataset prior (docs/conventions.md
    section 12): 1 to 11 spots, sin-uniform latitude, uniform longitude,
    radius uniform on (6, 12) degrees, contrast uniform on (0.5, 0.9),
    Lanczos taper on. Returns a list of complex coefficient vectors.
    """
    from starspot_sbi.surfaces import generate_spotted_surface
    out = []
    for _ in range(n):
        k = int(rng.integers(1, 12))
        spots = [{'theta': float(np.arccos(rng.uniform(-1.0, 1.0))),
                  'phi': float(rng.uniform(0.0, 2.0 * np.pi)),
                  'radius': float(np.radians(rng.uniform(6.0, 12.0))),
                  'contrast': float(rng.uniform(0.5, 0.9))}
                 for _ in range(k)]
        out.append(generate_spotted_surface(L, spots, lanczos=True))
    return out
