"""
The full inclination grid. The fast suite samples three or four betas; the
models run at all 91 integer degrees, so a failure at an unsampled beta is a
failure of the released pipeline. Everything here sweeps all 91.

The flux-invariance sweep runs at L = 16 rather than the production 30
because the Wigner stack costs ~0.3 s per beta at L = 16 and ~0.7 s at
L = 30 (91 betas would exceed the per-test budget); L = 30 is spot-checked
at four betas including both endpoints.
"""

import numpy as np
import pytest

from starspot_sbi.indexing import n_coeffs
from starspot_sbi.kernels import precompute_kernels_fast
from starspot_sbi.design_matrix import build_design_matrix, forward_model
from starspot_sbi.metrics import VIS_TOL, visibility_mask, weights

pytestmark = pytest.mark.slow

N_THETA = 120
THETA = np.linspace(0, np.pi, N_THETA)


def test_visibility_mask_all_integer_betas():
    """
    The mask at every integer beta against sin(theta - beta) recomputed from
    the grid colatitudes, and two structural facts: the mask has no gaps,
    and from beta = 1 onwards the visible count never increases. beta = 0
    is the documented exception (conventions section 11): sin(pi) evaluates
    to 1.2e-16 under VIS_TOL, so the theta = pi row is hidden there and the
    count steps 118 -> 119 between beta = 0 and beta = 1.
    """
    counts = []
    for bdeg in range(91):
        beta = np.radians(bdeg)
        mask = visibility_mask(beta, N_THETA)
        expected = np.sin(THETA - beta) > VIS_TOL
        assert np.array_equal(mask, expected), f"mask wrong at beta={bdeg}"

        vis_idx = np.flatnonzero(mask)
        assert vis_idx.size > 0
        assert np.all(mask[vis_idx[0]:vis_idx[-1] + 1]), f"gap at beta={bdeg}"
        counts.append(int(mask.sum()))

    assert counts[0] == N_THETA - 2          # both pole rows out at beta = 0
    for bdeg in range(2, 91):
        assert counts[bdeg] <= counts[bdeg - 1], \
            f"visible count grew at beta={bdeg}"
    print(f"visible rows: {counts[0]} at beta=0, {counts[1]} at beta=1, "
          f"{counts[-1]} at beta=90")


def test_weights_all_integer_betas_all_kinds():
    """
    Weights at every integer beta under all three weightings: normalised to
    one, non-negative, constant in longitude, 'vis' supported exactly on the
    mask, 'wmean' negligible inside the hidden cap, 'full' independent of
    beta. Normalisation deviations are pure roundoff, measured below 1e-15.
    The 'wmean' rows just inside the cap are not exactly zero: the partial
    branch of lat_mu_grid leaves ~1e-36 residuals where mu_max has already
    fallen under VIS_TOL, so the bound there is 1e-30 rather than equality.
    """
    w_full_ref = weights(0.0, N_THETA, 240, kind='full')
    worst_norm = 0.0
    for bdeg in range(91):
        beta = np.radians(bdeg)
        mask = visibility_mask(beta, N_THETA)
        for kind in ('full', 'vis', 'wmean'):
            w = weights(beta, N_THETA, 240, kind=kind)
            worst_norm = max(worst_norm, abs(w.sum() - 1.0))
            assert np.all(w >= 0)
            assert np.max(np.ptp(w, axis=1)) == 0.0
            if kind == 'vis':
                assert np.array_equal(w[:, 0] > 0, mask)
            if kind == 'wmean':
                assert np.all(w[~mask, :] < 1e-30)
        assert np.array_equal(weights(beta, N_THETA, 240, 'full'), w_full_ref)
    print(f"worst |sum(w) - 1| over 91 betas x 3 kinds: {worst_norm:.2e} "
          f"(bound 1e-12)")
    assert worst_norm < 1e-12


def _uniform_flux_check(L, betas_deg, kernels):
    """Uniform unit map through the design matrix at each beta."""
    s = np.zeros(n_coeffs(L), dtype=complex)
    s[0] = 2 * np.sqrt(np.pi)
    t_obs = np.linspace(0, 1.0, 24, endpoint=False)
    worst_f, worst_a = 0.0, 0.0
    for bdeg in betas_deg:
        A = build_design_matrix(L, np.radians(bdeg), 2 * np.pi, t_obs, kernels)
        mu = forward_model(s, A).reshape(3, -1)
        worst_a = max(worst_a, np.max(np.abs(mu[:2])))
        worst_f = max(worst_f, np.max(np.abs(mu[2] - 1.0)))
    return worst_f, worst_a


def test_uniform_map_flux_invariance_all_betas_L16():
    """
    F_0 = 1 and zero astrometry for the uniform map at all 91 integer betas.
    The fast suite checks three betas at L = 4. The flux error is the l = 0
    quadrature floor: measured max |F - 1| = 4.18e-9 over the sweep
    (conventions section 6 gives 4.2e-9), bound 1e-7 as in the fast suite.
    The astrometric channels are zero by the selection rules; pure roundoff.
    """
    kx, ky, kp = precompute_kernels_fast(16)
    worst_f, worst_a = _uniform_flux_check(16, range(91), [kx, ky, kp])
    print(f"max |F - 1| = {worst_f:.2e} (bound 1e-7), "
          f"max |astro| = {worst_a:.2e} (bound 1e-12)")
    assert worst_f < 1e-7
    assert worst_a < 1e-12


def test_uniform_map_flux_invariance_L30_spot_betas(kernels_L30):
    """The same invariance at the production degree, endpoints included."""
    ks = [kernels_L30['x'], kernels_L30['y'], kernels_L30['phot']]
    worst_f, worst_a = _uniform_flux_check(30, [0, 27, 63, 90], ks)
    print(f"L=30: max |F - 1| = {worst_f:.2e} (bound 1e-7), "
          f"max |astro| = {worst_a:.2e} (bound 1e-12)")
    assert worst_f < 1e-7
    assert worst_a < 1e-12


def test_pole_on_flux_constancy_approach_L30(kernels_L30):
    """
    Photometric modulation of a spotted surface as beta approaches 90
    degrees. Rotation cannot modulate the flux pole-on, so std/mean must
    fall towards the quadrature floor and be below 1e-6 at exactly pi/2
    (the fast suite's bound, measured 4e-8 at L = 4). Monotonicity is
    asserted with a 1.05 slack factor since the decrease is geometric.
    """
    from starspot_sbi.surfaces import generate_spotted_surface
    spots = [{'theta': 1.2, 'phi': 0.3, 'radius': np.deg2rad(10), 'contrast': 0.6},
             {'theta': 2.1, 'phi': -1.4, 'radius': np.deg2rad(8), 'contrast': 0.7}]
    s = generate_spotted_surface(30, spots, lanczos=True)
    t_obs = np.linspace(0, 1.0, 216, endpoint=False)

    ratios = []
    betas = [np.radians(b) for b in (85, 86, 87, 88, 89)] + [np.pi / 2]
    for beta in betas:
        A = build_design_matrix(30, beta, 2 * np.pi, t_obs, kernels_L30['phot'])
        mu = forward_model(s, A)
        ratios.append(np.std(mu) / abs(np.mean(mu)))
    for beta, r in zip(betas, ratios):
        print(f"beta = {np.degrees(beta):5.1f} deg: std/mean = {r:.3e}")
    assert ratios[-1] < 1e-6
    for a, b in zip(ratios, ratios[1:]):
        assert b < a * 1.05
