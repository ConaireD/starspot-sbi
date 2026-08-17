"""
Numerical stress: extreme inputs the fast suite avoids. Each test asserts
the function gives the right answer or raises, never a silently wrong
number. One test is written to fail deliberately: ssim_map on a grid with
fewer rows than the filter half-window returns an image of the wrong shape
rather than raising (see its docstring).
"""

import numpy as np
import pytest

from starspot_sbi.indexing import (coeffs_to_real, idx_to_lm, lm_to_idx,
                                   lm_indices, n_coeffs)
from starspot_sbi.design_matrix import build_B, build_design_matrix, forward_model
from starspot_sbi.metrics import (VIS_TOL, crps, detection_operating_points,
                                  mae, pr_auc, rmse, scalar_metrics, ssim_map,
                                  visibility_mask, weights, wmean)
from starspot_sbi.render import render, render_coeffs
from starspot_sbi.surfaces import generate_spotted_surface, place_spot

pytestmark = pytest.mark.slow

L = 30
OMEGA = 2 * np.pi
T_OBS = np.linspace(0.0, 1.0, 216, endpoint=False)


def test_visibility_and_weights_at_exact_beta_extremes():
    """
    beta exactly 0 and exactly pi/2 on the production grid, derived from the
    grid colatitudes rather than hardcoded row numbers. At beta = 0 both
    pole rows are excluded (theta = 0 geometrically, theta = pi because
    sin(pi) evaluates to 1.2e-16 under VIS_TOL); at beta = pi/2 exactly the
    lower half survives.
    """
    theta = np.linspace(0, np.pi, 120)

    m0 = visibility_mask(0.0, 120)
    assert np.array_equal(m0, np.sin(theta) > VIS_TOL)
    assert not m0[0] and not m0[-1] and m0[1:-1].all()

    m90 = visibility_mask(np.pi / 2, 120)
    assert np.array_equal(m90, np.sin(theta - np.pi / 2) > VIS_TOL)
    assert m90.sum() == np.sum(theta > np.pi / 2)

    for beta in (0.0, np.pi / 2):
        for kind in ('full', 'vis', 'wmean'):
            w = weights(beta, 120, 240, kind=kind)
            assert abs(w.sum() - 1.0) < 1e-12
            assert np.all(w >= 0)
    print(f"beta=0 visible rows {int(m0.sum())}, "
          f"beta=pi/2 visible rows {int(m90.sum())}")


def test_indexing_bijection_high_degree():
    """
    lm_to_idx / idx_to_lm at block boundaries up to l = 300, where a naive
    floor(sqrt(idx)) could misfloor. All four boundary indices per degree.
    """
    for l in range(301):
        for idx in (l * l, l * l + l, (l + 1) ** 2 - 1):
            ll, mm = idx_to_lm(idx)
            assert lm_to_idx(ll, mm) == idx
            assert ll == l
    print("bijection holds at block boundaries to l = 300")


@pytest.mark.parametrize('theta_s', [0.0, np.pi])
def test_polar_spot_is_axisymmetric_and_flux_constant(theta_s, kernels_L30):
    """
    A spot centred on either pole keeps only m = 0 coefficients, so its
    light curve is constant in time at any inclination. The residual
    modulation is the kernel quadrature floor, measured 3e-9 relative.
    """
    s = generate_spotted_surface(L, [{'theta': theta_s, 'phi': 0.7,
                                      'radius': np.deg2rad(10),
                                      'contrast': 0.6}], lanczos=True)
    scale = np.max(np.abs(s))
    for l, m in lm_indices(L):
        if m != 0:
            assert abs(s[lm_to_idx(l, m)]) < 1e-14 * scale

    A = build_design_matrix(L, 0.7, OMEGA, T_OBS, kernels_L30['phot'])
    mu = forward_model(s, A)
    rel = np.std(mu) / abs(np.mean(mu))
    print(f"pole {theta_s:.2f}: flux std/mean = {rel:.2e} (bound 1e-7)")
    assert rel < 1e-7


def test_extreme_radius_and_contrast_spots():
    """
    Spots of radius 1 and 45 degrees, contrast 0 and 1, at a pole and at
    the equator. Checks: the exact sphere-mean identity
    s_0^0 / (2 sqrt(pi)) = 1 + delta (1 - cos rho) / 2 (the l = 0 cap
    coefficient is closed-form, so tolerance is roundoff); contrast 1 is
    exactly no spot; renders stay finite; a 1-degree polar spot, far below
    the L = 30 resolution, still puts the image minimum at the pole.
    """
    for rho_deg in (1.0, 45.0):
        for contrast in (0.0, 1.0):
            for theta_s in (0.0, np.pi / 2):
                spot = {'theta': theta_s, 'phi': 0.0,
                        'radius': np.radians(rho_deg), 'contrast': contrast}
                s = generate_spotted_surface(L, [spot], lanczos=True)
                delta = contrast - 1.0
                expected = 1.0 + delta * (1 - np.cos(np.radians(rho_deg))) / 2
                got = s[0].real / (2 * np.sqrt(np.pi))
                assert abs(got - expected) < 1e-12
                if contrast == 1.0:
                    assert np.max(np.abs(s[1:])) == 0.0
                img = render_coeffs(s, 120, 240, L=L)
                assert np.all(np.isfinite(img))

    tiny = generate_spotted_surface(L, [{'theta': 0.0, 'phi': 0.0,
                                         'radius': np.radians(1.0),
                                         'contrast': 0.0}], lanczos=True)
    img = render_coeffs(tiny, 120, 240, L=L)
    row = np.unravel_index(np.argmin(img), img.shape)[0]
    depth = 1.0 - img.min()
    print(f"1-degree polar black spot: argmin row {row}, depth {depth:.2e}")
    assert row < 10
    assert 0 < depth < 0.1        # far below resolution; measured 3.1e-2


def test_black_wide_spot_flux_bounds(kernels_L30):
    """
    A 45-degree black (contrast 0) equatorial spot, edge-on. The cap covers
    (1 - cos 45)/2 = 14.6 per cent of the sphere, so the flux must modulate
    strongly while staying near (0, 1]. Measured min 0.5010 and max 1.0002:
    the truncated cap's Gibbs overshoot brightens the ring around the spot,
    so the flux may exceed 1 by a few 1e-4 when the spot is hidden. Rendered
    minimum -0.018.
    """
    s = generate_spotted_surface(L, [{'theta': np.pi / 2, 'phi': 0.0,
                                      'radius': np.radians(45.0),
                                      'contrast': 0.0}], lanczos=True)
    A = build_design_matrix(L, 0.0, OMEGA, T_OBS, kernels_L30['phot'])
    mu = forward_model(s, A)
    img = render_coeffs(s, 120, 240, L=L)
    print(f"flux range ({mu.min():.4f}, {mu.max():.4f}), "
          f"modulation {mu.max() - mu.min():.4f}, render min {img.min():.3f}")
    assert np.all(np.isfinite(mu))
    assert 0.2 < mu.min() and mu.max() < 1.01
    assert mu.max() - mu.min() > 0.05
    assert img.min() > -0.5


def test_unspotted_surface_through_every_metric():
    """
    A uniform surface through every metric. Right answers: SSIM 1, RMSE and
    MAE 0, CRPS 0 for a point mass at the truth. pr_auc and err_unc_corr
    return their documented nan. detection_operating_points has no documented
    contract for the nothing-to-detect case; its current -1 / nan values are
    loop initialisers leaking out, so the assertion is only that no
    plausible score comes back (nan or negative), not the sentinel itself,
    which is free to change to nan without breaking this test. A plausible
    finite score here would be the silent failure this test exists to catch.
    """
    flat = np.ones((120, 240))
    beta = 0.6

    out = scalar_metrics(flat, flat, beta, recon_std=np.zeros_like(flat),
                         samples=np.ones((4, 120, 240)))
    for key in ('ssim_aa_vis', 'ssim_aa_wmean', 'ssim_aa_full'):
        assert abs(out[key] - 1.0) < 1e-12
    for key in ('rmse_vis', 'rmse_full', 'mae_vis', 'mae_full',
                'crps_vis', 'crps_full'):
        assert abs(out[key]) < 1e-12
    assert np.isnan(out['pr_auc_vis']) and np.isnan(out['pr_auc_full'])
    assert np.isnan(out['err_unc_corr'])

    ops = detection_operating_points(flat, flat, beta)
    assert np.isnan(ops['f1_max']) or ops['f1_max'] < 0
    assert (np.isnan(ops['recall_at_precision_floor'])
            or ops['recall_at_precision_floor'] < 0)
    print("unspotted surface: identities exact, no plausible detection score")


def test_metrics_on_tiny_grids():
    """
    n_theta = 4, n_phi = 8, the smallest grid the 7-pixel SSIM window can
    serve (the polar continuation needs win_size//2 = 3 interior rows).
    Weights, SSIM identity/symmetry/bounds, RMSE, MAE, CRPS and pr_auc all
    behave; SSIM identity is exact by construction.
    """
    rng = np.random.default_rng(0)
    a = 1.0 + 0.2 * rng.normal(size=(4, 8))
    b = 1.0 + 0.2 * rng.normal(size=(4, 8))
    beta = 0.4

    for kind in ('full', 'vis', 'wmean'):
        w = weights(beta, 4, 8, kind=kind)
        assert abs(w.sum() - 1.0) < 1e-12

    s_ab, s_ba = ssim_map(a, b), ssim_map(b, a)
    assert s_ab.shape == (4, 8)
    assert np.array_equal(s_ab, s_ba)
    assert np.max(np.abs(ssim_map(a, a) - 1.0)) < 1e-12
    assert np.all(s_ab <= 1.0 + 1e-12) and np.all(s_ab >= -1.0 - 1e-12)

    assert rmse(a, a, beta) == 0.0
    assert mae(a, a, beta) == 0.0
    assert abs(crps(a, a[None].repeat(3, axis=0), beta)) < 1e-12
    auc = pr_auc(a - 0.3, b - 0.3, beta)
    assert np.isnan(auc) or 0.0 <= auc <= 1.0
    print("4 x 8 grid: all metrics behave")


def test_ssim_map_rejects_too_few_theta_rows():
    """
    The 7-pixel window's polar continuation reflects win_size // 2 = 3
    interior rows, so n_theta = 3 cannot be served: _box_theta_polar_fix
    used to return the wrong shape silently ((1, 8) for a (3, 8) input) and
    now raises. n_theta = 4 is the smallest servable grid and must keep its
    shape.
    """
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match='rows'):
        ssim_map(1.0 + 0.2 * rng.normal(size=(3, 8)),
                 1.0 + 0.2 * rng.normal(size=(3, 8)))
    a = 1.0 + 0.2 * rng.normal(size=(4, 8))
    assert ssim_map(a, a).shape == (4, 8)
    print("(3, 8) raises, (4, 8) keeps its shape")


def test_ssim_map_rejects_odd_phi_count():
    """
    The pole continuation rolls by half the phi grid, which assumes an even
    sample count; an odd count would roll by floor(n/2) and silently
    misalign the virtual rows. Verified that (8, 9) raises rather than
    returning something plausible.
    """
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match='even'):
        ssim_map(1.0 + 0.2 * rng.normal(size=(8, 9)),
                 1.0 + 0.2 * rng.normal(size=(8, 9)))
    print("(8, 9) raises on the odd phi axis")


def test_single_sample_crps_equals_weighted_mae():
    """
    With one sample the sharpness coefficient 2i - n - 1 vanishes, so CRPS
    must equal the weighted absolute error of that sample. Algebraic
    identity, so the tolerance is roundoff.
    """
    rng = np.random.default_rng(2)
    truth = 1.0 + 0.1 * rng.normal(size=(60, 120))
    sample = truth + 0.1 * rng.normal(size=truth.shape)
    beta = 0.5
    got = crps(truth, sample[None], beta)
    expected = wmean(np.abs(sample - truth), weights(beta, 60, 120, 'vis'))
    print(f"single-sample crps {got:.6e} vs weighted MAE {expected:.6e}, "
          f"diff {abs(got - expected):.2e}")
    assert abs(got - expected) < 1e-14


def test_design_matrix_at_exact_beta_extremes_L30(kernels_L30):
    """
    B at beta exactly 0 must reduce to the kernel laid out by (l, m); the
    guarded logarithms leave off-diagonal d(0) entries at ~1e-300 rather
    than zero, which must stay harmless. At beta exactly pi/2 everything is
    finite. The fast suite makes the beta = 0 check at L = 4.
    """
    B0 = build_B(L, 0.0, kernels_L30['phot'])
    scale = np.max(np.abs(kernels_L30['phot']))
    worst = 0.0
    for l, m in lm_indices(L):
        col = lm_to_idx(l, m)
        expect = np.zeros(2 * L + 1, dtype=complex)
        expect[m + L] = kernels_L30['phot'][col]
        worst = max(worst, np.max(np.abs(B0[:, col] - expect)) / scale)
    print(f"B(0) vs kernel layout, worst relative entry: {worst:.2e} "
          f"(bound 1e-12)")
    assert worst < 1e-12

    A = build_design_matrix(L, np.pi / 2, OMEGA, T_OBS,
                            [kernels_L30['x'], kernels_L30['y'],
                             kernels_L30['phot']])
    assert np.all(np.isfinite(A.real)) and np.all(np.isfinite(A.imag))


def test_render_and_place_spot_on_tiny_grid():
    """
    Rendering on n_theta = 4, n_phi = 8 with a spotted L = 30 surface:
    finite, pole rows constant in longitude, and the whole-sphere weighted
    mean still matches s_0^0 / (2 sqrt(pi)) to the coarse-grid quadrature
    error, measured 2 per cent.
    """
    s = generate_spotted_surface(L, [{'theta': 1.0, 'phi': 0.5,
                                      'radius': np.radians(20.0),
                                      'contrast': 0.5}], lanczos=True)
    img = render_coeffs(s, 4, 8, L=L)
    assert np.all(np.isfinite(img))
    assert np.ptp(img[0]) < 1e-10 and np.ptp(img[-1]) < 1e-10

    w = weights(0.0, 4, 8, kind='full')
    grid_mean = wmean(img, w)
    exact_mean = s[0].real / (2 * np.sqrt(np.pi))
    print(f"4 x 8 sphere mean {grid_mean:.4f} vs exact {exact_mean:.4f}")
    assert abs(grid_mean - exact_mean) < 0.1
