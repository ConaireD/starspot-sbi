"""
Statistical properties over hundreds of surfaces drawn from the documented
dataset prior, at the production degree L = 30. These are per-surface
assertions the fast suite makes once on a single random vector, plus a
population check of the generator against the measured statistics in
docs/conventions.md section 10.
"""

import numpy as np
import pytest

from conftest import sample_prior_surfaces
from starspot_sbi.indexing import (coeffs_to_real, lm_indices, n_coeffs,
                                   real_to_coeffs)
from starspot_sbi.design_matrix import build_design_matrix, forward_model
from starspot_sbi.render import build_Ylm_matrix

pytestmark = pytest.mark.slow

L = 30
OMEGA = 2 * np.pi
T_OBS = np.linspace(0.0, 1.0, 216, endpoint=False)


def test_reality_condition_500_surfaces():
    """
    s_l^{-m} = (-1)^m conj(s_l^m) and real s_l^0 for 500 prior surfaces, and
    the real-packing round trip. place_spot builds the two halves through
    separate phase evaluations, so agreement is floating point rather than
    bitwise; measured deviations are ~1e-16 of the coefficient scale.
    """
    rng = np.random.default_rng(20260817)
    surfaces = sample_prior_surfaces(rng, 500, L)

    idx = {(l, m): l * l + l + m for l, m in lm_indices(L)}
    worst_pair, worst_m0, worst_rt = 0.0, 0.0, 0.0
    for s in surfaces:
        scale = np.max(np.abs(s))
        for l in range(L + 1):
            worst_m0 = max(worst_m0, abs(s[idx[(l, 0)]].imag) / scale)
            for m in range(1, l + 1):
                dev = abs(s[idx[(l, -m)]] - (-1) ** m * np.conj(s[idx[(l, m)]]))
                worst_pair = max(worst_pair, dev / scale)
        rt = real_to_coeffs(coeffs_to_real(s))
        worst_rt = max(worst_rt, np.max(np.abs(rt - s)) / scale)
    print(f"reality pairs {worst_pair:.2e}, m=0 imag {worst_m0:.2e}, "
          f"round trip {worst_rt:.2e} (bounds 1e-13)")
    assert worst_pair < 1e-13
    assert worst_m0 < 1e-13
    assert worst_rt < 1e-13


def test_forward_model_imaginary_residual_300_surfaces(kernels_L30):
    """
    max|Im| / max|Re| of A s for 300 prior surfaces, all three channels, two
    inclinations. Conventions section 8 measures 1e-19 to 3e-15 on single
    draws; the bound is the fast suite's 1e-12.
    """
    rng = np.random.default_rng(11)
    surfaces = sample_prior_surfaces(rng, 300, L)
    ks = [kernels_L30['x'], kernels_L30['y'], kernels_L30['phot']]

    worst = 0.0
    for beta in (0.3, 1.2):
        A = build_design_matrix(L, beta, OMEGA, T_OBS, ks)
        for s in surfaces:
            mu = A @ s
            worst = max(worst, np.max(np.abs(mu.imag)) / np.max(np.abs(mu.real)))
    print(f"worst max|Im| / max|Re| over 300 surfaces x 2 betas: {worst:.2e} "
          f"(bound 1e-12)")
    assert worst < 1e-12


def test_design_matrix_linearity_200_pairs(kernels_L30):
    """
    Linearity of the forward model over 200 random coefficient pairs with
    random scalings. Deviations are roundoff; measured below 1e-14 relative
    to the signal scale.
    """
    rng = np.random.default_rng(12)
    A = build_design_matrix(L, 0.7, OMEGA, T_OBS,
                            [kernels_L30['x'], kernels_L30['phot']])
    worst = 0.0
    for _ in range(200):
        a = real_to_coeffs(rng.normal(size=n_coeffs(L)))
        b = real_to_coeffs(rng.normal(size=n_coeffs(L)))
        ca, cb = rng.normal(), rng.normal()
        lhs = forward_model(ca * a + cb * b, A)
        rhs = ca * forward_model(a, A) + cb * forward_model(b, A)
        worst = max(worst, np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs)))
    print(f"worst relative nonlinearity over 200 pairs: {worst:.2e} "
          f"(bound 1e-12)")
    assert worst < 1e-12


def test_rotational_equivariance_200_surfaces(kernels_L30):
    """
    s_l^m -> s_l^m e^{-i m D} equals t -> t + D/omega for 200 prior
    surfaces, all three channels, at the production degree. The fast suite
    asserts this once at L = 4. Measured relative deviation ~1e-13.
    """
    rng = np.random.default_rng(13)
    surfaces = sample_prior_surfaces(rng, 200, L)
    ks = [kernels_L30['x'], kernels_L30['y'], kernels_L30['phot']]
    D = 0.37
    phase = np.array([np.exp(-1j * m * D) for l, m in lm_indices(L)])

    A = build_design_matrix(L, 0.7, OMEGA, T_OBS, ks)
    A_shift = build_design_matrix(L, 0.7, OMEGA, T_OBS + D / OMEGA, ks)

    worst = 0.0
    for s in surfaces:
        lhs = forward_model(s * phase, A)
        rhs = forward_model(s, A_shift)
        worst = max(worst, np.max(np.abs(lhs - rhs)) / np.max(np.abs(rhs)))
    print(f"worst relative equivariance deviation over 200 surfaces: "
          f"{worst:.2e} (bound 1e-10)")
    assert worst < 1e-10


def test_min_intensity_distribution_500_surfaces():
    """
    The generator's population against the measured table in conventions
    section 10 (median minimum intensity 0.536, deepest -0.59, 0.055 per
    cent below zero, on the (L+1) x (2L+2) grid). With 500 draws the median
    standard error is ~0.02, so the bounds are (0.45, 0.62); the deepest
    minimum must stay above -0.6 and the below-zero fraction under 1 per
    cent.
    """
    rng = np.random.default_rng(14)
    surfaces = sample_prior_surfaces(rng, 500, L)

    Y = build_Ylm_matrix(L, L + 1, 2 * L + 2)
    S = np.column_stack(surfaces)
    mins = (Y @ S).real.min(axis=0)

    median = float(np.median(mins))
    frac_neg = float(np.mean(mins < 0))
    print(f"min intensity: median {median:.3f} (documented 0.536), "
          f"deepest {mins.min():.3f}, below zero {100 * frac_neg:.2f}%")
    assert 0.45 < median < 0.62
    assert mins.min() > -0.6
    assert frac_neg < 0.01
