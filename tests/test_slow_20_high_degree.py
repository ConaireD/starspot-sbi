"""
High-degree checks the fast suite cannot afford: kernels, Wigner-d, the design
matrix and SSIM at L = 20 to 30 and at the production grid size.

References are independent of the package's fast paths: adaptive weighted
quadrature (scipy.integrate.quad with the sqrt(1-u^2) weight built into the
rule, exact to ~1e-14 for these polynomial-times-weight integrands) for the
theta integrals, sympy for the Wigner matrices, and a pixel-space integration
built from n_hat alone (docs/conventions.md D5, D7) for the design matrix.

Every tolerance derives from a measurement recorded in the docstring, with
around 100x headroom unless stated otherwise.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, lpmv, sph_harm_y

from starspot_sbi.indexing import coeffs_to_real, lm_indices, n_coeffs
from starspot_sbi.kernels import (I_phi_x_fast, I_phi_y_fast,
                                  precompute_kernels_fast)
from starspot_sbi.design_matrix import build_design_matrix, forward_model
from starspot_sbi.render import render
from starspot_sbi.metrics import ssim_map
from starspot_sbi.surfaces import generate_spotted_surface
from starspot_sbi.wigner import wigner_d_matrix_exact, wigner_d_matrix_fast

pytestmark = pytest.mark.slow

L = 30
OMEGA = 2 * np.pi
T_OBS = np.linspace(0.0, 1.0, 216, endpoint=False)


def _quad_theta(l, m, channel):
    """
    Theta integral by adaptive quadrature with the sqrt(1-u^2) endpoint weight
    built into the rule, so the remaining integrand is a polynomial and the
    result is exact to roundoff. Validated below against sympy at (20, 2).
    lpmv is unnormalised, reaching ~1e40 at l = m = 30 and overflowing above
    l ~ 90, which is why kernels.py evaluates sph_harm_y instead; this
    reference therefore does not extend to higher degrees.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if channel == 'x':
            val, _ = quad(lambda u: (1 - u * u) * lpmv(m, l, u), -1, 1)
        elif channel == 'y':
            val, _ = quad(lambda u: u * lpmv(m, l, u), -1, 1,
                          weight='alg', wvar=(0.5, 0.5))
        else:
            val, _ = quad(lambda u: lpmv(m, l, u), -1, 1,
                          weight='alg', wvar=(0.5, 0.5))
    return val


def test_quad_reference_agrees_with_sympy_anchors():
    """
    The quadrature reference itself, against two anchors: the closed form
    pi/2 at (0, 0) and the sympy value at (20, 2), which took 27 s to compute
    once and is frozen here. Measured errors 4.4e-16 and 1.4e-14.
    """
    a = _quad_theta(0, 0, 'phot')
    b = _quad_theta(20, 2, 'phot')
    print(f"(0,0): {a - np.pi/2:.2e}   (20,2): {b - 0.2939959933131937:.2e}")
    assert abs(a - np.pi / 2) < 1e-12
    assert abs(b - 0.2939959933131937) < 1e-11


def test_phi_integrals_closed_form_to_m30():
    """
    The closed-form phi integrals against numeric quadrature for every order
    up to 30, real and imaginary parts. The fast suite checks |m| <= 12
    against sympy; this covers the full production range. Measured maximum
    error 6.0e-16.
    """
    worst = 0.0
    for m in range(-L, L + 1):
        re, _ = quad(lambda p: np.sin(p) * np.cos(p) * np.cos(m * p),
                     -np.pi / 2, np.pi / 2)
        im, _ = quad(lambda p: np.sin(p) * np.cos(p) * np.sin(m * p),
                     -np.pi / 2, np.pi / 2)
        worst = max(worst, abs(I_phi_x_fast(m) - (re + 1j * im)))
        re, _ = quad(lambda p: np.cos(p) * np.cos(m * p), -np.pi / 2, np.pi / 2)
        im, _ = quad(lambda p: np.cos(p) * np.sin(m * p), -np.pi / 2, np.pi / 2)
        worst = max(worst, abs(I_phi_y_fast(m) - (re + 1j * im)))
    print(f"max abs error over m = -30..30: {worst:.2e} (bound 1e-13)")
    assert worst < 1e-13


def test_kernels_fast_vs_adaptive_quadrature_L30(kernels_L30):
    """
    Full kernel vectors at the production degree against a reference built
    from the closed-form phi integrals and the adaptive weighted quadrature.
    The fast suite compares against sympy at L = 8 only, where the measured
    relative errors are 2.7e-11 (kx) and ~5e-7 (ky, kphot).

    Measured here at L = 30, relative to the largest entry of each channel:
    kx 1.1e-11, ky 7.4e-8, kphot 3.3e-8. Bounds carry ~100x headroom.
    """
    n = n_coeffs(L)
    ref = {c: np.zeros(n, dtype=complex) for c in ('x', 'y', 'phot')}
    for idx, (l, m) in enumerate(lm_indices(L)):
        am = abs(m)
        Nm = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(
            0.5 * (gammaln(l - am + 1) - gammaln(l + am + 1)))
        cs = (-1) ** am if m < 0 else 1
        ipx, ipy = I_phi_x_fast(m), I_phi_y_fast(m)
        if abs(ipx) > 1e-15:
            ref['x'][idx] = (Nm / np.pi) * ipx * _quad_theta(l, am, 'x') * cs
        if abs(ipy) > 1e-15:
            ref['y'][idx] = (Nm / np.pi) * ipy * _quad_theta(l, am, 'y') * cs
            ref['phot'][idx] = (Nm / np.pi) * ipy * _quad_theta(l, am, 'phot') * cs

    bounds = {'x': 1e-9, 'y': 1e-5, 'phot': 1e-5}
    for c in ('x', 'y', 'phot'):
        rel = np.max(np.abs(kernels_L30[c] - ref[c])) / np.max(np.abs(ref[c]))
        print(f"k{c}: max|fast - ref| / max|ref| = {rel:.2e} "
              f"(bound {bounds[c]:.0e})")
        assert rel < bounds[c]


def test_kernel_selection_rules_L30(kernels_L30):
    """
    Selection rules (docs/conventions.md K7) at the production degree. The
    fast suite checks them at low L; a parity error that grows with degree
    would pass there. Analytically-zero entries measure ~1e-16 absolute
    (conventions section 6); the bound is relative to each channel's largest
    entry.
    """
    def allowed(c, l, m):
        if c == 'x':
            return (l % 2 == 1 and m % 2 == 1) or (l, abs(m)) == (2, 2)
        if c == 'y':
            return (l % 2 == 1 and m % 2 == 0) or (l, abs(m)) == (2, 1)
        return (l % 2 == 0 and m % 2 == 0) or (l, abs(m)) == (1, 1)

    for c in ('x', 'y', 'phot'):
        k = kernels_L30[c]
        scale = np.max(np.abs(k))
        worst = 0.0
        for idx, (l, m) in enumerate(lm_indices(L)):
            if not allowed(c, l, m):
                worst = max(worst, abs(k[idx]))
        print(f"k{c}: largest forbidden entry {worst:.2e} of scale {scale:.2e}")
        assert worst / scale < 1e-12


@pytest.mark.parametrize('l', [16, 20])
def test_wigner_fast_matches_sympy(l):
    """
    Fast log-factorial d^l(beta) against sympy at degrees beyond the fast
    suite's l <= 12. Measured max abs difference 1.2e-12 at l = 16 and
    1.1e-11 at l = 20 (beta = 0.7).
    """
    for beta in (0.7, 1.2):
        de = wigner_d_matrix_exact(l, beta).real
        df = wigner_d_matrix_fast(l, beta)
        diff = np.max(np.abs(df - de))
        print(f"l={l} beta={beta}: max|fast - sympy| = {diff:.2e} (bound 1e-9)")
        assert diff < 1e-9


def test_wigner_orthogonality_l20_to_l30():
    """
    d(beta)^T d(beta) = I and d(-beta) d(beta) = I for l = 20, 25, 30 over
    the inclination range including both endpoints. Cancellation grows with
    l times beta: measured 4.3e-13 at (30, 0.2), 1.7e-7 at (30, 1.2), and
    conventions section 7 gives 5.5e-7 at (30, pi/2), the working accuracy
    of the rotation. The bound is 2e-6 everywhere, ~4x above the worst
    measured point and four orders below the mission noise.
    """
    bound = 2e-6
    for l in (20, 25, 30):
        eye = np.eye(2 * l + 1)
        for beta in (0.2, 0.7, 1.2, 1.55, np.pi / 2):
            d = wigner_d_matrix_fast(l, beta)
            dev_o = np.max(np.abs(d.T @ d - eye))
            dev_i = np.max(np.abs(wigner_d_matrix_fast(l, -beta) @ d - eye))
            print(f"l={l} beta={beta:.3f}: |d^T d - I| = {dev_o:.2e}, "
                  f"|d(-b) d(b) - I| = {dev_i:.2e} (bound {bound:.0e})")
            assert dev_o < bound
            assert dev_i < bound


def _direct_signals(s, beta, t_vals, n_th, n_ph):
    """
    Pixel-space integration of the three observables, built from n_hat and
    the sky axes alone (docs/conventions.md D5, D7): Gauss-Legendre in
    cos(theta), uniform periodic grid in phi. Shares no code with the
    kernel/Wigner/design path beyond sph_harm_y.
    """
    u, wu = np.polynomial.legendre.leggauss(n_th)
    theta = np.arccos(u)
    phi = np.linspace(-np.pi, np.pi, n_ph, endpoint=False)
    dphi = 2 * np.pi / n_ph

    g = np.zeros((2 * L + 1, n_th), dtype=complex)      # g_m(theta)
    for l, m in lm_indices(L):
        g[m + L] += s[l * l + l + m] * sph_harm_y(l, m, theta, 0.0)
    m_vals = np.arange(-L, L + 1)
    intensity = np.real(g.T @ np.exp(1j * np.outer(m_vals, phi)))

    sin_t, cos_t = np.sin(theta)[:, None], np.cos(theta)[:, None]
    cp, sp = np.cos(phi)[None, :], np.sin(phi)[None, :]
    cb, sb = np.cos(beta), np.sin(beta)

    out = np.zeros((3, len(t_vals)))                    # x, y, phot
    for i, t in enumerate(t_vals):
        cw, sw = np.cos(OMEGA * t), np.sin(OMEGA * t)
        mu = cb * cw * sin_t * cp - cb * sw * sin_t * sp - sb * cos_t
        x_obs = sw * sin_t * cp + cw * sin_t * sp
        y_obs = sb * cw * sin_t * cp - sb * sw * sin_t * sp + cb * cos_t
        base = intensity * np.maximum(mu, 0.0) * dphi / np.pi
        out[0, i] = np.einsum('t,tp->', wu, base * x_obs)
        out[1, i] = np.einsum('t,tp->', wu, base * y_obs)
        out[2, i] = np.einsum('t,tp->', wu, base)
    return out


def test_design_matrix_vs_pixel_quadrature_L30(kernels_L30):
    """
    The full forward model at the production degree against direct pixel
    integration, all three channels, four inclinations including both exact
    endpoints, eight rotation phases.

    This is the one test that would catch a convention error anywhere in the
    kernel / Wigner / design-matrix chain at L = 30 rather than at L = 4.
    Measured relative disagreement at (400, 800) quadrature points is at most
    1.2e-6 per channel (the terminator kink limits the quadrature); the bound
    is 2e-5.
    """
    spots = [{'theta': 1.1, 'phi': 0.5, 'radius': np.deg2rad(10), 'contrast': 0.7},
             {'theta': 2.0, 'phi': -2.0, 'radius': np.deg2rad(8), 'contrast': 0.6},
             {'theta': 0.4, 'phi': 2.5, 'radius': np.deg2rad(11), 'contrast': 0.8}]
    s = generate_spotted_surface(L, spots, lanczos=True)
    ks = [kernels_L30['x'], kernels_L30['y'], kernels_L30['phot']]
    t_sub = T_OBS[::27]

    for beta in (0.0, 0.3, 1.2, np.pi / 2):
        A = build_design_matrix(L, beta, OMEGA, T_OBS, ks)
        mu = forward_model(s, A).reshape(3, -1)[:, ::27]
        direct = _direct_signals(s, beta, t_sub, 400, 800)
        scale = np.max(np.abs(mu), axis=1)
        rel = np.max(np.abs(direct - mu), axis=1) / scale
        print(f"beta={beta:.4f}: rel err x={rel[0]:.2e} y={rel[1]:.2e} "
              f"phot={rel[2]:.2e} (bound 2e-5)")
        assert np.all(rel < 2e-5)


def test_render_matches_separable_direct_sum_L30_production_grid():
    """
    render() at (120, 240) and L = 30 against an independent separable
    evaluation, I = Re(sum_m g_m(theta) e^{i m phi}). The fast suite makes
    this comparison at L = 8 on a 20x40 grid. Differences are floating-point
    accumulation order only, measured below 1e-11.
    """
    spots = [{'theta': 0.9, 'phi': -1.0, 'radius': np.deg2rad(9), 'contrast': 0.6}]
    s = generate_spotted_surface(L, spots, lanczos=True)
    img = render(coeffs_to_real(s))

    theta = np.linspace(0, np.pi, 120)
    phi = np.linspace(-np.pi, np.pi, 240)
    g = np.zeros((2 * L + 1, 120), dtype=complex)
    for l, m in lm_indices(L):
        g[m + L] += s[l * l + l + m] * sph_harm_y(l, m, theta, 0.0)
    direct = np.real(g.T @ np.exp(1j * np.outer(np.arange(-L, L + 1), phi)))

    diff = np.max(np.abs(img - direct))
    print(f"render vs separable sum: {diff:.2e} (bound 1e-10)")
    assert diff < 1e-10


def test_ssim_pole_continuation_production_grid():
    """
    SSIM pole behaviour at (120, 240). At the pole row the filter window is
    physically symmetric under phi -> phi + pi, so rows 0 and -1 of the SSIM
    map must be invariant under a half-grid roll. Measured deviation 3.2e-12
    with the pole-correct continuation, against 6.3e-2 for a continuation
    that reflects without rolling, so this discriminates at O(1). The
    identity ssim_map(a, a) = 1 is exact by construction.
    """
    a = render(coeffs_to_real(generate_spotted_surface(L, [
        {'theta': 0.3, 'phi': 1.0, 'radius': np.deg2rad(11), 'contrast': 0.6},
        {'theta': 2.2, 'phi': -2.0, 'radius': np.deg2rad(9), 'contrast': 0.7},
    ], lanczos=True)))
    b = render(coeffs_to_real(generate_spotted_surface(L, [
        {'theta': 0.5, 'phi': 2.0, 'radius': np.deg2rad(10), 'contrast': 0.65},
    ], lanczos=True)))

    F = ssim_map(a, b)
    half = F.shape[1] // 2
    for r in (0, -1):
        dev = np.max(np.abs(F[r] - np.roll(F[r], half)))
        print(f"half-roll invariance, row {r}: {dev:.2e} (bound 1e-9)")
        assert dev < 1e-9

    ident = np.max(np.abs(ssim_map(a, a) - 1.0))
    print(f"ssim_map(a, a) identity deviation: {ident:.2e} (bound 1e-12)")
    assert ident < 1e-12
