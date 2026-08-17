"""
Tests for surfaces.py.

Cap coefficients are checked against direct quadrature of the cap indicator,
against the sympy path (patch A2), and against the l = 0 closed form. Spot
placement is checked against an explicit Wigner rotation and, on a rendered
grid, against the injected position. The position test is the sharp guard on
patch A1: the old (-1)^m shifts every spot by pi in longitude, and the test
measures this directly.
"""

import numpy as np
import pytest
from scipy.special import sph_harm_y

from starspot_sbi.indexing import lm_to_idx, lm_indices, n_coeffs
from starspot_sbi.wigner import wigner_d_matrix_fast
from starspot_sbi.surfaces import (
    spherical_cap_coeffs,
    spherical_cap_coeffs_exact,
    place_spot,
    generate_spotted_surface,
)

L = 8
RHO = np.deg2rad(20)
DELTA = -0.4


##########################
# Cap coefficients (C1)  #
##########################

def test_cap_l0_closed_form():
    """s_0^0 = delta sqrt(pi) (1 - cos rho), from C1 with P_0 = 1."""
    s = spherical_cap_coeffs(L, RHO, DELTA)
    expected = DELTA * np.sqrt(np.pi) * (1 - np.cos(RHO))
    print(f"l = 0: {s[0]:.12f}  expected {expected:.12f}")
    assert abs(s[0] - expected) < 1e-14


def test_cap_against_direct_quadrature():
    '''
    C1 against 2D quadrature of the cap indicator function. The cap edge is a
    step function, so quadrature converges slowly and the error is the
    quadrature's, not the formula's. The formula is verified against sympy to
    1e-12 in test_fast_matches_exact; this test checks it computes the right
    integral.
    '''
    nu, nphi = 400, 800
    u, wu = np.polynomial.legendre.leggauss(nu)
    phi = (np.arange(nphi) + 0.5) * (2 * np.pi / nphi)
    TH, PH = np.meshgrid(np.arccos(u), phi, indexing='ij')
    dO = (wu[:, None] * np.ones((1, nphi)) * (2 * np.pi / nphi)).ravel()
    cap = np.where(TH.ravel() < RHO, DELTA, 0.0)

    fast = spherical_cap_coeffs(L, RHO, DELTA)
    quad = np.array([
        np.dot(cap * dO, np.conj(sph_harm_y(l, 0, TH, PH).ravel()))
        for l in range(L + 1)
    ])
    err = np.max(np.abs(quad.real - fast))
    print(f"max |cap_coeffs - quadrature| = {err:.2e}  (quadrature limited)")
    assert err < 1e-2


def test_cap_only_m0():
    """A polar cap is axisymmetric, so place_spot at the pole has only m = 0."""
    s = place_spot(L, RHO, DELTA, 0.0, 0.0)
    for l, m in lm_indices(L):
        if m != 0:
            assert abs(s[lm_to_idx(l, m)]) < 1e-14


##########################
# Fast vs exact (A2)     #
##########################

def test_fast_matches_exact():
    """Patch A2: the exact path divided by (2l+1) twice. Now they agree."""
    fast = spherical_cap_coeffs(L, RHO, DELTA)
    exact = spherical_cap_coeffs_exact(L, RHO, DELTA)
    err = np.max(np.abs(fast - exact))
    print(f"fast vs exact: {err:.2e}")
    assert err < 1e-12


@pytest.mark.parametrize('rho_deg', [5, 15, 30, 45])
def test_fast_exact_across_radii(rho_deg):
    rho = np.deg2rad(rho_deg)
    fast = spherical_cap_coeffs(L, rho, -0.3)
    exact = spherical_cap_coeffs_exact(L, rho, -0.3)
    assert np.max(np.abs(fast - exact)) < 1e-12


##########################
# Lanczos taper (C3)     #
##########################

def test_lanczos_taper_values():
    """C3: the taper is sinc(l / (L+1)), with np.sinc's normalised convention."""
    untapered = spherical_cap_coeffs(L, RHO, DELTA)
    tapered = spherical_cap_coeffs(L, RHO, DELTA, lanczos=True)
    expected_ratio = np.sinc(np.arange(L + 1) / (L + 1))
    ratio = tapered / untapered
    print(f"max |ratio - sinc(l/(L+1))| = {np.max(np.abs(ratio - expected_ratio)):.2e}")
    assert np.max(np.abs(ratio - expected_ratio)) < 1e-15


def test_lanczos_taper_at_l0_is_one():
    """sinc(0) = 1, so the taper does not change the monopole."""
    assert spherical_cap_coeffs(L, RHO, DELTA, lanczos=True)[0] == \
           spherical_cap_coeffs(L, RHO, DELTA, lanczos=False)[0]


##########################
# Placement (C4, C5)     #
##########################

def test_place_spot_against_wigner():
    """C4: place_spot equals the full Wigner rotation of the polar cap."""
    theta_s, phi_s = np.deg2rad(63), np.deg2rad(37)
    s_pkg = place_spot(L, RHO, DELTA, theta_s, phi_s)
    s_pol = spherical_cap_coeffs(L, RHO, DELTA)
    s_ref = np.zeros(n_coeffs(L), dtype=complex)
    for l in range(L + 1):
        ms = np.arange(-l, l + 1)
        s_ref[l**2:(l + 1)**2] = (np.exp(-1j * ms * phi_s)
                                   * wigner_d_matrix_fast(l, theta_s)[:, l]
                                   * s_pol[l])
    err = np.max(np.abs(s_pkg - s_ref))
    print(f"|place_spot - Wigner rotation| = {err:.2e}")
    assert err < 1e-13


def test_place_spot_obeys_reality():
    """S4: placed spot satisfies s_l^{-m} = (-1)^m conj(s_l^m)."""
    s = place_spot(L, RHO, DELTA, np.deg2rad(60), np.deg2rad(45))
    viol = max(abs(s[lm_to_idx(l, -m)] - (-1)**m * np.conj(s[lm_to_idx(l, m)]))
               for l, m in lm_indices(L) if m > 0)
    print(f"max reality violation = {viol:.2e}")
    assert viol < 1e-13


def test_place_spot_recovers_position():
    """
    The spot renders at (theta_s, phi_s), not at (theta_s, phi_s - pi). This is
    the sharp test for patch A1. Grid spacing sets the error floor; 1 deg steps
    give a ~0.5 deg bound.
    """
    LL = 20
    nth, nph = 180, 360
    th = np.linspace(0.5 * np.pi / nth, np.pi - 0.5 * np.pi / nth, nth)
    ph = np.linspace(-np.pi, np.pi, nph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    Ycache = {(l, m): sph_harm_y(l, m, TH, PH) for l, m in lm_indices(LL)}

    positions = [(np.deg2rad(t), np.deg2rad(p))
                 for t in [30, 60, 90, 120, 150]
                 for p in [-150, -60, 0, 45, 135]]

    worst_th, worst_ph = 0.0, 0.0
    for theta_s, phi_s in positions:
        s = place_spot(LL, np.deg2rad(12), -0.5, theta_s, phi_s)
        img = np.real(sum(s[lm_to_idx(l, m)] * Ycache[(l, m)]
                          for l, m in lm_indices(LL)))
        i, j = np.unravel_index(np.argmin(img), img.shape)
        worst_th = max(worst_th, abs(th[i] - theta_s))
        dphi = (ph[j] - phi_s + np.pi) % (2 * np.pi) - np.pi
        worst_ph = max(worst_ph, abs(dphi))

    print(f"25 positions: worst colatitude error {np.rad2deg(worst_th):.2f} deg, "
          f"worst longitude error {np.rad2deg(worst_ph):.2f} deg  "
          f"(grid spacing {180/nth:.1f} / {360/nph:.1f} deg)")
    assert np.rad2deg(worst_th) < 1.5
    assert np.rad2deg(worst_ph) < 1.5


def test_old_sign_fails_by_pi():
    """
    The source's (-1)^m in place_spot is e^{-im pi}, a longitude shift of pi.
    Applying it to the corrected output and rendering recovers a spot at
    phi_s - pi rather than phi_s.
    """
    LL = 20
    theta_s, phi_s = np.deg2rad(60), np.deg2rad(45)
    s = place_spot(LL, np.deg2rad(12), -0.5, theta_s, phi_s)

    # apply the old (-1)^m
    s_old = s.copy()
    for l, m in lm_indices(LL):
        s_old[lm_to_idx(l, m)] *= (-1) ** m

    nth, nph = 90, 180
    th = np.linspace(0.5 * np.pi / nth, np.pi - 0.5 * np.pi / nth, nth)
    ph = np.linspace(-np.pi, np.pi, nph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    img = np.real(sum(s_old[lm_to_idx(l, m)] * sph_harm_y(l, m, TH, PH)
                      for l, m in lm_indices(LL)))
    i, j = np.unravel_index(np.argmin(img), img.shape)
    dphi = (ph[j] - phi_s + np.pi) % (2 * np.pi) - np.pi
    print(f"old (-1)^m: rendered longitude error = {np.rad2deg(dphi):+.1f} deg  (expect ~180)")
    assert abs(abs(np.rad2deg(dphi)) - 180) < 5


##########################
# d_{m0} identity (C5)   #
##########################

@pytest.mark.parametrize('l', [1, 3, 6, 10])
def test_d_m0_identity(l):
    """C5: d^l_{m0}(beta) = sqrt(4 pi / (2l+1)) Y_l^m(beta, 0), no (-1)^m."""
    beta = 1.1
    d_col = wigner_d_matrix_fast(l, beta)[:, l]
    ident = np.array([np.sqrt(4 * np.pi / (2 * l + 1))
                      * sph_harm_y(l, m, beta, 0.0).real
                      for m in range(-l, l + 1)])
    err = np.max(np.abs(d_col - ident))
    print(f"l = {l}: |d_m0 - identity| = {err:.2e}")
    assert err < 5e-12

    bad = ident * np.array([(-1.0) ** m for m in range(-l, l + 1)])
    assert np.max(np.abs(d_col - bad)) > 0.01

##########################
# Background (C6)        #
##########################

def test_background():
    """C6: s_0^0 = 2 sqrt(pi) I_bg for a surface with no spots."""
    for I_bg in [0.5, 1.0, 1.7]:
        s = generate_spotted_surface(L, [], I_background=I_bg)
        assert abs(s[0] - np.sqrt(4 * np.pi) * I_bg) < 1e-14
        assert np.max(np.abs(s[1:])) < 1e-14


##########################
# Superposition (C6)     #
##########################

def test_superposition():
    """C6: generate_spotted_surface equals background plus individual place_spot calls."""
    spots = [
        {'theta': 1.0, 'phi': 0.4, 'radius': np.deg2rad(10), 'contrast': 0.6},
        {'theta': 2.0, 'phi': -1.2, 'radius': np.deg2rad(7), 'contrast': 0.8},
    ]
    I_bg = 1.0
    tot = generate_spotted_surface(L, spots, I_background=I_bg)

    manual = np.zeros(n_coeffs(L), dtype=complex)
    manual[0] = np.sqrt(4 * np.pi) * I_bg
    for sp in spots:
        manual += place_spot(L, sp['radius'], sp['contrast'] - I_bg,
                             sp['theta'], sp['phi'])

    err = np.max(np.abs(tot - manual))
    print(f"|generate_spotted_surface - manual sum| = {err:.2e}")
    assert err < 1e-14


def test_linearity_in_delta():
    """Two calls with delta and 2*delta differ by exactly 2x."""
    s1 = place_spot(L, RHO, -0.3, 1.0, 0.5)
    s2 = place_spot(L, RHO, -0.6, 1.0, 0.5)
    assert np.max(np.abs(s2 - 2 * s1)) < 1e-14