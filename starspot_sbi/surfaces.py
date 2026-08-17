"""
Spot surfaces: spherical caps in the spherical-harmonic basis.

A circular spot of angular radius rho and contrast delta is a spherical cap.
Centred on +z it is axisymmetric, so only the m = 0 coefficients are non-zero
(C1); it is moved to (theta_s, phi_s) by the single-column Wigner rotation
D^l_{m0}(phi_s, theta_s, 0) = e^{-i m phi_s} d^l_{m0}(theta_s) (C4),with
d^l_{m0}(theta) = sqrt(4 pi / (2l+1)) Y_l^m(theta, 0) (C5). Spots add in
coefficient space on top of a uniform background s_0^0 = 2 sqrt(pi) I (C6).
"""

###########
# Imports #
###########
# standard
import numpy as np

# symbolic
import sympy
from sympy import Symbol, Rational, N, pi, sqrt, legendre

# special
from scipy.special import eval_legendre, lpmv, gammaln

# self
from starspot_sbi.indexing import n_coeffs


###################
# Polar cap (m=0) #
###################

def spherical_cap_coeffs(L, rho, delta, lanczos=False):
    """
    m = 0 coefficients of a cap of angular radius rho at +z with contrast
    delta (C1), using the Legendre identity (C2). lanczos multiplies by
    sinc(l / (L+1)) (C3). Background not included. Returns shape (L+1,).
    """
    x = np.cos(rho)
    s = np.zeros(L + 1)

    for l in range(L + 1):
        if l == 0:
            integral = 1 - x
        else:
            integral = (eval_legendre(l - 1, x) - eval_legendre(l + 1, x)) / (2 * l + 1)
        s[l] = delta * 2 * np.pi * np.sqrt((2 * l + 1) / (4 * np.pi)) * integral

    if lanczos:
        s *= np.sinc(np.arange(L + 1) / (L + 1))

    return s


def spherical_cap_coeffs_exact(L, rho, delta, lanczos=False):
    """
    Same as spherical_cap_coeffs, evaluated in sympy and cast to float.
    Patch A2: a single division by (2l+1); the source divided twice.
    """
    x = sympy.cos(rho)
    s = np.zeros(L + 1)
    _u = Symbol('u')

    for l in range(L + 1):
        if l == 0:
            integral_val = 1 - x
        else:
            p_lm1 = legendre(l - 1, _u).subs(_u, x)
            p_lp1 = legendre(l + 1, _u).subs(_u, x)
            integral_val = (p_lm1 - p_lp1) / (2 * l + 1)
        coeff = Rational(2, 1) * pi * sqrt(Rational(2 * l + 1, 1) / (4 * pi))
        s[l] = float(N(delta * coeff * integral_val))

    if lanczos:
        s *= np.sinc(np.arange(L + 1) / (L + 1))

    return s


#############
# Placement #
#############

def place_spot(L, rho, delta, theta_s, phi_s, lanczos=False):
    """
    Cap of radius rho, contrast delta, centred at colatitude theta_s, longitude
    phi_s (C4). Uses d^l_{m0}(theta) from C5, evaluated as
    sqrt((l-|m|)!/(l+|m|)!) P_l^{|m|}(cos theta) times (-1)^{|m|} for m < 0
    only (S6). Patch A1: the source applied (-1)^m to every m.

    Returns the complex coefficient vector, length (L+1)^2, obeying S4.
    """
    s_polar   = spherical_cap_coeffs(L, rho, delta, lanczos=lanczos)
    s         = np.zeros(n_coeffs(L), dtype=complex)
    cos_theta = np.cos(theta_s)

    for l in range(L + 1):
        if np.abs(s_polar[l]) < 1e-15:
            continue

        m_vals   = np.arange(-l, l + 1)
        abs_m    = np.abs(m_vals)
        Plm      = lpmv(abs_m, l, cos_theta)
        log_norm = 0.5 * (gammaln(l - abs_m + 1) - gammaln(l + abs_m + 1))
        sign     = np.where(m_vals < 0, (-1.0) ** abs_m, 1.0)      # A1: was m_vals >= 0
        d_col    = sign * np.exp(log_norm) * Plm
        phase    = np.exp(-1j * m_vals * phi_s)

        idx = l * l
        s[idx:idx + 2 * l + 1] = phase * d_col * s_polar[l]

    return s


def generate_spotted_surface(L, spots, I_background=1, lanczos=False):
    """
    Background s_0^0 = 2 sqrt(pi) I_background (C6) plus a place_spot per entry.

    spots : list of dicts with keys 'theta', 'phi', 'radius' (radians) and
    'contrast' (spot intensity; delta = contrast - I_background).

    The returned coefficient vector is not guaranteed positive everywhere on the
    sphere. Overlapping spots add their deficits, and can drive the intensity below 
    zero. 
    """
    s = np.zeros(n_coeffs(L), dtype=complex)
    s[0] = np.sqrt(4 * np.pi) * I_background

    for spot in spots:
        delta = spot['contrast'] - I_background
        s += place_spot(L, spot['radius'], delta, spot['theta'], spot['phi'], lanczos=lanczos)

    return s