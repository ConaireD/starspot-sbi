"""
Kernels for the photometric and astrometric forward model.

Each observable is a linear* functional of the surface intensity, so its
spherical-harmonic kernel factorises into a phi integral and a theta integral:

    k^h_{l,m} = (1/pi) * I_phi^h(m) * I_u^h(l,m)

with geometric weights (observer along +x, u = cos(theta)):

    photometry   V = (1/pi) sin(theta) cos(phi)
    x astrometry x_obs * V,  x_obs = sin(theta) sin(phi)
    y astrometry y_obs * V,  y_obs = cos(theta)

The 1/pi normalises a uniform unit-intensity map to unit flux, so F_0 = 1 and
sigma_phot is a fractional flux precision.

Every function has an _exact (sympy) and a _fast (closed form / Gauss-Legendre)
implementation. The _exact ones are for verification, as computation for higher L
takes ages.

Derivations: docs/conventions.md and notebooks/01_kernels.ipynb.
Conventions follow indexing.py: orthonormal complex SH with Condon-Shortley.

============================================================================
RE: linear* see the linearity.md document for details. 
"""

###########
# Imports #
###########

# standard
import numpy as np

# symbolic /science
import sympy 
from sympy import Symbol, sin, cos, sqrt, pi, factorial, Abs, N, integrate, assoc_legendre
from scipy.special import lpmv, sph_harm_y

# helpers
from tqdm.auto import tqdm
from starspot_sbi.indexing import lm_indices


#################
# Phi integrals #
#################

# Exact
_phi_sym = Symbol('phi')

def I_phi_x_exact(m_val):
    """Exact phi-integral for x-kernel via sympy integration."""
    expr = sin(_phi_sym) * cos(_phi_sym) * sympy.exp(sympy.I * m_val * _phi_sym)
    return complex(N(integrate(expr, (_phi_sym, -pi/2, pi/2))))

def I_phi_y_exact(m_val):
    """Exact phi-integral for y-kernel via sympy integration."""
    expr = cos(_phi_sym) * sympy.exp(sympy.I * m_val * _phi_sym)
    return complex(N(integrate(expr, (_phi_sym, -pi/2, pi/2))))

def I_phi_p_exact(m_val):
    """Exact phi-integral for photometry-kernel (same as y-kernel)."""
    return I_phi_y_exact(m_val)

# Fast
def I_phi_x_fast(m_val):
    """Closed-form phi-integral for the x-kernel."""
    if m_val == 0:
        return 0.0j
    if abs(m_val) == 2:
        return 1j * np.pi / 4 * np.sign(m_val)
    if m_val % 2 != 0:
        return -2j * np.sin(np.pi * m_val / 2) / (m_val**2 - 4)
    return 0.0j

def I_phi_y_fast(m_val):
    """Closed-form phi-integral for the y-kernel."""
    if abs(m_val) == 1:
        return np.pi / 2
    if m_val % 2 == 0:
        return -2 * np.cos(np.pi * m_val / 2) / (m_val**2 - 1)
    return 0.0

def I_phi_p_fast(m_val):
    """Closed-form phi-integral for photometry-kernel (same as y-kernel)."""
    return I_phi_y_fast(m_val)

###################
# Theta integrals #
###################

# Exact
_u_sym = Symbol('u')

def I_u_x_exact(l_val, m_val):
    """Exact theta-integral for x-kernel: int_{-1}^{1} (1-u^2) P_l^|m|(u) du."""
    P = assoc_legendre(l_val, Abs(m_val), _u_sym)
    return complex(N(integrate((1 - _u_sym**2) * P, (_u_sym, -1, 1))))

def I_u_y_exact(l_val, m_val):
    """Exact theta-integral for y-kernel: int_{-1}^{1} u sqrt(1-u^2) P_l^|m|(u) du."""
    P = assoc_legendre(l_val, Abs(m_val), _u_sym)
    return complex(N(integrate(_u_sym * sqrt(1 - _u_sym**2) * P, (_u_sym, -1, 1))))

def I_u_phot_exact(l_val, m_val):
    """Exact theta-integral for photometry-kernel: int_{-1}^{1} sqrt(1-u^2) P_l^|m|(u) du."""
    P = assoc_legendre(l_val, Abs(m_val), _u_sym)
    return complex(N(integrate(sqrt(1 - _u_sym**2) * P, (_u_sym, -1, 1))))



# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature for the theta integrals.
#
# _GL_N is a pinned constant: 500 is what generated the published
# dataset, and changing it changes the stored signals slightly.
# It is one of the two module-level constants the package allows 
# (see docs/conventions.md).
#
# Gauss-Legendre with N nodes is exact for polynomials of degree <= 2N-1.
# The x-channel integrand (1-u^2) P_l^|m|(u) is a polynomial, so kx is exact to roundoff.
# The y and photometric integrands have sqrt(1-u^2), which has a half-integer power
# at u = +-1 i.e. not a polynomial and not analytic at the endpoints, so the quadrature
# converges algebraically as N^-3. 
# More nodes doesn't necessarilly help the analytically-zero entries grow from ~1e-16
# at N=500 to ~1e-15 at N=2000 because the dot product has four times as many terms
# to cancel.
#
# Measured, this package, L = 4, uniform unit-intensity map:
#
#     N = 500    F_0 - 1 = 4.182e-9
#     N = 2000   F_0 - 1 = 6.538e-11      ratio 63.97 against the predicted 4^3 = 64
#
# F_0 - 1 = 4.2e-9 is the accuracy floor of the whole forward model at the
# production setting, five orders below the mission photometric noise 1e-4.
# Every downstream tolerance in the test suite descends from this number, so
# tests that guard it are written relative to the signal scale, not absolute.
#
# If a future change needs more accuracy, raising N is the may be wrong. Perhaps
# substitute back to theta (the sqrt becomes sin theta and the integrand is
# smooth on [0, pi]) or use a rule built for endpoint singularities, e.g.
# tanh-sinh?  Either would also require you to regenerate the dataset.
# ---------------------------------------------------------------------------

# Fast
_GL_N = 500   # 2000 gives ~64x better ky/kphot; 500 reproduces the published dataset
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_N)
_GL_THETA = np.arccos(_GL_NODES)

def I_u_x_fast(l, m):
    """int_{-1}^{1} (1-u^2) P_l^|m|(u) du  via Gauss-Legendre."""
    u, w = _GL_NODES, _GL_WEIGHTS
    return np.dot(w, (1 - u**2) * lpmv(abs(m), l, u))

def I_u_y_fast(l, m):
    """int_{-1}^{1} u sqrt(1-u^2) P_l^|m|(u) du  via Gauss-Legendre."""
    u, w = _GL_NODES, _GL_WEIGHTS
    return np.dot(w, u * np.sqrt(1 - u**2) * lpmv(abs(m), l, u))

def I_u_phot_fast(l, m):
    """int_{-1}^{1} sqrt(1-u^2) P_l^|m|(u) du  via Gauss-Legendre."""
    u, w = _GL_NODES, _GL_WEIGHTS
    return np.dot(w, np.sqrt(1 - u**2) * lpmv(abs(m), l, u))

######################
# Kernel computation #
######################

def _N_lm_sympy(l, m):
    """Sympy normalisation factor for spherical harmonics."""
    return sqrt((2*l + 1) / (4 * pi) * factorial(l - abs(m)) / factorial(l + abs(m)))

def _Ylm_at_phi0(l, m):
    """Y_l^m(theta, 0) at GL nodes.  Numerically stable for all l."""
    return sph_harm_y(l, m, _GL_THETA, np.zeros(_GL_N))

def precompute_kernels_exact(L_max):
    """
    Exact kernels via sympy integration.
    Slow but arbitrary precision. Practical for L <= ~15.
    """
    u_s = Symbol('u')
    indices = lm_indices(L_max)
    n = len(indices)

    kx = np.zeros(n, dtype=complex)
    ky = np.zeros(n, dtype=complex)
    kphot = np.zeros(n, dtype=complex)

    for idx, (l, m) in tqdm(enumerate(indices), total=n, desc='Kernels (exact)'):
        Nm = _N_lm_sympy(l, m)
        ipx = I_phi_x_exact(m)
        ipy = I_phi_y_exact(m)

        P = assoc_legendre(l, Abs(m), u_s)
        cs_phase = (-1)**abs(m) if m < 0 else 1

        if abs(ipx) > 1e-15:
            iux = complex(N(integrate((1 - u_s**2) * P, (u_s, -1, 1))))
            kx[idx] = (Nm/pi) * ipx * iux * cs_phase
        if abs(ipy) > 1e-15:
            iuy = complex(N(integrate(u_s * sqrt(1 - u_s**2) * P, (u_s, -1, 1))))
            iup = complex(N(integrate(sqrt(1 - u_s**2) * P, (u_s, -1, 1))))
            ky[idx]    = (Nm / pi) * ipy * iuy * cs_phase
            kphot[idx] = (Nm / pi) * ipy * iup * cs_phase

    return kx, ky, kphot

def precompute_kernels_fast(L):
    """
    Fast kernels: closed-form I_phi + GL quadrature with sph_harm_y.
    Stable to l ~ 100+.

    Accuracy at _GL_N = 500, relative, measured against the sympy path at L=8:
        kx     2.7e-11    (polynomial integrand, exact to roundoff)
        ky     4.3e-07    (integrand carries sqrt(1-u^2))
        kphot  8.6e-07
    These are maxima over (l, m), dominated by high l. At l = 0, where the
    normalisation is set, the photometric error is 4.2e-9. See the note above
    _GL_N for the N^-3 convergence and for why 500 is pinned.
    
    Error falls as N^-3. _GL_N = 2000 gives 6.7e-09 and 1.3e-08. The default
    stays at 500 because it reproduces the published dataset bitwise; both are
    far below the 1e-4 mission noise floor.
    """
    u, w = _GL_NODES, _GL_WEIGHTS
    wf_x    = w * (1 - u**2)
    wf_y    = w * u * np.sqrt(1 - u**2)
    wf_phot = w * np.sqrt(1 - u**2)

    indices = lm_indices(L)
    n = len(indices)
    kx    = np.zeros(n, dtype=complex)
    ky    = np.zeros(n, dtype=complex)
    kphot = np.zeros(n, dtype=complex)

    for idx, (l, m) in tqdm(enumerate(indices), total=n, desc='Kernels (fast)'):
        ipx = I_phi_x_fast(m)
        ipy = I_phi_y_fast(m)
        if abs(ipx) > 1e-15 or abs(ipy) > 1e-15:
            Ylm = _Ylm_at_phi0(l, m)
        if abs(ipx) > 1e-15:
            kx[idx] = (1/np.pi) * ipx * np.dot(wf_x, Ylm)
        if abs(ipy) > 1e-15:
            ky[idx]    = (1/np.pi) * ipy * np.dot(wf_y, Ylm)
            kphot[idx] = (1/np.pi) * ipy * np.dot(wf_phot, Ylm)

    return kx, ky, kphot
