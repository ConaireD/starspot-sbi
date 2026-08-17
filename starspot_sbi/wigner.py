"""
Wigner small-d and D matrices for rotating spherical-harmonic coefficients.

Convention (z-y-z active):

    D^l_{m,m'}(alpha, beta, gamma) = e^{-i m alpha} d^l_{m,m'}(beta) e^{-i m' gamma}

"""

###########
# Imports #
###########

# standard
import numpy as np
from tqdm.auto import tqdm

# special
from scipy.special import gammaln
from sympy.physics.wigner import wigner_d_small

#######################
# Full D from small d #
#######################

def wigner_D_from_d(l, alpha, gamma, d):
    """Full Wigner D-matrix from the small-d matrix and the two z rotations."""
    ms = np.arange(-l, l + 1)
    return np.diag(np.exp(-1j*ms*alpha)) @ d @ np.diag(np.exp(-1j*ms*gamma))

#########
# Exact #
#########

def wigner_d_matrix_exact(l, beta):
    """Exact d^l(beta) via sympy.  Returns complex; use for verification only."""
    return np.array(wigner_d_small(l, beta).tolist(), dtype=complex)


def precompute_wigner_d_exact(L_max, beta):
    """Exact d^l(beta) for l = 0..L_max, as a list indexed by l."""
    return [wigner_d_matrix_exact(l, beta)
            for l in tqdm(range(L_max + 1), desc='Wigner-d (exact)')]

########
# Fast #
########

# TODO: tie mathematical equations to specific funtions
# TODO: lint in my preferences

def wigner_d_element_fast(l, m, mp, beta):
    """
    Single element d^l_{m,m'}(beta), summed in log-factorial form.

    The logs are guarded by +1e-300, so at beta = 0 the off-diagonal entries
    come back as ~1e-300 rather than exact zero.
    """
    s_min = int(max(0, m - mp))
    s_max = int(min(l + m, l - mp))

    if s_min > s_max:
        return 0.0

    s = np.arange(s_min, s_max + 1)
    cos_half = np.cos(beta / 2)
    sin_half = np.sin(beta / 2)

    log_prefactor = 0.5*(
        gammaln(l + m + 1) + gammaln(l - m + 1)
        + gammaln(l + mp + 1) + gammaln(l - mp + 1)
    )

    log_den = (
        gammaln(l + m - s + 1) + gammaln(s + 1)
        + gammaln(mp - m + s + 1) + gammaln(l - mp - s + 1)
    )

    power_cos = 2*l + m - mp - 2*s
    power_sin = mp - m + 2*s

    log_trig = (
        power_cos * np.log(np.abs(cos_half) + 1e-300)
        + power_sin * np.log(np.abs(sin_half) + 1e-300)
    )

    sign = (-1.0)**s
    if cos_half < 0:
        sign *= (-1.0)**power_cos
    if sin_half < 0:
        sign *= (-1.0)**power_sin

    return np.sum(sign * np.exp(log_prefactor - log_den + log_trig))


def wigner_d_matrix_fast(l, beta):
    """Real (2l+1) x (2l+1) matrix d^l(beta), rows and columns m = -l..+l."""
    size = 2 * l + 1
    d = np.zeros((size, size))
    for i, m in enumerate(range(-l, l + 1)):
        for j, mp in enumerate(range(-l, l + 1)):
            d[i, j] = wigner_d_element_fast(l, m, mp, beta)
    return d


def precompute_wigner_d_fast(L_max, beta):
    """Fast d^l(beta) for l = 0..L_max, as a list indexed by l."""
    return [wigner_d_matrix_fast(l, beta)
            for l in tqdm(range(L_max + 1), desc='Wigner-d (fast)')]