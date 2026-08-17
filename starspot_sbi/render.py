"""
Rendering spherical-harmonic coefficient vectors onto a latitude-longitude grid.

    theta = linspace(0, pi, n_theta)        row 0 is theta = 0, the +z pole
    phi   = linspace(-pi, pi, n_phi)
    Y[:, l^2 + l + m] = sph_harm_y(l, m, THETA, PHI)

Both axes are endpoint-inclusive, so the poles and the phi = +-pi meridian are
each sampled and the grid is not area-uniform.  Area weighting is done via
metrics.py

Row 0 is the north rotational pole, latitude +90.  Under the package
inclination convention (docs/conventions.md section 2) the permanently
unobservable region is the north cap theta < beta, so a visibility mask excludes
the FIRST rows of a rendered image. 

The grid used in the paper is (n_theta, n_phi) = (120, 240).  Images are displayed 
with origin='upper' so that row 0 appears at the top.
"""

###########
# Imports #
###########

# standard
import numpy as np
from functools import lru_cache # memory efficient thingo

# special
from scipy.special import sph_harm_y

# self
from starspot_sbi.indexing import lm_indices, n_coeffs, real_to_coeffs

#############
# Constants #
#############

# grid size, affects metrics and rendering
N_THETA, N_PHI = 120, 240


###############
# Build Basis #
###############

def build_Ylm_matrix(L, n_theta=N_THETA, n_phi=N_PHI):
    """
    Complex spherical harmonics on the render grid, shape
    (n_theta * n_phi, (L+1)^2), columns in flat-index order.
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    Y = np.zeros((n_theta * n_phi, n_coeffs(L)), dtype=np.complex128)
    for l, m in lm_indices(L):
        Y[:, l**2 + l + m] = sph_harm_y(l, m, THETA.ravel(), PHI.ravel())
    return Y

@lru_cache(maxsize=4)
def get_Ylm(L, n_theta=N_THETA, n_phi=N_PHI):
    """
    Cached build_Ylm_matrix.  The returned array is read-only, since the cache
    shares it between callers. 
    """
    Y = build_Ylm_matrix(L, n_theta, n_phi)
    Y.setflags(write=False)
    return Y

#############
# Rendering #
#############

def render(real_vec, n_theta=N_THETA, n_phi=N_PHI, L=None):
    """
    Render a real-packed coefficient vector to an (n_theta, n_phi) image.

    The imaginary part is discarded.  It is zero to machine precision here by
    construction, because real_to_coeffs imposes the reality condition.
    """
    real_vec = np.asarray(real_vec)
    if L is None:
        L = int(round(np.sqrt(real_vec.shape[-1]))) - 1
    Y = get_Ylm(L, n_theta, n_phi)
    return (Y @ real_to_coeffs(real_vec)).real.reshape(n_theta, n_phi)

def render_coeffs(coeffs, n_theta=N_THETA, n_phi=N_PHI, L=None):
    """
    Render a complex coefficient vector, for callers holding coefficients
    rather than the real packing.  Equivalent to render(coeffs_to_real(c)) for
    any c obeying the reality condition.
    """
    coeffs = np.asarray(coeffs)
    if L is None:
        L = int(round(np.sqrt(coeffs.shape[-1]))) - 1
    Y = get_Ylm(L, n_theta, n_phi)
    return (Y @ coeffs).real.reshape(n_theta, n_phi)

def render_normed(normed_vec, mu_data, std_data, dc_value, include_dc=False,
                  n_theta=N_THETA, n_phi=N_PHI):
    """
    Un-standardise a coefficient vector and render it.

        raw = normed_vec * std_data + mu_data

    include_dc=True   normed_vec holds all (L+1)^2 entries and dc_value is
                      unused.  This is the setting the canonical checkpoint
                      n640000_seed101 was trained under.
    include_dc=False  normed_vec holds (L+1)^2 - 1 entries with the DC term
                      dropped, and dc_value supplies entry 0.

    mu_data and std_data must match normed_vec in length, so they too exclude
    the DC term when include_dc is False.
    """
    raw = np.asarray(normed_vec) * std_data + mu_data
    if include_dc:
        return render(raw, n_theta, n_phi)

    full = np.empty(raw.shape[-1] + 1, dtype=np.float64)
    full[0] = dc_value
    full[1:] = raw
    return render(full, n_theta, n_phi)

#####################
# Grid coordinates  #
#####################

def grid_coordinates(n_theta=N_THETA, n_phi=N_PHI):
    """
    Colatitude and longitude of the render grid, in degrees, as 1D arrays.

    Latitude is 90 - colatitude, so lat[0] = +90 at row 0.  Provided so that
    masks and latitude profiles do not each rebuild the grid and risk
    disagreeing with the renderer
    """
    colat = np.degrees(np.linspace(0, np.pi, n_theta))
    lon = np.degrees(np.linspace(-np.pi, np.pi, n_phi))
    return colat, lon