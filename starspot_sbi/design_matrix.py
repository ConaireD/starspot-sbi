"""
Design matrix for the photometric and astrometric forward model
"""

###########
# Imports #
###########

# standard
import numpy as np

# self
from starspot_sbi.indexing import n_coeffs
from starspot_sbi.wigner   import wigner_d_matrix_fast

#############
# Functions #
#############

def build_W(L, omega, t_obs):
    """Time matrix W_omega.  Shape: (N_obs, 2L+1)."""
    m_vals = np.arange(-L, L + 1)
    return np.exp(-1j * np.outer(omega * t_obs, m_vals))


def build_B(L, beta, k_h):
    """Kernel/inclination matrix B^h_beta.  Shape: (2L+1, (L+1)^2)."""
    n_rows = 2 * L + 1
    n_cols = n_coeffs(L)
    B = np.zeros((n_rows, n_cols), dtype=complex)

    for l in range(L + 1):
        idx_start = l**2
        idx_end = (l + 1)**2

        k_l = k_h[idx_start:idx_end]
        C_l = wigner_d_matrix_fast(l, beta)

        mixed = C_l @ k_l
        for i, m in enumerate(range(-l, l + 1)):
            B[m + L, idx_start + i] = mixed[i]

    return B

def build_design_matrix(L, beta, omega, t_obs, kernels):
    """
    Design matrix A(beta) = vstack(W @ B_h for each kernel h).

    kernels : array or list of arrays
        Single array (e.g. photometry) or list (e.g. [k_x, k_y]).
    Returns shape (K*N_obs, (L+1)^2) where K = number of kernels.
    """
    W = build_W(L, omega, t_obs)
    if not isinstance(kernels, list):
        kernels = [kernels]
    return np.vstack([W @ build_B(L, beta, k) for k in kernels])

def forward_model(s, A):
    """Signal mu = Re(A @ s)."""
    return np.real(A @ s)
