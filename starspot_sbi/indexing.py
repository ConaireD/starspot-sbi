"""
Indexing and packing for spherical-harmonic coefficient vectors.

Canonical ordering: a degree-L map is a length (L+1)^2 vector indexed by
lm_to_idx(l, m), l ascending and, within each l, m from -l to +l.

Two representations:
  - complex: c[lm_to_idx(l, m)] = s_l^m, with s_l^{-m} = (-1)^m conj(s_l^m)
  - real:    blocks [s_l^0 | Re s_l^{m>0} | Im s_l^{m>0}], same length

The m<0 half is redundant for a real map, so coeffs_to_real drops it and
real_to_coeffs regenerates it. L is inferred from array length, never global.
"""

###########
# Imports #
###########

import numpy as np
from functools import lru_cache

#############
# Functions #
#############

def lm_to_idx(l, m):
    """Flat index for coefficient (l, m)."""
    return l**2 + l + m


def idx_to_lm(idx):
    """Inverse of lm_to_idx."""
    l = int(np.floor(np.sqrt(idx)))
    return l, idx - l**2 - l


def lm_indices(L):
    """List of (l, m) pairs in flat-index order."""
    return [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]


def n_coeffs(L):
    """Number of SH coefficients up to degree L: (L+1)^2."""
    return (L + 1) ** 2


def _L_from_len(n):
    """Degree L implied by a vector of length n = (L+1)^2."""
    L = int(round(np.sqrt(n))) - 1
    if (L + 1) ** 2 != n:
        raise ValueError(f"length {n} is not (L+1)^2 for integer L")
    return L


@lru_cache(maxsize=8)
def _build_pack_indices(L):
    """
    Flat indices used by the complex <-> real packing at degree L.

    Returns (m0_idx, pos_idx, neg_idx, sign): the m=0 indices, the m>0 indices,
    their m<0 conjugate partners aligned elementwise, and the (-1)^m factor
    relating the two. Cached on L; returned arrays are read-only because the
    cache shares them between callers.
    """
    m0_idx, pos_idx, neg_idx, sign = [], [], [], []
    for l in range(L + 1):
        m0_idx.append(l*l + l)                # m=0 is the centre of the l-block
        for m in range(1, l + 1):
            pos_idx.append(l*l + l + m)
            neg_idx.append(l*l + l - m)
            sign.append((-1)**m)

    out = (np.array(m0_idx, dtype=np.int32),
           np.array(pos_idx, dtype=np.int32),
           np.array(neg_idx, dtype=np.int32),
           np.array(sign, dtype=np.int8))
    for a in out:
        a.setflags(write=False)
    return out


def coeffs_to_real(coeffs):
    """
    Pack a complex coefficient vector (n,) into its real form (n,).

    Keeps the m=0 entries and the real and imaginary parts of the m>0 half;
    the m<0 entries are dropped. Raises ValueError if n is not (L+1)^2.
    """
    n = coeffs.shape[0]
    m0_idx, pos_idx, _, _ = _build_pack_indices(_L_from_len(n))
    n_m0, n_pos = m0_idx.size, pos_idx.size

    out = np.empty(n, dtype=np.float64)
    out[:n_m0]             = coeffs[m0_idx].real
    out[n_m0:n_m0 + n_pos] = coeffs[pos_idx].real
    out[n_m0 + n_pos:]     = coeffs[pos_idx].imag
    return out


def real_to_coeffs(vec):
    """
    Unpack a real vector (n,) back to complex coefficients (n,).

    Inverse of coeffs_to_real. The m<0 half is rebuilt from the reality
    condition, so the result always describes a real-valued map.
    """
    n = vec.shape[0]
    m0_idx, pos_idx, neg_idx, neg_sign = _build_pack_indices(_L_from_len(n))
    n_m0, n_pos = m0_idx.size, pos_idx.size

    coeffs = np.zeros(n, dtype=np.complex128)
    coeffs[m0_idx] = vec[:n_m0]

    re = vec[n_m0:n_m0 + n_pos]
    im = vec[n_m0 + n_pos:]
    coeffs[pos_idx] = re + 1j*im
    coeffs[neg_idx] = neg_sign * (re - 1j*im)   # (-1)^m conj(s_l^m)
    return coeffs