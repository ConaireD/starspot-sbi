"""
Forward-operator construction and spectral primitives.

The real operator Re(A(beta) T) acts on real-packed coefficient vectors. T is
orthogonal but not normalised (T^H T is 1 on m = 0 columns and 2 elsewhere),
so norms and projectors here use the real-packed metric.
"""

import numpy as np

from starspot_sbi.indexing import idx_to_lm, n_coeffs, real_to_coeffs

# Graded-projector regularisation, absolute, not relative to sv[0]^2. Matches
# NewMetrics/crap/02_InformationContent_S.ipynb cell 3 (LAM = 1e-9), which is
# the implementation that produced Paper 2's numbers. The constant's provenance
# is unexplained there (OPERATOR_MACHINERY.md part 1B); it is kept for
# comparability.
LAM = 1e-9

# Channel order used everywhere below, including the rows of the spectrum
# files. The cache itself stores (astro_x, astro_y, phot).
CHANNELS = ('x', 'y', 'astro', 'phot', 'combined')

# Singular values below this fraction of sv[0] are denormal junk from the SVD,
# not part of the spectrum. The true-zero cluster of a well-conditioned
# operator sits at ~1e-16 relative (rank_scan.py, L = 4-8)
JUNK_FLOOR = 1e-20

# A gap this wide in decades is treated as trustworthy. Measured on the
# production cache at L = 30: the gap is 5.0 decades at
# beta = 45 deg where the rank is right, and 3.2 decades at beta = 60 deg where
# the numerically non-zero x m = 0 row is miscounted. 4 sits between.
GAP_TRUST_DECADES = 4.0

######################
# Basis and indexing #
######################

def build_T(l_max):
    """
    Real-to-complex transform T, so that Re(A T) acts on section-packed real
    vectors. Column j is real_to_coeffs of the j-th unit vector.
    """
    n = n_coeffs(l_max)
    T = np.zeros((n, n), dtype=complex)
    e = np.zeros(n)
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        T[:, j] = real_to_coeffs(e)
    return T


def degree_map(l_max):
    """
    Degree of each entry of a section-packed real vector, derived by pushing
    unit vectors through real_to_coeffs rather than assuming the packing.

    floor(sqrt(arange(n))) is the degree map for the complex l^2 + l + m index
    and is wrong for this basis: at L = 30 the two agree on 45 of 961 entries,
    and both give 2l + 1 entries per degree, so a count check does not catch it
    (OPERATOR_MACHINERY.md part 1C).
    """
    n = n_coeffs(l_max)
    ell = np.empty(n, dtype=int)
    e = np.zeros(n)
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        nz = np.flatnonzero(np.abs(real_to_coeffs(e)) > 0)
        degrees = {idx_to_lm(int(i))[0] for i in nz}
        if len(degrees) != 1:
            raise ValueError(f'real index {j} maps to degrees {degrees}')
        ell[j] = degrees.pop()
    return ell


def channel_operators(A_beta, T):
    """
    Real operators per channel from one cache slice of shape
    (3, n_obs, n_coeffs), cache channel order (astro_x, astro_y, phot).
    """
    Gx = np.real(np.asarray(A_beta[0]) @ T)
    Gy = np.real(np.asarray(A_beta[1]) @ T)
    Gp = np.real(np.asarray(A_beta[2]) @ T)
    return {'x': Gx, 'y': Gy, 'astro': np.vstack([Gx, Gy]),
            'phot': Gp, 'combined': np.vstack([Gx, Gy, Gp])}

# Rows of each family's operator, in channel_operators' keys. The null-space
# statistics were computed with this order; the order does not change the null
# space.
FAMILY_ROWS = {'phot': ('phot',), 'phot_ax': ('x', 'phot'),
               'phot_ay': ('y', 'phot'), 'phot_axay': ('x', 'y', 'phot')}


def family_operator(A_beta, T, family):
    """Stacked real operator of a signal family from one cache slice."""
    if family not in FAMILY_ROWS:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(FAMILY_ROWS)}")
    ops = channel_operators(A_beta, T)
    return np.vstack([ops[c] for c in FAMILY_ROWS[family]])

###################
# Rank            #
###################

def rank_by_gap(sv):
    """
    Rank at the largest gap in the log10 spectrum.

    Returns (rank, sv_above, sv_below, gap_decades). The gap is 13 to 15
    decades wide wherever the operator is well conditioned (rank_scan.py,
    L = 4-8) and narrows smoothly toward pole-on; judge it by gap_decades
    against GAP_TRUST_DECADES rather than trusting the rank unconditionally.
    """
    sv = np.asarray(sv, dtype=float)
    sv = sv[sv > JUNK_FLOOR * sv[0]]
    if sv.size < 2:
        return sv.size, float(sv[0]) if sv.size else np.nan, np.nan, np.inf
    logs = np.log10(sv)
    gaps = logs[:-1] - logs[1:]
    i = int(np.argmax(gaps))
    return i + 1, float(sv[i]), float(sv[i + 1]), float(gaps[i])


def rank_by_tol(sv, tol):
    """
    Rank at a fixed relative tolerance. Kept only as the comparison the gap
    method exists to replace: at pole-on the Wigner-d cancellation floor is
    5e-7 at l = 30 (docs/conventions.md section 7), so 1e-10 counts that floor
    as signal and over-counts, 64 where the true rank is 3.
    """
    sv = np.asarray(sv, dtype=float)
    return int(np.sum(sv > tol * sv[0]))


###################
# Projectors      #
###################

def graded_svd(G, lam=LAM):
    """
    SVD and grading, matching NewMetrics/crap/02_InformationContent_S.ipynb
    cell 3. Returns (Vt, sv, g) with g = sv^2 / (sv^2 + lam), lam absolute.
    """
    _, sv, Vt = np.linalg.svd(G, full_matrices=False)
    return Vt, sv, sv ** 2 / (sv ** 2 + lam)


def hard_weights(sv, rank):
    """
    Grading of the hard-cut projector: 1 on the first rank directions and 0
    below, the cut taken at the spectral gap the rank is already read from. It
    carries no free parameter, which is what makes it the honest object for an
    operator-level ceiling.
    """
    g = np.zeros(np.asarray(sv).size)
    g[:rank] = 1.0
    return g


def apply_projector(Vt, g, f):
    """P f for the graded projector P = V diag(g) V^T, without forming P."""
    return Vt.T @ (g * (Vt @ f))
