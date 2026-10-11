"""
Null-space statistics: the correlation rho between the null components of the
true surfaces and of the posterior means, the null power fraction f_null, and
the summed posterior shrinkage.

The definitions are fixed here as the paper's numbers were computed:
- Operator: the real, unwhitened family operator Re(A(beta) T) on real-packed
  coefficients at each surface's integer inclination. Norms and projectors use
  the real-packed metric (see starspot_sbi.operators).
- Null space: right singular vectors of the full SVD beyond the rank, with the
  rank at the largest log10 gap. The full SVD is required: for an operator with
  fewer rows than coefficients a reduced SVD omits part of the null space.
- Mean removal: each stack (true surfaces, or posterior means) has its own mean
  removed within each inclination.
- rho: one uncentred cosine, pooled over all surfaces and all coordinates of the
  mean-removed null components.
- f_null: per surface, the null power of the mean-removed vector over its total
  power.
"""

import numpy as np
import scipy.linalg

from starspot_sbi.operators import rank_by_gap


def _full_svd(G):
    """
    Full SVD. LAPACK's divide-and-conquer driver (gesdd, numpy's default) can
    fail to converge on particular matrices, depending on the build; the slower
    QR-based driver (gesvd) is used when it does.
    """
    try:
        return np.linalg.svd(G, full_matrices=True)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(G, full_matrices=True, lapack_driver='gesvd')

def null_basis(G):
    """
    Orthonormal rows spanning the null space of G. Returns (Vn, rank,
    gap_decades), with Vn of shape (n - rank, n).
    """
    _, sv, Vt = _full_svd(G)
    rank, _, _, gap = rank_by_gap(sv)
    return Vt[rank:], rank, gap


def project(X, Vn):
    """Rows of X projected onto the span of Vn's rows, in the original coordinates."""
    return (X @ Vn.T) @ Vn


def demean_by_group(X, groups):
    """X with each group's own mean removed."""
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    out = np.empty_like(X)
    for g in np.unique(groups):
        sel = groups == g
        out[sel] = X[sel] - X[sel].mean(axis=0, keepdims=True)
    return out


def pooled_correlation(a, b):
    """One uncentred cosine over every entry of a and b."""
    a, b = np.ravel(a), np.ravel(b)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def _inclinations(betas, bases):
    """Sorted inclinations present, after checking each has a basis and at
    least two surfaces (mean removal leaves a single surface at zero)."""
    labels, counts = np.unique(np.asarray(betas), return_counts=True)
    missing = [int(b) for b in labels if b not in bases]
    if missing:
        raise KeyError(f"no null basis for inclinations {missing}")
    if np.any(counts < 2):
        raise ValueError(f"inclinations with one surface: {labels[counts < 2].tolist()}")
    return labels


def null_power_fraction(X, betas, bases):
    """Per-surface f_null of the rows of X (fnull_three_ways.py, variant b)."""
    X = np.asarray(X, dtype=float)
    betas = np.asarray(betas)
    f = np.empty(X.shape[0])
    for b in _inclinations(betas, bases):
        sel = np.where(betas == b)[0]
        Xd = X[sel] - X[sel].mean(axis=0, keepdims=True)
        f[sel] = np.sum(project(Xd, bases[b]) ** 2, axis=1) / np.sum(Xd ** 2, axis=1)
    return f


def null_correlation(truths, post, betas, bases, n_boot=500, seed=0):
    """
    rho, its bootstrap standard error over surfaces, and a within-inclination
    permutation baseline (rho_check.py). The random streams are consumed in the
    same order as there, so the same inputs and seed reproduce its values.
    """
    truths = np.asarray(truths, dtype=float)
    post = np.asarray(post, dtype=float)
    betas = np.asarray(betas)
    labels = _inclinations(betas, bases)
    pt = np.empty_like(truths)
    pp = np.empty_like(post)
    for b in labels:
        sel = np.where(betas == b)[0]
        t = project(truths[sel], bases[b])
        p = project(post[sel], bases[b])
        pt[sel] = t - t.mean(axis=0, keepdims=True)
        pp[sel] = p - p.mean(axis=0, keepdims=True)
    rho = pooled_correlation(pt, pp)

    n = truths.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n, n)
        boot[i] = pooled_correlation(pt[pick], pp[pick])

    rng_p = np.random.default_rng(seed + 1)
    perm = pp.copy()
    for b in labels:
        sel = np.where(betas == b)[0]
        perm[sel] = pp[sel[rng_p.permutation(len(sel))]]

    return {'rho': rho, 'rho_se': float(boot.std()),
            'rho_perm': pooled_correlation(pt, perm)}


def shrinkage_sum(G, lam=1e6):
    """
    Posterior shrinkage S = 1 - diag[(G^T G + I/lam)^-1] / lam summed over
    coefficients, for prior variance lam on each real-packed coefficient and
    unit noise (paper2_counts_and_rho.ipynb cell 6).
    """
    G = np.asarray(G, dtype=float)
    F = G.T @ G + np.eye(G.shape[1]) / lam
    return float(np.sum(1.0 - np.diag(np.linalg.inv(F)) / lam))