"""
Tests for starspot_sbi.nullspace, on small hand-built operators with known
structure, so no design cache or weights are needed.
"""

import numpy as np
import pytest

from starspot_sbi.nullspace import (
    demean_by_group,
    null_basis,
    null_correlation,
    null_power_fraction,
    pooled_correlation,
    project,
    shrinkage_sum,
)
from starspot_sbi.operators import family_operator


def wide_operator(n_rows=6, n_cols=15, rank=4, seed=0):
    """Rank-4 operator with fewer rows than columns: a reduced SVD would return
    only 6 right singular vectors and miss 9 null directions."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, rank)) @ rng.normal(size=(rank, n_cols))


def two_inclinations(n_per=4, seed=2):
    ops = {10: wide_operator(seed=0), 20: wide_operator(seed=3)}
    bases = {b: null_basis(G)[0] for b, G in ops.items()}
    betas = np.repeat([10, 20], n_per)
    return ops, bases, betas, np.random.default_rng(seed)


def test_null_basis_spans_the_whole_null_space_of_a_wide_operator():
    G = wide_operator()
    Vn, rank, gap = null_basis(G)
    assert rank == 4
    assert Vn.shape == (15 - 4, 15)
    assert np.allclose(Vn @ Vn.T, np.eye(11), atol=1e-12)
    assert np.max(np.abs(G @ Vn.T)) < 1e-10


def test_projection_keeps_the_null_space_and_removes_the_row_space():
    G = wide_operator()
    Vn = null_basis(G)[0]
    rng = np.random.default_rng(1)
    in_null = rng.normal(size=(5, Vn.shape[0])) @ Vn
    in_row = rng.normal(size=(5, G.shape[0])) @ G
    assert np.allclose(project(in_null, Vn), in_null, atol=1e-12)
    assert np.max(np.abs(project(in_row, Vn))) < 1e-10


def test_demean_by_group_removes_each_groups_own_mean():
    X = np.arange(12.0).reshape(6, 2)
    groups = np.array([0, 0, 1, 1, 1, 0])
    Xd = demean_by_group(X, groups)
    for g in (0, 1):
        sel = groups == g
        assert np.allclose(Xd[sel], X[sel] - X[sel].mean(axis=0))


def test_pooled_correlation_is_an_uncentred_cosine():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(7, 3))
    assert pooled_correlation(a, 3.0 * a) == pytest.approx(1.0)
    assert pooled_correlation(a, -a) == pytest.approx(-1.0)
    b = a + 5.0          # a centred Pearson would ignore the offset; this must not
    want = np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert pooled_correlation(a, b) == pytest.approx(want)


def test_null_power_fraction_is_one_in_the_null_space_and_zero_in_the_row_space():
    ops, bases, betas, rng = two_inclinations()
    null_stack = np.vstack([rng.normal(size=(4, bases[b].shape[0])) @ bases[b] for b in (10, 20)])
    row_stack = np.vstack([rng.normal(size=(4, ops[b].shape[0])) @ ops[b] for b in (10, 20)])
    assert np.allclose(null_power_fraction(null_stack, betas, bases), 1.0)
    assert np.allclose(null_power_fraction(row_stack, betas, bases), 0.0, atol=1e-10)


def test_null_power_fraction_ignores_a_shared_offset():
    """The population mean at each inclination is removed first, so adding the
    same vector to every surface changes nothing."""
    ops, bases, betas, rng = two_inclinations()
    X = rng.normal(size=(8, 15))
    offset = rng.normal(size=15)
    assert np.allclose(null_power_fraction(X + offset, betas, bases),
                       null_power_fraction(X, betas, bases))


def test_a_single_surface_inclination_raises():
    _, bases, _, rng = two_inclinations()
    with pytest.raises(ValueError):
        null_power_fraction(rng.normal(size=(5, 15)), np.array([10, 10, 10, 10, 20]), bases)


def test_null_correlation_of_a_perfect_posterior_is_one():
    _, bases, betas, rng = two_inclinations(n_per=200)
    truths = rng.normal(size=(400, 15))
    out = null_correlation(truths, 2.0 * truths, betas, bases, n_boot=50, seed=0)
    assert out['rho'] == pytest.approx(1.0)
    assert out['rho_se'] == pytest.approx(0.0, abs=1e-12)
    assert abs(out['rho_perm']) < 0.1


def test_shrinkage_sum_matches_its_spectral_form():
    G = wide_operator()
    lam = 1e6
    sv = np.linalg.svd(G, compute_uv=False)
    want = np.sum(sv ** 2 / (sv ** 2 + 1.0 / lam))
    assert shrinkage_sum(G, lam) == pytest.approx(want, rel=1e-6)
    assert shrinkage_sum(G, lam) <= 4 + 1e-6

def test_family_operator_stacks_the_family_channels():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3, 4, 9)) + 1j * rng.normal(size=(3, 4, 9))
    G = family_operator(A, np.eye(9), 'phot_ax')
    assert np.allclose(G, np.vstack([A[0].real, A[2].real]))
    with pytest.raises(ValueError):
        family_operator(A, np.eye(9), 'phot_az')