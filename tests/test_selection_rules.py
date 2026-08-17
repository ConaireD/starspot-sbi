"""
Tests for the kernel selection rules.

These assert which (l, m) have non-zero kernel entries at beta = 0, verified
numerically to L=12 in the 2026-08-15 validation audit:

    kphot != 0  <=>  (l+m) even  and  (m even or |m| = 1)
    kx    != 0  <=>  (l+m) even  and  (m odd  or |m| = 2)
    ky    != 0  <=>  (l+m) odd   and  (m even or |m| = 1)

with three exceptions where an extra orthogonality kills the entry:
    kphot at |m| = 1 survives only for l = 1
    kx    at |m| = 2 survives only for l = 2
    ky    at |m| = 1 survives only for l = 2

Both directions are asserted: non-zero where the rule predicts non-zero, and
exactly zero everywhere else. A one-directional test would pass on a kernel
that was non-zero everywhere.

These are zeros of k at beta = 0. The observable subspace of the full design
matrix is different, because build_B mixes m within each l through d^l(beta).
"""

import numpy as np
import pytest

from starspot_sbi.indexing import lm_indices
from starspot_sbi.kernels import (precompute_kernels_fast,
                                   I_phi_x_fast, I_phi_y_fast)

# Entries below this are treated as analytically zero. Kernel magnitudes at
# L=12 span roughly 1e-1 to 1e-4, and the quadrature floor is ~1e-16, so any
# threshold between them separates the two populations cleanly. Verified by
# test_zero_nonzero_populations_are_separated below.
ZTOL = 1e-12

LS = [4, 8, 12]


def phot_rule(l, m):
    if (l + m) % 2 != 0:
        return False
    if abs(m) == 1:
        return l == 1
    return m % 2 == 0


def x_rule(l, m):
    if (l + m) % 2 != 0:
        return False
    if abs(m) == 2:
        return l == 2
    return abs(m) % 2 == 1


def y_rule(l, m):
    if (l + m) % 2 != 1:
        return False
    if abs(m) == 1:
        return l == 2
    return m % 2 == 0


RULES = {"kphot": phot_rule, "kx": x_rule, "ky": y_rule}


@pytest.fixture(scope="module")
def kernels():
    """Kernels at each L, computed once for the whole module."""
    return {L: dict(zip(("kx", "ky", "kphot"), precompute_kernels_fast(L)))
            for L in LS}


@pytest.mark.parametrize("L", LS)
@pytest.mark.parametrize("name", ["kphot", "kx", "ky"])
def test_selection_rule(kernels, L, name):
    """Non-zero exactly where the rule says, and zero everywhere else."""
    k = kernels[L][name]
    rule = RULES[name]

    wrong = [(l, m, abs(k[i]))
             for i, (l, m) in enumerate(lm_indices(L))
             if (abs(k[i]) > ZTOL) != rule(l, m)]

    assert not wrong, f"{name} at L={L}: {len(wrong)} mismatches, first 10: {wrong[:10]}"


@pytest.mark.parametrize("L", LS)
@pytest.mark.parametrize("name", ["kphot", "kx", "ky"])
def test_zero_nonzero_populations_are_separated(kernels, L, name):
    """
    The threshold is not load-bearing: the smallest 'non-zero' entry and the
    largest 'zero' entry are separated by many orders of magnitude.
    """
    k = np.abs(kernels[L][name])
    keep = np.array([RULES[name](l, m) for l, m in lm_indices(L)])

    assert keep.any(), "rule predicts no non-zero entries"
    smallest_kept = k[keep].min()
    largest_dropped = k[~keep].max() if (~keep).any() else 0.0

    assert smallest_kept > 1e6 * max(largest_dropped, 1e-300), (
        f"{name} at L={L}: populations not separated — "
        f"smallest kept {smallest_kept:.3e}, largest dropped {largest_dropped:.3e}"
    )


@pytest.mark.parametrize("L", LS)
def test_dropped_entries_vanish(L, kernels):
    """
    Entries the rule excludes are zero to quadrature precision.

    Two mechanisms produce them, and both are checked. Where the phi integral
    vanishes the implementation never runs the quadrature and the entry is
    exactly 0.0. Where the phi integral is non-zero but the theta integral
    vanishes by orthogonality (the three exceptions), the quadrature runs and
    returns a rounding residual of order 1e-18.
    """
    for name in ("kphot", "kx", "ky"):
        k = kernels[L][name]
        scale = np.max(np.abs(k))
        worst = 0.0
        for i, (l, m) in enumerate(lm_indices(L)):
            if not RULES[name](l, m):
                worst = max(worst, abs(k[i]))
                assert abs(k[i]) < 1e-13 * scale, f"{name}[{l},{m}] = {k[i]!r}"
        print(f"L={L:2d} {name:5s} largest excluded entry {worst:.2e}, "
              f"scale {scale:.3f}, bound {1e-13*scale:.2e}")

@pytest.mark.parametrize("L", LS)
def test_phi_guard_entries_are_exactly_zero(L, kernels):
    """
    Where the phi integral itself vanishes, the entry is exactly 0.0 because
    the implementation skips it rather than computing a cancelling quadrature.
    This pins the short-circuit in precompute_kernels_fast.
    """
    for name, phi in (("kx", I_phi_x_fast), ("ky", I_phi_y_fast),
                      ("kphot", I_phi_y_fast)):
        k = kernels[L][name]
        for i, (l, m) in enumerate(lm_indices(L)):
            if abs(phi(m)) <= 1e-15:
                assert k[i] == 0.0, f"{name}[{l},{m}] = {k[i]!r}, expected exact 0"


@pytest.mark.parametrize("L", LS)
def test_parity_complementarity(kernels, L):
    """
    The paper's complementarity claim, stated as a test: photometry and the x
    astrometric channel occupy (l+m) even; the y channel occupies (l+m) odd.
    No (l, m) is non-zero in both y and either of the other two.
    """
    kx, ky, kphot = (kernels[L][n] for n in ("kx", "ky", "kphot"))

    for i, (l, m) in enumerate(lm_indices(L)):
        if abs(kphot[i]) > ZTOL or abs(kx[i]) > ZTOL:
            assert (l + m) % 2 == 0, f"phot/x non-zero at odd (l+m): ({l},{m})"
        if abs(ky[i]) > ZTOL:
            assert (l + m) % 2 == 1, f"y non-zero at even (l+m): ({l},{m})"

    overlap = [(l, m) for i, (l, m) in enumerate(lm_indices(L))
               if abs(ky[i]) > ZTOL and (abs(kphot[i]) > ZTOL or abs(kx[i]) > ZTOL)]
    assert not overlap, f"y overlaps phot/x at {overlap[:10]}"


def test_exceptions_are_real():
    """
    The three exceptions survive a change of threshold. Excluded entries vanish
    while their rule-allowed neighbours at the same |m| do not.

    Bounds are relative to the largest entry of each kernel. An absolute bound
    tracks _GL_N rather than the physics: the analytically zero entries are
    ~1e-16 at _GL_N = 500 and ~1e-15 at 2000, since the quadrature sum
    accumulates four times as many terms.
    """
    L = 12
    kx, ky, kphot = precompute_kernels_fast(L)
    idx = {lm: i for i, lm in enumerate(lm_indices(L))}

    zero_x, zero_y, zero_p = (1e-13 * np.max(np.abs(k)) for k in (kx, ky, kphot))
    print(f"L={L} relative zero bounds: kx {zero_x:.2e}, ky {zero_y:.2e}, kphot {zero_p:.2e}")

    # kphot at |m|=1: l=1 survives, odd l >= 3 vanish by orthogonality
    assert abs(kphot[idx[(1, 1)]]) > ZTOL
    worst = max(abs(kphot[idx[(l, s)]]) for l in (3, 5, 7, 9, 11) for s in (1, -1))
    print(f"  kphot |m|=1, odd l>=3: largest {worst:.2e}, survivor (1,1) {abs(kphot[idx[(1, 1)]]):.3e}")
    for l in (3, 5, 7, 9, 11):
        assert abs(kphot[idx[(l, 1)]]) < zero_p
        assert abs(kphot[idx[(l, -1)]]) < zero_p

    # kx at |m|=2: l=2 survives, even l >= 4 vanish
    assert abs(kx[idx[(2, 2)]]) > ZTOL
    worst = max(abs(kx[idx[(l, s)]]) for l in (4, 6, 8, 10, 12) for s in (2, -2))
    print(f"  kx    |m|=2, even l>=4: largest {worst:.2e}, survivor (2,2) {abs(kx[idx[(2, 2)]]):.3e}")
    for l in (4, 6, 8, 10, 12):
        assert abs(kx[idx[(l, 2)]]) < zero_x
        assert abs(kx[idx[(l, -2)]]) < zero_x

    # ky at |m|=1: l=2 survives, even l >= 4 vanish
    assert abs(ky[idx[(2, 1)]]) > ZTOL
    worst = max(abs(ky[idx[(l, s)]]) for l in (4, 6, 8, 10, 12) for s in (1, -1))
    print(f"  ky    |m|=1, even l>=4: largest {worst:.2e}, survivor (2,1) {abs(ky[idx[(2, 1)]]):.3e}")
    for l in (4, 6, 8, 10, 12):
        assert abs(ky[idx[(l, 1)]]) < zero_y
        assert abs(ky[idx[(l, -1)]]) < zero_y
        

@pytest.mark.parametrize("L", LS)
def test_hermitian_symmetry(kernels, L):
    """
    A real map has real signals, so the kernel must satisfy
    k_{l,-m} = (-1)^m conj(k_{l,m}) in the same convention as the coefficient
    packing. This is what makes Re(A @ s) lossless.
    """
    for name in ("kphot", "kx", "ky"):
        k = kernels[L][name]
        idx = {lm: i for i, lm in enumerate(lm_indices(L))}
        for l in range(L + 1):
            for m in range(1, l + 1):
                lhs = k[idx[(l, -m)]]
                rhs = (-1) ** m * np.conj(k[idx[(l, m)]])
                assert np.isclose(lhs, rhs, rtol=1e-10, atol=1e-14), (
                    f"{name} at l={l}, m={m}: {lhs!r} vs {rhs!r}"
                )