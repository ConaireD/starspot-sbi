"""Tests for starspot_sbi.indexing: ordering helpers and complex <-> real packing."""

import numpy as np
import pytest

from starspot_sbi.indexing import (
    lm_indices, n_coeffs, lm_to_idx, idx_to_lm,
    coeffs_to_real, real_to_coeffs, _L_from_len,
)


class TestNCoeffs:
    def test_L0(self):
        assert n_coeffs(0) == 1

    def test_L1(self):
        assert n_coeffs(1) == 4

    def test_L2(self):
        assert n_coeffs(2) == 9

    def test_L5(self):
        assert n_coeffs(5) == 36

    def test_formula(self):
        for L in range(20):
            assert n_coeffs(L) == (L + 1) ** 2


class TestLmIndices:
    def test_L0(self):
        assert lm_indices(0) == [(0, 0)]

    def test_L1(self):
        assert lm_indices(1) == [(0, 0), (1, -1), (1, 0), (1, 1)]

    def test_L2(self):
        idx = lm_indices(2)
        assert len(idx) == 9
        assert idx[0] == (0, 0)
        assert idx[-1] == (2, 2)

    def test_length_matches_n_coeffs(self):
        for L in range(10):
            assert len(lm_indices(L)) == n_coeffs(L)

    def test_m_range(self):
        """Each l should have m in [-l, l]."""
        for L in range(8):
            for l, m in lm_indices(L):
                assert -l <= m <= l

    def test_ordering(self):
        """Indices should be sorted by l, then m."""
        idx = lm_indices(5)
        for i in range(len(idx) - 1):
            assert idx[i] < idx[i + 1]


class TestLmToIdx:
    def test_l0_m0(self):
        assert lm_to_idx(0, 0) == 0

    def test_l1(self):
        assert lm_to_idx(1, -1) == 1
        assert lm_to_idx(1, 0) == 2
        assert lm_to_idx(1, 1) == 3

    def test_l2(self):
        assert lm_to_idx(2, -2) == 4
        assert lm_to_idx(2, 0) == 6
        assert lm_to_idx(2, 2) == 8

    def test_consistent_with_lm_indices(self):
        """lm_to_idx should give the position in lm_indices."""
        for L in range(8):
            for pos, (l, m) in enumerate(lm_indices(L)):
                assert lm_to_idx(l, m) == pos

    def test_unique(self):
        """All indices for a given L should be unique."""
        for L in range(10):
            idxs = [lm_to_idx(l, m) for l, m in lm_indices(L)]
            assert len(idxs) == len(set(idxs))


class TestIdxToLm:
    def test_inverse_of_lm_to_idx(self):
        for L in range(8):
            for l, m in lm_indices(L):
                assert idx_to_lm(lm_to_idx(l, m)) == (l, m)

    def test_covers_range(self):
        """Every flat index below n_coeffs(L) maps to a valid (l, m)."""
        L = 6
        for idx in range(n_coeffs(L)):
            l, m = idx_to_lm(idx)
            assert 0 <= l <= L
            assert -l <= m <= l


class TestLFromLen:
    def test_valid_lengths(self):
        for L in range(10):
            assert _L_from_len(n_coeffs(L)) == L

    @pytest.mark.parametrize("n", [2, 3, 5, 50, 960, 962])
    def test_invalid_raises(self, n):
        with pytest.raises(ValueError):
            _L_from_len(n)


class TestPacking:
    @pytest.mark.parametrize("L", [0, 1, 2, 4, 6, 30])
    def test_real_roundtrip(self, L):
        """real -> complex -> real is exact for any real vector."""
        rng = np.random.default_rng(L)
        v = rng.standard_normal(n_coeffs(L))
        assert np.allclose(coeffs_to_real(real_to_coeffs(v)), v)

    @pytest.mark.parametrize("L", [1, 2, 4, 6])
    def test_complex_roundtrip_on_real_maps(self, L):
        """complex -> real -> complex is exact if the input is already a real map."""
        rng = np.random.default_rng(L + 100)
        c = real_to_coeffs(rng.standard_normal(n_coeffs(L)))
        assert np.allclose(real_to_coeffs(coeffs_to_real(c)), c)

    @pytest.mark.parametrize("L", [1, 2, 4, 6])
    def test_reality_condition(self, L):
        """Output of real_to_coeffs satisfies s_l^{-m} = (-1)^m conj(s_l^m)."""
        rng = np.random.default_rng(L + 200)
        c = real_to_coeffs(rng.standard_normal(n_coeffs(L)))
        for l in range(L + 1):
            assert c[lm_to_idx(l, 0)].imag == 0.0
            for m in range(1, l + 1):
                assert np.isclose(c[lm_to_idx(l, -m)],
                                  (-1) ** m * np.conj(c[lm_to_idx(l, m)]))

    def test_block_layout(self):
        """Real vector is [s_l^0 | Re s_l^{m>0} | Im s_l^{m>0}]."""
        L = 4
        rng = np.random.default_rng(0)
        c = real_to_coeffs(rng.standard_normal(n_coeffs(L)))
        v = coeffs_to_real(c)

        n_m0 = L + 1
        n_pos = L * (L + 1) // 2
        assert v.size == n_m0 + 2 * n_pos

        for l in range(L + 1):
            assert v[l] == c[lm_to_idx(l, 0)].real

        pos = [(l, m) for l in range(L + 1) for m in range(1, l + 1)]
        for k, (l, m) in enumerate(pos):
            assert v[n_m0 + k] == c[lm_to_idx(l, m)].real
            assert v[n_m0 + n_pos + k] == c[lm_to_idx(l, m)].imag

    def test_dtypes(self):
        v = np.zeros(n_coeffs(4))
        assert real_to_coeffs(v).dtype == np.complex128
        assert coeffs_to_real(real_to_coeffs(v)).dtype == np.float64

    def test_length_preserved(self):
        for L in [1, 4, 30]:
            v = np.zeros(n_coeffs(L))
            assert real_to_coeffs(v).size == n_coeffs(L)
            assert coeffs_to_real(real_to_coeffs(v)).size == n_coeffs(L)

    @pytest.mark.parametrize("n", [50, 960])
    def test_bad_length_raises(self, n):
        with pytest.raises(ValueError):
            coeffs_to_real(np.zeros(n, dtype=np.complex128))
        with pytest.raises(ValueError):
            real_to_coeffs(np.zeros(n))

    def test_dc_only_map(self):
        """A pure DC real vector gives a pure DC complex vector."""
        v = np.zeros(n_coeffs(4))
        v[0] = 3.5
        c = real_to_coeffs(v)
        assert c[0] == 3.5 + 0j
        assert np.all(c[1:] == 0)


class TestPackIndicesImmutable:
    def test_cached_arrays_readonly(self):
        """Cached index arrays are shared; mutation must not be possible."""
        from starspot_sbi.indexing import _build_pack_indices
        for a in _build_pack_indices(4):
            with pytest.raises(ValueError):
                a[0] = 0