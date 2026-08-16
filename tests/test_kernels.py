"""Tests for kernel computation: shapes, sparsity, exact vs fast agreement."""
import numpy as np
import pytest
from starspot_sbi.indexing import n_coeffs, lm_indices, lm_to_idx
from starspot_sbi.kernels import precompute_kernels_fast


@pytest.fixture(scope="module")
def kernels_L5():
    """Precompute fast kernels at L=5 (shared across tests in this module)."""
    return precompute_kernels_fast(5)


class TestKernelShapes:
    def test_output_length(self, kernels_L5):
        kx, ky, kphot = kernels_L5
        n = n_coeffs(5)
        assert kx.shape == (n,)
        assert ky.shape == (n,)
        assert kphot.shape == (n,)

    def test_dtype(self, kernels_L5):
        kx, ky, kphot = kernels_L5
        assert kx.dtype == complex
        assert ky.dtype == complex
        assert kphot.dtype == complex


class TestKernelSparsity:
    """Many kernel entries should be zero due to phi-integral selection rules."""

    def test_kx_m0_is_zero(self, kernels_L5):
        """I_phi_x(0) = 0, so all m=0 entries of kx should vanish."""
        kx, _, _ = kernels_L5
        for l in range(6):
            idx = lm_to_idx(l, 0)
            assert abs(kx[idx]) < 1e-15

    def test_kx_even_m_gt2_is_zero(self, kernels_L5):
        """I_phi_x vanishes for even |m| > 2."""
        kx, _, _ = kernels_L5
        for l, m in lm_indices(5):
            if m != 0 and abs(m) != 2 and m % 2 == 0:
                assert abs(kx[lm_to_idx(l, m)]) < 1e-15

    def test_ky_odd_m_gt1_is_zero(self, kernels_L5):
        """I_phi_y vanishes for odd |m| > 1."""
        ky = kernels_L5[1]
        for l, m in lm_indices(5):
            if abs(m) > 1 and m % 2 != 0:
                assert abs(ky[lm_to_idx(l, m)]) < 1e-15

    def test_kphot_odd_m_gt1_is_zero(self, kernels_L5):
        """kphot has same phi integral as ky."""
        kphot = kernels_L5[2]
        for l, m in lm_indices(5):
            if abs(m) > 1 and m % 2 != 0:
                assert abs(kphot[lm_to_idx(l, m)]) < 1e-15


class TestKernelFinite:
    """No NaN or Inf in kernel output."""

    def test_all_finite(self, kernels_L5):
        kx, ky, kphot = kernels_L5
        assert np.all(np.isfinite(kx))
        assert np.all(np.isfinite(ky))
        assert np.all(np.isfinite(kphot))


class TestKernelNonTrivial:
    """At least some kernel entries should be nonzero."""

    def test_kx_has_nonzero(self, kernels_L5):
        assert np.any(np.abs(kernels_L5[0]) > 1e-15)

    def test_ky_has_nonzero(self, kernels_L5):
        assert np.any(np.abs(kernels_L5[1]) > 1e-15)

    def test_kphot_has_nonzero(self, kernels_L5):
        assert np.any(np.abs(kernels_L5[2]) > 1e-15)


class TestKernelHighDegree:
    """Kernels should stay finite at high L."""

    def test_L30_finite(self):
        kx, ky, kphot = precompute_kernels_fast(30)
        assert np.all(np.isfinite(kx))
        assert np.all(np.isfinite(ky))
        assert np.all(np.isfinite(kphot))
        assert kx.shape == (n_coeffs(30),)
