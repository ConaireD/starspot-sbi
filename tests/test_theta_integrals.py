"""Tests for theta integrals: exact vs fast, parity, known values."""
import numpy as np
import pytest
from starspot_sbi.indexing import n_coeffs, lm_indices, lm_to_idx
from starspot_sbi.kernels import (
    I_u_x_exact, I_u_y_exact, I_u_phot_exact,
    I_u_x_fast, I_u_y_fast, I_u_phot_fast,
)

ATOL = 1e-8  # sqrt(1-u^2) integrands are not polynomial; GL has ~1e-8 error


class TestThetaExactVsFast:
    """GL quadrature (fast) should match sympy integration (exact)."""

    @pytest.mark.parametrize("l,m", [
        (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 2),
    ])
    def test_I_u_x(self, l, m):
        exact = I_u_x_exact(l, m)
        fast = I_u_x_fast(l, m)
        assert abs(exact - fast) < ATOL, f"l={l},m={m}: exact={exact}, fast={fast}"

    @pytest.mark.parametrize("l,m", [
        (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ])
    def test_I_u_y(self, l, m):
        exact = I_u_y_exact(l, m)
        fast = I_u_y_fast(l, m)
        assert abs(exact - fast) < ATOL, f"l={l},m={m}: exact={exact}, fast={fast}"

    @pytest.mark.parametrize("l,m", [
        (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ])
    def test_I_u_phot(self, l, m):
        exact = I_u_phot_exact(l, m)
        fast = I_u_phot_fast(l, m)
        assert abs(exact - fast) < ATOL, f"l={l},m={m}: exact={exact}, fast={fast}"


class TestThetaParity:
    """
    Parity checks on the theta integrals.
    P_l^m(-u) = (-1)^{l+m} P_l^m(u), so integrands with
    definite parity should vanish when the overall parity is odd.
    """

    def test_I_u_x_l0_m0(self):
        """int_{-1}^1 (1-u^2) P_0(u) du = int (1-u^2) du = 4/3."""
        val = I_u_x_fast(0, 0)
        assert abs(val - 4.0 / 3.0) < 1e-12

    def test_I_u_y_parity(self):
        """I_u_y for l=0,m=0: integrand u*sqrt(1-u^2) is odd -> should be 0."""
        val = I_u_y_fast(0, 0)
        assert abs(val) < 1e-14

    def test_I_u_phot_l0_m0(self):
        """int_{-1}^1 sqrt(1-u^2) du = pi/2."""
        val = I_u_phot_fast(0, 0)
        assert abs(val - np.pi / 2) < 1e-8  # GL approximation for non-polynomial


class TestThetaHighDegree:
    """Fast theta integrals should remain stable at moderately high l."""

    def test_stable_at_l50(self):
        """Should not produce NaN or Inf."""
        for m in [0, 1, 5, 10]:
            val = I_u_x_fast(50, m)
            assert np.isfinite(val)
            val = I_u_y_fast(50, m)
            assert np.isfinite(val)
            val = I_u_phot_fast(50, m)
            assert np.isfinite(val)
