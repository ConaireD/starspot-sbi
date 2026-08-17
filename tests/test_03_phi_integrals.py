"""Tests for phi integrals: known values, symmetry, sparsity, self-consistency.

Note: The 'fast' phi integrals include intentional sign/conjugate corrections
relative to the raw sympy integrals.  The corrections were determined empirically
to make the full forward model match observations.  Therefore we do NOT test
exact-vs-fast equality; instead we test each variant against known analytic values
and internal consistency.
"""
import numpy as np
import pytest
from starspot_sbi.indexing import n_coeffs, lm_indices, lm_to_idx
from starspot_sbi.kernels import (
    I_phi_x_exact, I_phi_y_exact, I_phi_p_exact,
    I_phi_x_fast, I_phi_y_fast, I_phi_p_fast,
)

ATOL = 1e-12


class TestPhiXKnownValues:
    """I_phi_x should vanish for m=0 and even |m|>2."""

    def test_m0_is_zero(self):
        assert I_phi_x_fast(0) == 0.0j

    def test_even_m_gt2_is_zero(self):
        for m in [-4, 4, -6, 6, -8, 8]:
            assert I_phi_x_fast(m) == 0.0j

    def test_m_pm2_is_imaginary(self):
        val_p2 = I_phi_x_fast(2)
        val_m2 = I_phi_x_fast(-2)
        # Should be purely imaginary
        assert abs(val_p2.real) < 1e-15
        assert abs(val_m2.real) < 1e-15
        # |value| = pi/4
        assert abs(abs(val_p2) - np.pi / 4) < 1e-14

    def test_odd_m_nonzero(self):
        """Odd m values should produce nonzero results."""
        for m in [-3, -1, 1, 3, 5]:
            assert abs(I_phi_x_fast(m)) > 1e-15


class TestPhiYKnownValues:
    """I_phi_y should vanish for odd |m|>1."""

    def test_odd_m_gt1_is_zero(self):
        for m in [-3, 3, -5, 5, -7, 7]:
            assert I_phi_y_fast(m) == 0.0

    def test_m_pm1(self):
        # CORRECTED (was: assert abs(val - (-np.pi/2)) < 1e-14).
        # The old assertion carried a spurious minus sign. The defining integral is
        #     I_y(m) = int_{-pi/2}^{+pi/2} cos(phi) e^{i m phi} dphi,
        # so I_y(1) = int cos^2 + i int cos sin = +pi/2 + 0i. Verified three ways
        # (scipy.integrate.quad on Re and Im, sympy I_phi_y_exact, and the closed
        # form) all agreeing to 15 digits at +1.570796326794897.
        val = I_phi_y_fast(1)
        assert abs(val - (np.pi / 2)) < 1e-14
        # I_y is even in m, so m=-1 must give the same value.
        assert abs(I_phi_y_fast(-1) - I_phi_y_fast(1)) < 1e-15

    def test_m0(self):
        # CORRECTED (was: assert abs(val - (-2.0)) < 1e-14, with the comment
        # "-1 * -2 * cos(0) / (0^2 - 1) = 2 / (-1) = -2"). The arithmetic in that
        # comment is itself wrong: the implementation is -2*cos(pi*m/2)/(m^2-1),
        # which at m=0 is -2*1/(-1) = +2. There is no leading -1 factor.
        # Independently: the integrand cos(phi) is strictly positive on
        # (-pi/2, +pi/2), so I_y(0) = int cos = +2 cannot be negative.
        val = I_phi_y_fast(0)
        assert abs(val - 2.0) < 1e-14

    def test_even_m_nonzero(self):
        """Even m values should produce nonzero results."""
        for m in [-2, 0, 2, 4]:
            assert abs(I_phi_y_fast(m)) > 1e-15


class TestPhiPEqualsPhiY:
    """Photometry phi integral should equal y-kernel phi integral."""

    def test_p_equals_y(self):
        for m in range(-8, 9):
            assert I_phi_p_fast(m) == I_phi_y_fast(m)

    def test_p_exact_equals_y_exact(self):
        for m in range(-4, 5):
            assert I_phi_p_exact(m) == I_phi_y_exact(m)


class TestPhiExactSelfConsistency:
    """Exact phi integrals should satisfy known identities."""

    def test_x_exact_m0_is_zero(self):
        """sin(phi)*cos(phi) integrated symmetrically -> 0 for m=0."""
        assert abs(I_phi_x_exact(0)) < 1e-14

    def test_y_exact_has_correct_magnitude(self):
        """Check |I_phi_y_exact(m)| matches the closed-form magnitude."""
        for m in range(-6, 7):
            ex = I_phi_y_exact(m)
            fa = I_phi_y_fast(m)
            # Fast has sign corrections; magnitudes should still match
            assert abs(abs(ex) - abs(fa)) < 1e-12, \
                f"m={m}: |exact|={abs(ex)}, |fast|={abs(fa)}"

    def test_x_exact_has_correct_magnitude(self):
        """Magnitudes should match between exact and fast for phi_x."""
        for m in range(-6, 7):
            ex = I_phi_x_exact(m)
            fa = I_phi_x_fast(m)
            assert abs(abs(ex) - abs(fa)) < 1e-12, \
                f"m={m}: |exact|={abs(ex)}, |fast|={abs(fa)}"


class TestPhiFastSparsity:
    """Selection rules: which m values give nonzero results."""

    @pytest.mark.parametrize("m", range(-8, 9))
    def test_phi_x_sparsity(self, m):
        val = I_phi_x_fast(m)
        if m == 0 or (m % 2 == 0 and abs(m) != 2):
            assert abs(val) < 1e-15, f"m={m} should be zero"
        elif abs(m) == 2 or m % 2 != 0:
            # These CAN be nonzero (not all will be, but the function should return)
            assert isinstance(val, (complex, float, np.complexfloating, np.floating))

    @pytest.mark.parametrize("m", range(-8, 9))
    def test_phi_y_sparsity(self, m):
        val = I_phi_y_fast(m)
        if abs(m) > 1 and m % 2 != 0:
            assert abs(val) < 1e-15, f"m={m} should be zero"
