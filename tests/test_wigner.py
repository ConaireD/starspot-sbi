import numpy as np
import pytest

from starspot_sbi.wigner import (
    wigner_D_from_d,
    wigner_d_matrix_exact,
    wigner_d_matrix_fast,
    precompute_wigner_d_fast,
)

BETAS = [0.1, 0.7, np.pi / 3, 1.9, 3.0]


def d1_closed_form(beta):
    """d^1(beta), rows and columns m = -1, 0, +1."""
    c, s, r2 = np.cos(beta), np.sin(beta), np.sqrt(2)
    return np.array([[(1 + c) / 2,  s / r2, (1 - c) / 2],
                     [-s / r2,      c,       s / r2],
                     [(1 - c) / 2, -s / r2, (1 + c) / 2]])


############
# Values   #
############

@pytest.mark.parametrize('beta', BETAS)
def test_l1_closed_form(beta):
    assert np.max(np.abs(wigner_d_matrix_fast(1, beta) - d1_closed_form(beta))) < 1e-14


def test_l0_is_one():
    assert wigner_d_matrix_fast(0, 1.234) == pytest.approx(np.array([[1.0]]))


@pytest.mark.parametrize('l', [0, 1, 6, 12])
def test_identity_at_zero_beta(l):
    d = wigner_d_matrix_fast(l, 0.0)
    assert np.max(np.abs(d - np.eye(2 * l + 1))) < 1e-13


def test_dtype_is_real(): 
    assert wigner_d_matrix_fast(4, 0.7).dtype == np.float64


##############
# Identities #
##############

@pytest.mark.parametrize('l', [1, 3, 5, 8, 12])
@pytest.mark.parametrize('beta', BETAS)
def test_orthogonality(l, beta):
    d = wigner_d_matrix_fast(l, beta)
    assert np.max(np.abs(d.T @ d - np.eye(2 * l + 1))) < 1e-12


@pytest.mark.parametrize('l', [1, 5, 12])
@pytest.mark.parametrize('beta', BETAS)
def test_inverse_is_negative_beta(l, beta):
    d = wigner_d_matrix_fast(l, beta)
    assert np.max(np.abs(d @ wigner_d_matrix_fast(l, -beta) - np.eye(2 * l + 1))) < 5e-12


@pytest.mark.parametrize('l', [1, 5, 12])
@pytest.mark.parametrize('beta', BETAS)
def test_transpose_is_negative_beta(l, beta):
    assert np.max(np.abs(wigner_d_matrix_fast(l, -beta)
                         - wigner_d_matrix_fast(l, beta).T)) < 5e-12


@pytest.mark.parametrize('l', [1, 4, 7])
@pytest.mark.parametrize('beta', [0.7, 1.9])
def test_symmetry_index_swap(l, beta):
    """d^l_{m,m'} = (-1)^(m - m') d^l_{m',m}."""
    d = wigner_d_matrix_fast(l, beta)
    ms = np.arange(-l, l + 1)
    sign = (-1.0) ** (ms[:, None] - ms[None, :])
    assert np.max(np.abs(d - sign * d.T)) < 1e-13


@pytest.mark.parametrize('l', [1, 4, 7])
@pytest.mark.parametrize('beta', [0.7, 1.9])
def test_symmetry_index_reversal(l, beta):
    """d^l_{m,m'} = d^l_{-m',-m}."""
    d = wigner_d_matrix_fast(l, beta)
    assert np.max(np.abs(d - d[::-1, ::-1].T)) < 1e-13


###################
# Fast vs exact   #
###################

@pytest.mark.parametrize('l', [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('beta', BETAS)
def test_fast_matches_exact(l, beta):
    err = np.max(np.abs(wigner_d_matrix_fast(l, beta)
                        - wigner_d_matrix_exact(l, beta).real))
    assert err < 1e-13


#########################
# Large-l stability     #
#########################

def test_l30_finite_and_bounded():
    for beta in [0.4, np.pi / 2, 2.6]:
        d = wigner_d_matrix_fast(30, beta)
        assert np.all(np.isfinite(d))
        assert np.max(np.abs(d)) <= 1.0


def test_l30_orthogonality_at_measured_tolerance():
    """Measured 2.2e-11 at beta = 0.4, 5.5e-7 at beta = pi/2 (FORWARD_MODEL T5e)."""
    d = wigner_d_matrix_fast(30, 0.4)
    assert np.max(np.abs(d.T @ d - np.eye(61))) < 1e-10

    d = wigner_d_matrix_fast(30, np.pi / 2)
    assert np.max(np.abs(d.T @ d - np.eye(61))) < 1e-6


######################
# Euler convention   #
######################

@pytest.mark.parametrize('l', [1, 3, 6])
def test_phase_attaches_to_column_index(l):
    alpha, beta, gamma = 0.4, 1.4, 2.2
    d = wigner_d_matrix_fast(l, beta)
    D = wigner_D_from_d(l, alpha, gamma, d)
    ms = np.arange(-l, l + 1)

    correct = np.exp(-1j * ms * alpha)[:, None] * d * np.exp(-1j * ms * gamma)[None, :]
    assert np.max(np.abs(D - correct)) < 1e-14

    swapped = np.exp(-1j * ms * gamma)[:, None] * d * np.exp(-1j * ms * alpha)[None, :]
    assert np.max(np.abs(D - swapped)) > 0.1     # the alternative is wrong by O(1)


def test_D_is_unitary():
    l, d = 5, wigner_d_matrix_fast(5, 1.1)
    D = wigner_D_from_d(l, 0.3, 0.9, d)
    assert np.max(np.abs(D.conj().T @ D - np.eye(2 * l + 1))) < 1e-12


##################
# Precompute     #
##################

def test_precompute_fast_shapes():
    ds = precompute_wigner_d_fast(4, 0.7)
    assert len(ds) == 5
    assert [d.shape for d in ds] == [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]
    assert np.allclose(ds[1], wigner_d_matrix_fast(1, 0.7))