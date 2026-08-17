"""
Tests for design.py
"""
import numpy as np
import pytest
 
from starspot_sbi.indexing import lm_indices, lm_to_idx, n_coeffs, real_to_coeffs
from starspot_sbi.kernels import precompute_kernels_fast
from starspot_sbi.design_matrix import build_W, build_B, build_design_matrix, forward_model



L = 4
P_ROT = 10.0
OMEGA = 2 * np.pi / P_ROT
N_OBS = 216
T_OBS = np.linspace(0, P_ROT, N_OBS, endpoint=False)
 
 
@pytest.fixture(scope='module')
def kernels():
    kx, ky, kphot = precompute_kernels_fast(L)
    return {'x': kx, 'y': ky, 'phot': kphot}
 
 
@pytest.fixture(scope='module')
def s_random():
    rng = np.random.default_rng(0)
    return real_to_coeffs(rng.normal(size=n_coeffs(L)))
 
 
#####################
# W — time matrix   #
#####################
 
def test_W_shape_and_modulus():
    W = build_W(L, OMEGA, T_OBS)
    assert W.shape == (N_OBS, 2 * L + 1)
    assert np.allclose(np.abs(W), 1.0)
 
 
def test_W_m_zero_column_is_unity():
    assert np.allclose(build_W(L, OMEGA, T_OBS)[:, L], 1.0)
 
 
def test_W_first_row_is_unity():
    """t_obs[0] = 0, so no phase has accumulated."""
    assert np.allclose(build_W(L, OMEGA, T_OBS)[0, :], 1.0)
 
 
def test_W_phase_sign():
    """exp(-i m omega t); the opposite sign is wrong by O(1)."""
    W = build_W(L, OMEGA, T_OBS)
    m = np.arange(-L, L + 1)
    correct = np.exp(-1j * np.outer(OMEGA * T_OBS, m))
    assert np.max(np.abs(W - correct)) < 1e-14
    assert np.max(np.abs(W - correct.conj())) > 0.1
 
 
###########################
# B — geometry matrix     #
###########################
 
def test_B_shape(kernels):
    assert build_B(L, 0.7, kernels['phot']).shape == (2 * L + 1, n_coeffs(L))
 
 
def test_B_one_nonzero_row_per_column(kernels):
    """Column l^2 + i can only occupy row m = (i - l) + L."""
    B = build_B(L, 0.7, kernels['phot'])
    for l, m in lm_indices(L):
        col = lm_to_idx(l, m)
        mask = np.ones(2 * L + 1, dtype=bool)
        mask[m + L] = False
        assert np.all(B[mask, col] == 0.0)
 
 
def test_B_at_zero_beta_is_the_kernel(kernels):
    """d^l(0) = I, so B reduces to the kernel laid out by (l, m)."""
    B = build_B(L, 0.0, kernels['phot'])
    for l, m in lm_indices(L):
        col = lm_to_idx(l, m)
        assert B[m + L, col] == pytest.approx(kernels['phot'][col], abs=1e-12)
 
 
###########################
# A — stacking            #
###########################
 
def test_design_matrix_shape_and_blocks(kernels):
    ks = [kernels['x'], kernels['y'], kernels['phot']]
    A = build_design_matrix(L, 0.7, OMEGA, T_OBS, ks)
    assert A.shape == (3 * N_OBS, n_coeffs(L))
 
    W = build_W(L, OMEGA, T_OBS)
    for j, k in enumerate(ks):
        block = A[j * N_OBS:(j + 1) * N_OBS]
        assert np.max(np.abs(block - W @ build_B(L, 0.7, k))) < 1e-14
 
 
def test_single_kernel_equals_list_of_one(kernels):
    a = build_design_matrix(L, 0.7, OMEGA, T_OBS, kernels['phot'])
    b = build_design_matrix(L, 0.7, OMEGA, T_OBS, [kernels['phot']])
    assert np.array_equal(a, b)
 
 
###########################
# Forward model           #
###########################
 
def test_linearity(kernels, s_random):
    rng = np.random.default_rng(1)
    s2 = real_to_coeffs(rng.normal(size=n_coeffs(L)))
    A = build_design_matrix(L, 0.7, OMEGA, T_OBS, kernels['phot'])
    lhs = forward_model(2.3 * s_random - 0.4 * s2, A)
    rhs = 2.3 * forward_model(s_random, A) - 0.4 * forward_model(s2, A)
    assert np.max(np.abs(lhs - rhs)) < 1e-14
 
 
@pytest.mark.parametrize('beta', [0.0, 0.7, np.pi / 2])
def test_imaginary_part_is_machine_noise(kernels, s_random, beta):
    """MEASURED: ratio 1e-19 to 3e-15 for a real map (FORWARD_MODEL T4.7)."""
    ks = [kernels['x'], kernels['y'], kernels['phot']]
    A = build_design_matrix(L, beta, OMEGA, T_OBS, ks)
    mu = A @ s_random
    ratio = np.max(np.abs(mu.imag)) / np.max(np.abs(mu.real))
    print(f"beta={beta:.3f}  max|Im| / max|Re| = {ratio:.2e} (bound 1e-12)")
    assert ratio < 1e-12 
 
def test_uniform_map_gives_unit_flux_and_zero_astrometry(kernels):
    """
    s_0^0 = 2 sqrt(pi) for unit intensity; F_0 = 1 pins the normalisation.

    MEASURED F_0 - 1 = 4.2e-9 at _GL_N = 500, the N^-3 quadrature error of the
    l = 0 photometric kernel (kernels.py).  Confirmed by raising to N = 2000,
    which reduces it 64-fold.  The astrometric channels are exactly zero by the
    selection rules (k^x_00 = k^y_00 = 0), so they keep a machine bound.
    """
    
    s = np.zeros(n_coeffs(L), dtype=complex)
    s[0] = 2 * np.sqrt(np.pi)
    for beta in [0.0, 0.7, np.pi / 2]:
        A = build_design_matrix(L, beta, OMEGA, T_OBS,
                                [kernels['x'], kernels['y'], kernels['phot']])
        mu = forward_model(s, A).reshape(3, N_OBS)
        err_x, err_y = np.max(np.abs(mu[0])), np.max(np.abs(mu[1]))
        err_f = np.max(np.abs(mu[2] - 1.0))
        print(f"beta={beta:.3f}  F_0 - 1 = {err_f:.3e} (bound 1e-7)  "
              f"|mu_x| = {err_x:.1e}  |mu_y| = {err_y:.1e} (bound 1e-12)")
        assert err_x < 1e-12
        assert err_y < 1e-12
        assert err_f < 1e-7

 
 
def test_rotational_equivariance(kernels, s_random):
    """s_l^m -> s_l^m exp(-i m D) is the same as t -> t + D/omega."""
    D = 0.37
    phase = np.array([np.exp(-1j * m * D) for l, m in lm_indices(L)])
 
    A = build_design_matrix(L, 0.7, OMEGA, T_OBS, kernels['phot'])
    A_shift_t = build_design_matrix(L, 0.7, OMEGA, T_OBS + D / OMEGA, kernels['phot'])
 
    assert np.max(np.abs(forward_model(s_random * phase, A)
                         - forward_model(s_random, A_shift_t))) < 1e-13
    assert np.max(np.abs(forward_model(s_random * phase.conj(), A)
                         - forward_model(s_random, A_shift_t))) > 1e-3
 
 
######################
# beta convention    #
######################
 
def test_pole_on_flux_is_constant(kernels, s_random):
    """
    At beta = pi/2 the spin axis points at the observer, so rotation cannot
    change the disc-integrated flux. At beta = 0 it must. An inverted beta
    convention anywhere upstream fails this at O(1).

    Residual modulation is 4.0e-8 of the mean at _GL_N = 500, inherited from
    the kernel quadrature, and 64 times smaller at N = 2000.
    """
    A_pole = build_design_matrix(L, np.pi / 2, OMEGA, T_OBS, kernels['phot'])
    A_edge = build_design_matrix(L, 0.0, OMEGA, T_OBS, kernels['phot'])

    mu_pole = forward_model(s_random, A_pole)
    mu_edge = forward_model(s_random, A_edge)
    rel_pole = np.std(mu_pole) / np.abs(np.mean(mu_pole))
    rel_edge = np.std(mu_edge) / np.abs(np.mean(mu_edge))
    print(f"pole-on  std/mean = {rel_pole:.3e} (bound 1e-6)")
    print(f"edge-on  std/mean = {rel_edge:.3e} (bound 1e-3, must exceed)")
    assert rel_pole < 1e-6
    assert rel_edge > 1e-3
 
 
def test_pole_on_centroid_rotates_rigidly(kernels, s_random):
    """
    Pole-on, the sky-projected map rotates as a rigid body, so the centroid
    traces a circle: x^2 + y^2 is constant in time.
    """
    A = build_design_matrix(L, np.pi / 2, OMEGA, T_OBS,
                            [kernels['x'], kernels['y']])
    mu = forward_model(s_random, A).reshape(2, N_OBS)
    r2 = mu[0]**2 + mu[1]**2
    rel = np.std(r2) / np.mean(r2)
    print(f"pole-on  std(x^2+y^2)/mean = {rel:.3e} (bound 1e-6)")
    assert rel < 1e-6