"""
Tests for starspot_sbi.pipeline.

Two tiers. The stub tier replaces the decoder and the density estimator with
maps whose outputs are analytic functions of their inputs, which checks the
standardisation, the axis handling and the batching in sample_draws without
weights. The checkpoint tier loads the released files and skips when weights/
is absent.

The stub estimator reads its draws from the conditioning vector, so the test
comparing two batchings of the same signals is sensitive to the noise
realisation rather than to the output shape alone.

Rendering is checked at a reduced grid and a reduced degree against a
double-precision numpy evaluation, since the production path runs in complex64
on the device.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from starspot_sbi.indexing import lm_indices, coeffs_to_real, real_to_coeffs
from starspot_sbi.models import (FAMILIES, GAIN_PHOT, GAIN_AST, N_COEFFS,
                                 LOG10_SIGMA_PHOT, LOG10_SIGMA_ASTRO,
                                 beta_norm_from_deg, make_context,
                                 load_vae, load_flow)
from starspot_sbi.render import N_THETA, N_PHI, build_Ylm_matrix
from starspot_sbi.pipeline import (STORED_TO_MODEL,
                                   LOG10_SIGMA_PHOT_MISSION,
                                   LOG10_SIGMA_ASTRO_MISSION,
                                   select_channels, to_model_order,
                                   render_draws, power_spectrum, sample_draws,
                                   posterior_mean, reconstruct, decode_ceiling)

WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'
SUFFIX = '_temp'                 # marks the checkpoints as pending verification
T_SIGNAL = 216
LATENT_DIM = 96

L_SMALL = 6
N_THETA_SMALL = 24
N_PHI_SMALL = 48

needs_weights = pytest.mark.skipif(
    not (WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt').exists(),
    reason=f'released checkpoints not present in {WEIGHTS}')


############################
# Stubs and helpers        #
############################

class StubVAE:
    """
    Decoder copying the latent into the leading coefficients and zeroing the
    rest, so an un-standardised output inverts to the latent that produced it.
    The encoder is the identity and the latent head takes the leading entries,
    which gives decode_ceiling a fixed point.
    """

    def __init__(self, n_out, latent_dim=LATENT_DIM):
        self.n_out = n_out
        self.latent_dim = latent_dim

    def decoder(self, z):
        out = torch.zeros(z.shape[0], self.n_out, dtype=torch.float32)
        out[:, :self.latent_dim] = z[:, :self.latent_dim]
        return out

    def encoder(self, x):
        return x

    def latent(self, h):
        return None, h[:, :self.latent_dim], None


class StubEstimator:
    """
    Density estimator with three sampling modes.

    'ctx' returns the leading entries of the conditioning vector, so the output
    depends on the noise realisation. 'index' returns j + 100 d for surface j
    and draw d, which distinguishes the batch axis from the draw axis.
    'constant' returns one value everywhere, which makes the un-standardisation
    checkable in closed form.

    With batch_first the return is transposed to (b, n_draws, d), which sbi
    0.26.1 does not produce and which sample_draws rejects.
    """

    def __init__(self, mode='ctx', constant=0.0, latent_dim=LATENT_DIM,
                 batch_first=False):
        self.mode = mode
        self.constant = constant
        self.latent_dim = latent_dim
        self.batch_first = batch_first

    def sample(self, shape, ctx):
        n_draws = int(shape[0])
        b = ctx.shape[0]
        if self.mode == 'ctx':
            out = ctx[:, :self.latent_dim].unsqueeze(0).repeat(n_draws, 1, 1)
        elif self.mode == 'index':
            j = torch.arange(b, dtype=torch.float32).reshape(1, b, 1)
            d = torch.arange(n_draws, dtype=torch.float32).reshape(n_draws, 1, 1)
            out = (j + 100.0 * d).repeat(1, 1, self.latent_dim)
        else:
            out = torch.full((n_draws, b, self.latent_dim),
                             float(self.constant), dtype=torch.float32)
        return out.transpose(0, 1) if self.batch_first else out


def make_stats(n, include_dc=True, dc_value=None, seed=7):
    rng = np.random.default_rng(seed)
    return {'mu_data': rng.normal(size=n).astype(np.float32),
            'std_data': (0.5 + rng.random(n)).astype(np.float32),
            'include_dc': include_dc,
            'dc_value': dc_value}


def make_meta(family):
    return {'ch': FAMILIES[family], 'T': T_SIGNAL,
            'gain_phot': GAIN_PHOT, 'gain_ast': GAIN_AST,
            'log10_sigma_phot': LOG10_SIGMA_PHOT,
            'log10_sigma_astro': LOG10_SIGMA_ASTRO,
            'temperature': 1.0}


def make_signals(B, n_ch, T=T_SIGNAL, seed=0):
    """
    Signals in model channel order, a relative flux near one and mean-zero
    astrometric offsets.
    """
    rng = np.random.default_rng(seed)
    sig = rng.normal(scale=1e-3, size=(B, n_ch, T))
    sig[:, 0] += 1.0
    return sig.astype(np.float32)


def make_stored_signals(B, T=T_SIGNAL, seed=0):
    """Three channels in stored order, (astro_x, astro_y, phot)."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(scale=1e-3, size=(B, 3, T))
    sig[:, 2] += 1.0
    return sig.astype(np.float32)


def single_mode(L, l, m, amplitude):
    """Real-packed vector holding one degree and order and its conjugate."""
    idx = {lm: i for i, lm in enumerate(lm_indices(L))}
    c = np.zeros((L + 1) ** 2, dtype=complex)
    c[idx[(l, m)]] = amplitude
    if m != 0:
        c[idx[(l, -m)]] = (-1) ** m * np.conj(amplitude)
    return coeffs_to_real(c)


############################
# Constants                #
############################

def test_mission_reference_point():
    """100 ppm photometry and 1 muas per epoch at alpha Cen B."""
    assert LOG10_SIGMA_PHOT_MISSION == -4.0
    assert LOG10_SIGMA_ASTRO_MISSION == -3.5


def test_stored_to_model_is_the_permutation_it_claims():
    """Stored files hold (astro_x, astro_y, phot) and the models index
    (phot, astro_x, astro_y)."""
    stored = np.array([0.0, 1.0, 2.0])
    assert list(stored[STORED_TO_MODEL]) == [2.0, 0.0, 1.0]


############################
# Channel selection        #
############################

@pytest.mark.parametrize('family,rows', [
    ('phot',      [2]),
    ('phot_ax',   [2, 0]),
    ('phot_ay',   [2, 1]),
    ('phot_axay', [2, 0, 1]),
])
def test_select_channels_takes_the_stored_rows(family, rows):
    """Channel c of the stored array carries the value c, so the output reads
    off the permutation directly."""
    sig = np.tile(np.arange(3, dtype=np.float32).reshape(1, 3, 1), (5, 1, 8))
    out = select_channels(sig, family)
    print(f"{family}: rows {out[0, :, 0].tolist()}")
    assert out.shape == (5, len(rows), 8)
    assert out[0, :, 0].tolist() == [float(r) for r in rows]


def test_select_channels_accepts_a_single_signal():
    sig = np.tile(np.arange(3, dtype=np.float32).reshape(3, 1), (1, 8))
    assert select_channels(sig, 'phot_ax').shape == (2, 8)


@pytest.mark.parametrize('family', ['phot', 'phot_ax', 'phot_ay', 'phot_axay'])
def test_to_model_order_permutes_a_stored_array(family):
    sig = np.tile(np.arange(3, dtype=np.float32).reshape(1, 3, 1), (2, 1, 8))
    out = to_model_order(sig, family, stored=True)
    assert np.array_equal(out, select_channels(sig, family))


def test_to_model_order_leaves_a_selected_three_channel_array_alone():
    """A phot_axay array already in model order is returned unchanged when the
    caller says so, which is the case for the legacy holdfull_sig cache."""
    sig = np.tile(np.arange(3, dtype=np.float32).reshape(1, 3, 1), (2, 1, 8))
    assert np.array_equal(to_model_order(sig, 'phot_axay', stored=False), sig)


def test_to_model_order_rejects_a_channel_count_it_cannot_interpret():
    with pytest.raises(ValueError):
        to_model_order(np.zeros((2, 2, 8)), 'phot', stored=False)


############################
# Rendering                #
############################

def test_render_shape_for_one_vector_and_for_a_stack():
    v = single_mode(L_SMALL, 3, 2, 0.4 + 0.1j)
    one = render_draws(v, N_THETA_SMALL, N_PHI_SMALL)
    many = render_draws(np.stack([v, 2 * v, 3 * v]), N_THETA_SMALL, N_PHI_SMALL)
    assert one.shape == (1, N_THETA_SMALL, N_PHI_SMALL)
    assert many.shape == (3, N_THETA_SMALL, N_PHI_SMALL)


def test_render_matches_a_double_precision_evaluation():
    """The complex64 path agrees with a complex128 evaluation of the same
    basis to the precision complex64 allows."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=(4, (L_SMALL + 1) ** 2)).astype(np.float32)
    got = render_draws(v, N_THETA_SMALL, N_PHI_SMALL)

    Y = build_Ylm_matrix(L_SMALL, N_THETA_SMALL, N_PHI_SMALL)
    ref = np.stack([(real_to_coeffs(u) @ Y.T).real.reshape(N_THETA_SMALL,
                                                           N_PHI_SMALL)
                    for u in v])
    err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    print(f"relative error {err:.3e}")
    assert err < 1e-5


def test_render_discards_a_vanishing_imaginary_part():
    """The real-packed vector maps to a Hermitian coefficient array, so the
    imaginary part of the synthesis is zero to numerical precision."""
    rng = np.random.default_rng(2)
    v = rng.normal(size=(L_SMALL + 1) ** 2)
    Y = build_Ylm_matrix(L_SMALL, N_THETA_SMALL, N_PHI_SMALL)
    img = real_to_coeffs(v) @ Y.T
    ratio = np.max(np.abs(img.imag)) / np.max(np.abs(img.real))
    print(f"imaginary over real {ratio:.3e}")
    assert ratio < 1e-12


def test_the_grid_is_theta_major():
    """A degree-one zonal mode varies down the rows and is constant along
    them, which fixes the reshape in render_draws."""
    v = single_mode(3, 1, 0, 1.0)
    img = render_draws(v, N_THETA_SMALL, N_PHI_SMALL)[0]
    along = np.max(np.abs(img - img[:, :1]))
    down = np.max(np.abs(img - img[:1, :]))
    print(f"variation along a row {along:.3e}, down a column {down:.3e}")
    assert along < 1e-6
    assert down > 1e-2


def test_the_basis_cache_returns_the_degree_it_was_asked_for():
    """Two degrees alternate through a cache holding two entries."""
    va = single_mode(4, 2, 1, 0.3)
    vb = single_mode(8, 5, 3, 0.3)
    first = render_draws(va, N_THETA_SMALL, N_PHI_SMALL)
    render_draws(vb, N_THETA_SMALL, N_PHI_SMALL)
    again = render_draws(va, N_THETA_SMALL, N_PHI_SMALL)
    assert np.array_equal(first, again)


def test_a_non_square_coefficient_count_raises():
    with pytest.raises(ValueError):
        render_draws(np.zeros(50), N_THETA_SMALL, N_PHI_SMALL)


############################
# Power spectrum           #
############################

def test_power_spectrum_isolates_a_single_degree():
    """One order and its conjugate carry 2|a|^2 and every other degree zero."""
    a = 0.4 + 0.1j
    v = single_mode(L_SMALL, 3, 2, a)
    C = power_spectrum(v)[0]
    print(f"C_3 {C[3]:.6f}, expected {2 * abs(a) ** 2:.6f}, "
          f"max elsewhere {np.max(np.delete(C, 3)):.3e}")
    assert C.shape == (L_SMALL + 1,)
    assert C[3] == pytest.approx(2 * abs(a) ** 2, rel=1e-10)
    assert np.max(np.delete(C, 3)) < 1e-12


def test_power_spectrum_sums_to_the_total_power():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(5, (L_SMALL + 1) ** 2))
    total = np.array([np.sum(np.abs(real_to_coeffs(u)) ** 2) for u in v])
    assert np.allclose(power_spectrum(v).sum(axis=1), total, rtol=1e-12)


def test_power_spectrum_is_invariant_under_rotation_about_the_spin_axis():
    """A rotation by alpha about z multiplies s_l^m by exp(-i m alpha) and
    leaves every C_l unchanged."""
    rng = np.random.default_rng(4)
    v = rng.normal(size=(L_SMALL + 1) ** 2)
    ms = np.array([m for _, m in lm_indices(L_SMALL)])
    rotated = coeffs_to_real(real_to_coeffs(v) * np.exp(-1j * ms * 0.7))
    diff = np.max(np.abs(power_spectrum(v) - power_spectrum(rotated)))
    print(f"max difference {diff:.3e}")
    assert diff < 1e-12


def test_power_spectrum_accepts_an_explicit_degree():
    v = np.zeros((2, (L_SMALL + 1) ** 2))
    assert power_spectrum(v, L=L_SMALL).shape == (2, L_SMALL + 1)


############################
# Posterior mean           #
############################

def test_posterior_mean_averages_the_draw_axis():
    rng = np.random.default_rng(5)
    draws = rng.normal(size=(3, 7, 11))
    out = posterior_mean(draws)
    assert out.shape == (3, 11)
    assert np.allclose(out, draws.mean(axis=1))


############################
# Sampling                 #
############################

def test_sample_draws_shape_and_dtype():
    B, n_draws = 3, 5
    draws = sample_draws(make_signals(B, 2), np.full(B, 45.0), 'phot_ax',
                         StubVAE(N_COEFFS), make_stats(N_COEFFS),
                         StubEstimator(), make_meta('phot_ax'),
                         n_draws=n_draws, batch_size=2, stored=False)
    assert draws.shape == (B, n_draws, N_COEFFS)
    assert draws.dtype == np.float32


@pytest.mark.parametrize('constant', [0.0, 1.0])
def test_the_standardisation_is_inverted(constant):
    """A constant latent decodes to a known vector, so the output is
    mu + constant std over the leading entries and mu elsewhere."""
    B, n_draws = 3, 5
    stats = make_stats(N_COEFFS)
    draws = sample_draws(make_signals(B, 1), np.full(B, 30.0), 'phot',
                         StubVAE(N_COEFFS), stats,
                         StubEstimator(mode='constant', constant=constant),
                         make_meta('phot'), n_draws=n_draws, stored=False)
    expected = stats['mu_data'].copy()
    expected[:LATENT_DIM] += constant * stats['std_data'][:LATENT_DIM]
    err = np.max(np.abs(draws - expected))
    print(f"constant {constant}: max error {err:.3e}")
    assert err < 1e-5


def test_the_draw_axis_and_the_batch_axis_are_not_exchanged():
    """The estimator returns j + 100 d for surface j and draw d, which recovers
    as itself after un-standardisation."""
    B, n_draws = 3, 5
    stats = make_stats(N_COEFFS)
    draws = sample_draws(make_signals(B, 1), np.full(B, 30.0), 'phot',
                         StubVAE(N_COEFFS), stats,
                         StubEstimator(mode='index'), make_meta('phot'),
                         n_draws=n_draws, stored=False)
    z = ((draws[:, :, :LATENT_DIM] - stats['mu_data'][:LATENT_DIM])
         / stats['std_data'][:LATENT_DIM])
    want = np.arange(B).reshape(B, 1) + 100.0 * np.arange(n_draws)
    err = np.max(np.abs(z - want[:, :, None]))
    print(f"max error {err:.3e}")
    assert err < 1e-3


def test_a_transposed_estimator_return_raises():
    """sbi 0.26.1 returns (n_draws, b, d). A (b, n_draws, d) return would
    associate draws with the wrong surfaces, so it is rejected rather than
    corrected."""
    B, n_draws = 3, 5
    with pytest.raises(ValueError):
        sample_draws(make_signals(B, 1), np.full(B, 30.0), 'phot',
                     StubVAE(N_COEFFS), make_stats(N_COEFFS),
                     StubEstimator(mode='index', batch_first=True),
                     make_meta('phot'), n_draws=n_draws, stored=False)


def test_the_dc_coefficient_is_reinstated_when_the_decoder_omits_it():
    B, n_draws = 2, 4
    stats = make_stats(N_COEFFS - 1, include_dc=False, dc_value=3.5449077)
    draws = sample_draws(make_signals(B, 1), np.full(B, 30.0), 'phot',
                         StubVAE(N_COEFFS - 1), stats,
                         StubEstimator(mode='constant', constant=0.0),
                         make_meta('phot'), n_draws=n_draws, stored=False)
    assert draws.shape == (B, n_draws, N_COEFFS)
    assert np.allclose(draws[:, :, 0], stats['dc_value'])
    assert np.allclose(draws[:, :, 1:], stats['mu_data'], atol=1e-5)


def test_a_missing_dc_value_raises():
    """A checkpoint trained without the DC coefficient and carrying no value
    for it cannot be reassembled."""
    stats = make_stats(N_COEFFS - 1, include_dc=False, dc_value=None)
    with pytest.raises(ValueError):
        sample_draws(make_signals(2, 1), np.full(2, 30.0), 'phot',
                     StubVAE(N_COEFFS - 1), stats,
                     StubEstimator(mode='constant'), make_meta('phot'),
                     n_draws=2, stored=False)


def test_an_inclination_count_mismatch_raises():
    with pytest.raises(ValueError):
        sample_draws(make_signals(4, 1), np.full(3, 45.0), 'phot',
                     StubVAE(N_COEFFS), make_stats(N_COEFFS),
                     StubEstimator(), make_meta('phot'), n_draws=2,
                     stored=False)


def test_repeated_calls_with_the_same_seed_agree():
    args = (make_signals(4, 2), np.full(4, 55.0), 'phot_ax',
            StubVAE(N_COEFFS), make_stats(N_COEFFS), StubEstimator(),
            make_meta('phot_ax'))
    a = sample_draws(*args, n_draws=3, seed=11, batch_size=2, stored=False)
    b = sample_draws(*args, n_draws=3, seed=11, batch_size=2, stored=False)
    assert np.array_equal(a, b)


def test_a_resumed_run_reproduces_an_uninterrupted_one():
    """The generator is seeded with seed + lo, where lo is the row offset within
    the call. A resumed call passing the matching seed offset and the same batch
    size agrees element by element."""
    sig, betas = make_signals(8, 2), np.linspace(10, 80, 8)
    common = ('phot_ax', StubVAE(N_COEFFS), make_stats(N_COEFFS),
              StubEstimator(), make_meta('phot_ax'))
    whole = sample_draws(sig, betas, *common, n_draws=3, seed=0, batch_size=4,
                         stored=False)
    first = sample_draws(sig[:4], betas[:4], *common, n_draws=3, seed=0,
                         batch_size=4, stored=False)
    second = sample_draws(sig[4:], betas[4:], *common, n_draws=3, seed=4,
                          batch_size=4, stored=False)
    err = np.max(np.abs(whole - np.concatenate([first, second])))
    print(f"max difference {err:.3e}")
    assert err == 0.0


def test_the_batch_size_changes_the_noise_realisation():
    """The noise for a chunk is drawn in one call of shape (b, n_ch, T), which
    is not the concatenation of two half-sized draws from the same seed, so two
    batch sizes give different draws from the same signals."""
    sig, betas = make_signals(8, 2), np.linspace(10, 80, 8)
    common = ('phot_ax', StubVAE(N_COEFFS), make_stats(N_COEFFS),
              StubEstimator(), make_meta('phot_ax'))
    a = sample_draws(sig, betas, *common, n_draws=3, seed=0, batch_size=4,
                     stored=False)
    b = sample_draws(sig, betas, *common, n_draws=3, seed=0, batch_size=8,
                     stored=False)
    assert not np.allclose(a, b)


@pytest.mark.parametrize('family', ['phot', 'phot_ax', 'phot_ay', 'phot_axay'])
def test_a_stored_signal_reaches_the_flow_in_model_order(family):
    """Passing the stored three channels gives the same draws as passing the
    selected channels with stored=False, for every family including the
    three-channel one."""
    B = 3
    stored = make_stored_signals(B, seed=9)
    common = (family, StubVAE(N_COEFFS), make_stats(N_COEFFS),
              StubEstimator(), make_meta(family))
    a = sample_draws(stored, np.full(B, 45.0), *common, n_draws=3, seed=0,
                     stored=True)
    b = sample_draws(select_channels(stored, family), np.full(B, 45.0), *common,
                     n_draws=3, seed=0, stored=False)
    assert np.array_equal(a, b)


def test_the_stored_flag_changes_the_result_for_the_three_channel_family():
    """The two interpretations of a three-channel array are distinguishable, so
    the flag is not decorative."""
    B = 3
    stored = make_stored_signals(B, seed=10)
    common = ('phot_axay', StubVAE(N_COEFFS), make_stats(N_COEFFS),
              StubEstimator(), make_meta('phot_axay'))
    a = sample_draws(stored, np.full(B, 45.0), *common, n_draws=3, seed=0,
                     stored=True)
    b = sample_draws(stored, np.full(B, 45.0), *common, n_draws=3, seed=0,
                     stored=False)
    assert not np.allclose(a, b)


def test_the_context_matches_a_direct_call_to_make_context():
    """sample_draws builds the same conditioning vector as models.make_context
    given the same generator seed."""
    B, n_draws = 3, 4
    sig, betas = make_signals(B, 2), np.array([10.0, 45.0, 80.0])
    meta, stats = make_meta('phot_ax'), make_stats(N_COEFFS)
    draws = sample_draws(sig, betas, 'phot_ax', StubVAE(N_COEFFS), stats,
                         StubEstimator(), meta, n_draws=n_draws, seed=0,
                         batch_size=B, stored=False)

    gen = torch.Generator().manual_seed(0)
    ctx = make_context(torch.tensor(sig, dtype=torch.float32),
                       torch.tensor(beta_norm_from_deg(betas),
                                    dtype=torch.float32),
                       torch.full((B,), LOG10_SIGMA_PHOT_MISSION),
                       torch.full((B,), LOG10_SIGMA_ASTRO_MISSION),
                       meta['ch'], gain_phot=meta['gain_phot'],
                       gain_ast=meta['gain_ast'], gen=gen,
                       log10_sigma_phot=meta['log10_sigma_phot'],
                       log10_sigma_astro=meta['log10_sigma_astro'])
    z = ((draws[:, 0, :LATENT_DIM] - stats['mu_data'][:LATENT_DIM])
         / stats['std_data'][:LATENT_DIM])
    ref = ctx[:, :LATENT_DIM].numpy()
    scale = float(np.max(np.abs(ref)))
    err = np.max(np.abs(z - ref)) / scale
    print(f"relative error {err:.3e}, context scale {scale:.3g}")
    assert err < 1e-4


############################
# Wrappers                 #
############################

def test_reconstruct_renders_the_posterior_mean():
    B = 2
    args = (make_signals(B, 1), np.full(B, 30.0), 'phot', StubVAE(N_COEFFS),
            make_stats(N_COEFFS), StubEstimator(), make_meta('phot'))
    draws = sample_draws(*args, n_draws=4, seed=2, stored=False)
    direct = render_draws(posterior_mean(draws), N_THETA_SMALL, N_PHI_SMALL)
    via = reconstruct(*args, n_theta=N_THETA_SMALL, n_phi=N_PHI_SMALL,
                      n_draws=4, seed=2, stored=False)
    assert np.array_equal(direct, via)


def test_decode_ceiling_returns_one_map_per_surface():
    B = 3
    rng = np.random.default_rng(6)
    v = rng.normal(size=(B, N_COEFFS))
    coeffs = np.stack([real_to_coeffs(u) for u in v])
    out = decode_ceiling(coeffs, StubVAE(N_COEFFS), make_stats(N_COEFFS),
                         n_theta=N_THETA_SMALL, n_phi=N_PHI_SMALL,
                         batch_size=2)
    assert out.shape == (B, N_THETA_SMALL, N_PHI_SMALL)
    assert np.isfinite(out).all()


def test_decode_ceiling_is_deterministic():
    rng = np.random.default_rng(6)
    coeffs = np.stack([real_to_coeffs(u)
                       for u in rng.normal(size=(2, N_COEFFS))])
    args = (coeffs, StubVAE(N_COEFFS), make_stats(N_COEFFS))
    a = decode_ceiling(*args, n_theta=N_THETA_SMALL, n_phi=N_PHI_SMALL)
    b = decode_ceiling(*args, n_theta=N_THETA_SMALL, n_phi=N_PHI_SMALL)
    assert np.array_equal(a, b)


############################
# Released checkpoints     #
############################

@needs_weights
@pytest.mark.parametrize('family', ['phot', 'phot_ax', 'phot_ay', 'phot_axay'])
def test_sample_draws_with_the_released_models(family):
    vae, _, stats = load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')
    est, meta = load_flow(WEIGHTS / f'flow_{family}{SUFFIX}.pt',
                          latent_dim=LATENT_DIM)
    B, n_draws = 3, 8
    draws = sample_draws(make_stored_signals(B, meta['T']),
                         np.array([10.0, 45.0, 80.0]), family, vae, stats,
                         est, meta, n_draws=n_draws, seed=0, batch_size=2)
    spread = float(draws.std(axis=1).mean())
    print(f"{family}: draws {draws.shape}, mean spread over draws {spread:.4g}")
    assert draws.shape == (B, n_draws, N_COEFFS)
    assert draws.dtype == np.float32
    assert np.isfinite(draws).all()
    assert spread > 0


@needs_weights
def test_the_released_posterior_mean_renders_to_a_finite_map():
    vae, _, stats = load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')
    est, meta = load_flow(WEIGHTS / f'flow_phot_axay{SUFFIX}.pt',
                          latent_dim=LATENT_DIM)
    B = 2
    draws = sample_draws(make_stored_signals(B, meta['T']), np.full(B, 45.0),
                         'phot_axay', vae, stats, est, meta, n_draws=4, seed=0)
    img = render_draws(posterior_mean(draws), N_THETA_SMALL, N_PHI_SMALL)
    print(f"intensity range {img.min():.4f} to {img.max():.4f}")
    assert img.shape == (B, N_THETA_SMALL, N_PHI_SMALL)
    assert np.isfinite(img).all()


@needs_weights
def test_the_released_draws_carry_power_at_every_degree():
    vae, _, stats = load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')
    est, meta = load_flow(WEIGHTS / f'flow_phot_axay{SUFFIX}.pt',
                          latent_dim=LATENT_DIM)
    draws = sample_draws(make_stored_signals(2, meta['T']), np.full(2, 45.0),
                         'phot_axay', vae, stats, est, meta, n_draws=4, seed=0)
    C = power_spectrum(draws.reshape(-1, N_COEFFS))
    print(f"C_0 {C[:, 0].mean():.4g}, C_1 {C[:, 1].mean():.4g}, "
          f"C_30 {C[:, 30].mean():.4g}")
    assert C.shape == (8, 31)
    assert np.all(C[:, 1:] > 0)


@needs_weights
def test_the_full_resolution_render_has_the_grid_the_module_declares():
    vae, _, stats = load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')
    est, meta = load_flow(WEIGHTS / f'flow_phot{SUFFIX}.pt',
                          latent_dim=LATENT_DIM)
    draws = sample_draws(make_stored_signals(1, meta['T']), np.array([45.0]),
                         'phot', vae, stats, est, meta, n_draws=2, seed=0)
    img = render_draws(posterior_mean(draws))
    assert img.shape == (1, N_THETA, N_PHI)
    assert np.isfinite(img).all()