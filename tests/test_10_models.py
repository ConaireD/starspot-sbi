"""
Tests for models.py.

The architecture and context-builder tests construct models from
scratch and need no data. The checkpoint tests need the nine released files in
weights/ and skip when that directory is absent, so a clone without weights
still runs a meaningful suite.

"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from starspot_sbi.models import (
    L_MAX, N_COEFFS, N_BETA_CLASSES,
    CH_PHOT, CH_AX, CH_AY, FAMILIES,
    LOG10_SIGMA_PHOT, LOG10_SIGMA_ASTRO, GAIN_PHOT, GAIN_AST,
    FLOW, EMB, AUX_DIM,
    ArchConfig, EncoderFlex, DecoderFlex, LatentSpace, VAE,
    AttnSignalEmbeddingAux, BetaClassifier,
    band_index_sets, order_index_sets, input_dim,
    _u, beta_norm_from_deg, beta_deg_from_norm,
    n_aux_for_flow, n_aux_for_clf,
    make_context, clf_context,
    build_flow, make_prior, make_posterior,
    load_vae, load_flow, load_classifier, clf_predict,
)

WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'
SUFFIX = '_temp'                 # marks the checkpoints as pending verification
T_SIGNAL = 216
LATENT_DIM = 96

needs_weights = pytest.mark.skipif(
    not (WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt').exists(),
    reason=f'released checkpoints not present in {WEIGHTS}')


############################
# Index sets and packing   #
############################

@pytest.mark.parametrize('l_min', [0, 1])
def test_band_index_sets_partition_the_vector(l_min):
    """Every entry of the real-packed vector belongs to exactly one degree."""
    sets = band_index_sets(L_MAX, l_min)
    assert len(sets) == L_MAX - l_min + 1
    flat = np.concatenate(sets)
    assert len(flat) == len(np.unique(flat))
    assert sorted(flat) == list(range(input_dim(l_min)))


@pytest.mark.parametrize('l_min', [0, 1])
def test_band_index_set_sizes(l_min):
    """Degree l contributes 2l + 1 entries: one m = 0, l real, l imaginary."""
    for l, s in zip(range(l_min, L_MAX + 1), band_index_sets(L_MAX, l_min)):
        assert len(s) == 2 * l + 1


@pytest.mark.parametrize('l_min', [0, 1])
def test_order_index_sets_partition_the_vector(l_min):
    sets = order_index_sets(L_MAX, l_min)
    flat = np.concatenate(sets)
    assert len(flat) == len(np.unique(flat))
    assert sorted(flat) == list(range(input_dim(l_min)))


def test_input_dim():
    assert input_dim(0) == N_COEFFS == 961
    assert input_dim(1) == N_COEFFS - 1


############################
# VAE architecture         #
############################

@pytest.fixture(scope='module')
def released_config():
    """The architecture of the released checkpoint, from its stored config."""
    return ArchConfig(tokenizer='band', pooling='cls', funnel='none',
                      d_model=128, output_dim=128, latent_dim=96, l_min=0,
                      n_heads=4, layers_per_stage=2, head_hidden_dims=[512],
                      attn_dropout=0.0, head_dropout=0.0)


def test_vae_round_trips_shape(released_config):
    vae = VAE(EncoderFlex(released_config),
              LatentSpace(released_config.output_dim, released_config.latent_dim),
              DecoderFlex(released_config))
    x = torch.randn(4, N_COEFFS)
    recon, mu, log_var = vae(x)
    assert recon.shape == (4, N_COEFFS)
    assert mu.shape == log_var.shape == (4, released_config.latent_dim)


def test_encoder_output_width(released_config):
    enc = EncoderFlex(released_config)
    assert enc(torch.randn(2, N_COEFFS)).shape == (2, released_config.output_dim)


def test_cls_pooling_adds_one_token(released_config):
    """With pooling='cls' the encoder prepends a learned token."""
    enc = EncoderFlex(released_config)
    assert enc.use_cls
    assert enc.cls.shape == (1, 1, released_config.d_model)
    assert enc.tok.n_tokens == L_MAX + 1        # l_min = 0, one token per degree


def test_latent_space_reparameterisation():
    """The draw is mu + sigma * eps, so it is stochastic while mu is not."""
    lat = LatentSpace(64, 16)
    h = torch.randn(8, 64)
    z1, mu1, lv1 = lat(h)
    z2, mu2, lv2 = lat(h)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(lv1, lv2)
    assert not torch.allclose(z1, z2)


def test_log_var_is_clamped():
    lat = LatentSpace(64, 16, log_var_clamp=(-2.0, 2.0))
    _, _, lv = lat(torch.randn(8, 64) * 1e4)
    assert lv.min() >= -2.0 and lv.max() <= 2.0


def test_decoder_is_deterministic(released_config):
    dec = DecoderFlex(released_config)
    dec.eval()
    z = torch.randn(4, released_config.latent_dim)
    with torch.no_grad():
        assert torch.allclose(dec(z), dec(z))


############################
# Conditioning             #
############################

def test_beta_normalisation_round_trip():
    for deg in [0.0, 30.0, 45.0, 90.0]:
        assert beta_deg_from_norm(beta_norm_from_deg(deg)) == pytest.approx(deg)
    assert beta_norm_from_deg(0.0) == pytest.approx(-1.0)
    assert beta_norm_from_deg(90.0) == pytest.approx(1.0)
    assert beta_norm_from_deg(45.0) == pytest.approx(0.0)


def test_sigma_normalisation():
    """u maps the training range to [-1, 1]; sigma = 1e-4 gives +0.2."""
    lo, hi = LOG10_SIGMA_PHOT
    assert _u(lo, lo, hi) == pytest.approx(-1.0)
    assert _u(hi, lo, hi) == pytest.approx(1.0)
    assert _u(-4.0, lo, hi) == pytest.approx(0.2)


@pytest.mark.parametrize('family,n_flow,n_clf', [
    ('phot', 2, 1),
    ('phot_ax', 3, 2),
    ('phot_ay', 3, 2),
    ('phot_axay', 3, 2),
])
def test_auxiliary_counts(family, n_flow, n_clf):
    """The classifier has one fewer auxiliary, since beta is its target."""
    ch = FAMILIES[family]
    assert n_aux_for_flow(ch) == n_flow
    assert n_aux_for_clf(ch) == n_clf


@pytest.mark.parametrize('family,ctx_flow,ctx_clf', [
    ('phot', 218, 217),
    ('phot_ax', 435, 434),
    ('phot_ay', 435, 434),
    ('phot_axay', 651, 650),
])
def test_context_lengths(family, ctx_flow, ctx_clf):
    """Measured against the released checkpoints' standardisation layers."""
    ch = FAMILIES[family]
    B = 3
    sig = torch.randn(B, len(ch), T_SIGNAL) * 0.003 + 1.0
    beta = torch.full((B,), beta_norm_from_deg(45.0))
    lp = torch.full((B,), -4.0)
    la = torch.full((B,), -3.5)

    x_flow = make_context(sig, beta, lp, la, ch)
    x_clf = clf_context(sig, lp, la, ch)
    print(f"{family}: flow {tuple(x_flow.shape)}, classifier {tuple(x_clf.shape)}")
    assert x_flow.shape == (B, ctx_flow)
    assert x_clf.shape == (B, ctx_clf)


def test_context_auxiliary_values():
    """The auxiliaries are beta then the normalised log sigmas, in that order."""
    ch = FAMILIES['phot_axay']
    B = 2
    sig = torch.randn(B, 3, T_SIGNAL) * 0.003 + 1.0
    beta = torch.full((B,), beta_norm_from_deg(30.0))
    lp = torch.full((B,), -4.0)
    la = torch.full((B,), -3.5)

    x = make_context(sig, beta, lp, la, ch)
    aux = x[:, 3 * T_SIGNAL:]
    assert aux.shape == (B, 3)
    assert aux[0, 0] == pytest.approx(beta_norm_from_deg(30.0))
    assert aux[0, 1] == pytest.approx(_u(-4.0, *LOG10_SIGMA_PHOT))
    assert aux[0, 2] == pytest.approx(_u(-3.5, *LOG10_SIGMA_ASTRO))


def test_clf_context_has_no_beta():
    """The classifier's first auxiliary is the photometric sigma, not beta."""
    ch = FAMILIES['phot_axay']
    B = 2
    sig = torch.randn(B, 3, T_SIGNAL) * 0.003 + 1.0
    x = clf_context(sig, torch.full((B,), -4.0), torch.full((B,), -3.5), ch)
    aux = x[:, 3 * T_SIGNAL:]
    assert aux.shape == (B, 2)
    assert aux[0, 0] == pytest.approx(_u(-4.0, *LOG10_SIGMA_PHOT))
    assert aux[0, 1] == pytest.approx(_u(-3.5, *LOG10_SIGMA_ASTRO))


def test_photometric_channel_is_relative_flux():
    """
    The photometric channel becomes (y / mean(y) - 1) * gain, so a noiseless
    constant series maps to zero and the gain sets the scale of the departure.
    """
    ch = [CH_PHOT]
    B = 2
    sig = torch.ones(B, 1, T_SIGNAL)
    x = make_context(sig, torch.zeros(B), torch.full((B,), -12.0),
                     torch.full((B,), -12.0), ch)
    assert torch.max(torch.abs(x[:, :T_SIGNAL])) < 1e-3


def test_astrometric_channel_is_mean_subtracted():
    """An astrometric channel with a constant offset loses it."""
    ch = [CH_PHOT, CH_AX]
    B = 2
    sig = torch.ones(B, 2, T_SIGNAL)
    sig[:, 1] = 5.0
    x = make_context(sig, torch.zeros(B), torch.full((B,), -12.0),
                     torch.full((B,), -12.0), ch)
    ax = x[:, T_SIGNAL:2 * T_SIGNAL]
    assert torch.max(torch.abs(ax)) < 1e-2


def test_gain_scales_the_signal():
    """Doubling the gain doubles the preprocessed signal."""
    ch = [CH_PHOT]
    B = 2
    sig = 1.0 + 0.01 * torch.randn(B, 1, T_SIGNAL)
    kw = dict(l10_p=torch.full((B,), -12.0), l10_a=torch.full((B,), -12.0),
              ch_sel=ch)
    a = make_context(sig, torch.zeros(B), gain_phot=GAIN_PHOT, **kw)
    b = make_context(sig, torch.zeros(B), gain_phot=2 * GAIN_PHOT, **kw)
    assert torch.allclose(b[:, :T_SIGNAL], 2 * a[:, :T_SIGNAL], atol=1e-4)


def test_noise_is_reproducible_with_a_generator():
    ch = FAMILIES['phot_ax']
    B = 2
    sig = 1.0 + 0.01 * torch.randn(B, 2, T_SIGNAL)
    args = (sig, torch.zeros(B), torch.full((B,), -4.0), torch.full((B,), -4.0), ch)
    a = make_context(*args, gen=torch.Generator().manual_seed(0))
    b = make_context(*args, gen=torch.Generator().manual_seed(0))
    c = make_context(*args, gen=torch.Generator().manual_seed(1))
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


############################
# Embeddings               #
############################

@pytest.mark.parametrize('n_channels,n_aux', [(1, 2), (2, 3), (3, 3)])
def test_embedding_output_width(n_channels, n_aux):
    emb = AttnSignalEmbeddingAux(T=T_SIGNAL, n_channels=n_channels, n_aux=n_aux,
                                 **EMB, aux_dim=AUX_DIM)
    x = torch.randn(4, n_channels * T_SIGNAL + n_aux)
    assert emb(x).shape == (4, EMB['embedding_dim'])


def test_embedding_patch_arithmetic():
    """T = 216 over patches of 16 gives 14 patches with 8 samples of padding."""
    emb = AttnSignalEmbeddingAux(T=216, n_channels=2, n_aux=3, **EMB, aux_dim=AUX_DIM)
    assert emb.n_patches == 14
    assert emb.pad == 8
    assert emb.n_tokens == 28
    assert emb.signal_len == 432


def test_classifier_returns_91_logits():
    clf = BetaClassifier(T=T_SIGNAL, n_channels=2, n_aux=2)
    x = torch.randn(4, 2 * T_SIGNAL + 2)
    assert clf(x).shape == (4, N_BETA_CLASSES) == (4, 91)


def test_clf_predict_normalises_and_temperature_softens():
    """A temperature above one flattens the posterior."""
    clf = BetaClassifier(T=T_SIGNAL, n_channels=1, n_aux=1)
    clf.eval()
    x = torch.randn(4, T_SIGNAL + 1)
    p1 = clf_predict(clf, x, temperature=1.0)
    p2 = clf_predict(clf, x, temperature=2.0)
    assert torch.allclose(p1.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(p2.sum(-1), torch.ones(4), atol=1e-6)
    assert p2.max(-1).values.mean() < p1.max(-1).values.mean()


############################
# Flow construction        #
############################

def test_build_flow_accepts_its_context():
    est = build_flow(n_channels=1, n_aux=2, T=T_SIGNAL, latent_dim=LATENT_DIM)
    x = torch.randn(4, T_SIGNAL + 2)
    z = torch.randn(4, LATENT_DIM)
    with torch.no_grad():
        lp = est.log_prob(z, x)
    assert torch.isfinite(lp).all()


def test_make_prior_dimensions():
    prior = make_prior(LATENT_DIM)
    assert prior.sample((5,)).shape == (5, LATENT_DIM)
    assert prior.log_prob(torch.zeros(3, LATENT_DIM)).shape == (3,)


############################
# Released checkpoints     #
############################

@needs_weights
def test_vae_checkpoint_loads_strictly():
    vae, cfg, stats = load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')
    print(f"latent_dim {cfg.latent_dim}, l_min {cfg.l_min}, "
          f"tokenizer {cfg.tokenizer!r}, pooling {cfg.pooling!r}, "
          f"include_dc {stats['include_dc']}, dc_value {stats['dc_value']}")
    assert cfg.latent_dim == LATENT_DIM
    assert cfg.l_min == 0
    assert stats['include_dc'] is True
    assert stats['dc_value'] is None
    assert stats['mu_data'].shape == (N_COEFFS,)
    assert stats['std_data'].shape == (N_COEFFS,)
    assert np.all(stats['std_data'] > 0)


@needs_weights
def test_vae_forward_pass():
    vae, cfg, _ = load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')
    x = torch.randn(4, N_COEFFS)
    with torch.no_grad():
        recon, mu, log_var = vae(x)
    assert recon.shape == (4, N_COEFFS)
    assert mu.shape == (4, LATENT_DIM)
    assert torch.isfinite(recon).all()


@needs_weights
def test_vae_is_frozen():
    vae, _, _ = load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')
    assert not any(p.requires_grad for p in vae.parameters())
    assert not vae.training


@needs_weights
@pytest.mark.parametrize('family,ch,n_aux,ctx', [
    ('phot',      [0],       2, 218),
    ('phot_ax',   [0, 1],    3, 435),
    ('phot_ay',   [0, 2],    3, 435),
    ('phot_axay', [0, 1, 2], 3, 651),
])
def test_flow_checkpoints_load_strictly(family, ch, n_aux, ctx):
    est, meta = load_flow(WEIGHTS / f'flow_{family}{SUFFIX}.pt', latent_dim=LATENT_DIM)
    print(f"{family}: ch {meta['ch']}, n_aux {meta['n_aux']}, T {meta['T']}, "
          f"context {meta['context_dim']}")
    assert meta['ch'] == ch
    assert meta['n_aux'] == n_aux
    assert meta['T'] == T_SIGNAL
    assert meta['context_dim'] == ctx
    assert meta['gain_phot'] == pytest.approx(GAIN_PHOT)
    assert meta['gain_ast'] == pytest.approx(GAIN_AST)
    assert meta['log10_sigma_phot'] == LOG10_SIGMA_PHOT
    assert meta['log10_sigma_astro'] == LOG10_SIGMA_ASTRO


@needs_weights
@pytest.mark.parametrize('family', ['phot', 'phot_ax', 'phot_ay', 'phot_axay'])
def test_flow_forward_pass(family):
    """A context built by make_context is accepted and gives finite densities."""
    est, meta = load_flow(WEIGHTS / f'flow_{family}{SUFFIX}.pt', latent_dim=LATENT_DIM)
    ch, B = meta['ch'], 4

    sig = 1.0 + 0.003 * torch.randn(B, len(ch), meta['T'])
    x = make_context(sig, torch.full((B,), beta_norm_from_deg(45.0)),
                     torch.full((B,), -4.0), torch.full((B,), -3.5), ch,
                     gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'])
    assert x.shape == (B, meta['context_dim'])

    z = torch.randn(B, LATENT_DIM)
    with torch.no_grad():
        lp = est.log_prob(z, x)
        draws = est.sample((3,), x)
    print(f"{family}: log_prob {tuple(lp.shape)}, sample {tuple(draws.shape)}")
    assert torch.isfinite(lp).all()
    assert draws.shape[-1] == LATENT_DIM
    assert torch.isfinite(draws).all()


@needs_weights
@pytest.mark.parametrize('family,ch,n_aux,ctx', [
    ('phot',      [0],       1, 217),
    ('phot_ax',   [0, 1],    2, 434),
    ('phot_ay',   [0, 2],    2, 434),
    ('phot_axay', [0, 1, 2], 2, 650),
])
def test_classifier_checkpoints_load_strictly(family, ch, n_aux, ctx):
    clf, meta = load_classifier(WEIGHTS / f'clf_{family}{SUFFIX}.pt', T=T_SIGNAL)
    print(f"{family}: ch {meta['ch']}, n_aux {meta['n_aux']}, "
          f"context {meta['context_dim']}, temperature {meta['temperature']:.6f}")
    assert meta['ch'] == ch
    assert meta['n_aux'] == n_aux
    assert meta['context_dim'] == ctx
    assert 0.9 < meta['temperature'] < 1.2


@needs_weights
@pytest.mark.parametrize('family', ['phot', 'phot_ax', 'phot_ay', 'phot_axay'])
def test_classifier_forward_pass(family):
    clf, meta = load_classifier(WEIGHTS / f'clf_{family}{SUFFIX}.pt', T=T_SIGNAL)
    ch, B = meta['ch'], 4

    sig = 1.0 + 0.003 * torch.randn(B, len(ch), T_SIGNAL)
    x = clf_context(sig, torch.full((B,), -4.0), torch.full((B,), -3.5), ch,
                    gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'])
    assert x.shape == (B, meta['context_dim'])

    p = clf_predict(clf, x, temperature=meta['temperature'])
    assert p.shape == (B, N_BETA_CLASSES)
    assert torch.allclose(p.sum(-1), torch.ones(B), atol=1e-5)
    assert torch.isfinite(p).all()


@needs_weights
def test_classifier_context_is_one_shorter_than_the_flow():
    """The same family gives a classifier context one entry shorter."""
    for family in ['phot', 'phot_ax', 'phot_ay', 'phot_axay']:
        _, mf = load_flow(WEIGHTS / f'flow_{family}{SUFFIX}.pt', latent_dim=LATENT_DIM)
        _, mc = load_classifier(WEIGHTS / f'clf_{family}{SUFFIX}.pt', T=T_SIGNAL)
        assert mf['ch'] == mc['ch']
        assert mc['context_dim'] == mf['context_dim'] - 1


@needs_weights
def test_posterior_samples_from_a_loaded_flow():
    est, meta = load_flow(WEIGHTS / f'flow_phot_axay{SUFFIX}.pt', latent_dim=LATENT_DIM)
    post = make_posterior(est, LATENT_DIM)

    ch, B = meta['ch'], 1
    sig = 1.0 + 0.003 * torch.randn(B, len(ch), meta['T'])
    x = make_context(sig, torch.full((B,), beta_norm_from_deg(45.0)),
                     torch.full((B,), -4.0), torch.full((B,), -3.5), ch,
                     gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'])

    draws = post.sample((16,), x=x[0], show_progress_bars=False)
    assert draws.shape == (16, LATENT_DIM)
    assert torch.isfinite(draws).all()


@needs_weights
def test_flow_recovers_T_without_being_told():
    """T comes from the context standardisation layer when not supplied."""
    _, auto = load_flow(WEIGHTS / f'flow_phot_ax{SUFFIX}.pt', latent_dim=LATENT_DIM)
    _, told = load_flow(WEIGHTS / f'flow_phot_ax{SUFFIX}.pt', latent_dim=LATENT_DIM,
                        T=T_SIGNAL)
    assert auto['T'] == told['T'] == T_SIGNAL