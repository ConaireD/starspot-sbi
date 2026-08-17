"""
End-to-end regression test on one explicit-holdout surface.

The rest of the suite exercises the pipeline on random noise, so it checks
shapes and strict loading but not meaning: a transposed channel order or a
mis-applied standardisation produces finite, plausible numbers and passes.
This test runs the real thing on real data and asserts that the recovered
surface actually resembles the truth.

The fixture is tests/fixtures/e2e_*, one surface the VAE and the flows never
saw, selected by the deterministic rule recorded in e2e_meta.json. The
reference noise level is (log10 sigma_phot, log10 sigma_astro) = (-4.0, -3.5)
for the phot_axay family.

Bounds are set well below the measured values, and each measured value is
printed, so a failure shows how far a number moved rather than only that it
moved. What each bound protects is stated on its test.

Needs the released checkpoints; skips cleanly without them.
"""

###########
# Imports #
###########

# standard
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

# self
from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.render import render, render_normed
from starspot_sbi.metrics import ssim_aa_vis
from starspot_sbi.models import (
    beta_norm_from_deg, clf_context, clf_predict, load_classifier, load_flow,
    load_vae, make_context, make_posterior,
)


#############
# Constants #
#############

WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
SUFFIX = '_temp'
FAMILY = 'phot_axay'

T_SIGNAL = 216
LATENT_DIM = 96
N_DRAWS = 256
SEED = 20260817

# Mission reference noise level, in log10 of the per-sample sigma.
L10_PHOT, L10_ASTRO = -4.0, -3.5

# The stored .npy files hold (astro_x, astro_y, phot); the models want
# (phot, astro_x, astro_y). See docs/conventions.md section 9.
STORED_TO_MODEL = [2, 0, 1]

# Bounds. Each is far below the value a working pipeline gives, so only a
# genuine regression trips them. The measured values at the time of writing are
# quoted on each test.
MIN_CEILING_SSIM = 0.85
MIN_FLOW_SSIM = 0.70
MAX_BETA_ERROR_DEG = 15
MIN_BETA_MASS_5DEG = 0.20
MIN_SWAP_PENALTY = 0.08

needs_weights = pytest.mark.skipif(
    not (WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt').exists(),
    reason=f'released checkpoints not present in {WEIGHTS}')


#####################
# Fixture loading   #
#####################

@lru_cache(maxsize=1)
def _fixture():
    """
    The committed surface, its signals, and the inclination this test scores at.

    Returns (meta, coeffs, signals, slot, beta_deg). The slot is the stored
    inclination nearest 60 degrees, chosen so the visible region is well inside
    the domain rather than at either pole-on or edge-on extreme.
    """
    meta = json.loads((FIXTURES / 'e2e_meta.json').read_text())
    coeffs = np.load(FIXTURES / 'e2e_surface.npy')
    signals = np.load(FIXTURES / 'e2e_signals.npy')
    betas = np.asarray(meta['betas_deg'])
    slot = int(np.argmin(np.abs(betas - 60.0)))
    return meta, coeffs, signals, slot, int(betas[slot])


@lru_cache(maxsize=1)
def _vae():
    """Loaded VAE with its standardisation statistics."""
    return load_vae(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt')


@lru_cache(maxsize=1)
def _flow():
    """Loaded flow wrapped in its sbi posterior, with the checkpoint metadata."""
    est, meta = load_flow(WEIGHTS / f'flow_{FAMILY}{SUFFIX}.pt',
                          latent_dim=LATENT_DIM)
    return make_posterior(est, LATENT_DIM), meta


@lru_cache(maxsize=1)
def _truth():
    """Rendered truth image and the inclination in radians."""
    _, coeffs, _, _, beta_deg = _fixture()
    return render(coeffs_to_real(coeffs)), np.radians(beta_deg)


def _signal(order=STORED_TO_MODEL):
    """
    The stored signal at the scored inclination, as a (1, 3, T) batch permuted
    into the given channel order. The default is the correct permutation.
    """
    _, _, signals, slot, _ = _fixture()
    sig = torch.tensor(signals[slot], dtype=torch.float32).unsqueeze(0)
    return sig[:, order, :]


#####################
# Reconstruction    #
#####################

def _decode_and_render(latent):
    """Decode a (1, latent_dim) latent, un-standardise, and render."""
    vae, _, stats = _vae()
    with torch.no_grad():
        vec = vae.decoder(latent).squeeze(0).numpy()
    return render_normed(vec, stats['mu_data'], stats['std_data'],
                         stats['dc_value'], include_dc=stats['include_dc'])


def _flow_reconstruction(order=STORED_TO_MODEL, seed=SEED):
    """
    Full inference path for one signal: build the context, sample the flow,
    decode the posterior mean latent, render.

    Both stochastic steps are seeded, the noise draw inside make_context by an
    explicit generator and the flow sampler by the global torch seed, so the
    returned image is a deterministic function of (order, seed).
    """
    post, meta = _flow()
    _, _, _, _, beta_deg = _fixture()

    gen = torch.Generator().manual_seed(seed)
    ctx = make_context(_signal(order),
                       torch.tensor([beta_norm_from_deg(beta_deg)]),
                       torch.tensor([L10_PHOT]), torch.tensor([L10_ASTRO]),
                       meta['ch'], gain_phot=meta['gain_phot'],
                       gain_ast=meta['gain_ast'], gen=gen)

    torch.manual_seed(seed)
    draws = post.sample((N_DRAWS,), x=ctx[0], show_progress_bars=False)
    return _decode_and_render(draws.mean(0, keepdim=True))


#####################
# The fixture files #
#####################

def test_fixture_files_are_present_and_self_consistent():
    """The committed fixture is intact and matches what the models expect."""
    meta, coeffs, signals, slot, beta_deg = _fixture()
    assert coeffs.shape == (961,) and np.iscomplexobj(coeffs)
    assert signals.shape == (len(meta['betas_deg']), 3, T_SIGNAL)
    assert meta['channel_order'] == ['astro_x', 'astro_y', 'phot']
    assert 3 <= meta['n_spots'] <= 6
    assert len(meta['spots']) == meta['n_spots']
    assert 0 <= beta_deg <= 90
    print(f"fixture {meta['surface_file']}: {meta['n_spots']} spots, "
          f"scoring at beta = {beta_deg} deg (slot {slot})")


#####################
# The bounds        #
#####################

@needs_weights
def test_decoder_ceiling():
    """
    Encode the truth, take the encoder mean, decode, and score.

    This is the best any flow can do, since every reconstruction passes through
    the same decoder. It protects the standardisation: feeding the VAE raw
    rather than standardised coefficients, or un-standardising with mismatched
    mu_data and std_data, collapses this number while leaving every shape and
    strict-load check intact. Measured 0.9894.
    """
    vae, _, stats = _vae()
    truth, beta_rad = _truth()
    _, coeffs, _, _, _ = _fixture()

    normed = (coeffs_to_real(coeffs) - stats['mu_data']) / stats['std_data']
    with torch.no_grad():
        _, z_mu, _ = vae.latent(
            vae.encoder(torch.tensor(normed, dtype=torch.float32).unsqueeze(0)))

    score = ssim_aa_vis(truth, _decode_and_render(z_mu), beta_rad)
    print(f'decoder ceiling ssim_aa_vis = {score:.6f} '
          f'(bound {MIN_CEILING_SSIM})')
    assert score > MIN_CEILING_SSIM


@needs_weights
def test_flow_reconstruction():
    """
    Full inference on the stored signal: context, flow, decoder, render.

    Protects the whole path end to end, and in particular the gains and the
    signal preprocessing. Passing un-gained signals, or the checkpoint defaults
    where the stored gains differ, presents the frozen standardisation layer
    with inputs hundreds of times too small and the posterior degenerates.
    Measured 0.9536, against a paper median of about 0.97.
    """
    truth, beta_rad = _truth()
    score = ssim_aa_vis(truth, _flow_reconstruction(), beta_rad)
    print(f'flow ssim_aa_vis = {score:.6f} (bound {MIN_FLOW_SSIM})')
    assert score > MIN_FLOW_SSIM


@needs_weights
def test_classifier_recovers_the_inclination():
    """
    Classify the same signal and check the posterior sits on the true beta.

    The classifier sees no beta, so this protects the parts of the signal path
    that carry geometry: the channel permutation, the per-channel
    self-normalisation, and the gains. A scrambled signal still yields a valid
    91-way softmax, just one pointing nowhere near the truth. Measured argmax
    61 against a truth of 59, with 0.9732 of the mass within 5 degrees.
    """
    _, _, _, _, beta_deg = _fixture()
    clf, meta = load_classifier(WEIGHTS / f'clf_{FAMILY}{SUFFIX}.pt', T=T_SIGNAL)

    gen = torch.Generator().manual_seed(SEED)
    ctx = clf_context(_signal(), torch.tensor([L10_PHOT]),
                      torch.tensor([L10_ASTRO]), meta['ch'],
                      gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'],
                      gen=gen)
    probs = clf_predict(clf, ctx, temperature=meta['temperature'])[0].numpy()

    grid = np.arange(probs.size)
    argmax = int(probs.argmax())
    mass = float(probs[np.abs(grid - beta_deg) <= 5].sum())
    print(f'classifier argmax = {argmax} deg, truth = {beta_deg} deg, '
          f'error = {abs(argmax - beta_deg)} deg (bound {MAX_BETA_ERROR_DEG})')
    print(f'classifier mass within 5 deg = {mass:.6f} '
          f'(bound {MIN_BETA_MASS_5DEG})')

    assert abs(argmax - beta_deg) <= MAX_BETA_ERROR_DEG
    assert mass > MIN_BETA_MASS_5DEG


#####################
# Negative control  #
#####################

@needs_weights
def test_swapping_the_astrometric_channels_degrades_the_reconstruction():
    """
    Repeat the flow reconstruction with astro_x and astro_y exchanged.

    Without this the suite cannot distinguish a correct channel order from a
    wrong one, because a wrong order is still finite and still scores something.
    The swap is the failure mode the permutation in section 9 of the conventions
    exists to prevent. Measured 0.9536 correct against 0.7659 swapped, a penalty
    of 0.1877.
    """
    truth, beta_rad = _truth()
    correct = ssim_aa_vis(truth, _flow_reconstruction(), beta_rad)
    swapped = ssim_aa_vis(truth, _flow_reconstruction(order=[2, 1, 0]), beta_rad)
    penalty = correct - swapped
    print(f'correct ssim_aa_vis = {correct:.6f}, '
          f'axay-swapped = {swapped:.6f}, penalty = {penalty:.6f} '
          f'(bound {MIN_SWAP_PENALTY})')
    assert penalty > MIN_SWAP_PENALTY


#####################
# Determinism       #
#####################

@needs_weights
def test_flow_reconstruction_is_reproducible():
    """
    Two seeded runs give the same image to the bit.

    Every stochastic step is seeded, so a drift here means a source of
    randomness has escaped the seeds and the printed numbers above are no
    longer comparable between runs.
    """
    truth, beta_rad = _truth()
    a = _flow_reconstruction()
    b = _flow_reconstruction()
    print(f'repeat ssim_aa_vis = {ssim_aa_vis(truth, a, beta_rad):.10f} '
          f'and {ssim_aa_vis(truth, b, beta_rad):.10f}')
    assert np.array_equal(a, b)
