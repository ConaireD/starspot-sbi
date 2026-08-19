"""
Tests for encode_latents, decode_latents and decode_ceiling_coeffs.

Assumptions these tests cannot check:
- vae.latent returns (draw, mu, log_var) in that order, with sigma =
  exp(log_var / 2). The stub reproduces this contract; a checkpoint whose
  latent head orders its returns differently would pass the stub tier and
  produce wrong sigmas in production.
- stats['mu_data'] and stats['std_data'] are the standardisation the
  checkpoint was trained with. The tests verify the algebra of applying
  them, not their values.
- decode_ceiling_coeffs duplicates the encode-decode loop rather than
  composing encode_latents and decode_latents, so that a stub latent head
  returning log_var = None (as test_12's StubVAE does) still works through
  decode_ceiling. The equivalence of the two paths is asserted here for a
  stub whose log_var is real, and holds for the released checkpoint by the
  same algebra, but is not asserted per checkpoint.

The checkpoint tier loads the released weights and skips when weights/ is
absent, matching test_12.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.models import load_vae
from starspot_sbi.pipeline import (encode_latents, decode_latents,
                                   decode_ceiling_coeffs, decode_ceiling,
                                   render_draws)
from starspot_sbi.surfaces import generate_spotted_surface

WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'
SUFFIX = '_temp'

L_SMALL = 6
N_SMALL = (L_SMALL + 1) ** 2
LATENT_SMALL = 8
LOG_VAR = -1.5

needs_weights = pytest.mark.skipif(
    not (WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt').exists(),
    reason=f'released checkpoints not present in {WEIGHTS}')


class StubVAE:
    """
    Encoder is the identity, the latent mean is the leading entries of the
    input, log_var is a constant, and the decoder embeds the latent into the
    leading coefficients and zeroes the rest.
    """

    def __init__(self, n_out, latent_dim=LATENT_SMALL, log_var=LOG_VAR):
        self.n_out = n_out
        self.latent_dim = latent_dim
        self.log_var = log_var

    def encoder(self, x):
        return x

    def latent(self, h):
        mu = h[:, :self.latent_dim]
        lv = torch.full_like(mu, self.log_var)
        return mu, mu, lv

    def decoder(self, z):
        out = torch.zeros(z.shape[0], self.n_out, dtype=torch.float32)
        out[:, :self.latent_dim] = z[:, :self.latent_dim]
        return out


def make_stats(n, include_dc=True, dc_value=None, seed=7):
    rng = np.random.default_rng(seed)
    return {'mu_data': rng.normal(size=n).astype(np.float32),
            'std_data': (0.5 + rng.random(n)).astype(np.float32),
            'include_dc': include_dc,
            'dc_value': dc_value}


def make_coeffs(B, seed=3):
    """Complex stored-format coefficient vectors from the spot simulator."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        spots = [{'theta': rng.uniform(0.3, 2.8), 'phi': rng.uniform(-3, 3),
                  'radius': rng.uniform(0.1, 0.3),
                  'contrast': rng.uniform(0.5, 0.9)}]
        out.append(generate_spotted_surface(L_SMALL, spots, lanczos=True))
    return np.stack(out)


def test_encode_latents_returns_the_standardised_leading_entries():
    B = 5
    coeffs = make_coeffs(B)
    stats = make_stats(N_SMALL)
    vae = StubVAE(N_SMALL)
    mu, sigma = encode_latents(coeffs, vae, stats, batch_size=2)

    packed = np.stack([coeffs_to_real(c) for c in coeffs])
    expect = ((packed - stats['mu_data']) / stats['std_data'])[:, :LATENT_SMALL]
    err = np.abs(mu - expect).max()
    print(f'encode mu error {err:.2e}, sigma {sigma[0, 0]:.6f} against '
          f'{np.exp(0.5 * LOG_VAR):.6f}')
    assert mu.shape == (B, LATENT_SMALL) and sigma.shape == (B, LATENT_SMALL)
    assert err < 1e-5
    assert np.allclose(sigma, np.exp(0.5 * LOG_VAR), rtol=1e-6)


def test_decode_latents_inverts_the_standardisation():
    B = 4
    rng = np.random.default_rng(0)
    z = rng.normal(size=(B, LATENT_SMALL)).astype(np.float32)
    stats = make_stats(N_SMALL)
    vae = StubVAE(N_SMALL)
    vecs = decode_latents(z, vae, stats, batch_size=3)

    expect = np.tile(stats['mu_data'], (B, 1))
    expect[:, :LATENT_SMALL] += z * stats['std_data'][:LATENT_SMALL]
    err = np.abs(vecs - expect).max()
    print(f'decode error {err:.2e}')
    assert vecs.shape == (B, N_SMALL)
    assert err < 1e-5


def test_decode_latents_reinstates_the_dc_coefficient():
    z = np.zeros((2, LATENT_SMALL), dtype=np.float32)
    stats = make_stats(N_SMALL - 1, include_dc=False, dc_value=3.5449077)
    vae = StubVAE(N_SMALL - 1)
    vecs = decode_latents(z, vae, stats)
    print(f'dc column {vecs[:, 0]}')
    assert vecs.shape == (2, N_SMALL)
    assert np.allclose(vecs[:, 0], 3.5449077)


def test_decode_ceiling_coeffs_composes_encode_and_decode():
    coeffs = make_coeffs(6)
    stats = make_stats(N_SMALL)
    vae = StubVAE(N_SMALL)
    direct = decode_ceiling_coeffs(coeffs, vae, stats, batch_size=4)
    mu, _ = encode_latents(coeffs, vae, stats, batch_size=4)
    composed = decode_latents(mu, vae, stats)
    err = np.abs(direct - composed).max()
    print(f'compose error {err:.2e}')
    assert err < 1e-6


def test_decode_ceiling_renders_decode_ceiling_coeffs():
    coeffs = make_coeffs(3)
    stats = make_stats(N_SMALL)
    vae = StubVAE(N_SMALL)
    imgs = decode_ceiling(coeffs, vae, stats, n_theta=24, n_phi=48)
    expect = render_draws(decode_ceiling_coeffs(coeffs, vae, stats), 24, 48)
    err = np.abs(imgs - expect).max()
    print(f'render path error {err:.2e}')
    assert imgs.shape == (3, 24, 48)
    assert err < 1e-6


@needs_weights
def test_checkpoint_encode_and_decode_shapes():
    vae, _, stats = load_vae(
        str(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt'))
    rng = np.random.default_rng(1)
    out = []
    for _ in range(3):
        spots = [{'theta': rng.uniform(0.3, 2.8), 'phi': rng.uniform(-3, 3),
                  'radius': rng.uniform(0.10, 0.21),
                  'contrast': rng.uniform(0.5, 0.9)}]
        out.append(generate_spotted_surface(30, spots, lanczos=True))
    coeffs = np.stack(out)

    mu, sigma = encode_latents(coeffs, vae, stats)
    vecs = decode_ceiling_coeffs(coeffs, vae, stats)
    imgs = decode_ceiling(coeffs, vae, stats)
    rerendered = render_draws(vecs)
    err = np.abs(imgs - rerendered).max()
    print(f'latent {mu.shape}, sigma range {sigma.min():.4f} to '
          f'{sigma.max():.4f}, coeffs {vecs.shape}, render path err {err:.2e}')
    assert mu.shape[1] == 96 and sigma.shape == mu.shape
    assert (sigma > 0).all()
    assert vecs.shape == (3, 961)
    assert err < 1e-6
