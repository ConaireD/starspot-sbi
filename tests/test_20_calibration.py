"""
Tests for starspot_sbi.calibration and pipeline.sample_latents.

Assumptions these tests cannot check:
- The ecdf_band simulation and the width inversion reproduce the legacy
  cell's constructions; they are asserted against distributional facts
  (uniform ranks under a matched sampler, the c = 1 fixed point), not
  against the legacy npz caches.
- sample_latents is asserted equal to sample_draws through the stub
  decoder and through the released checkpoint; equality relies on both
  functions seeding torch.manual_seed(seed + lo) immediately before
  est.sample, which is a convention, not an API.

The stubs reproduce the contracts of test_12_pipeline.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from starspot_sbi.calibration import (sbc_ranks, ecdf_band,
                                      width_from_rank_variance,
                                      tarp_coverage, pit_pixels)
from starspot_sbi.models import FAMILIES, GAIN_PHOT, GAIN_AST, load_vae, load_flow
from starspot_sbi.pipeline import sample_draws, sample_latents, decode_latents

WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'
SUFFIX = '_temp'
T_SIGNAL = 216
LATENT = 8

needs_weights = pytest.mark.skipif(
    not (WEIGHTS / f'flow_phot_axay{SUFFIX}.pt').exists(),
    reason=f'released checkpoints not present in {WEIGHTS}')


class StubVAE:
    def __init__(self, n_out, latent_dim=LATENT):
        self.n_out, self.latent_dim = n_out, latent_dim

    def decoder(self, z):
        out = torch.zeros(z.shape[0], self.n_out, dtype=torch.float32)
        out[:, :self.latent_dim] = z[:, :self.latent_dim]
        return out


class StubEstimator:
    """Draws are standard normal from the global torch state, so two calls
    after the same torch.manual_seed agree."""

    def __init__(self, latent_dim=LATENT):
        self.latent_dim = latent_dim

    def sample(self, shape, ctx):
        return torch.randn(int(shape[0]), ctx.shape[0], self.latent_dim)


def make_meta(family):
    return {'ch': FAMILIES[family], 'T': T_SIGNAL,
            'gain_phot': GAIN_PHOT, 'gain_ast': GAIN_AST}


def make_stats(n, seed=7):
    rng = np.random.default_rng(seed)
    return {'mu_data': rng.normal(size=n).astype(np.float32),
            'std_data': (0.5 + rng.random(n)).astype(np.float32),
            'include_dc': True, 'dc_value': None}


def make_stored_signals(B, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.normal(scale=1e-3, size=(B, 3, T_SIGNAL))
    sig[:, 2] += 1.0
    return sig.astype(np.float32)


def test_sbc_ranks_are_uniform_for_a_matched_sampler():
    rng = np.random.default_rng(0)
    N, L, d = 400, 64, 3
    z_true = rng.standard_normal((N, d))
    z_draws = rng.standard_normal((N, L, d))
    r = sbc_ranks(z_true, z_draws)
    v = (r / L).var(axis=0)
    print(f'rank variance {v} against 1/12 = {1 / 12:.4f}')
    assert r.shape == (N, d)
    assert np.all(np.abs(v - 1 / 12) < 0.02)


def test_the_width_inversion_fixes_the_calibrated_point():
    c, cov = width_from_rank_variance(1.0 / 12.0)
    print(f'c_hat {c:.4f}, coverage {cov:.4f}')
    assert abs(c - 1.0) < 0.02
    assert abs(cov - 0.683) < 0.01
    c2, cov2 = width_from_rank_variance(0.0867)
    print(f'at the paper variance 0.0867: c {c2:.3f}, coverage {cov2:.3f}')
    assert c2 > 1.0 and cov2 < 0.683


def test_ecdf_band_shrinks_with_sample_size():
    _, b1 = ecdf_band(200, 64, n_sim=400)
    _, b2 = ecdf_band(2000, 64, n_sim=400)
    print(f'band {b1:.4f} at N=200, {b2:.4f} at N=2000')
    assert b2 < b1


def test_tarp_is_calibrated_for_a_matched_sampler():
    rng = np.random.default_rng(1)
    N, L, d = 500, 128, 4
    z_true = rng.standard_normal((N, d))
    z_draws = rng.standard_normal((N, L, d))
    refs = rng.standard_normal((N, d))
    levels, cov, ci = tarp_coverage(z_draws, z_true, refs, n_boot=100)
    dev = np.abs(cov - levels).max()
    print(f'max coverage deviation {dev:.4f}')
    assert dev < 0.05


def test_pit_is_uniform_for_a_matched_sampler_and_skewed_for_a_biased_one():
    rng = np.random.default_rng(2)
    truth = rng.standard_normal((24, 48))
    draws = rng.standard_normal((256, 24, 48))
    p = pit_pixels(draws, truth, rng)
    print(f'pit mean {p.mean():.4f}')
    assert abs(p.mean() - 0.5) < 0.02
    p_hi = pit_pixels(draws + 1.0, truth, rng)
    assert p_hi.mean() < 0.25


def test_pit_breaks_ties_randomly():
    rng = np.random.default_rng(3)
    truth = np.zeros((8, 8))
    draws = np.zeros((64, 8, 8))
    p = pit_pixels(draws, truth, rng)
    print(f'tied pit mean {p.mean():.3f}, spread {p.std():.3f}')
    assert 0.3 < p.mean() < 0.7
    assert p.std() > 0.1


def test_sample_latents_matches_sample_draws_through_the_stub():
    family = 'phot_axay'
    n = (30 + 1) ** 2
    sig = make_stored_signals(5)
    betas = np.array([10.0, 30.0, 45.0, 60.0, 80.0])
    vae, stats = StubVAE(n), make_stats(n)
    est, meta = StubEstimator(), make_meta(family)
    z = sample_latents(sig, betas, family, est, meta, n_draws=7, seed=3,
                       batch_size=2)
    d = sample_draws(sig, betas, family, vae, stats, est, meta, n_draws=7,
                     seed=3, batch_size=2)
    dec = decode_latents(z.reshape(-1, LATENT), vae, stats).reshape(d.shape)
    err = np.abs(dec - d).max()
    print(f'stub latent-vs-draw error {err:.2e}')
    assert z.shape == (5, 7, LATENT)
    assert err < 1e-6


@needs_weights
def test_sample_latents_matches_sample_draws_through_the_checkpoint():
    family = 'phot_axay'
    sig = make_stored_signals(3, seed=5)
    betas = np.array([20.0, 45.0, 70.0])
    vae, _, stats = load_vae(str(WEIGHTS / f'vae_n640000_seed101{SUFFIX}.pt'))
    est, meta = load_flow(str(WEIGHTS / f'flow_phot_axay{SUFFIX}.pt'),
                          latent_dim=96)
    z = sample_latents(sig, betas, family, est, meta, n_draws=8, seed=11,
                       batch_size=2)
    d = sample_draws(sig, betas, family, vae, stats, est, meta, n_draws=8,
                     seed=11, batch_size=2)
    dec = decode_latents(z.reshape(-1, 96), vae, stats).reshape(d.shape)
    err = np.abs(dec - d).max()
    scale = np.abs(d).max()
    print(f'checkpoint latent-vs-draw error {err:.2e} against scale {scale:.2e}')
    assert err < 1e-5 * max(scale, 1.0)
