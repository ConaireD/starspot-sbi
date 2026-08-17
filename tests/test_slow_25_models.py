"""
Model consistency on many held-out surfaces. The fast end-to-end test
scores one committed fixture surface; here the same pipeline runs over the
first surfaces of the explicit holdout split, so the assertions are on
distributions rather than one draw. All bounds sit far below the values
measured at the time of writing, quoted per test, so only a genuine
regression trips them.

Needs the released checkpoints and the holdout dataset; skips cleanly
without either. Scoring inclination is the stored slot nearest 60 degrees,
as in the fast end-to-end test.
"""

import csv
from functools import lru_cache

import numpy as np
import pytest
import torch

from conftest import HOLDOUT_DIR, WEIGHTS_DIR
from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.metrics import ssim_aa_vis
from starspot_sbi.models import (beta_norm_from_deg, clf_context, clf_predict,
                                 load_classifier, load_flow, load_vae,
                                 make_context, make_posterior)
from starspot_sbi.render import render, render_normed

pytestmark = pytest.mark.slow

SUFFIX = '_temp'
LATENT_DIM = 96
T_SIGNAL = 216
L10_PHOT, L10_ASTRO = -4.0, -3.5
SEED = 20260817
N_DRAWS = 96

# Stored .npy channel order is (astro_x, astro_y, phot); the models want the
# channels of their family in model order. docs/conventions.md section 9.
STORED = {'phot': [2], 'phot_ax': [2, 0], 'phot_ay': [2, 1],
          'phot_axay': [2, 0, 1]}
SWAPPED = {'phot_ax': [2, 1], 'phot_ay': [2, 0], 'phot_axay': [2, 1, 0]}

needs_weights = pytest.mark.skipif(
    not (WEIGHTS_DIR / f'vae_n640000_seed101{SUFFIX}.pt').exists(),
    reason=f'released checkpoints not present in {WEIGHTS_DIR}')
needs_holdout = pytest.mark.skipif(
    not (HOLDOUT_DIR / 'metadata.csv').exists(),
    reason=f'explicit holdout dataset not present at {HOLDOUT_DIR}')


@lru_cache(maxsize=1)
def _vae():
    return load_vae(WEIGHTS_DIR / f'vae_n640000_seed101{SUFFIX}.pt')


@lru_cache(maxsize=4)
def _flow(family):
    est, meta = load_flow(WEIGHTS_DIR / f'flow_{family}{SUFFIX}.pt',
                          latent_dim=LATENT_DIM)
    return make_posterior(est, LATENT_DIM), meta


@lru_cache(maxsize=1)
def _holdout(n=30):
    """
    The first n holdout surfaces by surface index: rendered truth, stored
    signal at the slot nearest 60 degrees, and that inclination.
    """
    rows = {}
    with open(HOLDOUT_DIR / 'metadata.csv') as f:
        for r in csv.DictReader(f):
            rows[int(r['surface_idx'])] = r
    out = []
    for idx in sorted(rows)[:n]:
        r = rows[idx]
        coeffs = np.load(HOLDOUT_DIR / 'surfaces' / r['surface_file'])
        sig = np.load(HOLDOUT_DIR / 'signals' / r['signal_file'])
        betas = np.array([int(b) for b in r['betas_deg'].split(';')])
        slot = int(np.argmin(np.abs(betas - 60)))
        out.append((idx, coeffs, render(coeffs_to_real(coeffs)),
                    sig[slot], int(betas[slot])))
    return out


def _reconstruct(family, order, idx, sig, beta_deg, n_draws=N_DRAWS):
    """Context -> flow -> posterior-mean latent -> decoder -> render."""
    post, meta = _flow(family)
    vae, _, stats = _vae()
    x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)[:, order, :]
    gen = torch.Generator().manual_seed(SEED + idx)
    ctx = make_context(x, torch.tensor([beta_norm_from_deg(beta_deg)]),
                       torch.tensor([L10_PHOT]), torch.tensor([L10_ASTRO]),
                       meta['ch'], gain_phot=meta['gain_phot'],
                       gain_ast=meta['gain_ast'], gen=gen)
    torch.manual_seed(SEED + idx)
    draws = post.sample((n_draws,), x=ctx[0], show_progress_bars=False)
    with torch.no_grad():
        vec = vae.decoder(draws.mean(0, keepdim=True)).squeeze(0).numpy()
    return render_normed(vec, stats['mu_data'], stats['std_data'],
                         stats['dc_value'], include_dc=stats['include_dc'])


def _scores(family, order, surfaces, n_draws=N_DRAWS):
    out = []
    for idx, _, truth, sig, bdeg in surfaces:
        recon = _reconstruct(family, order, idx, sig, bdeg, n_draws)
        out.append(ssim_aa_vis(truth, recon, np.radians(bdeg)))
    return np.array(out)


@needs_weights
@needs_holdout
def test_flow_reconstruction_distribution_30_surfaces():
    """
    Full inference on 30 held-out surfaces with the phot_axay flow. The
    paper reports a median ssim_aa_vis of about 0.97 over the holdout;
    measured here 0.940 median, 0.853 minimum, over the first 12 at 96
    draws. The bound of 0.85 on the median is loose so that only a genuine
    regression (gains, permutation, standardisation) trips it.
    """
    scores = _scores('phot_axay', STORED['phot_axay'], _holdout(30),
                     n_draws=128)
    q = np.percentile(scores, [0, 25, 50, 75, 100])
    print(f"phot_axay over {scores.size} surfaces: min {q[0]:.4f}, "
          f"q25 {q[1]:.4f}, median {q[2]:.4f}, q75 {q[3]:.4f}, "
          f"max {q[4]:.4f} (paper median ~0.97, bound 0.85)")
    assert np.median(scores) > 0.85


@needs_weights
@needs_holdout
@pytest.mark.parametrize('family', ['phot', 'phot_ax', 'phot_ay'])
def test_all_families_reconstruct(family):
    """
    The remaining three flow families over 12 held-out surfaces each.
    Measured medians at 96 draws: phot 0.924, phot_ax 0.935, phot_ay 0.937.
    The bound of 0.80 per family is loose for the same reason as above.
    """
    scores = _scores(family, STORED[family], _holdout(30)[:12])
    print(f"{family}: median {np.median(scores):.4f}, "
          f"min {scores.min():.4f} (bound 0.80 on the median)")
    assert np.median(scores) > 0.80


@needs_weights
@needs_holdout
def test_astrometric_swaps_degrade_the_reconstruction():
    """
    Both astrometric swaps over 12 surfaces: phot_ax fed astro_y instead of
    astro_x, phot_ay fed astro_x, and phot_axay with the pair exchanged.
    Measured median penalties 0.060, 0.086 and 0.224; bounds 0.015, 0.02
    and 0.08. Without this the suite cannot distinguish a correct channel
    order from a wrong one on more than the single fixture surface.
    """
    surfaces = _holdout(30)[:12]
    bounds = {'phot_ax': 0.015, 'phot_ay': 0.02, 'phot_axay': 0.08}
    for family, bound in bounds.items():
        correct = np.median(_scores(family, STORED[family], surfaces))
        swapped = np.median(_scores(family, SWAPPED[family], surfaces))
        penalty = correct - swapped
        print(f"{family}: correct median {correct:.4f}, swapped "
              f"{swapped:.4f}, penalty {penalty:.4f} (bound {bound})")
        assert penalty > bound


@needs_weights
@needs_holdout
def test_vae_ceiling_distribution_30_surfaces():
    """
    Encode each truth, decode the encoder mean, score: the ceiling any flow
    can reach. Measured median 0.986, minimum 0.969; bound 0.95 on the
    median. A collapse here with the flow tests intact points at the
    standardisation statistics rather than the flows.
    """
    vae, _, stats = _vae()
    scores = []
    for _, coeffs, truth, _, bdeg in _holdout(30):
        normed = (coeffs_to_real(coeffs) - stats['mu_data']) / stats['std_data']
        with torch.no_grad():
            _, z, _ = vae.latent(vae.encoder(
                torch.tensor(normed, dtype=torch.float32).unsqueeze(0)))
            vec = vae.decoder(z).squeeze(0).numpy()
        img = render_normed(vec, stats['mu_data'], stats['std_data'],
                            stats['dc_value'], include_dc=stats['include_dc'])
        scores.append(ssim_aa_vis(truth, img, np.radians(bdeg)))
    scores = np.array(scores)
    print(f"vae ceiling: median {np.median(scores):.4f}, "
          f"min {scores.min():.4f} (bound 0.95 on the median)")
    assert np.median(scores) > 0.95


@needs_weights
@needs_holdout
@pytest.mark.parametrize('family', ['phot', 'phot_ax', 'phot_ay', 'phot_axay'])
def test_classifier_beta_recovery_30_surfaces(family):
    """
    Each classifier over 30 held-out surfaces, batched. Measured median
    absolute errors on the first 12: phot 3, phot_ax 1, phot_ay 1,
    phot_axay 0 degrees (single-surface outliers reach ~38 for phot_ax, so
    the assertion is on the median). Bound 10 degrees.
    """
    clf, meta = load_classifier(WEIGHTS_DIR / f'clf_{family}{SUFFIX}.pt',
                                T=T_SIGNAL)
    surfaces = _holdout(30)
    sigs = np.stack([sig[STORED[family], :] for _, _, _, sig, _ in surfaces])
    truths = np.array([bdeg for *_, bdeg in surfaces])

    x = torch.tensor(sigs, dtype=torch.float32)
    B = x.shape[0]
    gen = torch.Generator().manual_seed(SEED)
    ctx = clf_context(x, torch.full((B,), L10_PHOT),
                      torch.full((B,), L10_ASTRO), meta['ch'],
                      gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'],
                      gen=gen)
    probs = clf_predict(clf, ctx, temperature=meta['temperature']).numpy()
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5)

    err = np.abs(probs.argmax(1) - truths)
    print(f"{family}: median |error| {np.median(err):.1f} deg, "
          f"max {err.max()} deg over {B} surfaces (bound 10 on the median)")
    assert np.median(err) <= 10
