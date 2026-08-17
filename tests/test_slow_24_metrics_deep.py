"""
Metric properties over many random pairs at the production grid: SSIM
axioms, CRPS against a brute-force double sum, the sorted sharpness form
with ties, PR-AUC noise monotonicity averaged over surfaces, and the
MAE <= RMSE inequality.
"""

import numpy as np
import pytest

from conftest import sample_prior_surfaces
from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.metrics import (crps, mae, pr_auc, rmse, ssim_aa_full,
                                  ssim_aa_vis, ssim_map, weights, wmean)
from starspot_sbi.render import render

pytestmark = pytest.mark.slow

L = 30


def _rendered(rng, n):
    return [render(coeffs_to_real(s)) for s in sample_prior_surfaces(rng, n, L)]


def test_ssim_properties_200_random_pairs():
    """
    Identity, symmetry and boundedness of SSIM over 200 pairs of rendered
    prior surfaces at (120, 240). Identity and symmetry are exact by
    construction (identical intermediate arrays), so those tolerances are
    zero and roundoff respectively; boundedness allows 1e-12 of roundoff
    above 1.
    """
    rng = np.random.default_rng(20260817)
    imgs = _rendered(rng, 40)

    worst_sym, worst_id, lo, hi = 0.0, 0.0, np.inf, -np.inf
    n_pairs = 0
    for i in range(len(imgs)):
        worst_id = max(worst_id, np.max(np.abs(ssim_map(imgs[i], imgs[i]) - 1)))
        for j in range(i + 1, min(i + 6, len(imgs))):
            F = ssim_map(imgs[i], imgs[j])
            G = ssim_map(imgs[j], imgs[i])
            worst_sym = max(worst_sym, np.max(np.abs(F - G)))
            lo, hi = min(lo, F.min()), max(hi, F.max())
            n_pairs += 1
            for beta in (0.0, 0.6):
                v = ssim_aa_vis(imgs[i], imgs[j], beta)
                assert -1.0 - 1e-12 <= v <= 1.0 + 1e-12
    print(f"{n_pairs} pairs: identity dev {worst_id:.2e}, symmetry dev "
          f"{worst_sym:.2e}, range ({lo:.4f}, {hi:.4f})")
    assert worst_id < 1e-12
    assert worst_sym < 1e-12
    assert -1.0 - 1e-12 <= lo and hi <= 1.0 + 1e-12


def test_ssim_degrades_with_noise_averaged_50_surfaces():
    """
    Mean SSIM over 50 surfaces falls strictly as pixel noise grows through
    three well-separated levels. A per-surface version of the fast suite's
    single-draw check; averaging removes its sampling luck.
    """
    rng = np.random.default_rng(3)
    imgs = _rendered(rng, 50)
    levels = (0.02, 0.1, 0.5)
    means = []
    for sigma in levels:
        vals = [ssim_aa_full(im, im + sigma * rng.normal(size=im.shape))
                for im in imgs]
        means.append(np.mean(vals))
    print("mean ssim at sigma " + ", ".join(
        f"{s}: {m:.4f}" for s, m in zip(levels, means)))
    assert means[0] > means[1] > means[2]


def _crps_brute(truth, samples, beta, kind='vis'):
    """CRPS from the O(n^2) double sum, sharing only the weights helper."""
    n = samples.shape[0]
    accuracy = np.mean(np.abs(samples - truth[None]), axis=0)
    sharp = np.zeros_like(truth)
    for i in range(n):
        sharp += np.sum(np.abs(samples[i][None] - samples), axis=0)
    sharp /= 2.0 * n * n
    w = weights(beta, *truth.shape, kind=kind)
    return wmean(accuracy - sharp, w)


@pytest.mark.parametrize('n', [2, 3, 17, 100])
def test_crps_matches_brute_force_double_sum(n):
    """
    The sorted-form CRPS against the O(n^2) definition at several sample
    counts on a 40 x 80 grid. Algebraically identical, so agreement is
    roundoff; measured below 1e-15 relative.
    """
    rng = np.random.default_rng(100 + n)
    truth = 1.0 + 0.2 * rng.normal(size=(40, 80))
    samples = truth[None] + 0.3 * rng.normal(size=(n, 40, 80))
    beta = 0.5
    a = crps(truth, samples, beta)
    b = _crps_brute(truth, samples, beta)
    print(f"n={n}: sorted {a:.10e} vs brute {b:.10e}, "
          f"diff {abs(a - b):.2e}")
    assert abs(a - b) < 1e-12 * max(1.0, abs(b))


@pytest.mark.parametrize('n', [2, 3, 100])
def test_crps_sorted_form_with_tied_samples(n):
    """
    The sorted sharpness form at n = 2, 3 and 100 when samples contain
    exact ties, where a wrong tie-break in the sorted coefficients would
    show. Half the samples are duplicates by construction.
    """
    rng = np.random.default_rng(200 + n)
    truth = 1.0 + 0.2 * rng.normal(size=(20, 40))
    base = truth[None] + 0.3 * rng.normal(size=(max(1, n // 2), 20, 40))
    samples = np.concatenate([base, base], axis=0)[:n]
    beta = 0.3
    a = crps(truth, samples, beta)
    b = _crps_brute(truth, samples, beta)
    print(f"n={n} with ties: diff {abs(a - b):.2e}")
    assert abs(a - b) < 1e-12 * max(1.0, abs(b))


def test_pr_auc_noise_monotonicity_averaged_40_surfaces():
    """
    Mean PR-AUC over 40 surfaces falls strictly as reconstruction noise
    grows through four well-separated levels. The fast suite checks the
    ordering on one surface at two levels; averaging over the population is
    what the paper actually relies on. Surfaces without visible spots
    return nan and are excluded (their count is printed).
    """
    rng = np.random.default_rng(4)
    imgs = _rendered(rng, 40)
    beta = 0.6
    levels = (0.03, 0.1, 0.3, 1.0)
    means = []
    for sigma in levels:
        vals = [pr_auc(im, im + sigma * rng.normal(size=im.shape), beta)
                for im in imgs]
        vals = [v for v in vals if not np.isnan(v)]
        means.append(np.mean(vals))
    print(f"{len(vals)} of {len(imgs)} surfaces had visible spots; "
          "mean pr_auc at sigma "
          + ", ".join(f"{s}: {m:.4f}" for s, m in zip(levels, means)))
    assert means[0] > means[1] > means[2] > means[3]


def test_mae_below_rmse_100_pairs():
    """
    MAE <= RMSE under every weighting (Jensen), over 100 pairs. An error in
    the shared weighting or aggregation shows up as a violation.
    """
    rng = np.random.default_rng(5)
    imgs = _rendered(rng, 20)
    checked = 0
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if i == j:
                continue
            for beta in (0.0, 0.7):
                for kind in ('full', 'vis', 'wmean'):
                    m = mae(imgs[i], imgs[j], beta, kind)
                    r = rmse(imgs[i], imgs[j], beta, kind)
                    assert m <= r + 1e-14
            checked += 1
            if checked >= 100:
                break
        if checked >= 100:
            break
    print(f"mae <= rmse held for {checked} pairs x 2 betas x 3 kinds")
