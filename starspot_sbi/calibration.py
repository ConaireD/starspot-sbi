"""
Calibration diagnostics from stored posterior draws.

Simulation-based calibration ranks (marginal, per coordinate), the
simultaneous ECDF band the ranks are judged against, the width factor the
rank variance inverts to, TARP expected coverage (joint), and the pixel
probability integral transform with randomised tie-breaking. Ported from
the legacy F10 calibration cell (PaperFigures_v7); the constructions are
unchanged, the array interfaces are new.

Latent-space quantities need latent draws, which pipeline.sample_latents
regenerates with the run's seeds; the pixel PIT reads rendered draws
directly.
"""

###########
# Imports #
###########

# standard
import numpy as np
from scipy.stats import norm


#####################
# SBC               #
#####################

def sbc_ranks(z_true, z_draws):
    """
    Rank of each truth among its draws, per coordinate: (N, d) truths
    against (N, L, d) draws, integer ranks in [0, L].
    """
    z_true = np.asarray(z_true)
    z_draws = np.asarray(z_draws)
    return (z_draws < z_true[:, None, :]).sum(axis=1)


def ecdf_band(n, n_draws, alpha=0.05, n_sim=2000, n_grid=201, seed=0):
    """
    Simultaneous (1 - alpha) band for the deviation of a normalised-rank
    ECDF from uniformity, by simulation under the uniform null. Returns
    (grid, band): a per-coordinate ECDF whose maximum deviation exceeds the
    band is discrepant at that level.
    """
    grid = np.linspace(0, 1, n_grid)
    rng = np.random.default_rng(seed)
    dev = np.empty(n_sim)
    for s in range(n_sim):
        u = np.sort(rng.integers(0, n_draws + 1, size=n) / n_draws)
        dev[s] = np.abs(np.searchsorted(u, grid, side='right') / n - grid).max()
    return grid, float(np.quantile(dev, 1 - alpha))


def width_from_rank_variance(rank_var, n_mc=400_000, c_max=1.4, n_c=401,
                             seed=0):
    """
    Invert the normalised-rank variance to an equivalent posterior width.

    For a Gaussian marginal reported too narrow, the normalised rank is
    Phi(c z) with z standard normal and c > 1 the narrowness factor; its
    variance rises above 1/12 with c. Returns (c_hat, coverage_68): the
    factor matching the measured variance, and the mass a nominal 68.3 per
    cent interval then covers. Draft Appendix B8; a biased mean also
    inflates the variance, so c_hat is a lower bound on the width deficit.
    """
    z = np.random.default_rng(seed).standard_normal(n_mc)
    cs = np.linspace(1.0, c_max, n_c)
    vs = np.array([norm.cdf(c * z).var() for c in cs])
    cov = np.array([np.mean(np.abs(c * z) < 1.0) for c in cs])
    c_hat = float(np.interp(rank_var, vs, cs))
    return c_hat, float(np.interp(rank_var, vs, cov))


#####################
# TARP              #
#####################

def tarp_coverage(z_draws, z_true, z_refs, n_boot=400, seed=0):
    """
    TARP expected coverage: for each truth, the fraction of its draws
    closer to a reference point than the truth is, against the credibility
    level. Returns (levels, coverage, ci) with a bootstrap 95 per cent
    band over the surfaces. Calibration is coverage equal to the level for
    every reference distribution; the reference choice is the caller's and
    should be reported with the curve.
    """
    z_draws = np.asarray(z_draws)
    z_true = np.asarray(z_true)
    z_refs = np.asarray(z_refs)
    d_t = np.linalg.norm(z_true - z_refs, axis=1)
    d_s = np.linalg.norm(z_draws - z_refs[:, None, :], axis=2)
    f = (d_s < d_t[:, None]).mean(axis=1)
    levels = np.linspace(0, 1, 101)
    cov = (f[None, :] <= levels[:, None]).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = np.stack([
        (f[rng.integers(0, len(f), len(f))][None, :]
         <= levels[:, None]).mean(axis=1)
        for _ in range(n_boot)])
    return levels, cov, np.quantile(boot, [0.025, 0.975], axis=0)


#####################
# Pixel PIT         #
#####################

def pit_pixels(draw_maps, truth_map, rng):
    """
    Randomised probability integral transform per pixel: the fraction of
    draws below the truth, with ties broken uniformly. draw_maps is
    (L, n_theta, n_phi) against a (n_theta, n_phi) truth; uniform under
    calibration.
    """
    draw_maps = np.asarray(draw_maps)
    below = (draw_maps < truth_map[None]).sum(axis=0)
    ties = (draw_maps == truth_map[None]).sum(axis=0)
    return (below + rng.random(below.shape) * ties) / draw_maps.shape[0]
