"""
Tests for metrics.py.

"""

import numpy as np
import pytest

from starspot_sbi.render import N_THETA, N_PHI, render_coeffs, grid_coordinates
from starspot_sbi.surfaces import place_spot, generate_spotted_surface
from starspot_sbi.metrics import (
    VIS_TOL,
    SPOT_THRESHOLD,
    lat_mu_grid,
    visibility_mask,
    cap_boundary_lat,
    weights,
    wmean,
    ssim_map,
    ssim_2d,
    ssim_aa_vis,
    ssim_aa_wmean,
    ssim_aa_full,
    rmse,
    mae,
    err_unc_corr,
    crps,
    spot_mask,
    pr_auc,
    detection_operating_points,
    scalar_metrics,
)

NT, NP = 120, 240


############################
# The mask, by row index   #
############################

@pytest.mark.parametrize('beta_deg,first,last', [
    (0,   1, 58),
    (15,  5, 59),
    (30, 10, 59),
    (45, 15, 59),
    (60, 20, 59),
    (75, 25, 59),
    (90, 30, 59),
])
def test_visible_rows_match_forward_geometry(beta_deg, first, last):
    """
    Measured ranges at n_theta = 60, from the audit's comparison against the
    forward model. nsf_lib gives [30,59] at beta = 0 and [0,59] at beta = 90;
    metrics_utils v10 and v11 both give [0, 59 - beta * 59 / 180].
    """
    rows = np.where(visibility_mask(np.radians(beta_deg), 60))[0]
    print(f"beta = {beta_deg:2d} deg: visible rows [{rows.min()}, {rows.max()}], "
          f"expected [{first}, {last}]")
    assert (rows.min(), rows.max()) == (first, last)


def test_mask_is_not_mirrored():
    """
    beta and 90 - beta must give different masks. The nsf_lib version agrees
    with the geometry only at beta = 45, where the mirror is the identity.
    """
    for beta_deg in [0, 15, 30, 60, 75, 90]:
        a = visibility_mask(np.radians(beta_deg), 60)
        b = visibility_mask(np.radians(90 - beta_deg), 60)
        assert not np.array_equal(a, b), f"mask at {beta_deg} equals its mirror"


def test_mask_is_not_the_metrics_utils_version():
    """
    metrics_utils v11 computes mu_max = cos(lat - beta) with lat the standard
    latitude, which hides the southern cap. v10 computed cos(lat_negated + beta),
    algebraically the same thing, which is why the v11 edit changed nothing. The
    forward model hides the northern cap, so the two disagree on half the sphere
    at pole-on.
    """
    n = 60
    colat, _ = grid_coordinates(n, 2)
    lat = 90.0 - colat
    for beta_deg in [0, 30, 60, 90]:
        theirs = np.cos(np.radians(lat - beta_deg)) > 0
        ours = visibility_mask(np.radians(beta_deg), n)
        n_diff = np.sum(theirs != ours)
        print(f"beta = {beta_deg:2d} deg: rows differing from metrics_utils = {n_diff}")
        if beta_deg > 0:
            assert n_diff > 0
    # at pole-on the two masks are complementary halves
    assert np.sum(np.cos(np.radians(lat - 90.0)) > 0) == 30
    assert np.sum(visibility_mask(np.radians(90), n)) == 30


def test_hidden_cap_is_northern():
    """
    Rows above latitude +(90 - beta) are never visible, and the southernmost row
    is always visible for beta > 0. A southern cap reverses both.
    """
    for beta_deg in [15, 45, 75]:
        beta = np.radians(beta_deg)
        lat, mu_max, _ = lat_mu_grid(beta, NT)
        hidden = mu_max <= VIS_TOL
        assert hidden[0], "row 0 is the north pole and must be hidden for beta > 0"
        assert not hidden[-1], "row -1 is the south pole and must be visible"
        assert np.all(lat[hidden] > cap_boundary_lat(beta) - 2.0)
        print(f"beta = {beta_deg:2d} deg: cap boundary at "
              f"{cap_boundary_lat(beta):+.1f} deg, {hidden.sum()} rows hidden")


def test_cap_boundary_latitude():
    for beta_deg in [0, 30, 60, 90]:
        assert cap_boundary_lat(np.radians(beta_deg)) == pytest.approx(90 - beta_deg)


def test_visible_fraction_falls_with_beta():
    """Essentially all of the sphere equator-on, exactly half pole-on."""
    fracs = []
    for beta_deg in [0, 30, 60, 90]:
        w = weights(np.radians(beta_deg), NT, NP, kind='vis')
        w_full = weights(np.radians(beta_deg), NT, NP, kind='full')
        area = np.sum(np.where(w > 0, w_full, 0.0))
        fracs.append(area)
        print(f"beta = {beta_deg:2d} deg: visible solid-angle fraction {area:.4f}")
    assert fracs[0] == pytest.approx(1.0, abs=1e-3)
    assert fracs[-1] == pytest.approx(0.5, abs=1e-2)
    assert all(fracs[i] > fracs[i + 1] for i in range(len(fracs) - 1))


def test_mu_max_closed_form():
    """mu_max = max(sin(theta - beta), 0), from the observer frame."""
    for beta_deg in [0, 37, 90]:
        beta = np.radians(beta_deg)
        colat, _ = grid_coordinates(NT, 2)
        _, mu_max, _ = lat_mu_grid(beta, NT)
        expected = np.maximum(np.sin(np.radians(colat) - beta), 0.0)
        assert np.max(np.abs(mu_max - expected)) < 1e-14


def test_mu_mean_against_numerical_phase_average():
    """The closed form for the phase average of max(n_hat . r_hat, 0)."""
    psi = np.linspace(0, 2 * np.pi, 40001)
    for beta_deg in [0, 30, 60, 90]:
        beta = np.radians(beta_deg)
        colat, _ = grid_coordinates(NT, 2)
        theta = np.radians(colat)
        _, _, mu_mean = lat_mu_grid(beta, NT)
        numeric = np.array([
            np.mean(np.maximum(np.sin(t) * np.cos(beta) * np.cos(psi)
                               - np.cos(t) * np.sin(beta), 0.0))
            for t in theta
        ])
        err = np.max(np.abs(mu_mean - numeric))
        print(f"beta = {beta_deg:2d} deg: max |closed form - numeric| = {err:.1e}")
        assert err < 1e-4


def test_mu_mean_and_mu_max_share_support():
    for beta_deg in [0, 15, 45, 75, 90]:
        _, mu_max, mu_mean = lat_mu_grid(np.radians(beta_deg), NT)
        assert np.array_equal(mu_mean > 0, mu_max > 0)


def test_mu_mean_at_pole_on():
    """At beta = 90 the sub-observer point is the south pole, where mu_mean = 1."""
    _, _, mu_mean = lat_mu_grid(np.radians(90), NT)
    assert mu_mean[-1] == pytest.approx(1.0, abs=1e-12)
    assert mu_mean[0] == pytest.approx(0.0, abs=1e-12)


############################
# Weights                  #
############################

@pytest.mark.parametrize('kind', ['full', 'vis', 'wmean'])
def test_weights_normalised_and_nonnegative(kind):
    w = weights(np.radians(40), NT, NP, kind=kind)
    assert w.shape == (NT, NP)
    assert np.all(w >= 0)
    assert np.sum(w) == pytest.approx(1.0)


@pytest.mark.parametrize('kind', ['full', 'vis', 'wmean'])
def test_weights_constant_in_longitude(kind):
    w = weights(np.radians(40), NT, NP, kind=kind)
    assert np.all(np.ptp(w, axis=1) < 1e-15)


def test_full_weights_approximate_solid_angle():
    w = weights(0.0, NT, NP, kind='full')
    colat, _ = grid_coordinates(NT, 2)
    assert np.sum(w[colat < 90]) == pytest.approx(0.5, abs=1e-2)


def test_full_weights_independent_of_beta():
    a = weights(0.0, NT, NP, kind='full')
    b = weights(np.radians(70), NT, NP, kind='full')
    assert np.array_equal(a, b)


def test_wmean_weights_vanish_inside_the_cap():
    beta = np.radians(60)
    w = weights(beta, NT, NP, kind='wmean')
    hidden = ~visibility_mask(beta, NT)
    assert np.all(w[hidden] == 0.0)


def test_unknown_weighting_raises():
    with pytest.raises(ValueError):
        weights(0.5, NT, NP, kind='nonsense')


def test_wmean_matches_plain_mean_for_uniform_weights():
    x = np.random.default_rng(0).normal(size=(10, 20))
    assert wmean(x, np.ones_like(x)) == pytest.approx(x.mean())


############################
# SSIM                     #
############################

def test_ssim_identical_inputs_is_one():
    rng = np.random.default_rng(0)
    img = 1.0 - 0.3 * rng.random((NT, NP))
    assert np.max(np.abs(ssim_map(img, img) - 1.0)) < 1e-10
    assert ssim_2d(img, img) == pytest.approx(1.0, abs=1e-10)
    for beta_deg in [0, 45, 90]:
        beta = np.radians(beta_deg)
        assert ssim_aa_vis(img, img, beta) == pytest.approx(1.0, abs=1e-10)
        assert ssim_aa_wmean(img, img, beta) == pytest.approx(1.0, abs=1e-10)
        assert ssim_aa_full(img, img) == pytest.approx(1.0, abs=1e-10)


def test_ssim_symmetric():
    rng = np.random.default_rng(1)
    a = 1.0 - 0.3 * rng.random((NT, NP))
    b = 1.0 - 0.3 * rng.random((NT, NP))
    assert np.allclose(ssim_map(a, b), ssim_map(b, a))


def test_ssim_bounded():
    rng = np.random.default_rng(3)
    a = 1.0 - 0.3 * rng.random((60, 120))
    b = 1.0 - 0.3 * rng.random((60, 120))
    s = ssim_map(a, b)
    assert np.all(s <= 1.0 + 1e-9)
    assert np.all(s >= -1.0 - 1e-9)


def test_ssim_degrades_with_noise():
    rng = np.random.default_rng(2)
    s = generate_spotted_surface(12, [{'theta': 1.2, 'phi': 0.3,
                                       'radius': np.deg2rad(12), 'contrast': 0.6}])
    img = render_coeffs(s, NT, NP, L=12)
    scores = [ssim_aa_vis(img, img + sigma * rng.normal(size=img.shape),
                          np.radians(30))
              for sigma in [0.0, 0.01, 0.05, 0.2]]
    print("SSIM against noise level:", [f"{v:.4f}" for v in scores])
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))


def test_ssim_filter_is_pole_correct():
    """
    A longitude-constant field has longitude-constant local statistics, so its
    SSIM against itself is exactly one at the pole rows. A filter using
    mode='nearest' along theta duplicates the pole row rather than continuing
    across it, and fails within win_size // 2 rows of each pole. That is the
    version reconstruction_utils uses; metrics_utils v11 uses the reflected and
    rolled continuation implemented here.
    """
    colat, _ = grid_coordinates(NT, 2)
    img = np.repeat(np.cos(np.radians(colat))[:, None], NP, axis=1)
    s = ssim_map(img, img)
    assert np.max(np.abs(s[:5] - 1.0)) < 1e-10
    assert np.max(np.abs(s[-5:] - 1.0)) < 1e-10


def test_ssim_filter_continues_across_the_pole():
    """
    A field antisymmetric under phi -> phi + pi has its own reflection across
    the pole, so the local mean there is zero rather than the pole row's value.
    """
    from starspot_sbi.metrics import _box
    _, lon = grid_coordinates(NT, NP)
    img = np.repeat(np.cos(np.radians(lon))[None, :], NT, axis=0)
    m = _box(img, 7)
    print(f"row 0 local mean {np.max(np.abs(m[0])):.2e}, "
          f"row 60 {np.max(np.abs(m[60])):.2e}")
    assert np.max(np.abs(m[0])) < np.max(np.abs(m[60]))


def test_ssim_vis_ignores_the_hidden_cap():
    rng = np.random.default_rng(11)
    beta = np.radians(70)
    img = 1.0 - 0.2 * rng.random((NT, NP))
    recon = img.copy()
    hidden = ~visibility_mask(beta, NT)
    win = 7
    recon[:max(0, hidden.sum() - win)] += 3.0        # stay clear of the window reach
    assert ssim_aa_vis(img, recon, beta) == pytest.approx(1.0, abs=1e-6)


############################
# Point estimates          #
############################

def test_rmse_and_mae_zero_for_identical():
    rng = np.random.default_rng(4)
    img = 1.0 - 0.3 * rng.random((NT, NP))
    for kind in ['full', 'vis', 'wmean']:
        assert rmse(img, img, np.radians(30), kind) == pytest.approx(0.0, abs=1e-14)
        assert mae(img, img, np.radians(30), kind) == pytest.approx(0.0, abs=1e-14)


def test_rmse_of_constant_offset():
    """A uniform offset d gives RMSE = MAE = d under every weighting."""
    img = np.ones((NT, NP))
    for beta_deg in [0, 45, 90]:
        for kind in ['full', 'vis', 'wmean']:
            assert rmse(img, img + 0.07, np.radians(beta_deg), kind) == pytest.approx(0.07)
            assert mae(img, img + 0.07, np.radians(beta_deg), kind) == pytest.approx(0.07)


def test_rmse_vis_ignores_the_hidden_cap():
    """Corrupting rows inside the unobservable cap must not move a vis-weighted score."""
    rng = np.random.default_rng(5)
    beta = np.radians(60)
    img = 1.0 - 0.2 * rng.random((NT, NP))
    recon = img.copy()
    hidden = ~visibility_mask(beta, NT)
    assert hidden.sum() > 0
    recon[hidden] += 5.0
    assert rmse(img, recon, beta, 'vis') == pytest.approx(0.0, abs=1e-14)
    assert rmse(img, recon, beta, 'full') > 1.0


def test_err_unc_corr_perfect_and_null():
    rng = np.random.default_rng(6)
    beta = np.radians(30)
    truth = np.ones((NT, NP))
    err = rng.random((NT, NP))
    recon = truth - err
    assert err_unc_corr(truth, recon, err, beta) == pytest.approx(1.0, abs=1e-10)
    assert np.isnan(err_unc_corr(truth, recon, np.ones_like(err), beta))


############################
# CRPS                     #
############################

def test_crps_zero_for_a_point_mass_at_the_truth():
    truth = np.ones((30, 60)) * 0.9
    samples = np.repeat(truth[None], 16, axis=0)
    assert crps(truth, samples, np.radians(30)) == pytest.approx(0.0, abs=1e-14)


def test_crps_equals_offset_for_a_displaced_point_mass():
    truth = np.ones((30, 60))
    samples = np.repeat((truth + 0.1)[None], 16, axis=0)
    assert crps(truth, samples, np.radians(30)) == pytest.approx(0.1, abs=1e-12)


def test_crps_sharpness_term_matches_the_double_sum():
    """The sorted form equals 0.5 E|Y - Y'| computed by brute force."""
    rng = np.random.default_rng(7)
    n = 12
    samples = rng.normal(size=(n, 4, 5))

    ordered = np.sort(samples, axis=0)
    coef = (2 * np.arange(1, n + 1) - n - 1)[:, None, None]
    fast = np.sum(coef * ordered, axis=0) / n ** 2

    slow = np.zeros((4, 5))
    for i in range(n):
        for j in range(n):
            slow += np.abs(samples[i] - samples[j])
    slow *= 0.5 / n ** 2

    assert np.max(np.abs(fast - slow)) < 1e-12


def test_crps_is_deterministic():
    """No random permutation, so repeated calls agree exactly."""
    rng = np.random.default_rng(8)
    truth = np.ones((30, 60))
    samples = truth[None] + 0.05 * rng.normal(size=(20, 30, 60))
    assert crps(truth, samples, np.radians(30)) == crps(truth, samples, np.radians(30))


def test_crps_rewards_sharpness_at_equal_accuracy():
    """Two sample sets centred on the truth: the narrower scores lower."""
    rng = np.random.default_rng(12)
    truth = np.ones((30, 60))
    tight = truth[None] + 0.01 * rng.normal(size=(64, 30, 60))
    wide = truth[None] + 0.10 * rng.normal(size=(64, 30, 60))
    assert crps(truth, tight, np.radians(30)) < crps(truth, wide, np.radians(30))


############################
# Spot detection           #
############################

def test_spot_mask_threshold():
    img = np.array([[0.5, 0.89, 0.9, 1.0]])
    assert np.array_equal(spot_mask(img), [[True, True, False, False]])
    assert np.array_equal(spot_mask(img, 0.6), [[True, False, False, False]])
    assert SPOT_THRESHOLD == 0.9


def test_pr_auc_perfect_recovery():
    """
    A perfect reconstruction scores at the discretisation ceiling rather than
    exactly 1: the sweep is n_thresholds evenly spaced values and the truth
    threshold falls between two of them, so the trapezoid cuts a corner near
    recall 1. Denser sweeps approach 1.
    """
    s = generate_spotted_surface(12, [{'theta': np.radians(60), 'phi': 0.3,
                                       'radius': np.deg2rad(15), 'contrast': 0.5}])
    truth = render_coeffs(s, NT, NP, L=12)
    coarse = pr_auc(truth, truth, np.radians(30), n_thresholds=200)
    fine = pr_auc(truth, truth, np.radians(30), n_thresholds=4000)
    print(f"PR-AUC for a perfect reconstruction: {coarse:.6f} at 200 thresholds, "
          f"{fine:.6f} at 4000")
    assert coarse > 0.99
    assert fine > coarse

def test_pr_auc_ordering_with_degradation():
    rng = np.random.default_rng(9)
    beta = np.radians(30)
    s = generate_spotted_surface(12, [
        {'theta': np.radians(60), 'phi': 0.3, 'radius': np.deg2rad(14), 'contrast': 0.5},
        {'theta': np.radians(110), 'phi': -1.0, 'radius': np.deg2rad(11), 'contrast': 0.6},
    ])
    truth = render_coeffs(s, NT, NP, L=12)
    scores = [pr_auc(truth, truth + sigma * rng.normal(size=truth.shape), beta)
              for sigma in [0.0, 0.02, 0.08]]
    print("PR-AUC against noise level:", [f"{v:.4f}" for v in scores])
    assert scores[0] > scores[1] > scores[2]


def test_pr_auc_nan_when_no_visible_spots():
    truth = np.ones((NT, NP))
    assert np.isnan(pr_auc(truth, truth, np.radians(30)))


def test_pr_auc_ignores_spots_in_the_hidden_cap():
    """
    A spot inside the unobservable cap must not enter a vis-weighted detection
    score, and must enter the whole-sphere one. Under the mirrored mask both
    statements reverse.

    The spot must fit inside the cap with margin: its centre sits a radius above
    the boundary plus a further allowance for the smoothing the band limit
    imposes on a hard-edged cap.
    """
    beta = np.radians(60)
    radius_deg = 12.0
    lat_spot = cap_boundary_lat(beta) + radius_deg + 12.0     # boundary +30, spot at +54
    s = generate_spotted_surface(20, [
        {'theta': np.radians(90 - lat_spot), 'phi': 0.0,
         'radius': np.deg2rad(radius_deg), 'contrast': 0.3}])
    truth = render_coeffs(s, NT, NP, L=20)

    hidden = ~visibility_mask(beta, NT)
    print(f"cap boundary +{cap_boundary_lat(beta):.1f} deg, spot centre +{lat_spot:.1f} deg, "
          f"radius {radius_deg} deg; image minimum {truth.min():.4f}, "
          f"visible minimum {truth[~hidden].min():.4f}, threshold {SPOT_THRESHOLD}")
    assert truth.min() < SPOT_THRESHOLD, "spot too shallow to register as spotted"
    assert np.all(truth[~hidden] > SPOT_THRESHOLD), "spot leaks into the visible region"

    assert np.isnan(pr_auc(truth, np.ones_like(truth), beta, 'vis'))
    assert pr_auc(truth, np.ones_like(truth), beta, 'full') == 0.0

    
def test_operating_points_recover_the_truth_threshold():
    """For a perfect reconstruction the F1-maximising threshold is the truth threshold."""
    s = generate_spotted_surface(12, [{'theta': np.radians(60), 'phi': 0.3,
                                       'radius': np.deg2rad(14), 'contrast': 0.5}])
    truth = render_coeffs(s, NT, NP, L=12)
    op = detection_operating_points(truth, truth, np.radians(30))
    print(f"F1 max {op['f1_max']:.4f} at threshold {op['f1_threshold']:.4f}, "
          f"truth threshold {op['truth_threshold']}")
    assert op['f1_max'] > 0.99
    assert abs(op['f1_threshold'] - op['truth_threshold']) < 0.02


############################
# Assembly                 #
############################

def test_scalar_metrics_keys_and_perfect_recovery():
    s = generate_spotted_surface(12, [{'theta': np.radians(60), 'phi': 0.3,
                                       'radius': np.deg2rad(14), 'contrast': 0.5}])
    truth = render_coeffs(s, NT, NP, L=12)
    out = scalar_metrics(truth, truth, np.radians(30))

    expected = {'ssim_aa_vis', 'ssim_aa_wmean', 'ssim_aa_full',
                'rmse_vis', 'rmse_full', 'mae_vis', 'mae_full',
                'pr_auc_vis', 'pr_auc_full'}
    assert set(out) == expected
    assert out['ssim_aa_vis'] == pytest.approx(1.0, abs=1e-9)
    assert out['rmse_vis'] == pytest.approx(0.0, abs=1e-12)
    assert out['pr_auc_vis'] > 0.99

def test_scalar_metrics_optional_arguments_and_prefix():
    rng = np.random.default_rng(10)
    s = generate_spotted_surface(12, [{'theta': np.radians(60), 'phi': 0.3,
                                       'radius': np.deg2rad(14), 'contrast': 0.5}])
    truth = render_coeffs(s, NT, NP, L=12)
    recon = truth + 0.02 * rng.normal(size=truth.shape)
    samples = recon[None] + 0.02 * rng.normal(size=(16,) + truth.shape)
    std = samples.std(axis=0)

    out = scalar_metrics(truth, recon, np.radians(30), recon_std=std,
                         samples=samples, prefix='map_')
    assert all(k.startswith('map_') for k in out)
    assert 'map_err_unc_corr' in out
    assert 'map_crps_vis' in out and 'map_crps_full' in out