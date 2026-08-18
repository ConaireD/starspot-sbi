"""
Tests for scripts/posterior_shrinkage.py.

Two layers. The linear-Gaussian identities are exercised on operators small
enough to check by hand, where the closed form and the exact solve can be made
to agree exactly or to disagree by a factor of a hundred on demand. The run
uses an L = 6 design cache built in tmp_path and synthetic draw files whose
per-direction sample variance is set to the analytic posterior variance, so the
whole chain has a known answer.

The draw files here follow the layout posterior_shrinkage.py expects:
<family>/chunk_*.csv with columns idx, slot, beta, and <family>/draws/*.npy of
shape (n_rows, n_draws, n_coeffs). Whether run_holdout.py writes that layout is
not checked by anything in this file.

"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from build_caches import build_design, load_design                  # noqa: E402
from operator_analysis import (build_T, channel_operators,          # noqa: E402
                               rank_by_gap, n_eff)
from posterior_shrinkage import (                                   # noqa: E402
    CALIBRATION_CAVEAT, FAMILY_CHANNELS,
    whitened_operator, operator_svd, prior_direction_variance,
    analytic_variance, analytic_shrinkage, exact_direction_variance,
    projected_variance, shrinkage_from_variance,
    iter_draw_rows, analyse,
)
from starspot_sbi.indexing import n_coeffs                          # noqa: E402

L6 = 6
N_OBS6 = 32
N_BETA = 91
P_ROT = 1.0
FAMILY = 'phot_axay'
SIGMA_PHOT = 1e-4
SIGMA_ASTRO = 10.0 ** -3.5
BETAS = (10, 40)


@pytest.fixture(scope='module')
def cache6(tmp_path_factory):
    """A full 91-inclination design cache at L = 6."""
    d = tmp_path_factory.mktemp('caches')
    build_design(str(d), L6, N_OBS6, N_BETA, P_ROT)
    return str(d)


@pytest.fixture(scope='module')
def design6(cache6):
    """(A, T) for the cache, loaded once."""
    return load_design(cache6, L6, N_OBS6, N_BETA), build_T(L6)


def operator_at(design, beta_deg, family=FAMILY):
    """The whitened operator, its padded spectrum and its right vectors."""
    A, T = design
    G = whitened_operator(channel_operators(A[beta_deg], T), family,
                          SIGMA_PHOT, SIGMA_ASTRO)
    sv, Vt = operator_svd(G)
    return G, sv, Vt


############################
# Linear-Gaussian identities #
############################

def test_shrinkage_is_one_minus_the_variance_ratio():
    """
    The two forms of the same quantity. analytic_shrinkage is written as
    s^2 / (s^2 + 1/tau2) so that a null direction gives exactly zero rather
    than a difference of two equal floats.
    """
    sv = np.array([10.0, 1.0, 0.1, 0.0])
    tau2 = np.array([1.0, 4.0, 0.25, 2.0])
    var = analytic_variance(sv, tau2)
    shr = analytic_shrinkage(sv, tau2)
    ratio = 1.0 - var / tau2
    print('shrinkage', np.array2string(shr, precision=8),
          'via 1 - var/tau2', np.array2string(ratio, precision=8))
    assert np.allclose(shr, ratio, rtol=0, atol=1e-12)
    assert shr[-1] == 0.0
    assert var[-1] == tau2[-1]


def test_unit_prior_shrinkage_sums_to_neff():
    """
    With tau_i = 1 the summed shrinkage is exactly the N_eff of
    operator_analysis, which is what makes the two scripts comparable.
    """
    sv = np.array([50.0, 7.0, 1.0, 0.3, 0.0, 0.0])
    total = float(np.sum(analytic_shrinkage(sv, np.ones_like(sv))))
    print(f'summed shrinkage {total:.10f}, n_eff {n_eff(sv, 1.0):.10f}')
    assert abs(total - n_eff(sv, 1.0)) < 1e-12


def test_prior_direction_variance_is_the_quadratic_form():
    """
    tau_i^2 = v_i^T diag(prior) v_i. An isotropic prior gives tau_i^2 = c in
    every direction, since the rows of Vt are unit vectors.
    """
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    Vt = Q.T
    prior = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tau2 = prior_direction_variance(Vt, prior)
    direct = np.array([v @ (prior * v) for v in Vt])
    iso = prior_direction_variance(Vt, np.full(5, 0.7))
    print('tau2', np.array2string(tau2, precision=6),
          'isotropic', np.array2string(iso, precision=12))
    assert np.allclose(tau2, direct, atol=1e-14)
    assert np.allclose(iso, 0.7, atol=1e-14)


def test_closed_form_is_exact_for_an_isotropic_prior():
    """
    An isotropic prior is diagonal in every basis, so the v_i are eigenvectors
    of the posterior precision and the per-direction closed form is the truth.
    Checked on a random operator and on the 1 x 2 case used below.
    """
    rng = np.random.default_rng(1)
    for G in (rng.normal(size=(6, 10)), np.array([[1.0, 1.0]])):
        sv, Vt = operator_svd(G)
        prior = np.full(G.shape[1], 0.3)
        tau2 = prior_direction_variance(Vt, prior)
        var_an = analytic_variance(sv, tau2)
        var_ex = exact_direction_variance(G, prior, Vt)
        rel = float(np.max(np.abs(var_an - var_ex) / var_ex))
        print(f'G {G.shape}: max relative difference {rel:.2e}')
        assert rel < 1e-10


def test_closed_form_departs_from_the_exact_solve_for_a_spread_prior():
    """
    G = [[1, 1]] with prior variances 100 and 0.01. The singular directions are
    (1, 1)/sqrt(2) and (1, -1)/sqrt(2), and both carry tau^2 = 50.005. The
    closed form leaves the null direction at its prior width, 50.005, where the
    exact posterior gives 52.005/101.01 = 0.51485: the prior couples the two
    directions and the per-direction form cannot see it. This is the quantity
    analyse reports as closed_form_vs_exact_max_rel, and it is why that number
    is reported rather than assumed small.
    """
    G = np.array([[1.0, 1.0]])
    prior = np.array([100.0, 0.01])
    sv, Vt = operator_svd(G)
    tau2 = prior_direction_variance(Vt, prior)
    var_an = analytic_variance(sv, tau2)
    var_ex = exact_direction_variance(G, prior, Vt)
    rel = np.abs(var_an - var_ex) / var_ex
    print(f'tau2 {tau2}, closed {var_an}, exact {var_ex}, relative {rel}')
    assert np.allclose(tau2, 50.005, atol=1e-9)
    assert abs(var_ex[1] - 0.5148500) < 1e-6
    assert abs(var_an[1] - 50.005) < 1e-6
    assert rel[1] > 50


def test_shrinkage_from_variance_goes_negative_when_the_posterior_is_wider():
    """A posterior wider than the prior is a diagnostic, not an error."""
    shr = shrinkage_from_variance(np.array([0.25, 1.0, 4.0]), np.ones(3))
    print('shrinkage', shr)
    assert shr[0] == 0.75 and shr[1] == 0.0 and shr[2] == -3.0


############################
# Projection               #
############################

def test_projected_variance_isolates_one_direction():
    """
    Draws confined to a single right singular vector have their whole sample
    variance in that direction and none in any other, exactly.
    """
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    Vt = Q.T
    a = rng.normal(size=32)
    draws = a[:, None] * Vt[3][None, :]
    var = projected_variance(draws, Vt)
    print(f'variance along direction 3: {var[3]:.10f}, sample variance of a '
          f'{a.var(ddof=1):.10f}, largest elsewhere {np.max(np.delete(var, 3)):.2e}')
    assert abs(var[3] - a.var(ddof=1)) < 1e-12
    assert np.max(np.delete(var, 3)) < 1e-24


############################
# Operator assembly        #
############################

def test_whitened_operator_stacks_the_family_channels_in_order(design6):
    """
    Channel order follows FAMILY_CHANNELS, each block divided by its own noise,
    so that the closed forms apply with sigma_n = 1.
    """
    A, T = design6
    ops = channel_operators(A[40], T)
    G = whitened_operator(ops, FAMILY, SIGMA_PHOT, SIGMA_ASTRO)
    n = n_coeffs(L6)
    print(f'G {G.shape}, channels {FAMILY_CHANNELS[FAMILY]}')
    assert G.shape == (3 * N_OBS6, n)
    assert np.allclose(G[:N_OBS6], ops['phot'] / SIGMA_PHOT, rtol=1e-12)
    assert np.allclose(G[N_OBS6:2 * N_OBS6], ops['x'] / SIGMA_ASTRO, rtol=1e-12)
    assert np.allclose(G[2 * N_OBS6:], ops['y'] / SIGMA_ASTRO, rtol=1e-12)

    assert whitened_operator(ops, 'phot', SIGMA_PHOT, SIGMA_ASTRO).shape[0] == N_OBS6


def test_operator_svd_spans_the_whole_domain(design6):
    """
    Every direction of the coefficient space appears, the unmeasured ones with
    sigma = 0 exactly, and Vt is orthogonal so projections lose nothing.
    """
    _, sv, Vt = operator_at(design6, 40)
    n = n_coeffs(L6)
    rank = rank_by_gap(sv[sv > 0])[0]
    print(f'n {n}, sv size {sv.size}, rank {rank}, '
          f'exact zeros {int(np.sum(sv == 0))}')
    assert sv.size == n and Vt.shape == (n, n)
    assert np.allclose(Vt @ Vt.T, np.eye(n), atol=1e-10)
    assert rank < n, 'a full-rank operator would make the null-space rows empty'


############################
# Draw files               #
############################

def write_chunk(fam_dir, name, rows, draws):
    """One chunk CSV and its sibling draw block."""
    os.makedirs(os.path.join(fam_dir, 'draws'), exist_ok=True)
    with open(os.path.join(fam_dir, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'slot', 'beta'])
        for r in rows:
            w.writerow(r)
    if draws is not None:
        np.save(os.path.join(fam_dir, 'draws', name.replace('.csv', '.npy')),
                draws)


def test_iter_draw_rows_pairs_csv_rows_with_draw_blocks(tmp_path):
    fam = str(tmp_path / 'fam')
    write_chunk(fam, 'chunk_000.csv', [(7, 0, 10), (8, 1, 40)],
                np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    write_chunk(fam, 'chunk_001.csv', [(9, 2, 10)],
                np.ones((1, 3, 5)))
    got = list(iter_draw_rows(fam))
    print([(i, s, b, d.shape) for i, s, b, d in got])
    assert [(i, s, b) for i, s, b, _ in got] == [(7, 0, 10), (8, 1, 40), (9, 2, 10)]
    assert got[0][3].shape == (3, 5)
    assert np.array_equal(got[2][3], np.ones((3, 5)))

    assert len(list(iter_draw_rows(fam, n_max=2))) == 2


def test_iter_draw_rows_skips_a_chunk_with_no_draw_block(tmp_path):
    """A CSV written without --save-draws is passed over, not guessed at."""
    fam = str(tmp_path / 'fam')
    write_chunk(fam, 'chunk_000.csv', [(1, 0, 10)], np.zeros((1, 3, 5)))
    write_chunk(fam, 'chunk_001.csv', [(2, 0, 10)], None)
    got = list(iter_draw_rows(fam))
    print(f'{len(got)} row(s) from two chunks')
    assert [i for i, _, _, _ in got] == [1]


def test_iter_draw_rows_refuses_a_length_mismatch(tmp_path):
    fam = str(tmp_path / 'fam')
    write_chunk(fam, 'chunk_000.csv', [(1, 0, 10), (2, 0, 10)],
                np.zeros((1, 3, 5)))
    with pytest.raises(ValueError, match='draw blocks'):
        list(iter_draw_rows(fam))


def test_iter_draw_rows_is_empty_for_a_missing_directory(tmp_path):
    assert list(iter_draw_rows(str(tmp_path / 'absent'))) == []


############################
# End to end               #
############################

def draws_with_variance(var, Vt, rng):
    """
    Two draws whose per-direction sample variance is exactly `var`. Antipodal
    points at +-sqrt(var/2) have sample variance var with ddof = 1, and Vt is
    orthogonal so rotating them into the coefficient basis preserves it.
    """
    a = np.sqrt(np.asarray(var) / 2.0)
    sign = rng.choice([-1.0, 1.0], size=a.size)
    y = np.vstack([a * sign, -a * sign])
    return y @ Vt


@pytest.fixture(scope='module')
def analysis(design6, cache6, tmp_path_factory):
    """
    One run over two inclinations and two surfaces each, with the draws built
    to carry exactly the analytic posterior variance under a unit prior. The
    SBI columns must then reproduce the analytic ones to rounding.
    """
    rng = np.random.default_rng(3)
    fam = str(tmp_path_factory.mktemp('holdout') / FAMILY)
    prior = np.ones(n_coeffs(L6))

    rows, blocks = [], []
    for beta in BETAS:
        _, sv, Vt = operator_at(design6, beta)
        var = analytic_variance(sv, prior_direction_variance(Vt, prior))
        for k in range(2):
            rows.append((100 * beta + k, k, beta))
            blocks.append(draws_with_variance(var, Vt, rng))
    write_chunk(fam, 'chunk_000.csv', rows, np.stack(blocks))

    out = str(tmp_path_factory.mktemp('shrinkage'))
    summary = analyse(fam, FAMILY, prior, cache6, out, SIGMA_PHOT, SIGMA_ASTRO,
                      L6, N_OBS6, N_BETA, progress=False)
    return out, summary


def test_analyse_recovers_the_analytic_baseline(analysis):
    """
    Draws carrying the analytic variance give N_eff_SBI = N_eff_analytic and a
    zero log-determinant ratio in both subspaces. A failure here is a wiring
    failure: a transposed projection, a lost whitening or the wrong Vt.
    """
    _, summary = analysis
    for beta in BETAS:
        r = summary['per_beta'][beta]
        print(f'beta {beta}: rank {r["rank"]}, N_eff analytic '
              f'{r["neff_analytic_prior_weighted"]:.6f}, SBI '
              f'{r["neff_sbi_mean"]:.6f}, difference {r["neff_difference"]:.2e}, '
              f'logdet row {r["logdet_ratio_row_space"]:.2e}, null '
              f'{r["logdet_ratio_null_space"]:.2e}, spread {r["neff_sbi_std"]:.2e}')
        assert r['n_surfaces'] == 2
        assert abs(r['neff_difference']) < 1e-6
        assert abs(r['logdet_ratio_row_space']) < 1e-6
        assert abs(r['logdet_ratio_null_space']) < 1e-6
        assert r['neff_sbi_std'] < 1e-9


def test_unit_prior_makes_the_two_neff_definitions_coincide(analysis):
    """
    With tau_i = 1 the prior-weighted N_eff is the unit-prior one. The two keys
    exist because they part company for any other prior.
    """
    _, summary = analysis
    for beta in BETAS:
        r = summary['per_beta'][beta]
        print(f'beta {beta}: prior-weighted {r["neff_analytic_prior_weighted"]:.9f}, '
              f'unit prior {r["neff_analytic_unit_prior"]:.9f}')
        assert abs(r['neff_analytic_prior_weighted']
                   - r['neff_analytic_unit_prior']) < 1e-9


def test_isotropic_prior_leaves_only_the_numerical_floor(analysis):
    """
    The prior is a multiple of the identity here, so the per-direction closed
    form is exact and every part of the reported difference is arithmetic. It
    measures 3e-7 at L = 6, from np.linalg.inv on a matrix mixing whitened
    sigma^2 near 1e8 with 1/tau^2 of order one. That is the floor of
    closed_form_vs_exact_max_rel: a departure smaller than it says nothing
    about the prior, and the floor rises at L = 30 where the prior variances
    span several orders of magnitude.
    """
    _, summary = analysis
    worst = max(summary['per_beta'][b]['closed_form_vs_exact_max_rel']
                for b in BETAS)
    print(f'largest closed-form against exact relative difference {worst:.2e}')
    assert worst < 1e-5


def test_null_space_shrinkage_is_zero_analytically(analysis):
    """
    The closed form leaves an unmeasured direction at its prior width, so the
    analytic null-space shrinkage is identically zero. Any SBI null-space
    shrinkage is therefore prior, not measurement; here the draws were built
    from the analytic variance, so it is zero too.
    """
    _, summary = analysis
    for beta in BETAS:
        r = summary['per_beta'][beta]
        print(f'beta {beta}: analytic null sum '
              f'{r["shrink_analytic_null_space_sum"]:.2e}, SBI null sum '
              f'{r["shrink_sbi_null_space_sum"]:.2e}, SBI row sum '
              f'{r["shrink_sbi_row_space_sum"]:.4f} of rank {r["rank"]}')
        assert abs(r['shrink_analytic_null_space_sum']) < 1e-12
        assert abs(r['shrink_sbi_null_space_sum']) < 1e-9


def test_analyse_writes_a_csv_per_inclination_and_a_summary(analysis):
    out, summary = analysis
    for beta in BETAS:
        path = os.path.join(out, f'perdir_beta{beta:03d}.csv')
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n_coeffs(L6)
        in_row = [r for r in rows if r['in_row_space'] == 'True']
        assert len(in_row) == summary['per_beta'][beta]['rank']
        # At L = 6 the operator has more rows than columns, so operator_svd
        # pads nothing and the unmeasured directions carry singular values at
        # the SVD floor rather than exact zeros. Their shrinkage is 1e-24. At
        # L = 30, where 961 columns exceed 648 rows, the padded directions are
        # exactly zero and this is an equality.
        null_shrink = [float(r['shrink_analytic']) for r in rows
                       if r['in_row_space'] == 'False']
        print(f'beta {beta}: largest analytic null-space shrinkage '
              f'{max(null_shrink):.2e} over {len(null_shrink)} directions')
        assert max(null_shrink) < 1e-12

    with open(os.path.join(out, 'summary.json')) as f:
        on_disk = json.load(f)
    print('summary keys', sorted(on_disk), 'per_beta keys',
          sorted(on_disk['per_beta']))
    assert set(on_disk['per_beta']) == {str(b) for b in BETAS}
    assert on_disk['settings']['family'] == FAMILY
    assert on_disk['settings']['n_surfaces'] == 4
    assert on_disk['settings']['calibration_caveat'] == CALIBRATION_CAVEAT
    assert set(on_disk['overall']) == {
        'neff_analytic_prior_weighted', 'neff_sbi_mean', 'neff_difference',
        'logdet_ratio_row_space', 'logdet_ratio_null_space'}


def test_overall_is_weighted_by_surface_count(design6, cache6, tmp_path):
    """
    Unequal surface counts per inclination weight the overall figure toward the
    better-sampled one, which is why the weights exist rather than a plain mean.
    """
    rng = np.random.default_rng(4)
    fam = str(tmp_path / FAMILY)
    prior = np.ones(n_coeffs(L6))
    rows, blocks = [], []
    for beta, n_surf in ((10, 1), (40, 3)):
        _, sv, Vt = operator_at(design6, beta)
        var = analytic_variance(sv, prior_direction_variance(Vt, prior))
        for k in range(n_surf):
            rows.append((k, k, beta))
            blocks.append(draws_with_variance(var, Vt, rng))
    write_chunk(fam, 'chunk_000.csv', rows, np.stack(blocks))

    summary = analyse(fam, FAMILY, prior, cache6, str(tmp_path / 'out'),
                      SIGMA_PHOT, SIGMA_ASTRO, L6, N_OBS6, N_BETA,
                      progress=False)
    pb = summary['per_beta']
    want = (1 * pb[10]['neff_analytic_prior_weighted']
            + 3 * pb[40]['neff_analytic_prior_weighted']) / 4
    plain = (pb[10]['neff_analytic_prior_weighted']
             + pb[40]['neff_analytic_prior_weighted']) / 2
    got = summary['overall']['neff_analytic_prior_weighted']
    print(f'weighted {got:.6f}, expected {want:.6f}, unweighted mean {plain:.6f}')
    assert abs(got - want) < 1e-9
    assert abs(got - plain) > 1e-6, 'the two inclinations must differ, or the '\
                                    'weighting is untested'


def test_analyse_refuses_when_there_are_no_saved_draws(cache6, tmp_path):
    """The message names run_holdout.py, since this script cannot re-sample."""
    with pytest.raises(FileNotFoundError, match='run_holdout'):
        analyse(str(tmp_path / 'nothing'), FAMILY, np.ones(n_coeffs(L6)),
                cache6, str(tmp_path / 'out'), SIGMA_PHOT, SIGMA_ASTRO,
                L6, N_OBS6, N_BETA, progress=False)