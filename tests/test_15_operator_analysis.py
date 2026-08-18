"""
Tests for scripts/operator_analysis.py.

Everything at small L. The closed forms and the gap method are exercised on
operators assembled directly with build_design_matrix; the end-to-end run uses
a cache built in tmp_path, following test_13_caches.py.

Numerical bounds are set at roughly ten times a measured value, and each test
prints what it measured so a later failure shows how far something moved.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from build_caches import build_design                          # noqa: E402
from operator_analysis import (                                # noqa: E402
    CHANNELS, GAP_TRUST_DECADES, LAM, NOISE_LEVELS, PROJECTORS,
    MISSION_LABEL, UNIFORM_LABEL,
    build_T, degree_map, channel_operators,
    rank_by_gap, rank_by_tol, closed_form_rank,
    graded_svd, hard_weights, apply_projector, observable_power_fraction,
    variance_reduction, per_degree_fraction, projector_trace,
    shape_correlation, n_eff, n_eff_mission,
    run_analysis,
)
from starspot_sbi.indexing import n_coeffs                     # noqa: E402
from starspot_sbi.kernels import precompute_kernels_fast       # noqa: E402
from starspot_sbi.design_matrix import build_W, build_B        # noqa: E402

L6 = 6
N_OBS6 = 32
N_BETA = 91          # pole-on lives at index 90, so the cache spans 0..90
P_ROT = 1.0

INTERIOR = (20, 33, 60)


def direct_slice(l_max, n_obs, beta_deg):
    """One cache-shaped slice (3, n_obs, n) built without the cache layer."""
    kx, ky, kp = precompute_kernels_fast(l_max)
    t_obs = np.linspace(0, P_ROT, n_obs, endpoint=False)
    W = build_W(l_max, 2 * np.pi / P_ROT, t_obs)
    beta = np.radians(beta_deg)
    return np.stack([W @ build_B(l_max, beta, k) for k in (kx, ky, kp)])


@pytest.fixture(scope='module')
def cache6(tmp_path_factory):
    """A full 91-inclination cache at L = 6, a few seconds to build."""
    d = tmp_path_factory.mktemp('caches')
    build_design(str(d), L6, N_OBS6, N_BETA, P_ROT)
    return str(d)


@pytest.fixture(scope='module')
def svd6():
    """One graded SVD of the combined operator at L = 6, beta = 33, shared."""
    A = direct_slice(L6, N_OBS6, 33)
    ops = channel_operators(A, build_T(L6))
    Vt, sv, g = graded_svd(ops['combined'], LAM)
    return ops, Vt, sv, g


############################
# Closed forms             #
############################

def test_gap_rank_reproduces_closed_forms():
    """
    Gap rank against the corrected closed forms at L = 4 to 8, per channel, at
    beta = 0, three interior inclinations and pole-on. L = 4 exercises the
    Lo = 3 special case (interior astro 11, not 4L - 3 = 13 and not Taaki's
    4L - 2 = 14). Nothing here is tuned: a disagreement is a finding.
    """
    for l_max in (4, 5, 6, 7, 8):
        n_obs = 4 * l_max + 8
        T = build_T(l_max)
        for beta_deg in (0,) + INTERIOR + (90,):
            ops = channel_operators(direct_slice(l_max, n_obs, beta_deg), T)
            for ch in CHANNELS:
                sv = np.linalg.svd(ops[ch], compute_uv=False)
                r, _, _, gap = rank_by_gap(sv)
                cf = closed_form_rank(l_max, ch, beta_deg)
                print(f'L={l_max} beta={beta_deg:>2} {ch:>9}: '
                      f'rank {r:>2} closed {cf:>2} gap {gap:.1f} dec')
                assert r == cf, (l_max, beta_deg, ch)
                assert gap >= GAP_TRUST_DECADES, (l_max, beta_deg, ch)


def test_l3_special_case():
    """
    At L = 3 the combined operator is astro + phot - 2 at beta = 0 and
    interior (a cross-channel dependency at m = +-2), and interior astro is
    4Lo + 1 - 2 = 11, one above Taaki's 4L - 2 = 10 (RANK_RESOLUTION.md).
    """
    T = build_T(3)
    for beta_deg, want_astro, want_comb in [(0, 11, 14), (33, 11, 14),
                                            (90, 2, 3)]:
        ops = channel_operators(direct_slice(3, 20, beta_deg), T)
        ra = rank_by_gap(np.linalg.svd(ops['astro'], compute_uv=False))[0]
        rc = rank_by_gap(np.linalg.svd(ops['combined'], compute_uv=False))[0]
        print(f'L=3 beta={beta_deg:>2}: astro {ra} (want {want_astro}), '
              f'combined {rc} (want {want_comb})')
        assert ra == closed_form_rank(3, 'astro', beta_deg) == want_astro
        assert rc == closed_form_rank(3, 'combined', beta_deg) == want_comb


############################
# Gap versus tolerance     #
############################

def test_fixed_tolerance_overcounts_at_pole_on():
    """
    The reason the gap method exists. At pole-on the fast Wigner path leaves a
    cancellation floor (~1e-8 relative at L = 6, measured below and printed),
    which a 1e-10 relative tolerance counts as signal. The gap does not.
    """
    ops = channel_operators(direct_slice(L6, N_OBS6, 90), build_T(L6))
    sv = np.linalg.svd(ops['combined'], compute_uv=False)
    r_gap, sv_above, sv_below, gap = rank_by_gap(sv)
    r_tol = rank_by_tol(sv, 1e-10)
    floor = sv_below / sv[0]
    print(f'pole-on combined: gap rank {r_gap} (gap {gap:.1f} dec, floor '
          f'{floor:.1e} relative), 1e-10 tolerance rank {r_tol}')
    assert r_gap == 3
    assert r_tol > r_gap
    assert 1e-10 < floor < 1e-6      # the floor sits between the two criteria


############################
# Projector                #
############################

def test_projector_symmetric_idempotent_trace(svd6):
    """
    P = V diag(g) V^T: symmetric; P P = V diag(g^2) V^T so the departure from
    idempotency is bounded by max g(1 - g); trace(P) is the rank up to the sum
    of the gradings lost, both of which are measured and printed.
    """
    ops, Vt, sv, g = svd6
    rank = rank_by_gap(sv)[0]
    P = Vt.T @ (g[:, None] * Vt)

    asym = np.max(np.abs(P - P.T))
    idem = np.max(np.abs(P @ P - P))
    shoulder = float(np.max(g * (1 - g)))
    trace_gap = abs(np.trace(P) - rank)
    print(f'rank {rank}, asymmetry {asym:.2e}, |PP - P| {idem:.2e}, '
          f'max g(1-g) {shoulder:.2e}, |trace - rank| {trace_gap:.2e}')

    assert asym < 1e-14
    assert idem <= 10 * shoulder + 1e-14
    assert trace_gap < 1e-3          # measured 6e-6 at L = 6, beta = 33

    twice = apply_projector(Vt, g, apply_projector(Vt, g, np.ones(P.shape[0])))
    with_g2 = apply_projector(Vt, g ** 2, np.ones(P.shape[0]))
    assert np.allclose(twice, with_g2, atol=1e-12)


def test_observable_fraction_extremes(svd6):
    """
    A surface built from right singular vectors above the gap is fully
    observable; one built from the null space is not observable at all.
    """
    ops, Vt, sv, g = svd6
    rank = rank_by_gap(sv)[0]

    f_obs = Vt[:rank].sum(0)
    f_null = Vt[rank:].sum(0)
    frac_obs = observable_power_fraction(Vt, g, f_obs)
    frac_null = observable_power_fraction(Vt, g, f_null)
    print(f'observable surface: {frac_obs:.10f}  null surface: {frac_null:.2e}')
    assert frac_obs > 1 - 1e-4       # 1 - max grading loss, measured 5e-6
    assert frac_null < 1e-6


############################
# Degree map               #
############################

def test_degree_map_partitions_and_differs_from_floor_sqrt():
    """
    The map derived from starspot_sbi.indexing partitions the vector with
    2l + 1 entries per degree. floor(sqrt(arange(961))), the complex-index map
    the old notebooks used on real vectors, agrees on exactly 45 of 961 entries
    at L = 30 and starts [0,1,1,1,2,2,2,2,2,3,3,3] where the true map starts
    [0..11] (OPERATOR_MACHINERY.md part 1C).
    """
    ell = degree_map(30)
    wrong = np.floor(np.sqrt(np.arange(n_coeffs(30)))).astype(int)

    counts = np.bincount(ell, minlength=31)
    assert counts.sum() == n_coeffs(30)
    assert all(counts[l] == 2 * l + 1 for l in range(31))
    assert all(np.bincount(wrong, minlength=31)[l] == 2 * l + 1
               for l in range(31)), 'the count check that cannot catch it'

    n_agree = int(np.sum(ell == wrong))
    print(f'first twelve true {ell[:12].tolist()}, '
          f'floor-sqrt {wrong[:12].tolist()}, agree on {n_agree} of 961')
    assert ell[:12].tolist() == list(range(12))
    assert wrong[:12].tolist() == [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
    assert n_agree == 45


def test_per_degree_fraction_shape_range_and_trace(svd6):
    """
    One value per degree, all in [0, 1], and the fractions weighted by their
    2l + 1 modes sum to trace(P), which is the rank up to the grading. Note
    the degree-0 fraction is measurably below 1 (0.80 at L = 6, beta = 33):
    the DC mode mixes with higher even harmonics in every observable, so part
    of it lies in the null space.
    """
    ops, Vt, sv, g = svd6
    rank = rank_by_gap(sv)[0]
    frac = per_degree_fraction(Vt, g, degree_map(L6), L6)
    weighted = sum(frac[l] * (2 * l + 1) for l in range(L6 + 1))
    print('per-degree fractions:', np.array2string(frac, precision=4),
          f'  weighted sum {weighted:.4f}, rank {rank}')
    assert frac.shape == (L6 + 1,)
    assert np.all((frac >= 0) & (frac <= 1 + 1e-12))
    assert abs(weighted - np.sum(g)) < 1e-9      # exact identity with trace(P)
    assert abs(weighted - rank) < 1e-3           # grading loss, measured 6e-6


############################
# Shape correlation        #
############################

def test_shape_correlation_uses_tied_ranks():
    """
    Ties carry no ordering information, and scipy.stats.spearmanr gives them
    their average rank. On [1,1,2,2] against [1,2,1,2] the tied-rank
    correlation is exactly 0, while the argsort-of-argsort rank used before
    breaks the ties by position and reports 0.8.
    """
    a = [1.0, 1.0, 2.0, 2.0]
    b = [1.0, 2.0, 1.0, 2.0]
    tied = shape_correlation(a, b)
    ordinal = float(np.corrcoef(np.argsort(np.argsort(a)),
                                np.argsort(np.argsort(b)))[0, 1])
    print(f'tied-rank {tied:.6f}, ordinal {ordinal:.6f}')
    assert abs(tied) < 1e-12
    assert abs(ordinal - 0.8) < 1e-12


def test_shape_correlation_ends_and_constants():
    """
    Monotone agreement is 1, reversal is -1, and a constant curve has no
    ranking, so the correlation is undefined and reported as nan rather than
    as a number.
    """
    up = [0.0, 1.0, 2.0, 3.0]
    print(f'monotone {shape_correlation(up, [1.0, 4.0, 9.0, 16.0]):.6f}, '
          f'reversed {shape_correlation(up, up[::-1]):.6f}')
    assert abs(shape_correlation(up, [1.0, 4.0, 9.0, 16.0]) - 1.0) < 1e-12
    assert abs(shape_correlation(up, up[::-1]) + 1.0) < 1e-12
    assert np.isnan(shape_correlation([1.0, 1.0, 1.0], up[:3]))


############################
# N_eff                    #
############################

def test_neff_monotone_in_noise_and_bounded_by_rank(svd6):
    """More noise constrains fewer directions; N_eff never exceeds the rank."""
    ops, Vt, sv, g = svd6
    rank = rank_by_gap(sv)[0]
    values = [n_eff(sv, s) for s in NOISE_LEVELS]
    print(f'rank {rank}, N_eff at {NOISE_LEVELS}: '
          + ', '.join(f'{v:.3f}' for v in values))
    assert values[0] > values[1] > values[2]
    assert all(v <= rank + 1e-6 for v in values)


############################
# End to end               #
############################

@pytest.fixture(scope='module')
def analysis6(cache6, tmp_path_factory):
    """One timed full run at L = 6 over five inclinations, shared."""
    out = str(tmp_path_factory.mktemp('operator'))
    t0 = time.perf_counter()
    rank_rows, neff_rows, proj_rows = run_analysis(
        cache6, out, [0, 20, 33, 60, 90], L6, N_OBS6, N_BETA, progress=False)
    elapsed = time.perf_counter() - t0
    return out, rank_rows, neff_rows, proj_rows, elapsed


def test_analysis_at_l6_is_fast(analysis6):
    """The whole analysis at L = 6 in under a few seconds."""
    elapsed = analysis6[-1]
    print(f'five inclinations at L = 6: {elapsed:.2f} s')
    assert elapsed < 10.0


def test_analysis_writes_everything(analysis6):
    out, rank_rows, neff_rows, _, _ = analysis6
    for b in (0, 20, 33, 60, 90):
        assert os.path.exists(os.path.join(out, f'spectrum_beta{b:03d}.npy'))
        assert os.path.exists(os.path.join(out, f'per_degree_beta{b:03d}.csv'))
    for name in ('rank_vs_beta.csv', 'neff_vs_beta.csv',
                 'projector_vs_beta.csv', 'manifest.json'):
        assert os.path.exists(os.path.join(out, name))

    with open(os.path.join(out, 'manifest.json')) as f:
        manifest = json.load(f)
    assert manifest['l_max'] == L6
    assert manifest['spectrum_channel_order'] == list(CHANNELS)
    assert manifest['wall_seconds'] > 0
    assert manifest['projector'] == 'graded'
    assert manifest['projectors_computed'] == list(PROJECTORS)
    assert manifest['lambda_is_absolute'] is True
    assert set(manifest['trace_vs_rank']) == {'0', '20', '33', '60', '90'}

    spec = np.load(os.path.join(out, 'spectrum_beta033.npy'))
    assert spec.shape == (len(CHANNELS), n_coeffs(L6))

    with open(os.path.join(out, 'rank_vs_beta.csv')) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 * len(CHANNELS)
    trusted = [r for r in rows if r['gap_trusted'] == 'True']
    print(f'{len(trusted)} of {len(rows)} rows have a trusted gap')
    assert trusted, 'no trusted gaps at all would mean the method broke'
    for r in trusted:
        assert r['agree'] == 'True', (r['beta_deg'], r['channel'])


def test_neff_rows_name_the_uniform_levels_and_the_mission_pair(analysis6):
    """
    The uniform rows apply one per-sample level to every row of the operator;
    the mission row whitens each channel by its own sigma. A reader taking the
    mission value for a fourth member of the uniform series would be comparing
    two different quantities, so the labels state which is which, in the
    returned rows, in the CSV and in the manifest.
    """
    out, _, neff_rows, _, _ = analysis6
    want = {UNIFORM_LABEL.format(s) for s in NOISE_LEVELS} | {MISSION_LABEL}
    print('labels:', sorted(want))
    assert {r['sigma_n'] for r in neff_rows} == want
    assert len(neff_rows) == 5 * (len(NOISE_LEVELS) + 1)

    with open(os.path.join(out, 'neff_vs_beta.csv')) as f:
        rows = list(csv.DictReader(f))
    assert {r['sigma_n'] for r in rows} == want

    with open(os.path.join(out, 'manifest.json')) as f:
        labels = json.load(f)['neff_row_labels']
    assert labels[-1] == MISSION_LABEL
    assert set(labels) == want


def test_mission_neff_is_the_whitened_operator_not_a_uniform_level(analysis6):
    """
    The mission row is routed through n_eff_mission, which whitens the astro
    and photometric rows separately before the SVD. It therefore reproduces a
    direct whitened SVD and matches none of the uniform values.
    """
    _, _, neff_rows, _, _ = analysis6
    ops = channel_operators(direct_slice(L6, N_OBS6, 33), build_T(L6))
    direct = n_eff_mission(ops['x'], ops['y'], ops['phot'])

    at33 = [r for r in neff_rows if r['beta_deg'] == 33]
    mission = [r['n_eff'] for r in at33 if r['sigma_n'] == MISSION_LABEL][0]
    uniform = [r['n_eff'] for r in at33 if r['sigma_n'] != MISSION_LABEL]
    rel = abs(mission - direct) / direct
    print(f'mission {mission:.6f}, direct {direct:.6f}, relative difference '
          f'{rel:.2e}; uniform ' + ', '.join(f'{u:.4f}' for u in uniform))
    assert rel < 1e-4          # cache and direct build agree to their dtype
    assert all(abs(mission - u) > 1e-6 for u in uniform)


def test_per_degree_csv_carries_both_projectors_and_a_trace_footer(analysis6):
    """
    The primary column follows --projector, both gradings are written whatever
    it is, and the footer row states the trace against the rank.
    """
    out, _, _, proj_rows, _ = analysis6
    with open(os.path.join(out, 'per_degree_beta033.csv')) as f:
        rows = list(csv.DictReader(f))
    degrees = [r for r in rows if r['degree'] != 'trace']
    footer = [r for r in rows if r['degree'] == 'trace'][0]

    assert len(degrees) == L6 + 1
    assert all(r['observable_fraction'] == r['observable_fraction_graded']
               for r in degrees), 'default primary column is the graded one'

    want = {r['beta_deg']: r for r in proj_rows}[33]
    print(f'footer: graded {footer["observable_fraction_graded"]}, hard '
          f'{footer["observable_fraction_hard"]}, rank {footer["n_modes"]}')
    assert int(footer['n_modes']) == want['rank']
    assert abs(float(footer['observable_fraction_hard']) - want['rank']) < 1e-9
    assert float(footer['observable_fraction_graded']) < want['rank']


def test_hard_projector_trace_is_the_rank(cache6):
    """
    The definition: a hard cut at the spectral gap has trace exactly equal to
    the rank, since its grading is a vector of ones and zeros.
    """
    T = build_T(L6)
    ell = degree_map(L6)
    for beta_deg in (0, 33, 60):
        ops = channel_operators(direct_slice(L6, N_OBS6, beta_deg), T)
        Vt, sv, _ = graded_svd(ops['combined'])
        rank = rank_by_gap(sv)[0]
        frac = per_degree_fraction(Vt, hard_weights(sv, rank), ell, L6)
        trace = projector_trace(frac)
        print(f'beta {beta_deg:>2}: rank {rank}, hard trace {trace:.12f}, '
              f'error {abs(trace - rank):.2e}')
        assert abs(trace - rank) < 1e-10       # measured 4e-13


def test_graded_trace_is_below_the_rank_and_rises_as_lambda_falls(cache6):
    """
    The deficit is the diagnostic: the graded trace never reaches the rank and
    approaches it monotonically as lambda falls. At L = 30, beta = 33 the
    measured deficits are 3 per cent at lambda = 1e-9 and 46 per cent at 1e-4;
    the same ordering is asserted here at L = 6.
    """
    T = build_T(L6)
    ell = degree_map(L6)
    ops = channel_operators(direct_slice(L6, N_OBS6, 33), T)
    Vt, sv, _ = graded_svd(ops['combined'])
    rank = rank_by_gap(sv)[0]

    lams = (1e-4, 1e-6, 1e-9, 1e-12)
    traces = []
    for lam in lams:
        g = sv ** 2 / (sv ** 2 + lam)
        traces.append(projector_trace(per_degree_fraction(Vt, g, ell, L6)))
        print(f'lambda {lam:.0e}: trace {traces[-1]:8.4f} of rank {rank}, '
              f'deficit {1 - traces[-1] / rank:6.1%}')

    assert all(t < rank for t in traces)
    assert all(a < b for a, b in zip(traces, traces[1:]))
    assert traces[-1] > 0.99 * rank            # 1e-12 is nearly a hard cut


def test_the_two_projectors_agree_on_shape(cache6):
    """
    Level and shape are different questions: the gradings differ most at high
    degree, but they must still order the degrees the same way, so a figure
    drawn with either identifies the same degrees as most and least
    observable.

    At L = 6 the default lambda = 1e-9 lies far below the smallest retained
    singular value squared, so the two projectors coincide to four figures and
    the comparison is empty. Lambda is therefore set here to ten times
    sv[rank-1]^2, which reproduces at L = 6 the level divergence measured at
    L = 30 with the inherited constant.
    """
    T = build_T(L6)
    ell = degree_map(L6)
    for beta_deg in (0, 20, 33, 60):
        ops = channel_operators(direct_slice(L6, N_OBS6, beta_deg), T)
        Vt, sv, _ = graded_svd(ops['combined'])
        rank = rank_by_gap(sv)[0]

        lam = 10 * sv[rank - 1] ** 2
        g = sv ** 2 / (sv ** 2 + lam)
        f_graded = per_degree_fraction(Vt, g, ell, L6)
        f_hard = per_degree_fraction(Vt, hard_weights(sv, rank), ell, L6)

        corr = shape_correlation(f_graded, f_hard)
        ratio = f_graded / np.maximum(f_hard, 1e-30)
        same_max = np.argmax(f_graded) == np.argmax(f_hard)
        same_min = np.argmin(f_graded) == np.argmin(f_hard)
        print(f'beta {beta_deg:>2}: lambda {lam:.2e}, shape correlation '
              f'{corr:.6f}, level ratio {ratio.min():.4f} to {ratio.max():.4f}, '
              f'trace {projector_trace(f_graded):.2f} of rank {rank}, '
              f'argmax agrees {same_max}, argmin agrees {same_min}')
        assert ratio.min() < 0.9, 'lambda is not biting; the test is empty'
        assert corr > 0.95
        assert same_max

    # At the inherited lambda the two coincide, including the weakest degree.
    ops = channel_operators(direct_slice(L6, N_OBS6, 33), T)
    Vt, sv, g = graded_svd(ops['combined'], LAM)
    rank = rank_by_gap(sv)[0]
    f_graded = per_degree_fraction(Vt, g, ell, L6)
    f_hard = per_degree_fraction(Vt, hard_weights(sv, rank), ell, L6)
    worst = float(np.max(np.abs(f_graded - f_hard)))
    print(f'at lambda = {LAM:.0e}, beta 33: max per-degree difference {worst:.2e}, '
          f'argmin agrees {np.argmin(f_graded) == np.argmin(f_hard)}')
    assert worst < 1e-4                        # measured 3e-6
    assert np.argmin(f_graded) == np.argmin(f_hard)


def test_projector_choice_moves_the_primary_column(cache6, tmp_path):
    """--projector hard fills observable_fraction with the hard values."""
    out = str(tmp_path / 'hard')
    _, _, proj_rows = run_analysis(cache6, out, [33], L6, N_OBS6, N_BETA,
                                   projector='hard', progress=False)
    with open(os.path.join(out, 'per_degree_beta033.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['degree'] != 'trace']
    assert all(r['observable_fraction'] == r['observable_fraction_hard']
               for r in rows)
    with open(os.path.join(out, 'manifest.json')) as f:
        assert json.load(f)['projector'] == 'hard'
    print(f'hard deficit {proj_rows[0]["deficit_hard"]:.2e}, '
          f'graded deficit {proj_rows[0]["deficit_graded"]:.2%}')
    assert abs(proj_rows[0]['deficit_hard']) < 1e-12


def test_run_analysis_refuses_an_unknown_projector(cache6, tmp_path):
    with pytest.raises(ValueError, match='projector'):
        run_analysis(cache6, str(tmp_path / 'x'), [0], L6, N_OBS6, N_BETA,
                     projector='soft', progress=False)


def test_run_analysis_refuses_an_absent_cache(tmp_path):
    """The failure is load_design's own message, naming the build command."""
    with pytest.raises(FileNotFoundError, match='build_caches'):
        run_analysis(str(tmp_path / 'nothing'), str(tmp_path / 'out'),
                     [0], L6, N_OBS6, N_BETA, progress=False)