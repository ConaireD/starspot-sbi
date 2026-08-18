"""
Analytic characterisation of the measurement operator A(beta), from the
design-matrix cache alone.

Everything computed here is a property of the measurement:

    rank            by the largest gap in the log10 singular spectrum, with the
                    bracketing singular values and the gap width in decades, so
                    a reader can see when the gap stops being trustworthy
    spectra         the singular values per inclination and channel, since the
                    conditioning argument rests on the decay of the small ones
    N_eff           sum_i sv_i^2 / (sv_i^2 + sigma_n^2), the covariance-weighted
                    count of constrained directions, which degrades smoothly
                    where the rank does not, reported at uniform per-sample
                    levels and at the whitened mission pair
    projector       the observable-power projector P = V diag(g) V^T, in two
                    gradings from the one SVD: graded, g = sv^2/(sv^2 + lambda),
                    and hard, g = 1 above the spectral gap and 0 below
    per degree      the observable power fraction of a unit-power surface at
                    each degree, an operator-level ceiling on recovery, with
                    the trace sum_l (2l+1) frac_l reported against the rank so
                    a reader can tell a near-hard curve from a regularised one

Usage:
    python scripts/operator_analysis.py --betas 0,5,15,30,45,60,75,85,90
    python scripts/operator_analysis.py --all-betas --out results/operator
    python scripts/operator_analysis.py --projector hard
    python scripts/operator_analysis.py --l-max 8 --check-closed-forms
"""

###########
# Imports #
###########

# python
import os
import csv
import json
import time
import argparse
from datetime import datetime

# standard
import numpy as np
from scipy.stats import spearmanr
from tqdm.auto import tqdm

# self
import starspot_sbi
from starspot_sbi.indexing import n_coeffs, real_to_coeffs, idx_to_lm
from build_caches import load_design, DEFAULT_CACHE_DIR, L_MAX, N_OBS, N_BETA


#############
# Constants #
#############

# Graded-projector regularisation, absolute, not relative to sv[0]^2. Matches
# NewMetrics/crap/02_InformationContent_S.ipynb cell 3 (LAM = 1e-9), which is
# the implementation that produced Paper 2's numbers. The constant's provenance
# is unexplained there (OPERATOR_MACHINERY.md part 1B); it is kept for
# comparability.
LAM = 1e-9

# 'graded' is the default for comparability with Paper 2; 'hard' cuts at the
# spectral gap the rank is read from and carries no free parameter.
PROJECTORS = ('graded', 'hard')
DEFAULT_PROJECTOR = 'graded'

# Noise levels of the existing figure, fractional per-sample. The mission point
# is the per-channel pair from starspot_sbi.pipeline: sigma_phot = 1e-4,
# sigma_astro = 10^-3.5.
NOISE_LEVELS = (1e-4, 1e-3, 1e-2)
SIGMA_PHOT_MISSION = 1e-4
SIGMA_ASTRO_MISSION = 10.0 ** -3.5

# Row labels of neff_vs_beta.csv. The uniform rows apply one per-sample level
# to every row of the operator; the mission row whitens each channel by its own
# sigma. Reading the mission value as a fourth member of the uniform series
# would be wrong, so the labels carry the distinction.
UNIFORM_LABEL = 'uniform_{:.0e}'
MISSION_LABEL = 'mission_whitened'

# Channel order used everywhere below, including the rows of the spectrum
# files. The cache itself stores (astro_x, astro_y, phot).
CHANNELS = ('x', 'y', 'astro', 'phot', 'combined')

# Singular values below this fraction of sv[0] are denormal junk from the SVD,
# not part of the spectrum. The true-zero cluster of a well-conditioned
# operator sits at ~1e-16 relative (rank_scan.py, L = 4-8)
JUNK_FLOOR = 1e-20

# A gap this wide in decades is treated as trustworthy. Measured on the
# production cache at L = 30 (RANK_RESOLUTION.md): the gap is 5.0 decades at
# beta = 45 deg where the rank is right, and 3.2 decades at beta = 60 deg where
# the numerically non-zero x m = 0 row is miscounted. 4 sits between.
GAP_TRUST_DECADES = 4.0

DEFAULT_OUT_DIR = os.path.join('results', 'operator')


######################
# Basis and indexing #
######################

def build_T(l_max):
    """
    Real-to-complex transform T, so that Re(A T) acts on section-packed real
    vectors. Column j is real_to_coeffs of the j-th unit vector.
    """
    n = n_coeffs(l_max)
    T = np.zeros((n, n), dtype=complex)
    e = np.zeros(n)
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        T[:, j] = real_to_coeffs(e)
    return T


def degree_map(l_max):
    """
    Degree of each entry of a section-packed real vector, derived by pushing
    unit vectors through real_to_coeffs rather than assuming the packing.

    floor(sqrt(arange(n))) is the degree map for the complex l^2 + l + m index
    and is wrong for this basis: at L = 30 the two agree on 45 of 961 entries,
    and both give 2l + 1 entries per degree, so a count check does not catch it
    (OPERATOR_MACHINERY.md part 1C).
    """
    n = n_coeffs(l_max)
    ell = np.empty(n, dtype=int)
    e = np.zeros(n)
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        nz = np.flatnonzero(np.abs(real_to_coeffs(e)) > 0)
        degrees = {idx_to_lm(int(i))[0] for i in nz}
        if len(degrees) != 1:
            raise ValueError(f'real index {j} maps to degrees {degrees}')
        ell[j] = degrees.pop()
    return ell


def channel_operators(A_beta, T):
    """
    Real operators per channel from one cache slice of shape
    (3, n_obs, n_coeffs), cache channel order (astro_x, astro_y, phot).
    """
    Gx = np.real(np.asarray(A_beta[0]) @ T)
    Gy = np.real(np.asarray(A_beta[1]) @ T)
    Gp = np.real(np.asarray(A_beta[2]) @ T)
    return {'x': Gx, 'y': Gy, 'astro': np.vstack([Gx, Gy]),
            'phot': Gp, 'combined': np.vstack([Gx, Gy, Gp])}


##################
# Rank and N_eff #
##################

def rank_by_gap(sv):
    """
    Rank at the largest gap in the log10 spectrum.

    Returns (rank, sv_above, sv_below, gap_decades). The gap is 13 to 15
    decades wide wherever the operator is well conditioned (rank_scan.py,
    L = 4-8) and narrows smoothly toward pole-on; judge it by gap_decades
    against GAP_TRUST_DECADES rather than trusting the rank unconditionally.
    """
    sv = np.asarray(sv, dtype=float)
    sv = sv[sv > JUNK_FLOOR * sv[0]]
    if sv.size < 2:
        return sv.size, float(sv[0]) if sv.size else np.nan, np.nan, np.inf
    logs = np.log10(sv)
    gaps = logs[:-1] - logs[1:]
    i = int(np.argmax(gaps))
    return i + 1, float(sv[i]), float(sv[i + 1]), float(gaps[i])


def rank_by_tol(sv, tol):
    """
    Rank at a fixed relative tolerance. Kept only as the comparison the gap
    method exists to replace: at pole-on the Wigner-d cancellation floor is
    5e-7 at l = 30 (docs/conventions.md section 7), so 1e-10 counts that floor
    as signal and over-counts, 64 where the true rank is 3.
    """
    sv = np.asarray(sv, dtype=float)
    return int(np.sum(sv > tol * sv[0]))


def n_eff(sv, sigma_n):
    """
    Covariance-weighted count of constrained directions,
    sum_i sv_i^2 / (sv_i^2 + sigma_n^2), for one per-sample noise level applied
    to every row of the operator.
    """
    s2 = np.asarray(sv, dtype=float) ** 2
    return float(np.sum(s2 / (s2 + sigma_n ** 2)))


def n_eff_mission(Gx, Gy, Gp, sigma_phot=SIGMA_PHOT_MISSION,
                  sigma_astro=SIGMA_ASTRO_MISSION):
    """
    N_eff of the noise-whitened combined operator at the mission noise pair,
    which is sum_i svw_i^2 / (svw_i^2 + 1) for the whitened singular values.
    """
    Gw = np.vstack([Gx / sigma_astro, Gy / sigma_astro, Gp / sigma_phot])
    svw = np.linalg.svd(Gw, compute_uv=False)
    return n_eff(svw, 1.0)


################
# Closed forms #
################

def closed_form_rank(l_max, channel, beta_deg):
    """
    Corrected closed-form rank (RANK_RESOLUTION.md, verified against gap-SVD at
    L = 3-8 and 30). Lo and Le are the largest odd and even degrees <= L.

        beta = 0     x Lo+3   y Lo+2   astro 2Lo+5   phot Le+3   combined sum
        interior     x 2Lo    y 2Lo+1  astro 4Lo-3   phot 2Le+1  combined sum
        beta = 90    x 2      y 2      astro 2       phot 1      combined 3

    Special cases inside Taaki's stated domain L > 2: at Lo = 3 (L = 3, 4) the
    (2, +-2) and (2, +-1) kernel entries leave only two proportional x/y row
    pairs interior, so astro is 4Lo+1-2; and at L = 3 the combined operator
    loses a further 2 to a cross-channel dependency at m = +-2, at beta = 0 and
    interior alike. Returns None below L = 3, where the forms are unverified.
    """
    if l_max < 3:
        return None
    Lo = l_max if l_max % 2 == 1 else l_max - 1
    Le = l_max if l_max % 2 == 0 else l_max - 1

    if beta_deg == 0:
        forms = {'x': Lo + 3, 'y': Lo + 2, 'astro': 2 * Lo + 5, 'phot': Le + 3}
    elif beta_deg == 90:
        forms = {'x': 2, 'y': 2, 'astro': 2, 'phot': 1}
    else:
        pairs_lost = 2 if Lo == 3 else 4
        forms = {'x': 2 * Lo, 'y': 2 * Lo + 1,
                 'astro': 2 * Lo + (2 * Lo + 1) - pairs_lost,
                 'phot': 2 * Le + 1}

    cross = 2 if (l_max == 3 and beta_deg != 90) else 0
    forms['combined'] = forms['astro'] + forms['phot'] - cross
    return forms[channel]


#####################
# Graded projector  #
#####################

def graded_svd(G, lam=LAM):
    """
    SVD and grading, matching NewMetrics/crap/02_InformationContent_S.ipynb
    cell 3. Returns (Vt, sv, g) with g = sv^2 / (sv^2 + lam), lam absolute.
    """
    _, sv, Vt = np.linalg.svd(G, full_matrices=False)
    return Vt, sv, sv ** 2 / (sv ** 2 + lam)


def hard_weights(sv, rank):
    """
    Grading of the hard-cut projector: 1 on the first rank directions and 0
    below, the cut taken at the spectral gap the rank is already read from. It
    carries no free parameter, which is what makes it the honest object for an
    operator-level ceiling.
    """
    g = np.zeros(np.asarray(sv).size)
    g[:rank] = 1.0
    return g


def apply_projector(Vt, g, f):
    """P f for the graded projector P = V diag(g) V^T, without forming P."""
    return Vt.T @ (g * (Vt @ f))


def observable_power_fraction(Vt, g, f):
    """||P f|| / ||f||, the amplitude fraction of f in the observable subspace."""
    nf = np.linalg.norm(f)
    return float(np.linalg.norm(apply_projector(Vt, g, f)) / nf) if nf > 0 else np.nan


def variance_reduction(Vt, g):
    """diag(V diag(g) V^T): per-coefficient observable power fraction."""
    return (Vt ** 2 * g[:, None]).sum(0)


def per_degree_fraction(Vt, g, ell, l_max):
    """
    Observable power fraction per degree: the mean of variance_reduction over
    the 2l + 1 entries of each degree, which is the expected observable power
    of an isotropic unit-power surface confined to that degree. An
    operator-level ceiling on per-degree recovery.
    """
    S = variance_reduction(Vt, g)
    return np.array([S[ell == l].mean() for l in range(l_max + 1)])


def projector_trace(frac):
    """
    sum_l (2l+1) frac_l, which is trace(P) and the sum of the grading weights.

    For the hard projector it is the rank exactly. For the graded one the
    shortfall against the rank measures how much of the projector is
    regulariser rather than geometry: measured at L = 30, beta = 33 deg,
    combined channels, rank 174, the trace is 168.55 at lambda = 1e-9 (a 3 per
    cent deficit) and 93.74 at lambda = 1e-4 (46 per cent).
    """
    frac = np.asarray(frac, dtype=float)
    return float(np.sum((2 * np.arange(frac.size) + 1) * frac))


def shape_correlation(a, b):
    """
    Spearman rank correlation between two per-degree curves, so that a
    disagreement in shape can be told from a disagreement in level.

    scipy.stats.spearmanr assigns tied values their average rank. The
    argsort-of-argsort rank used previously broke ties by position and reported
    an ordering where the curves carry none. Returns nan for a constant curve,
    where the correlation is undefined.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


############
# Analysis #
############

def analyse_beta(A_beta, T, ell, l_max, noise_levels=NOISE_LEVELS, lam=LAM):
    """
    Everything for one inclination: per-channel rank with the gap bracket and
    condition number, per-channel spectra, combined-operator N_eff at each
    uniform noise level and at the whitened mission pair, and the per-degree
    observable fraction of the combined operator.
    """
    ops = channel_operators(A_beta, T)
    out = {'rank': {}, 'spectrum': {}}

    for ch in CHANNELS:
        G = ops[ch]
        if ch == 'combined':
            Vt, sv, g_graded = graded_svd(G, lam)
        else:
            sv = np.linalg.svd(G, compute_uv=False)
        r, sv_above, sv_below, gap = rank_by_gap(sv)
        out['spectrum'][ch] = sv
        out['rank'][ch] = {'rank': r, 'sv_above': sv_above, 'sv_below': sv_below,
                           'gap_decades': gap,
                           'condition': float(sv[0] / sv[r - 1]) if r >= 1 else np.nan}

        if ch == 'combined':
            # Both projectors from the one SVD: the grading is the only thing
            # that differs, so the second costs a multiply.
            weights = {'graded': g_graded, 'hard': hard_weights(sv, r)}
            out['per_degree'] = {k: per_degree_fraction(Vt, w, ell, l_max)
                                 for k, w in weights.items()}
            out['trace'] = {k: projector_trace(f)
                            for k, f in out['per_degree'].items()}
            out['trace_deficit'] = {k: 1.0 - t / r if r else np.nan
                                    for k, t in out['trace'].items()}
            out['shape_correlation'] = shape_correlation(
                out['per_degree']['graded'], out['per_degree']['hard'])

    sv_c = out['spectrum']['combined']
    out['n_eff'] = {UNIFORM_LABEL.format(s): n_eff(sv_c, s) for s in noise_levels}
    out['n_eff'][MISSION_LABEL] = n_eff_mission(ops['x'], ops['y'], ops['phot'])
    return out


def run_analysis(cache_dir, out_dir, betas, l_max=L_MAX, n_obs=N_OBS,
                 n_beta=N_BETA, noise_levels=NOISE_LEVELS, lam=LAM,
                 projector=DEFAULT_PROJECTOR, write=True, progress=True):
    """
    The full sweep. Loads the design cache (raising with load_design's message
    if absent), analyses each requested inclination, writes the CSVs, spectra
    and manifest, and returns (rank_rows, neff_rows, projector_rows).

    Both projectors are computed whatever `projector` is set to, since they
    share the SVD; the argument selects which one fills the primary
    observable_fraction column and the terminal summary.
    """
    if projector not in PROJECTORS:
        raise ValueError(f'projector must be one of {PROJECTORS}, got {projector!r}')

    t0 = time.perf_counter()
    A = load_design(cache_dir, l_max, n_obs, n_beta)
    T = build_T(l_max)
    ell = degree_map(l_max)

    if write:
        os.makedirs(out_dir, exist_ok=True)

    rank_rows, neff_rows, proj_rows = [], [], []
    n = n_coeffs(l_max)

    it = tqdm(betas, desc=f'operator L{l_max} N{n_obs}', unit='beta',
              disable=not progress)
    for b in it:
        res = analyse_beta(A[b], T, ell, l_max, noise_levels, lam)

        for ch in CHANNELS:
            r = res['rank'][ch]
            cf = closed_form_rank(l_max, ch, b)
            rank_rows.append({
                'beta_deg': b, 'channel': ch, 'rank': r['rank'],
                'closed_form': cf,
                'agree': (r['rank'] == cf) if cf is not None else '',
                'gap_trusted': r['gap_decades'] >= GAP_TRUST_DECADES,
                'sv_above_gap': r['sv_above'], 'sv_below_gap': r['sv_below'],
                'gap_decades': r['gap_decades'], 'condition': r['condition'],
            })
        for label, value in res['n_eff'].items():
            neff_rows.append({'beta_deg': b, 'sigma_n': label, 'n_eff': value})

        rank_c = res['rank']['combined']['rank']
        proj_rows.append({
            'beta_deg': b, 'rank': rank_c,
            'trace_graded': res['trace']['graded'],
            'trace_hard': res['trace']['hard'],
            'deficit_graded': res['trace_deficit']['graded'],
            'deficit_hard': res['trace_deficit']['hard'],
            'shape_correlation': res['shape_correlation'],
        })

        if write:
            spec = np.full((len(CHANNELS), n), np.nan)
            for i, ch in enumerate(CHANNELS):
                sv = res['spectrum'][ch]
                spec[i, :sv.size] = sv
            np.save(os.path.join(out_dir, f'spectrum_beta{b:03d}.npy'), spec)

            primary = res['per_degree'][projector]
            with open(os.path.join(out_dir, f'per_degree_beta{b:03d}.csv'),
                      'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['degree', 'observable_fraction',
                            'observable_fraction_graded',
                            'observable_fraction_hard', 'n_modes'])
                for l in range(l_max + 1):
                    w.writerow([l, f'{primary[l]:.10g}',
                                f'{res["per_degree"]["graded"][l]:.10g}',
                                f'{res["per_degree"]["hard"][l]:.10g}',
                                2 * l + 1])
                # Footer: the trace identity sum_l (2l+1) frac_l against the
                # rank, which says whether a curve is nearly a hard projector
                # or mostly regulariser.
                w.writerow(['trace', f'{res["trace"][projector]:.10g}',
                            f'{res["trace"]["graded"]:.10g}',
                            f'{res["trace"]["hard"]:.10g}', rank_c])

    if write:
        _write_rows(os.path.join(out_dir, 'rank_vs_beta.csv'), rank_rows)
        _write_rows(os.path.join(out_dir, 'neff_vs_beta.csv'), neff_rows)
        _write_rows(os.path.join(out_dir, 'projector_vs_beta.csv'), proj_rows)
        manifest = {
            'l_max': int(l_max), 'n_obs': int(n_obs), 'n_beta': int(n_beta),
            'betas_deg': [int(b) for b in betas],
            'noise_levels': list(noise_levels),
            'neff_row_labels': ([UNIFORM_LABEL.format(s) for s in noise_levels]
                                + [MISSION_LABEL]),
            'sigma_phot_mission': SIGMA_PHOT_MISSION,
            'sigma_astro_mission': SIGMA_ASTRO_MISSION,
            'lambda': lam, 'lambda_is_absolute': True,
            'projector': projector, 'projectors_computed': list(PROJECTORS),
            'trace_vs_rank': {str(r['beta_deg']):
                              {'rank': r['rank'],
                               'trace_graded': r['trace_graded'],
                               'trace_hard': r['trace_hard'],
                               'deficit_graded': r['deficit_graded'],
                               'shape_correlation': r['shape_correlation']}
                              for r in proj_rows},
            'gap_trust_decades': GAP_TRUST_DECADES,
            'spectrum_channel_order': list(CHANNELS),
            'operator': 'Re(A T), section-packed real basis',
            'cache_dir': os.path.abspath(cache_dir),
            'package_version': getattr(starspot_sbi, '__version__', 'unknown'),
            'wall_seconds': round(time.perf_counter() - t0, 3),
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

    return rank_rows, neff_rows, proj_rows


def _write_rows(path, rows):
    """CSV from a list of homogeneous dicts."""
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


#############
# Reporting #
#############

def print_summary(rank_rows, neff_rows, proj_rows=None, projector=DEFAULT_PROJECTOR):
    """The rank table, the mission N_eff, and the projector trace against rank."""
    print(f'\n{"beta":>5} {"channel":>9} {"rank":>5} {"closed":>7} {"agree":>6} '
          f'{"gap(dec)":>9} {"bracket":>21} {"condition":>10}')
    for r in rank_rows:
        agree = {True: 'yes', False: 'NO', '': '-'}[r['agree']]
        trust = '' if r['gap_trusted'] else '  (gap untrusted)'
        print(f'{r["beta_deg"]:>5} {r["channel"]:>9} {r["rank"]:>5} '
              f'{str(r["closed_form"]):>7} {agree:>6} {r["gap_decades"]:>9.1f} '
              f'{r["sv_above_gap"]:>10.2e}/{r["sv_below_gap"]:<10.2e} '
              f'{r["condition"]:>10.2e}{trust}')
    mission = {r['beta_deg']: r['n_eff'] for r in neff_rows
               if r['sigma_n'] == MISSION_LABEL}
    if mission:
        print(f'\nN_eff at the whitened mission pair (sigma_phot '
              f'{SIGMA_PHOT_MISSION:.0e}, sigma_astro {SIGMA_ASTRO_MISSION:.2e}):')
        for b, v in mission.items():
            print(f'  beta {b:>3} deg: {v:8.2f}')

    if proj_rows:
        print(f'\nprojector trace against rank (primary: {projector})')
        print(f'{"beta":>5} {"rank":>5} {"tr graded":>10} {"deficit":>8} '
              f'{"tr hard":>8} {"deficit":>8} {"shape corr":>11}')
        for r in proj_rows:
            print(f'{r["beta_deg"]:>5} {r["rank"]:>5} {r["trace_graded"]:>10.2f} '
                  f'{r["deficit_graded"]:>7.1%} {r["trace_hard"]:>8.2f} '
                  f'{r["deficit_hard"]:>7.1%} {r["shape_correlation"]:>11.4f}')
        worst = max(proj_rows, key=lambda r: r['deficit_graded'])
        print(f'largest graded deficit: {worst["deficit_graded"]:.1%} at beta '
              f'{worst["beta_deg"]} deg. A deficit near zero means the graded '
              f'curve is nearly the hard projector; a large one means it is '
              f'mostly regulariser.')
        low = [r for r in proj_rows if r['shape_correlation'] < 0.9]
        if low:
            print('shape correlation below 0.9 at beta '
                  + ', '.join(str(r['beta_deg']) for r in low)
                  + ': the two projectors disagree in shape, not only in level.')


###############
# Entry point #
###############

def parse_betas(spec):
    """'0,5,15' to a sorted list of integer degrees."""
    return sorted({int(x) for x in spec.split(',') if x.strip() != ''})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', default=os.environ.get('STARSPOT_CACHE_DIR',
                                                          DEFAULT_CACHE_DIR))
    ap.add_argument('--out', default=DEFAULT_OUT_DIR)
    ap.add_argument('--l-max', type=int, default=L_MAX)
    ap.add_argument('--n-obs', type=int, default=N_OBS)
    ap.add_argument('--n-beta', type=int, default=N_BETA,
                    help='inclination count of the cache being read')
    ap.add_argument('--betas', default='0,5,15,30,45,60,75,85,90',
                    help='comma-separated integer degrees')
    ap.add_argument('--all-betas', action='store_true',
                    help='every inclination in the cache')
    ap.add_argument('--noise', default=','.join(f'{s:g}' for s in NOISE_LEVELS),
                    help='comma-separated per-sample noise levels for N_eff')
    ap.add_argument('--lam', type=float, default=LAM,
                    help='graded-projector regularisation, absolute')
    ap.add_argument('--projector', choices=PROJECTORS, default=DEFAULT_PROJECTOR,
                    help='which projector fills the primary column; both are '
                         'computed and written whichever is chosen')
    ap.add_argument('--check-closed-forms', action='store_true',
                    help='rank table only, no files; exit 1 if a trusted gap '
                         'disagrees with the closed form')
    args = ap.parse_args()

    betas = (list(range(args.n_beta)) if args.all_betas
             else parse_betas(args.betas))
    bad = [b for b in betas if not 0 <= b < args.n_beta]
    if bad:
        ap.error(f'betas {bad} outside the cache range 0..{args.n_beta - 1}')
    noise = tuple(float(s) for s in args.noise.split(','))

    rank_rows, neff_rows, proj_rows = run_analysis(
        args.cache_dir, args.out, betas, args.l_max, args.n_obs, args.n_beta,
        noise, args.lam, args.projector, write=not args.check_closed_forms)

    print_summary(rank_rows, neff_rows, proj_rows, args.projector)

    if args.check_closed_forms:
        bad = [r for r in rank_rows if r['gap_trusted'] and r['agree'] is False]
        if bad:
            print(f'\n{len(bad)} trusted-gap disagreement(s) with the closed forms')
            raise SystemExit(1)
        print('\nall trusted-gap ranks agree with the closed forms')
    else:
        print(f'\nwritten to {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()