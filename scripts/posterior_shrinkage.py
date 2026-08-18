"""
Posterior shrinkage of the SBI posterior against the analytic linear-Gaussian
baseline, per singular direction of the measurement operator.

Write G = Re(A(beta) T) for the operator in the real-packed basis, whitened per
channel by the noise point, with SVD singular values sigma_i and right singular
vectors v_i. With a Gaussian prior of variance tau_i^2 along v_i, the
linear-Gaussian posterior variance along v_i is

    var_i = sigma_n^2 / (sigma_i^2 + sigma_n^2 / tau_i^2)

so the shrinkage 1 - var_i / tau_i^2 = sigma_i^2 / (sigma_i^2 + sigma_n^2 /
tau_i^2), and its sum over i is N_eff. That identity is exact for a prior
diagonal in the v_i basis; with the training marginal, which is diagonal in the
coefficient basis instead, it is an approximation whose size is measured
against the exact posterior solve and reported in the summary.

The SBI posterior's per-direction variance comes from the coefficient draws
written by run_holdout.py --save-draws, projected onto the same v_i. Its
summed shrinkage is an effective dimension directly comparable to N_eff; the
difference is what the network learned, since no prior can add information in
the row space but a prior can and does narrow the null space.

Calibration caveat, printed with every run: the paper's section 7.4 measures
the SBI posterior as mildly overconfident, TARP coverage about three per cent
below nominal, so some of the measured shrinkage is overconfidence rather than
information. That section supports no variance correction factor, so raw
numbers only are reported here.

The operator construction and the gap rank are reused from
scripts/operator_analysis.py; the SVD is recomputed here because that script
stores only the spectra, not the right singular vectors.

Usage:
    python scripts/posterior_shrinkage.py --family phot_axay --n 200
    python scripts/posterior_shrinkage.py --family phot --holdout-results results/holdout
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
from tqdm.auto import tqdm

# self
import starspot_sbi
from starspot_sbi.indexing import n_coeffs
from build_caches import load_design, DEFAULT_CACHE_DIR, L_MAX, N_OBS, N_BETA
from operator_analysis import (build_T, channel_operators, rank_by_gap,
                               GAP_TRUST_DECADES)


#############
# Constants #
#############

LOG10_SIGMA_PHOT_MISSION = -4.0
LOG10_SIGMA_ASTRO_MISSION = -3.5

# Cache channel keys per family, matching starspot_sbi.models.FAMILIES.
FAMILY_CHANNELS = {
    'phot':      ('phot',),
    'phot_ax':   ('phot', 'x'),
    'phot_ay':   ('phot', 'y'),
    'phot_axay': ('phot', 'x', 'y'),
}

DEFAULT_VAE = os.path.join('weights', 'vae_n640000_seed101_temp.pt')
DEFAULT_OUT_DIR = os.path.join('results', 'shrinkage')

CALIBRATION_CAVEAT = (
    'Calibration caveat: the SBI posterior is mildly overconfident (TARP '
    'coverage about three per cent below nominal, paper section 7.4), so some '
    'of the measured shrinkage is overconfidence rather than information. No '
    'correction factor is supported by that section; the numbers below are raw.')


######################
# Analytic quantities #
######################

def whitened_operator(ops, family, sigma_phot, sigma_astro):
    """
    The stacked operator for one family, each channel divided by its noise, so
    the closed forms below apply with sigma_n = 1.
    """
    sigma = {'phot': sigma_phot, 'x': sigma_astro, 'y': sigma_astro}
    return np.vstack([ops[ch] / sigma[ch] for ch in FAMILY_CHANNELS[family]])


def operator_svd(G):
    """
    Full SVD of the operator: (sv, Vt) with sv zero-padded to the coefficient
    count, so every direction of the domain appears, the null ones with
    sigma_i = 0 exactly.
    """
    _, sv, Vt = np.linalg.svd(G, full_matrices=True)
    full = np.zeros(Vt.shape[0])
    full[:sv.size] = sv
    return full, Vt


def prior_direction_variance(Vt, prior_var):
    """tau_i^2 = v_i^T diag(prior_var) v_i, the prior variance along each v_i."""
    return (Vt ** 2) @ np.asarray(prior_var, dtype=float)


def analytic_variance(sv, tau2, sigma_n=1.0):
    """Posterior variance along v_i: sigma_n^2 / (sigma_i^2 + sigma_n^2 / tau_i^2)."""
    return sigma_n ** 2 / (np.asarray(sv) ** 2 + sigma_n ** 2 / np.asarray(tau2))


def analytic_shrinkage(sv, tau2, sigma_n=1.0):
    """
    Shrinkage along v_i, sigma_i^2 / (sigma_i^2 + sigma_n^2 / tau_i^2). Written
    in this form rather than as 1 - var / tau2 so that a null direction
    (sigma_i = 0) gives exactly zero.
    """
    s2 = np.asarray(sv, dtype=float) ** 2
    return s2 / (s2 + sigma_n ** 2 / np.asarray(tau2))


def exact_direction_variance(G, prior_var, Vt, sigma_n=1.0):
    """
    v_i^T Sigma_post v_i from the exact solve
    Sigma_post = (G^T G / sigma_n^2 + diag(1/prior_var))^-1, for measuring how
    far the per-direction closed form sits from the truth when the prior is
    not diagonal in the v_i basis.
    """
    n = G.shape[1]
    H = G.T @ G / sigma_n ** 2 + np.diag(1.0 / np.asarray(prior_var, dtype=float))
    Sigma = np.linalg.inv(H)
    return np.einsum('ij,jk,ik->i', Vt, Sigma, Vt)


#####################
# The SBI posterior #
#####################

def projected_variance(draws, Vt):
    """
    Per-direction sample variance of one surface's draws, (n_draws, n) against
    Vt rows, with ddof = 1.
    """
    y = np.asarray(draws, dtype=float) @ Vt.T
    return y.var(axis=0, ddof=1)


def shrinkage_from_variance(var, tau2):
    """1 - var / tau2. Can be negative where the posterior is wider than the prior."""
    return 1.0 - np.asarray(var, dtype=float) / np.asarray(tau2, dtype=float)


#####################
# Prior             #
#####################

def load_prior_from_checkpoint(vae_path):
    """
    Diagonal prior in the real-packed coefficient basis from the VAE
    checkpoint's standardisation statistics: variance std_data^2 per
    coefficient. This is the training-set marginal the network also had. The
    released checkpoint stores full-length (961) statistics with the DC
    coefficient included (l_min = 0).
    """
    import torch
    ckpt = torch.load(vae_path, map_location='cpu', weights_only=False)
    std = np.asarray(ckpt['std_data'], dtype=float).ravel()
    if ckpt['config'].get('l_min', 0) != 0:
        raise ValueError(f'{vae_path}: trained without the DC coefficient; '
                         'the prior would need its variance from elsewhere')
    return std ** 2


#####################
# Draws access      #
#####################

def iter_draw_rows(fam_dir, n_max=None):
    """
    Yield (surface_idx, slot, beta_deg, draws) per surface from the chunk CSVs
    and the sibling draws/chunk_*.npy written by run_holdout.py --save-draws,
    which are aligned row for row.
    """
    draw_dir = os.path.join(fam_dir, 'draws')
    chunks = sorted(f for f in os.listdir(fam_dir)
                    if f.startswith('chunk_') and f.endswith('.csv')) \
        if os.path.isdir(fam_dir) else []
    yielded = 0
    for name in chunks:
        npy = os.path.join(draw_dir, name.replace('.csv', '.npy'))
        if not os.path.exists(npy):
            continue
        draws = np.load(npy)
        with open(os.path.join(fam_dir, name)) as f:
            rows = list(csv.DictReader(f))
        if len(rows) != draws.shape[0]:
            raise ValueError(f'{name}: {len(rows)} rows against '
                             f'{draws.shape[0]} draw blocks')
        for k, r in enumerate(rows):
            yield int(r['idx']), int(r['slot']), int(round(float(r['beta']))), draws[k]
            yielded += 1
            if n_max is not None and yielded >= n_max:
                return


#####################
# The run           #
#####################

def analyse(fam_dir, family, prior_var, cache_dir, out_dir,
            sigma_phot, sigma_astro, l_max=L_MAX, n_obs=N_OBS, n_beta=N_BETA,
            n_max=None, progress=True):
    """
    Group the saved draws by inclination, compare per-direction shrinkage
    against the analytic baseline at each, write one CSV per inclination and a
    summary JSON, and return the summary.
    """
    t0 = time.perf_counter()
    A = load_design(cache_dir, l_max, n_obs, n_beta)
    T = build_T(l_max)
    os.makedirs(out_dir, exist_ok=True)

    rows = list(iter_draw_rows(fam_dir, n_max))
    if not rows:
        raise FileNotFoundError(
            f'no saved draws under {fam_dir}; run scripts/run_holdout.py '
            f'--family {family} --save-draws first. This script does not '
            f're-sample the flow.')

    by_beta = {}
    for idx, slot, beta, draws in rows:
        by_beta.setdefault(beta, []).append((idx, slot, draws))

    summary = {'per_beta': {}}
    it = tqdm(sorted(by_beta), desc=f'shrinkage {family}', unit='beta',
              disable=not progress)
    for beta in it:
        ops = channel_operators(A[beta], T)
        G = whitened_operator(ops, family, sigma_phot, sigma_astro)
        sv, Vt = operator_svd(G)
        rank, _, _, gap = rank_by_gap(sv[sv > 0])
        tau2 = prior_direction_variance(Vt, prior_var)

        var_an = analytic_variance(sv, tau2)
        shr_an = analytic_shrinkage(sv, tau2)
        var_ex = exact_direction_variance(G, prior_var, Vt)
        closed_vs_exact = float(np.max(np.abs(var_an - var_ex)
                                       / np.maximum(var_ex, 1e-300)))

        surfs = by_beta[beta]
        var_sbi = np.zeros_like(tau2)
        neff_sbi_each = []
        for _, _, draws in surfs:
            v = projected_variance(draws, Vt)
            var_sbi += v
            neff_sbi_each.append(float(np.sum(shrinkage_from_variance(v, tau2))))
        var_sbi /= len(surfs)
        shr_sbi = shrinkage_from_variance(var_sbi, tau2)

        row_sp = np.arange(sv.size) < rank
        with np.errstate(divide='ignore'):
            logdet_row = float(np.sum(np.log(var_sbi[row_sp] / var_an[row_sp])))
            logdet_null = float(np.sum(np.log(var_sbi[~row_sp] / var_an[~row_sp])))

        path = os.path.join(out_dir, f'perdir_beta{beta:03d}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['direction', 'in_row_space', 'sigma_whitened', 'tau2',
                        'var_analytic', 'shrink_analytic', 'var_sbi',
                        'shrink_sbi'])
            for i in range(sv.size):
                w.writerow([i, bool(row_sp[i]), f'{sv[i]:.10g}',
                            f'{tau2[i]:.10g}', f'{var_an[i]:.10g}',
                            f'{shr_an[i]:.10g}', f'{var_sbi[i]:.10g}',
                            f'{shr_sbi[i]:.10g}'])

        summary['per_beta'][beta] = {
            'n_surfaces': len(surfs),
            'rank': int(rank), 'gap_decades': float(gap),
            'gap_trusted': bool(gap >= GAP_TRUST_DECADES),
            'neff_analytic_prior_weighted': float(np.sum(shr_an)),
            'neff_analytic_unit_prior': float(np.sum(sv ** 2 / (sv ** 2 + 1.0))),
            'neff_sbi_mean': float(np.mean(neff_sbi_each)),
            'neff_sbi_median': float(np.median(neff_sbi_each)),
            'neff_sbi_std': float(np.std(neff_sbi_each)),
            'neff_difference': float(np.mean(neff_sbi_each) - np.sum(shr_an)),
            'shrink_sbi_row_space_sum': float(np.sum(shr_sbi[row_sp])),
            'shrink_sbi_null_space_sum': float(np.sum(shr_sbi[~row_sp])),
            'shrink_analytic_null_space_sum': float(np.sum(shr_an[~row_sp])),
            'logdet_ratio_row_space': logdet_row,
            'logdet_ratio_null_space': logdet_null,
            'closed_form_vs_exact_max_rel': closed_vs_exact,
        }

    pb = summary['per_beta']
    weights = np.array([pb[b]['n_surfaces'] for b in pb], dtype=float)
    weights /= weights.sum()

    def wmean(key):
        return float(sum(w * pb[b][key] for w, b in zip(weights, pb)))

    summary['overall'] = {k: wmean(k) for k in
                          ('neff_analytic_prior_weighted', 'neff_sbi_mean',
                           'neff_difference', 'logdet_ratio_row_space',
                           'logdet_ratio_null_space')}
    summary['settings'] = {
        'family': family, 'l_max': int(l_max), 'n_obs': int(n_obs),
        'sigma_phot': sigma_phot, 'sigma_astro': sigma_astro,
        'n_surfaces': int(len(rows)),
        'prior': 'diagonal, training marginal std_data^2 from the VAE checkpoint',
        'calibration_caveat': CALIBRATION_CAVEAT,
        'package_version': getattr(starspot_sbi, '__version__', 'unknown'),
        'wall_seconds': round(time.perf_counter() - t0, 3),
        'created': datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary):
    print('\n' + CALIBRATION_CAVEAT + '\n')
    print(f'{"beta":>5} {"n":>4} {"rank":>5} {"N_eff an.":>10} {"N_eff SBI":>10} '
          f'{"diff":>8} {"logdet row":>11} {"logdet null":>12}')
    for b, r in sorted(summary['per_beta'].items()):
        trust = '' if r['gap_trusted'] else '  (gap untrusted)'
        print(f'{b:>5} {r["n_surfaces"]:>4} {r["rank"]:>5} '
              f'{r["neff_analytic_prior_weighted"]:>10.2f} '
              f'{r["neff_sbi_mean"]:>10.2f} {r["neff_difference"]:>8.2f} '
              f'{r["logdet_ratio_row_space"]:>11.1f} '
              f'{r["logdet_ratio_null_space"]:>12.1f}{trust}')
    o = summary['overall']
    print(f'\noverall (surface-weighted): N_eff analytic '
          f'{o["neff_analytic_prior_weighted"]:.2f}, SBI '
          f'{o["neff_sbi_mean"]:.2f}, difference {o["neff_difference"]:.2f}')


###############
# Entry point #
###############

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', default='phot_axay',
                    help='one of ' + ', '.join(FAMILY_CHANNELS))
    ap.add_argument('--holdout-results', default=os.path.join('results', 'holdout'),
                    help='run_holdout.py output directory holding '
                         '<family>/chunk_*.csv and <family>/draws/')
    ap.add_argument('--vae', default=DEFAULT_VAE,
                    help='VAE checkpoint, read only for mu_data and std_data')
    ap.add_argument('--cache-dir', default=os.environ.get('STARSPOT_CACHE_DIR',
                                                          DEFAULT_CACHE_DIR))
    ap.add_argument('--out', default=DEFAULT_OUT_DIR)
    ap.add_argument('--n', type=int, default=None,
                    help='surfaces to use; default all with saved draws')
    ap.add_argument('--l-max', type=int, default=L_MAX)
    ap.add_argument('--n-obs', type=int, default=N_OBS)
    ap.add_argument('--n-beta', type=int, default=N_BETA)
    ap.add_argument('--log-sigma-phot', type=float, default=LOG10_SIGMA_PHOT_MISSION)
    ap.add_argument('--log-sigma-astro', type=float, default=LOG10_SIGMA_ASTRO_MISSION)
    args = ap.parse_args()

    if args.family not in FAMILY_CHANNELS:
        ap.error(f'unknown family {args.family}')

    prior_var = load_prior_from_checkpoint(args.vae)
    try:
        summary = analyse(
            os.path.join(args.holdout_results, args.family), args.family,
            prior_var, args.cache_dir, args.out,
            10.0 ** args.log_sigma_phot, 10.0 ** args.log_sigma_astro,
            args.l_max, args.n_obs, args.n_beta, args.n)
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)
    print_summary(summary)
    print(f'\nwritten to {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
