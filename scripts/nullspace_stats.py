"""
Null-space statistics of a holdout run, per family: the null-space correlation
rho with its bootstrap standard error and permutation baseline, the median null
power fraction of the posterior means and of the true surfaces, the mean rank,
and the summed shrinkage weighted over the run's inclinations.

Pairs and their order come from the run's full_<family>_lp<lp>_la<la>.csv.
Posterior means are the means of the draws saved with run_holdout --save-draws,
or, with --postmeans-dir, precomputed means postmeans_<family>_lp<lp>_la<la>.npy
in the same order.

The paper's numbers:
    python scripts/nullspace_stats.py --run-dir <path>/results/holdout \
        --holdout <path>/explicit_holdout
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy
from build_caches import DEFAULT_CACHE_DIR, L_MAX, load_design
from run_holdout import load_holdout_index, read_pair

from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.nullspace import (
    null_basis,
    null_correlation,
    null_power_fraction,
    shrinkage_sum,
)
from starspot_sbi.operators import (
    FAMILY_ROWS,
    GAP_TRUST_DECADES,
    build_T,
    family_operator,
)
from starspot_sbi.pipeline import LOG10_SIGMA_ASTRO_MISSION, LOG10_SIGMA_PHOT_MISSION


def run_csv(run_dir, family, lp, la):
    return os.path.join(run_dir, f'full_{family}_lp{lp}_la{la}.csv')


def load_truths(holdout_dir, pairs_csv):
    """True real-packed coefficients and integer inclinations, in the CSV's order."""
    pairs, meta = load_holdout_index(holdout_dir, pairs_csv)
    truths, betas = [], []
    for idx, slot in pairs:
        coeffs, _, beta, _ = read_pair(holdout_dir, meta, idx, slot)
        truths.append(coeffs_to_real(coeffs))
        betas.append(int(round(beta)))
    return pairs, np.array(truths), np.array(betas)


def posterior_means_from_draws(run_dir, family, pairs):
    """Mean of each pair's saved draws, in the order of pairs."""
    fam_dir = os.path.join(run_dir, family)
    means = {}
    names = sorted(f for f in os.listdir(fam_dir) if f.startswith('chunk_') and f.endswith('.csv'))
    for name in names:
        rows = pd.read_csv(os.path.join(fam_dir, name))
        block = np.load(os.path.join(fam_dir, 'draws', name.replace('.csv', '.npy')), mmap_mode='r')
        if len(rows) != block.shape[0]:
            raise ValueError(f'{name}: {len(rows)} rows against {block.shape[0]} draw blocks')
        for k, r in enumerate(rows.itertuples()):
            means[(int(r.idx), int(r.slot))] = np.asarray(block[k]).mean(axis=0)
    missing = [p for p in pairs if tuple(p) not in means]
    if missing:
        raise KeyError(f'{len(missing)} pairs have no saved draws, e.g. {missing[0]}')
    return np.array([means[tuple(p)] for p in pairs])


def family_stats(family, design, T, truths, post, betas, n_boot, seed):
    labels, counts = np.unique(betas, return_counts=True)
    bases, ranks, gaps, shrink = {}, {}, {}, 0.0
    for b, cnt in zip(labels, counts, strict=True):
        G = family_operator(design[int(b)], T, family)
        bases[int(b)], ranks[int(b)], gaps[int(b)] = null_basis(G)
        shrink += cnt * shrinkage_sum(G)
    rho = null_correlation(truths, post, betas, bases, n_boot=n_boot, seed=seed)
    return {'family': family, 'n': len(betas), **rho,
            'f_null_post_median': float(np.median(null_power_fraction(post, betas, bases))),
            'f_null_truth_median': float(np.median(null_power_fraction(truths, betas, bases))),
            'mean_rank': float(np.mean([ranks[int(b)] for b in betas])),
            'n_untrusted_gap': int(sum(g < GAP_TRUST_DECADES for g in gaps.values())),
            'shrinkage_sum': shrink / counts.sum()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', default='all', choices=['all', *FAMILY_ROWS])
    ap.add_argument('--run-dir', required=True,
                    help='a run_holdout --out directory, with saved draws unless --postmeans-dir')
    ap.add_argument('--holdout', default='data/explicit_holdout')
    ap.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    ap.add_argument('--log-sigma-phot', type=float, default=LOG10_SIGMA_PHOT_MISSION)
    ap.add_argument('--log-sigma-astro', type=float, default=LOG10_SIGMA_ASTRO_MISSION)
    ap.add_argument('--postmeans-dir', help='precomputed posterior means instead of saved draws')
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join('results', 'nullspace', 'nullspace_stats.csv'))
    args = ap.parse_args()

    lp, la = args.log_sigma_phot, args.log_sigma_astro
    families = list(FAMILY_ROWS) if args.family == 'all' else [args.family]
    design = load_design(args.cache_dir)
    T = build_T(L_MAX)

    rows = []
    for family in families:
        pairs, truths, betas = load_truths(args.holdout, run_csv(args.run_dir, family, lp, la))
        if args.postmeans_dir:
            post = np.load(os.path.join(args.postmeans_dir, f'postmeans_{family}_lp{lp}_la{la}.npy'))
            source = 'postmeans'
        else:
            post = posterior_means_from_draws(args.run_dir, family, pairs)
            source = 'draws'
        if post.shape != truths.shape:
            raise ValueError(f'{family}: posterior means {post.shape} against truths {truths.shape}')
        row = {'lp': lp, 'la': la, 'source': source,
            'numpy': np.__version__, 'scipy': scipy.__version__,
            **family_stats(family, design, T, truths, post, betas, args.n_boot, args.seed)}
        rows.append(row)
        print(f"{family}: rho {row['rho']:.5f} +- {row['rho_se']:.5f} (perm {row['rho_perm']:.5f}), "
              f"f_null post {row['f_null_post_median']:.4f} truth {row['f_null_truth_median']:.4f}, "
              f"sum S {row['shrinkage_sum']:.1f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()