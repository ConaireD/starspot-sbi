"""
Recovery against injected astrometric noise, per family, at the fixed
photometric reference.

For each family and each log10 sigma_astro on the grid, the flow is
resampled on a fixed set of holdout pairs, the posterior-mean map is scored
against the truth with visibility weighting, and one row per surface is
appended to the output CSV. The photometric noise stays at the mission
reference; the family without astrometric channels carries no dependence on
the swept value and provides the flat reference curve. The pairs are drawn
once from the saved holdout run's pairs, so every family and every noise
level scores the same surfaces and the comparison is paired.

Rows already present in the output are skipped, so an interrupted sweep
resumes where it stopped.

Usage:
    python scripts/noise_sweep.py --pairs-from results/holdout --n 400
    python scripts/noise_sweep.py --family phot_axay --draws 128
"""

###########
# Imports #
###########

# python
import os
import argparse

# standard
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# machine learning
import torch

# self
from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.render import render
from starspot_sbi.metrics import ssim_aa_vis, pr_auc, crps
from starspot_sbi.models import FAMILIES, load_vae, load_flow
from starspot_sbi.pipeline import (sample_draws, render_draws,
                                   LOG10_SIGMA_PHOT_MISSION)

from run_holdout import load_holdout_index, read_pair


#############
# Constants #
#############

WEIGHTS_SUFFIX = '_temp'
LATENT_DIM = 96
SNR_GRID = (-6.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0)
CHUNK = 25

###############
# Entry point #
###############

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', default='all',
                    help="one of phot, phot_ax, phot_ay, phot_axay, or 'all'")
    ap.add_argument('--holdout', default='data/explicit_holdout')
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--out', default=os.path.join('results', 'snr'))
    ap.add_argument('--pairs', help='CSV with idx and slot columns; the sweep '
                                    'subsamples it with --seed')
    ap.add_argument('--n', type=int, default=400,
                    help='surfaces per noise level')
    ap.add_argument('--draws', type=int, default=128)
    ap.add_argument('--seed', type=int, default=707)
    ap.add_argument('--log-sigma-phot', type=float,
                    default=LOG10_SIGMA_PHOT_MISSION)
    ap.add_argument('--grid', type=float, nargs='+', default=list(SNR_GRID),
                    help='log10 sigma_astro values to sweep, space separated: '
                         '--grid -6.0 -4.0 -2.0')
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    families = list(FAMILIES) if args.family == 'all' else [args.family]
    grid = list(args.grid)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(
        args.out, f'snr_lp{args.log_sigma_phot}_n{args.n}_d{args.draws}.csv')

    pairs, meta = load_holdout_index(args.holdout, args.pairs, args.n,
                                     args.seed)
    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.n:
        keep = np.sort(rng.choice(len(pairs), args.n, replace=False))
        pairs = [pairs[k] for k in keep]
    print(f'{len(pairs)} pairs, families {families}, grid {grid}, '
          f'{args.draws} draws, device {args.device}')

    done = set()
    if os.path.exists(out_csv):
        d = pd.read_csv(out_csv)
        done = set(zip(d['family'], d['log_sigma_astro'].round(2), d['idx']))
        print(f'{len(done)} rows already present; resuming')

    vae, _, stats = load_vae(os.path.join(
        args.weights, f'vae_n640000_seed101{WEIGHTS_SUFFIX}.pt'),
        device=args.device)

    truths, signals, betas = {}, {}, {}
    for surface_idx, slot in pairs:
        coeffs, sig, beta_deg, _ = read_pair(args.holdout, meta,
                                             surface_idx, slot)
        truths[(surface_idx, slot)] = render(coeffs_to_real(coeffs))
        signals[(surface_idx, slot)] = sig
        betas[(surface_idx, slot)] = beta_deg

    for family in families:
        est, fmeta = load_flow(os.path.join(
            args.weights, f'flow_{family}{WEIGHTS_SUFFIX}.pt'),
            latent_dim=LATENT_DIM, device=args.device)
        for la in grid:
            todo = [p for p in pairs
                    if (family, round(la, 2), p[0]) not in done]
            if not todo:
                continue
            for c0 in tqdm(range(0, len(todo), CHUNK),
                           desc=f'{family} la={la:.1f}', leave=False):
                block = todo[c0:c0 + CHUNK]
                sig = np.stack([signals[p] for p in block])
                bet = np.array([betas[p] for p in block], dtype=float)
                draws = sample_draws(
                    sig, bet, family, vae, stats, est, fmeta,
                    log10_sigma_phot=args.log_sigma_phot,
                    log10_sigma_astro=la, n_draws=args.draws,
                    seed=args.seed + c0, device=args.device)
                rows = []
                for k, p in enumerate(block):
                    imgs = render_draws(draws[k], device=args.device)
                    mean_map = imgs.mean(axis=0)
                    truth = truths[p]
                    beta = np.radians(bet[k])
                    rows.append({
                        'family': family, 'log_sigma_astro': la,
                        'idx': p[0], 'slot': p[1], 'beta': bet[k],
                        'ssim_vis': ssim_aa_vis(truth, mean_map, beta),
                        'pr_auc': pr_auc(truth, mean_map, beta, 'vis'),
                        'crps': crps(truth, imgs, beta, 'vis')})
                pd.DataFrame(rows).to_csv(out_csv, mode='a', index=False,
                                          header=not os.path.exists(out_csv))
    print(f'written to {os.path.abspath(out_csv)}')


if __name__ == '__main__':
    main()
