"""
Regenerate the per-surface metrics of the paper's results section, from the
released checkpoints and the stored holdout signals.

Given --pairs, the same (surface, inclination) pairs the notebook used are
scored, which makes the comparison paired: same surfaces, same inclinations,
different code.

Usage:
    python scripts/run_holdout.py --family phot_axay --n 200
    python scripts/run_holdout.py --family phot_axay --pairs figdata/full_phot_axay_lp-4.0_la-3.5.csv
    python scripts/run_holdout.py --family all --n 5000 --draws 256 --save-draws

With --save-draws the coefficient draws are kept, 1 MB per surface at 256 draws,
so 20 GB for four families at 5000 surfaces. 

Reproducibility: the noise draw and the flow sampling are both seeded on the
chunk's first row index, so a rerun reproduces the same numbers and a resumed
run agrees with an uninterrupted one.
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

# standard
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# machine learning
import torch

# self
from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.render import render, render_coeffs
from starspot_sbi.metrics import (ssim_aa_vis, ssim_aa_wmean, ssim_aa_full,
                                  rmse, mae, crps, pr_auc, err_unc_corr,
                                  spot_mask, SPOT_THRESHOLD)
from starspot_sbi.models import FAMILIES, load_vae, load_flow
from starspot_sbi.pipeline import (sample_draws, posterior_mean, render_draws,
                                   power_spectrum, select_channels,
                                   LOG10_SIGMA_PHOT_MISSION,
                                   LOG10_SIGMA_ASTRO_MISSION)


#############
# Constants #
#############

WEIGHTS_SUFFIX = '_temp'
LATENT_DIM = 96
CHUNK = 100

# Column names follow the notebook's figdata/full_*.csv so the two can be
# compared without a rename. ssim_wmean is new here; the notebook computed ten
# SSIM variants and reported two, and this is the second of them.
COLUMNS = ['idx', 'slot', 'beta', 'ssim_vis', 'ssim_wmean', 'ssim_full',
           'rmse_vis', 'rmse_full', 'mae_vis', 'pr_auc', 'pr_auc_full',
           'sff_true', 'sff_rec', 'crps', 'crps_full', 'err_unc',
           'n_spots_true']


#####################
# Holdout access    #
#####################

def load_holdout_index(holdout_dir, pairs_csv=None, n=None, seed=0):
    """
    The (surface_idx, slot) pairs to score.

    With pairs_csv, the notebook's own pairs are read, so the comparison is
    paired. Without it, n surfaces are drawn from metadata.csv and one slot per
    surface is drawn at random.

    metadata.csv holds duplicate rows from two concurrent generation runs, so it 
    is deduplicated on surface_idx here.
    """
    meta = pd.read_csv(os.path.join(holdout_dir, 'metadata.csv'))
    meta = meta.drop_duplicates('surface_idx', keep='first').set_index('surface_idx')

    if pairs_csv:
        pairs = pd.read_csv(pairs_csv)[['idx', 'slot']]
        pairs = pairs.astype({'idx': int, 'slot': int})
        return [(int(r.idx), int(r.slot)) for r in pairs.itertuples()], meta

    rng = np.random.default_rng(seed)
    idx = rng.choice(meta.index.values, size=min(n, len(meta)), replace=False)
    idx.sort()
    n_inc = int(meta.iloc[0]['n_inc'])
    slots = rng.integers(0, n_inc, size=idx.size)
    return list(zip(idx.tolist(), slots.tolist())), meta


def _column(meta, *candidates):
    """First column present out of several spellings, or None."""
    for c in candidates:
        if c in meta.columns:
            return c
    return None


def read_pair(holdout_dir, meta, surface_idx, slot):
    """
    The stored surface, its signal at this slot, and that inclination.

    Column names differ between generations of the dataset,
    this attempts to resolve differences. If a file name is absent the path
    is built from the index, which is the convention the generator uses.
    """
    row = meta.loc[surface_idx]

    scol = _column(meta, 'surface_file', 'surface', 'surface_path')
    gcol = _column(meta, 'signal_file', 'signals_file', 'signal_path')
    surf_name = row[scol] if scol else f'surface_{surface_idx:07d}.npy'
    sig_name = row[gcol] if gcol else f'signal_{surface_idx:07d}.npy'

    coeffs = np.load(os.path.join(holdout_dir, 'surfaces', surf_name))
    signal = np.load(os.path.join(holdout_dir, 'signals', sig_name))

    bcol = _column(meta, 'betas_deg', 'betas', 'beta_deg')
    betas = [float(b) for b in str(row[bcol]).replace(',', ';').split(';')]

    ncol = _column(meta, 'n_spots', 'n_spots_true', 'nspots')
    n_spots = int(row[ncol]) if ncol else -1

    return coeffs, signal[slot], round(betas[slot]), n_spots


#####################
# Scoring           #
#####################

def score(truth, recon, beta_deg, samples=None, recon_std=None,
          threshold=SPOT_THRESHOLD):
    """Every column of the output CSV apart from the identifiers."""
    if truth.shape != recon.shape:
        raise ValueError(f'truth is {truth.shape} and the reconstruction is '
                         f'{recon.shape}; both must be on the same render grid')
    beta = np.radians(beta_deg)
    out = {
        'ssim_vis':   ssim_aa_vis(truth, recon, beta),
        'ssim_wmean': ssim_aa_wmean(truth, recon, beta),
        'ssim_full':  ssim_aa_full(truth, recon),
        'rmse_vis':   rmse(truth, recon, beta, 'vis'),
        'rmse_full':  rmse(truth, recon, beta, 'full'),
        'mae_vis':    mae(truth, recon, beta, 'vis'),
        'pr_auc':     pr_auc(truth, recon, beta, 'vis', threshold=threshold),
        'pr_auc_full': pr_auc(truth, recon, beta, 'full', threshold=threshold),
        'sff_true':   float(np.mean(spot_mask(truth, threshold))),
        'sff_rec':    float(np.mean(spot_mask(recon, threshold))),
    }
    out['crps'] = crps(truth, samples, beta, 'vis') if samples is not None else np.nan
    out['crps_full'] = crps(truth, samples, beta, 'full') if samples is not None else np.nan
    out['err_unc'] = (err_unc_corr(truth, recon, recon_std, beta)
                      if recon_std is not None else np.nan)
    return out


#####################
# The run           #
#####################

def run_family(family, pairs, meta, holdout_dir, weights_dir, out_dir, args):
    """Score every pair for one family, writing one CSV per chunk."""
    fam_dir = os.path.join(out_dir, family)
    draw_dir = os.path.join(fam_dir, 'draws')
    spec_dir = os.path.join(fam_dir, 'spectra')
    for d in (fam_dir, draw_dir, spec_dir):
        os.makedirs(d, exist_ok=True)

    vae, _, stats = load_vae(os.path.join(
        weights_dir, f'vae_n640000_seed101{WEIGHTS_SUFFIX}.pt'), device=args.device)
    est, fmeta = load_flow(os.path.join(
        weights_dir, f'flow_{family}{WEIGHTS_SUFFIX}.pt'),
        latent_dim=LATENT_DIM, device=args.device)

    n_chunks = int(np.ceil(len(pairs) / CHUNK))
    t0 = time.time()
    for c in tqdm(range(n_chunks), desc=family):
        path = os.path.join(fam_dir, f'chunk_{c:05d}.csv')
        if os.path.exists(path):
            continue

        block = pairs[c * CHUNK:(c + 1) * CHUNK]
        truths, signals, betas, n_spots, ids = [], [], [], [], []
        for surface_idx, slot in block:
            coeffs, sig, beta_deg, ns = read_pair(holdout_dir, meta, surface_idx, slot)
            truths.append(render(coeffs_to_real(coeffs)))
            signals.append(select_channels(sig, family))
            betas.append(beta_deg)
            n_spots.append(ns)
            ids.append((surface_idx, slot))

        draws = sample_draws(
            np.stack(signals), np.array(betas), family, vae, stats, est, fmeta,
            log10_sigma_phot=args.log_sigma_phot,
            log10_sigma_astro=args.log_sigma_astro,
            n_draws=args.draws, seed=args.seed + c * CHUNK,
            batch_size=args.batch, device=args.device)

        if args.save_draws:
            np.save(os.path.join(draw_dir, f'chunk_{c:05d}.npy'), draws)

        recons = render_draws(posterior_mean(draws), device=args.device)

        rows, spectra = [], []
        for k, ((surface_idx, slot), truth) in enumerate(zip(ids, truths)):
            # rendered draws for this surface only: 29.5 MB at 256 draws, so
            # they are computed one surface at a time and discarded
            imgs = render_draws(draws[k], device=args.device)
            r = {'idx': surface_idx, 'slot': slot, 'beta': betas[k],
                 'n_spots_true': n_spots[k]}
            r.update(score(truth, recons[k], betas[k],
                           samples=imgs, recon_std=imgs.std(axis=0)))
            rows.append(r)
            if args.spectra:
                spectra.append(power_spectrum(draws[k]).mean(axis=0))

        pd.DataFrame(rows)[COLUMNS].to_csv(path, index=False)
        if args.spectra:
            np.save(os.path.join(spec_dir, f'chunk_{c:05d}.npy'),
                    np.stack(spectra))

    frames = [pd.read_csv(os.path.join(fam_dir, f))
              for f in sorted(os.listdir(fam_dir)) if f.startswith('chunk_')]
    df = pd.concat(frames, ignore_index=True).sort_values('idx')
    combined = os.path.join(
        out_dir,
        f'full_{family}_lp{args.log_sigma_phot}_la{args.log_sigma_astro}.csv')
    df.to_csv(combined, index=False)

    elapsed = time.time() - t0
    print(f'[{family}] {len(df)} surfaces in {elapsed / 60:.1f} min -> {combined}')
    return df, elapsed


def summarise(df, family):
    """Median of every metric, with the count of finite values."""
    print(f'\n{family}')
    for c in COLUMNS[3:]:
        v = df[c].to_numpy(dtype=float)
        good = v[np.isfinite(v)]
        if good.size == 0:
            print(f'  {c:14s} all nan')
        else:
            print(f'  {c:14s} median {np.median(good):9.5f}   n {good.size:5d}'
                  + ('' if good.size == v.size else f' ({v.size - good.size} nan)'))


###############
# Entry point #
###############

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', default='phot_axay',
                    help="one of phot, phot_ax, phot_ay, phot_axay, or 'all'")
    ap.add_argument('--holdout', default='data/explicit_holdout',
                    help='directory holding metadata.csv, surfaces/, signals/')
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--out', default='results/holdout')
    ap.add_argument('--pairs', help='CSV with idx and slot columns, to score the '
                                    'same pairs an earlier run used')
    ap.add_argument('--n', type=int, default=200,
                    help='surfaces to draw when --pairs is absent')
    ap.add_argument('--draws', type=int, default=256)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-sigma-phot', type=float, default=LOG10_SIGMA_PHOT_MISSION)
    ap.add_argument('--log-sigma-astro', type=float, default=LOG10_SIGMA_ASTRO_MISSION)
    ap.add_argument('--save-draws', action='store_true',
                    help='keep the coefficient draws, 1 MB per surface at 256 '
                         'draws. Everything else derives from them, so a figure '
                         'reworked later needs no rerun.')
    ap.add_argument('--spectra', action='store_true',
                    help='write the mean power spectrum per surface')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    families = list(FAMILIES) if args.family == 'all' else [args.family]
    os.makedirs(args.out, exist_ok=True)

    pairs, meta = load_holdout_index(args.holdout, args.pairs, args.n, args.seed)
    n_coeff_mb = args.draws * 961 * 4 / 1e6
    print(f'{len(pairs)} pairs, {len(families)} families, {args.draws} draws, '
          f'device {args.device}')
    if args.save_draws:
        print(f'draws kept: {n_coeff_mb:.1f} MB per surface, '
              f'{len(pairs) * len(families) * n_coeff_mb / 1e3:.1f} GB in total')
    print(f'noise point (log10 sigma_phot, log10 sigma_astro) = '
          f'({args.log_sigma_phot}, {args.log_sigma_astro})')

    manifest = {'n_pairs': len(pairs), 'families': families,
                'draws': args.draws, 'seed': args.seed,
                'saved_draws': bool(args.save_draws),
                'saved_spectra': bool(args.spectra),
                'log10_sigma_phot': args.log_sigma_phot,
                'log10_sigma_astro': args.log_sigma_astro,
                'pairs_source': args.pairs or f'random, seed {args.seed}',
                'weights_suffix': WEIGHTS_SUFFIX, 'elapsed_s': {}}

    for family in families:
        df, elapsed = run_family(family, pairs, meta, args.holdout, args.weights,
                                 args.out, args)
        manifest['elapsed_s'][family] = elapsed
        summarise(df, family)

    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()