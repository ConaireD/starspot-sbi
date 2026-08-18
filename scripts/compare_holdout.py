"""
Compare run_holdout.py output against the notebook's figdata, row for row.

The two files index surfaces differently. run_holdout writes surface_idx of the
grouped holdout; figdata writes the row of the legacy evaluation cache. The
translation table from translate_pairs.py carries both, so the merge goes
through it.

The noise draw and the flow sampling differ between the two implementations, so
agreement is statistical: the medians should sit close and the per-row
correlation should be high. Two exceptions. sff_true comes from the stored
surface with no noise and no sampling, so it must agree to the render
resolution. SSIM is computed on a fixed pixel window and is a property of the
grid: the notebook rendered at 60 x 120 and the package renders at 120 x 240,
which puts SSIM about 0.019 high with no disagreement behind it.

    python scripts/compare_holdout.py
    python scripts/compare_holdout.py --family phot
"""

import argparse
import os

import numpy as np
import pandas as pd

FAMILIES = ('phot', 'phot_ax', 'phot_ay', 'phot_axay')
MINE = 'results/holdout'
FIGDATA = '../Sydney/FromScratch/figdata'
MAP = '/tmp/pairs5000.csv'

METRICS = ['ssim_vis', 'ssim_full', 'rmse_vis', 'rmse_full', 'pr_auc',
           'pr_auc_full', 'sff_true', 'sff_rec', 'crps', 'crps_full',
           'err_unc']
GRID_DEPENDENT = ('ssim_vis', 'ssim_full')


def compare(family, mine_dir, figdata, mapping):
    name = f'full_{family}_lp-4.0_la-3.5.csv'
    a = pd.read_csv(os.path.join(mine_dir, name))
    b = pd.read_csv(os.path.join(figdata, name))
    t = pd.read_csv(mapping)[['idx', 'slot', 'cache_idx']]

    m = a.merge(t, on=['idx', 'slot'], how='inner').merge(
        b, left_on=['cache_idx', 'slot'], right_on=['idx', 'slot'],
        suffixes=('_new', '_old'), how='inner')
    print(f'\n=== {family}: {len(a)} new, {len(b)} reference, {len(m)} paired')
    if len(m) < 2:
        print('the translation table does not cover these rows')
        return

    bad = m[m['beta_new'].round() != m['beta_old'].round()]
    print(f'inclination mismatch on {len(bad)} of {len(m)} rows')

    print(f'{"metric":<13} {"median new":>11} {"median old":>11} '
          f'{"med |diff|":>11} {"corr":>7}')
    for c in [c for c in METRICS
              if f'{c}_new' in m.columns and f'{c}_old' in m.columns]:
        x = m[f'{c}_new'].to_numpy(float)
        y = m[f'{c}_old'].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            print(f'{c:<13} too few finite pairs ({ok.sum()})')
            continue
        x, y = x[ok], y[ok]
        note = '   grid' if c in GRID_DEPENDENT else ''
        if ok.sum() != len(m):
            note += f'   ({len(m) - ok.sum()} dropped)'
        print(f'{c:<13} {np.median(x):>11.5f} {np.median(y):>11.5f} '
              f'{np.median(np.abs(x - y)):>11.5f} '
              f'{np.corrcoef(x, y)[0, 1]:>7.4f}{note}')

    d = np.abs(m['sff_true_new'] - m['sff_true_old'])
    print(f'sff_true is noise-free: median difference {np.median(d):.3e}, '
          f'largest {d.max():.3e}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', default='all')
    ap.add_argument('--mine', default=MINE)
    ap.add_argument('--figdata', default=FIGDATA)
    ap.add_argument('--map', default=MAP)
    args = ap.parse_args()

    families = FAMILIES if args.family == 'all' else (args.family,)
    for f in families:
        compare(f, args.mine, args.figdata, args.map)
    print('\nSSIM rows are marked "grid": the reference was rendered at '
          '60 x 120 and these at 120 x 240.')


if __name__ == '__main__':
    main()