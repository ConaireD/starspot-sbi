"""
Compare run_holdout.py output against the notebook's figdata, row for row.

The two files index surfaces differently. run_holdout writes surface_idx of the
grouped holdout; figdata writes the row of the legacy evaluation cache. The
translation table written by translate_pairs.py carries both, so the merge goes
through it rather than on idx directly.

The noise draw and the flow sampling differ between the two implementations, so
agreement is statistical: the medians should sit close and the per-row
correlation should be high. sff_true is the exception. It comes from the stored
surface with no noise and no sampling, so it must agree to rounding, and a
disagreement there means the two runs read different surfaces.

    python scripts/compare_holdout.py
"""

import sys

import numpy as np
import pandas as pd

MINE = 'results/holdout_check/full_phot_axay_lp-4.0_la-3.5.csv'
REF = '../Sydney/FromScratch/figdata/full_phot_axay_lp-4.0_la-3.5.csv'
MAP = '/tmp/pairs200_translated.csv'

METRICS = ['ssim_vis', 'ssim_full', 'rmse_vis', 'rmse_full', 'pr_auc',
           'pr_auc_full', 'sff_true', 'sff_rec', 'crps', 'crps_full',
           'err_unc']


def main():
    a = pd.read_csv(MINE)
    b = pd.read_csv(REF)
    t = pd.read_csv(MAP)[['idx', 'slot', 'cache_idx']]

    a = a.merge(t, on=['idx', 'slot'], how='inner')
    m = a.merge(b, left_on=['cache_idx', 'slot'], right_on=['idx', 'slot'],
                suffixes=('_new', '_old'), how='inner')
    print(f'{len(pd.read_csv(MINE))} new rows, {len(b)} reference rows, '
          f'{len(m)} paired through the translation table\n')
    if len(m) < 2:
        print('the translation table does not cover these rows')
        return

    bad = m[m['beta_new'].round() != m['beta_old'].round()]
    print(f'inclination mismatch on {len(bad)} of {len(m)} rows')
    if len(bad):
        print(bad[['cache_idx', 'slot', 'beta_new', 'beta_old']].head(10)
              .to_string(index=False))
        print('every visibility-weighted metric below is meaningless while '
              'this is non-zero')

    shared = [c for c in METRICS
              if f'{c}_new' in m.columns and f'{c}_old' in m.columns]
    print(f'\n{"metric":<13} {"median new":>11} {"median old":>11} '
          f'{"med |diff|":>11} {"corr":>7}')
    for c in shared:
        x = m[f'{c}_new'].to_numpy(float)
        y = m[f'{c}_old'].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            print(f'{c:<13} too few finite pairs ({ok.sum()})')
            continue
        x, y = x[ok], y[ok]
        print(f'{c:<13} {np.median(x):>11.5f} {np.median(y):>11.5f} '
              f'{np.median(np.abs(x - y)):>11.5f} '
              f'{np.corrcoef(x, y)[0, 1]:>7.4f}'
              + ('' if ok.sum() == len(m) else f'   ({len(m) - ok.sum()} dropped)'))

    if 'sff_true' in shared:
        d = np.abs(m['sff_true_new'] - m['sff_true_old'])
        print(f'\nsff_true is noise-free and must agree exactly: largest '
              f'difference {d.max():.3e}, {int((d > 1e-6).sum())} rows above '
              f'1e-6')


if __name__ == '__main__':
    sys.exit(main())