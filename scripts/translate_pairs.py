"""
figdata idx is a row of the legacy evaluation cache, not a surface_idx.

holdfull_beta is (100000, 20) in normalised units, and the cache was built from
a permutation of the holdout surfaces with perm_seed 42. This recovers the
permutation by matching each cache row's twenty inclinations against the
betas_deg list of the grouped metadata, and writes a pairs file in surface_idx
terms that run_holdout.py reads.

The match is on the ordered tuple of twenty integer inclinations, which is
unique for all 100000 surfaces. Collisions are reported and dropped.

    python scripts/translate_pairs.py
    python scripts/translate_pairs.py --n 200 --out /tmp/pairs200.csv
    python scripts/translate_pairs.py --family phot
"""

import argparse
import os

import numpy as np
import pandas as pd

CACHE = '../Sydney/FromScratch/nsf_stage3_ladder/cache/'
GROUPED = '../Sydney/data/fast/L30_N216_grouped/explicit_holdout'
FIGDATA = '../Sydney/FromScratch/figdata'
FAMILIES = ('phot', 'phot_ax', 'phot_ay', 'phot_axay')
DEFAULT_OUT = '/tmp/pairs_translated.csv'


def ref_path(family):
    return os.path.join(FIGDATA, f'full_{family}_lp-4.0_la-3.5.csv')


def denorm(x):
    """Normalised inclination in [-1, 1] back to degrees."""
    return 90.0 * (np.asarray(x, dtype=float) + 1.0) / 2.0


def surface_lookup(grouped):
    """Ordered inclination tuple to surface_idx, with collisions dropped."""
    meta = pd.read_csv(os.path.join(grouped, 'metadata.csv'))
    meta = meta.drop_duplicates('surface_idx', keep='first')
    keys, dupes = {}, 0
    for sidx, s in zip(meta['surface_idx'].to_numpy(), meta['betas_deg']):
        t = tuple(int(round(float(v)))
                  for v in str(s).replace(',', ';').split(';'))
        if t in keys:
            dupes += 1
            keys[t] = None
        else:
            keys[t] = int(sidx)
    usable = sum(v is not None for v in keys.values())
    print(f'{len(meta)} surfaces, {dupes} inclination tuples seen more than '
          f'once, {usable} usable')
    return keys, meta.set_index('surface_idx')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--grouped', default=GROUPED)
    ap.add_argument('--family', default='phot_axay',
                    help='which figdata file supplies the pairs')
    ap.add_argument('--n', type=int, default=None,
                    help='rows to translate; default all')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--check-families', action='store_true',
                    help='report whether the four figdata files use the same '
                         'pairs, which decides whether one output serves all')
    args = ap.parse_args()

    deg = np.rint(denorm(np.load(
        os.path.join(args.cache, 'holdfull_beta.npy')))).astype(int)
    print(f'cache beta {deg.shape}, degrees {deg.min()} to {deg.max()}')

    if args.check_families:
        sets = {}
        for fam in FAMILIES:
            p = ref_path(fam)
            if not os.path.exists(p):
                print(f'{fam:<10} absent')
                continue
            d = pd.read_csv(p)[['idx', 'slot']].astype(int)
            sets[fam] = set(map(tuple, d.to_numpy()))
            print(f'{fam:<10} {len(d)} rows, {len(sets[fam])} distinct pairs')
        if len(sets) > 1:
            base = sets[args.family]
            for fam, s in sets.items():
                print(f'  {fam:<10} shares {len(s & base)} of {len(base)} '
                      f'with {args.family}')

    ref = pd.read_csv(ref_path(args.family))
    agree = np.mean([abs(deg[int(x.idx), int(x.slot)] - x.beta) < 0.51
                     for x in ref.itertuples()])
    print(f'\n{len(ref)} reference rows, beta[idx, slot] agrees on {agree:.4f}')
    if agree < 0.99:
        raise SystemExit('the cache indexing does not reproduce the reference '
                         'beta; nothing written')

    keys, meta = surface_lookup(args.grouped)
    want = ref if args.n is None else ref.head(args.n)

    rows, missing = [], 0
    for row in want.itertuples():
        sidx = keys.get(tuple(deg[int(row.idx)].tolist()))
        if sidx is None:
            missing += 1
            continue
        rows.append({'idx': sidx, 'slot': int(row.slot),
                     'cache_idx': int(row.idx), 'beta': row.beta})
    print(f'translated {len(rows)} of {len(want)}, {missing} unmatched')
    if not rows:
        raise SystemExit('nothing to write')

    out = pd.DataFrame(rows)
    bad = 0
    for x in out.itertuples():
        b = [float(v) for v in
             str(meta.loc[x.idx, 'betas_deg']).replace(',', ';').split(';')]
        if abs(b[x.slot] - x.beta) > 0.51:
            bad += 1
    print(f'slot check: {bad} of {len(out)} rows disagree on beta')
    if bad:
        raise SystemExit('the translation puts the wrong inclination at the '
                         'requested slot; nothing written')

    if out['idx'].duplicated().any():
        n = int(out['idx'].duplicated().sum())
        print(f'note: {n} surfaces appear more than once, at different slots')

    out.to_csv(args.out, index=False)
    print(f'written to {args.out}')
    print(out.head(6).to_string(index=False))


if __name__ == '__main__':
    main()