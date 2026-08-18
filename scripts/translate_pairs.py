"""
figdata's idx is a row of the legacy evaluation cache, not a surface_idx.

holdfull_beta is (100000, 20) in normalised units, and the cache was built from
a permutation of the holdout surfaces with perm_seed 42. This confirms that
beta[idx, slot] reproduces the reference beta, recovers the permutation by
matching each cache row's twenty inclinations against the betas_deg list of the
grouped metadata, and writes a pairs file in surface_idx terms that
run_holdout.py can read.

The match is on the ordered tuple of twenty integer inclinations. Collisions
are counted and reported; a surface whose tuple is not unique cannot be
identified this way and is dropped.

    python scripts/translate_pairs.py
"""

import os

import numpy as np
import pandas as pd

CACHE = '../Sydney/FromScratch/nsf_stage3_ladder/cache/'
GROUPED = '../Sydney/data/fast/L30_N216_grouped/explicit_holdout'
REF = '../Sydney/FromScratch/figdata/full_phot_axay_lp-4.0_la-3.5.csv'
OUT = '/tmp/pairs200_translated.csv'
N_PAIRS = 200


def denorm(x):
    """Normalised inclination in [-1, 1] back to degrees."""
    return 90.0 * (np.asarray(x, dtype=float) + 1.0) / 2.0


def main():
    beta = np.load(os.path.join(CACHE, 'holdfull_beta.npy'))
    deg = np.rint(denorm(beta)).astype(int)
    print(f'cache beta {beta.shape}, degrees {deg.min()} to {deg.max()}')

    ref = pd.read_csv(REF)
    r = ref.head(6)
    print('\nreference beta against beta[idx, slot]:')
    for row in r.itertuples():
        print(f'  idx {int(row.idx):>5} slot {int(row.slot):>2}: reference '
              f'{row.beta:>6.2f}, cache {deg[int(row.idx), int(row.slot)]:>3}')
    agree = np.mean([abs(deg[int(x.idx), int(x.slot)] - x.beta) < 0.51
                     for x in ref.itertuples()])
    print(f'  agreement over all {len(ref)} reference rows: {agree:.4f}')
    if agree < 0.99:
        print('  the reading is wrong; stop here')
        return

    meta = pd.read_csv(os.path.join(GROUPED, 'metadata.csv'))
    meta = meta.drop_duplicates('surface_idx', keep='first')
    keys = {}
    dupes = 0
    for sidx, s in zip(meta['surface_idx'].to_numpy(), meta['betas_deg']):
        t = tuple(int(round(float(v))) for v in str(s).replace(',', ';').split(';'))
        if t in keys:
            dupes += 1
            keys[t] = None
        else:
            keys[t] = int(sidx)
    print(f'\n{len(meta)} surfaces, {dupes} inclination tuples seen more than '
          f'once, {sum(v is not None for v in keys.values())} usable')

    want = ref.head(N_PAIRS)
    rows, missing = [], 0
    for row in want.itertuples():
        t = tuple(deg[int(row.idx)].tolist())
        sidx = keys.get(t)
        if sidx is None:
            missing += 1
            continue
        rows.append({'idx': sidx, 'slot': int(row.slot),
                     'cache_idx': int(row.idx), 'beta': row.beta})
    print(f'translated {len(rows)} of {len(want)}, {missing} unmatched')

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(OUT, index=False)
        print(f'written to {OUT}')
        print(out.head(6).to_string(index=False))

        # The translation must put the reference beta back at the same slot of
        # the grouped metadata, or the permutation is wrong in a way the tuple
        # match cannot see.
        m = meta.set_index('surface_idx')
        bad = 0
        for x in out.itertuples():
            b = [float(v) for v in
                 str(m.loc[x.idx, 'betas_deg']).replace(',', ';').split(';')]
            if abs(b[x.slot] - x.beta) > 0.51:
                bad += 1
        print(f'slot check: {bad} of {len(out)} rows disagree on beta')


if __name__ == '__main__':
    main()