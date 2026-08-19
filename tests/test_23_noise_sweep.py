"""
Tests for scripts/noise_sweep.py.

Follows test_14_run_holdout.py's convention: a small synthetic holdout
built in tmp_path, with load_vae, load_flow, sample_draws and render_draws
stubbed so the driver is exercised without the released weights or a GPU.
test_11_end_to_end and the real overnight run (results/snr/) cover the
real path; what is tested here is the script's own logic: which pairs it
selects, whether it resumes correctly across (family, noise level,
surface) rather than only (family, surface), and whether the output
schema matches what the figure notebook reads.

Assumptions this file cannot check:
- sample_draws' and render_draws' true output shapes and value ranges,
  covered by test_12_pipeline and test_17_vae_pipeline.
- ssim_aa_vis / pr_auc / crps's correctness, covered by test_09_metrics.
- that the real flow checkpoints accept the conditioning vector shape
  noise_sweep.py builds; that is exercised only by a real weights run,
  since the small fixture here cannot pass a released decoder.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from run_holdout import load_holdout_index                       # noqa: E402
from starspot_sbi.kernels import precompute_kernels_fast         # noqa: E402
from starspot_sbi.design_matrix import build_design_matrix, forward_model  # noqa: E402
from starspot_sbi.surfaces import generate_spotted_surface       # noqa: E402
from starspot_sbi.render import N_THETA, N_PHI                   # noqa: E402

L = 8
N_OBS = 32
N_INC = 4
N_SURF = 6
P_ROT = 1.0
WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'


############################
# A synthetic holdout      #
############################

@pytest.fixture(scope='module')
def holdout(tmp_path_factory):
    """Six surfaces, four inclinations each, matching test_14's fixture."""
    d = tmp_path_factory.mktemp('holdout')
    for sub in ('surfaces', 'signals'):
        os.makedirs(d / sub)

    kx, ky, kp = precompute_kernels_fast(L)
    omega = 2 * np.pi / P_ROT
    t_obs = np.linspace(0, P_ROT, N_OBS, endpoint=False)
    rng = np.random.default_rng(0)

    rows = []
    for i in range(N_SURF):
        spots = [{'theta': rng.uniform(0.4, 2.7), 'phi': rng.uniform(-3, 3),
                  'radius': np.radians(rng.uniform(8, 14)),
                  'contrast': rng.uniform(0.5, 0.8)}
                 for _ in range(rng.integers(1, 4))]
        s = generate_spotted_surface(L, spots, lanczos=True)
        betas = sorted(rng.choice(91, size=N_INC, replace=False).tolist())

        sig = np.empty((N_INC, 3, N_OBS), dtype=np.float32)
        for k, b in enumerate(betas):
            A = build_design_matrix(L, np.radians(b), omega, t_obs, [kx, ky, kp])
            sig[k] = forward_model(s, A).reshape(3, N_OBS)

        np.save(d / 'surfaces' / f'surface_{i:07d}.npy', s)
        np.save(d / 'signals' / f'signal_{i:07d}.npy', sig)
        rows.append({'surface_idx': i, 'n_inc': N_INC, 'n_spots': len(spots),
                     'betas_deg': ';'.join(str(b) for b in betas),
                     'surface_file': f'surface_{i:07d}.npy',
                     'signal_file': f'signal_{i:07d}.npy'})

    pd.DataFrame(rows).to_csv(d / 'metadata.csv', index=False)
    return str(d)


############################
# Stubs                    #
############################

def _stub(monkeypatch, ns, n_calls):
    """
    Patch load_vae/load_flow/sample_draws/render_draws so main() runs with
    no weights and no GPU. n_calls records one entry per sample_draws call,
    so a test can tell how many surface-level draws were actually sampled.
    """
    n_coeffs_L = (L + 1) ** 2
    monkeypatch.setattr(ns, 'load_vae',
                        lambda *a, **k: (None, None, {'std_data': 1.0,
                                                      'mu_data': 0.0}))
    monkeypatch.setattr(ns, 'load_flow',
                        lambda *a, **k: (None, {'ch': [0, 1, 2]}))

    def fake_sample_draws(signals, betas, *a, **k):
        n_calls.append(len(signals))
        return np.zeros((len(signals), k.get('n_draws', 4), n_coeffs_L),
                        dtype=np.float32)
    monkeypatch.setattr(ns, 'sample_draws', fake_sample_draws)
    monkeypatch.setattr(
        ns, 'render_draws',
        lambda v, *a, **k: np.ones((np.atleast_2d(v).shape[0], N_THETA, N_PHI)))


def _run(monkeypatch, argv, ns_module='noise_sweep'):
    import importlib
    ns = importlib.import_module(ns_module)
    n_calls = []
    _stub(monkeypatch, ns, n_calls)
    monkeypatch.setattr(sys, 'argv', ['noise_sweep.py'] + argv)
    ns.main()
    return ns, n_calls


def _base(holdout, out, n=N_SURF):
    """The arguments every run shares, up to but not including --grid."""
    return ['--family', 'phot', '--holdout', holdout,
            '--weights', str(WEIGHTS), '--out', out,
            '--n', str(n), '--draws', '4']


############################
# Output schema            #
############################

def test_writes_one_row_per_pair_per_family_per_level(holdout, tmp_path,
                                                      monkeypatch):
    out = str(tmp_path / 'snr')
    _run(monkeypatch, _base(holdout, out)
         + ['--grid', '-4.0', '-3.5', '--seed', '0'])
    df = pd.read_csv(os.path.join(out, 'snr_lp-4.0_n6_d4.csv'))
    print(f'{len(df)} rows, columns {list(df.columns)}')
    assert set(df.columns) == {'family', 'log_sigma_astro', 'idx', 'slot',
                               'beta', 'ssim_vis', 'pr_auc', 'crps'}
    assert len(df) == N_SURF * 2               # one family, two grid points
    assert set(df.log_sigma_astro.round(2)) == {-4.0, -3.5}
    assert set(df.family) == {'phot'}


def test_all_requested_families_appear(holdout, tmp_path, monkeypatch):
    out = str(tmp_path / 'snr')
    _run(monkeypatch, ['--family', 'all', '--holdout', holdout,
                       '--weights', str(WEIGHTS), '--out', out,
                       '--n', str(N_SURF), '--draws', '4',
                       '--grid', '-4.0', '--seed', '0'])
    df = pd.read_csv(os.path.join(out, 'snr_lp-4.0_n6_d4.csv'))
    print(f'families present: {sorted(df.family.unique())}')
    assert set(df.family) == {'phot', 'phot_ax', 'phot_ay', 'phot_axay'}
    assert len(df) == N_SURF * 4


############################
# Resume                   #
############################

def test_resuming_skips_completed_family_level_pairs(holdout, tmp_path,
                                                     monkeypatch):
    """
    A (family, level, surface) triple already in the CSV is not resampled;
    a new level for the same family and surfaces is.
    """
    out = str(tmp_path / 'snr')
    _, calls1 = _run(monkeypatch, _base(holdout, out)
                     + ['--grid', '-4.0', '--seed', '0'])
    assert sum(calls1) == N_SURF

    # same level again: nothing to do
    _, calls2 = _run(monkeypatch, _base(holdout, out)
                     + ['--grid', '-4.0', '--seed', '0'])
    print(f'resample calls on an already-done level: {sum(calls2)}')
    assert sum(calls2) == 0

    # a new level for the same family: freshly sampled
    _, calls3 = _run(monkeypatch, _base(holdout, out)
                     + ['--grid', '-3.5', '--seed', '0'])
    print(f'resample calls on a new level: {sum(calls3)}')
    assert sum(calls3) == N_SURF

    df = pd.read_csv(os.path.join(out, 'snr_lp-4.0_n6_d4.csv'))
    assert len(df) == N_SURF * 2
    assert not df.duplicated(subset=['family', 'log_sigma_astro', 'idx']).any()


############################
# Pair subsampling         #
############################

def test_a_pairs_csv_larger_than_n_is_subsampled_reproducibly(
        holdout, tmp_path, monkeypatch):
    """
    load_holdout_index returns every row of an explicit --pairs file
    regardless of --n, so noise_sweep.py's own truncation to --n is what
    is under test here: the same seed keeps the same rows, and a
    different seed keeps a different subset.
    """
    pairs_all, meta = load_holdout_index(holdout, n=N_SURF, seed=0)
    pairs_csv = tmp_path / 'pairs.csv'
    pd.DataFrame(pairs_all, columns=['idx', 'slot']).to_csv(
        pairs_csv, index=False)

    keep_n = 3
    outs = []
    for seed in (5, 5, 9):
        out = str(tmp_path / f'snr_{seed}_{len(outs)}')
        _run(monkeypatch, _base(holdout, out, n=keep_n)
             + ['--pairs', str(pairs_csv), '--grid', '-4.0',
                '--seed', str(seed)])
        df = pd.read_csv(os.path.join(out, f'snr_lp-4.0_n{keep_n}_d4.csv'))
        outs.append(sorted(df.idx.tolist()))

    print(f'seed 5 twice: {outs[0]} / {outs[1]}; seed 9: {outs[2]}')
    assert len(outs[0]) == keep_n
    assert outs[0] == outs[1]
    assert outs[0] != outs[2]


def test_n_above_the_pool_keeps_every_pair(holdout, tmp_path, monkeypatch):
    out = str(tmp_path / 'snr')
    _run(monkeypatch, _base(holdout, out, n=N_SURF * 10)
         + ['--grid', '-4.0', '--seed', '0'])
    df = pd.read_csv(os.path.join(out, f'snr_lp-4.0_n{N_SURF * 10}_d4.csv'))
    assert len(df) == N_SURF


############################
# Argument handling        #
############################

def test_the_grid_takes_several_negative_values_as_separate_tokens(
        holdout, tmp_path, monkeypatch):
    """
    Argparse accepts a token matching a plain negative number as a value, so
    the levels are passed space separated. A comma-joined list does not match
    that pattern and is read as an unknown option.
    """
    out = str(tmp_path / 'snr')
    _run(monkeypatch, _base(holdout, out)
         + ['--grid', '-6.0', '-4.0', '-2.0', '--seed', '0'])
    df = pd.read_csv(os.path.join(out, 'snr_lp-4.0_n6_d4.csv'))
    levels = sorted(df.log_sigma_astro.round(2).unique())
    print(f'levels parsed: {levels}')
    assert levels == [-6.0, -4.0, -2.0]


def test_a_single_level_is_a_list_of_one(holdout, tmp_path, monkeypatch):
    out = str(tmp_path / 'snr')
    _run(monkeypatch, _base(holdout, out) + ['--grid', '-3.5', '--seed', '0'])
    df = pd.read_csv(os.path.join(out, 'snr_lp-4.0_n6_d4.csv'))
    assert df.log_sigma_astro.round(2).unique().tolist() == [-3.5]


def test_a_comma_joined_grid_is_rejected(holdout, tmp_path, monkeypatch):
    """
    The old form fails at the float conversion rather than silently sweeping
    one nonsensical level, so a script or note still passing it is visible.
    """
    out = str(tmp_path / 'snr')
    with pytest.raises(SystemExit):
        _run(monkeypatch, _base(holdout, out)
             + ['--grid=-6.0,-4.0', '--seed', '0'])


def test_the_default_grid_is_the_module_constant(holdout, tmp_path,
                                                 monkeypatch):
    """Without --grid the sweep runs the eight levels the module declares."""
    import importlib
    ns = importlib.import_module('noise_sweep')
    out = str(tmp_path / 'snr')
    _run(monkeypatch, _base(holdout, out) + ['--seed', '0'])
    df = pd.read_csv(os.path.join(out, 'snr_lp-4.0_n6_d4.csv'))
    levels = sorted(df.log_sigma_astro.round(2).unique())
    print(f'default grid: {levels}')
    assert levels == sorted(round(g, 2) for g in ns.SNR_GRID)
    assert len(df) == N_SURF * len(ns.SNR_GRID)