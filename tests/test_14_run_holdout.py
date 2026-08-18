"""
Tests for scripts/run_holdout.py.

The reconstruction is covered by test_12_pipeline.py and the metrics by
test_09_metrics.py, so what is tested here is how it reads the
holdout, which rows it selects, whether it resumes correctly, and whether the
column list it writes agrees with what score produces.

A synthetic holdout is built in tmp_path rather than reading the real one, so
these run anywhere, and the reconstruction is stubbed where the driver is under
test. The released flows are fixed at T = 216 with a conditioning vector of
3 T + 3 = 651 entries and a decoder expecting 961 coefficients, so a small
fixture cannot pass through them at all. The real path is covered by
test_11_end_to_end and test_slow_25_models.
"""

import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from run_holdout import (                                        # noqa: E402
    COLUMNS, _column, read_pair, load_holdout_index, score, summarise,
)
from starspot_sbi.indexing import coeffs_to_real                 # noqa: E402
from starspot_sbi.kernels import precompute_kernels_fast         # noqa: E402
from starspot_sbi.design_matrix import build_design_matrix, forward_model  # noqa: E402
from starspot_sbi.surfaces import generate_spotted_surface       # noqa: E402
from starspot_sbi.render import render, N_THETA, N_PHI           # noqa: E402
from starspot_sbi.pipeline import (render_draws, select_channels,  # noqa: E402
                                   to_model_order)
from starspot_sbi.metrics import (spot_mask, weights, wmean,     # noqa: E402
                                  ssim_aa_vis, rmse as rmse_metric,
                                  cap_boundary_lat, SPOT_THRESHOLD)

L = 8
N_OBS = 32
N_INC = 4
N_SURF = 6
P_ROT = 1.0

# run_family takes a weights directory but never reads it here, since the
# reconstruction is stubbed. The released flows are fixed at T = 216 and L = 30,
# so this fixture cannot pass through them; test_11_end_to_end covers that path.
WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'


############################
# A synthetic holdout      #
############################

@pytest.fixture(scope='module')
def holdout(tmp_path_factory):
    """
    Six surfaces, four inclinations each, in the layout the real holdout uses.

    The metadata carries one duplicate row, so the deduplication the released
    dataset requires is exercised rather than assumed.
    """
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

    frame = pd.DataFrame(rows)
    pd.concat([frame, frame.iloc[[2]]], ignore_index=True).to_csv(
        d / 'metadata.csv', index=False)
    return str(d)


############################
# Column resolution        #
############################

def test_column_finds_the_first_present():
    df = pd.DataFrame(columns=['b', 'c'])
    assert _column(df, 'a', 'b', 'c') == 'b'
    assert _column(df, 'c', 'b') == 'c'


def test_column_returns_none_when_absent():
    assert _column(pd.DataFrame(columns=['x']), 'a', 'b') is None


############################
# Reading the holdout      #
############################

def test_read_pair_returns_the_requested_slot(holdout):
    """
    The signal and the inclination must come from the same slot. Reading the
    signal at one slot and the beta at another is silent and wrong.
    """
    meta = pd.read_csv(os.path.join(holdout, 'metadata.csv'))
    meta = meta.drop_duplicates('surface_idx', keep='first').set_index('surface_idx')

    raw = np.load(os.path.join(holdout, 'signals', 'signal_0000003.npy'))
    betas = [int(b) for b in meta.loc[3, 'betas_deg'].split(';')]

    for slot in range(N_INC):
        coeffs, sig, beta, n_spots = read_pair(holdout, meta, 3, slot)
        assert np.array_equal(sig, raw[slot])
        assert beta == betas[slot]
        assert coeffs.shape == (81,)
        assert n_spots >= 1
        print(f'slot {slot}: beta {beta} deg, {n_spots} spots')


def test_read_pair_signal_reproduces_the_forward_model(holdout):
    """The stored signal is the forward model of the stored surface."""
    meta = pd.read_csv(os.path.join(holdout, 'metadata.csv'))
    meta = meta.drop_duplicates('surface_idx', keep='first').set_index('surface_idx')
    coeffs, sig, beta, _ = read_pair(holdout, meta, 1, 2)

    kx, ky, kp = precompute_kernels_fast(L)
    A = build_design_matrix(L, np.radians(beta), 2 * np.pi / P_ROT,
                            np.linspace(0, P_ROT, N_OBS, endpoint=False),
                            [kx, ky, kp])
    expected = forward_model(coeffs, A).reshape(3, N_OBS)
    err = np.max(np.abs(sig - expected))
    print(f'stored signal against the forward model: {err:.2e}')
    assert err < 1e-6                          # stored float32


############################
# Index selection          #
############################

def test_index_deduplicates(holdout):
    """
    The released metadata carries 70,000 duplicate rows from two concurrent
    generation runs. A reader that does not deduplicate double-weights them.
    """
    raw = pd.read_csv(os.path.join(holdout, 'metadata.csv'))
    assert len(raw) == N_SURF + 1              # the fixture plants one duplicate

    pairs, meta = load_holdout_index(holdout, n=N_SURF, seed=0)
    assert len(meta) == N_SURF
    assert len(pairs) == N_SURF
    assert len({i for i, _ in pairs}) == N_SURF


def test_index_respects_n(holdout):
    pairs, _ = load_holdout_index(holdout, n=3, seed=0)
    assert len(pairs) == 3
    assert [i for i, _ in pairs] == sorted(i for i, _ in pairs)


def test_index_slots_are_in_range(holdout):
    _, _ = load_holdout_index(holdout, n=N_SURF, seed=0)
    for seed in range(5):
        pairs, _ = load_holdout_index(holdout, n=N_SURF, seed=seed)
        assert all(0 <= s < N_INC for _, s in pairs)


def test_index_is_reproducible(holdout):
    a, _ = load_holdout_index(holdout, n=4, seed=7)
    b, _ = load_holdout_index(holdout, n=4, seed=7)
    c, _ = load_holdout_index(holdout, n=4, seed=8)
    assert a == b
    assert a != c


def test_index_from_a_pairs_file(holdout, tmp_path):
    """
    With --pairs the same surfaces and inclinations an earlier analysis used are
    scored, which makes the comparison paired rather than distributional.
    """
    want = [(4, 1), (0, 3), (2, 0)]
    p = tmp_path / 'pairs.csv'
    pd.DataFrame(want, columns=['idx', 'slot']).to_csv(p, index=False)

    pairs, _ = load_holdout_index(holdout, pairs_csv=str(p))
    assert pairs == want


def test_pairs_file_ignores_its_other_columns(holdout, tmp_path):
    """A results CSV carries metrics alongside idx and slot; only the two matter."""
    p = tmp_path / 'pairs.csv'
    pd.DataFrame({'idx': [1, 5], 'slot': [0, 2], 'beta': [30.0, 60.0],
                  'ssim_vis': [0.9, 0.8]}).to_csv(p, index=False)
    pairs, _ = load_holdout_index(holdout, pairs_csv=str(p))
    assert pairs == [(1, 0), (5, 2)]


############################
# Scoring and the schema   #
############################

def test_score_fills_every_column():
    """
    COLUMNS against what score returns plus the identifiers. A mismatch raises
    only after the flow has run, so it is worth catching here.
    """
    rng = np.random.default_rng(0)
    truth = 1.0 - 0.2 * rng.random((120, 240))
    recon = truth + 0.01 * rng.normal(size=truth.shape)
    samples = recon[None] + 0.01 * rng.normal(size=(8,) + truth.shape)

    row = {'idx': 0, 'slot': 0, 'beta': 45.0, 'n_spots_true': 3}
    row.update(score(truth, recon, 45.0, samples=samples,
                     recon_std=samples.std(axis=0)))

    missing = [c for c in COLUMNS if c not in row]
    extra = [k for k in row if k not in COLUMNS]
    print(f'missing {missing}, extra {extra}')
    assert not missing
    assert not extra
    assert pd.DataFrame([row])[COLUMNS].shape == (1, len(COLUMNS))


def test_score_without_samples_gives_nan_not_a_missing_column():
    """
    CRPS and the error-uncertainty correlation need the draws. Without them the
    columns must be present and nan, so the CSV schema does not change.
    """
    rng = np.random.default_rng(1)
    truth = 1.0 - 0.2 * rng.random((60, 120))
    out = score(truth, truth, 30.0)
    assert 'crps' in out and np.isnan(out['crps'])
    assert 'err_unc' in out and np.isnan(out['err_unc'])


def test_score_of_a_perfect_reconstruction():
    rng = np.random.default_rng(2)
    truth = 1.0 - 0.3 * rng.random((60, 120))
    out = score(truth, truth, 45.0)
    assert out['ssim_vis'] == pytest.approx(1.0, abs=1e-9)
    assert out['rmse_vis'] == pytest.approx(0.0, abs=1e-12)
    assert out['sff_true'] == out['sff_rec']


def test_summarise_tolerates_all_nan(capsys):
    """A column that is nan throughout must report rather than raise."""
    df = pd.DataFrame({c: [np.nan] * 3 if c == 'crps' else [1.0, 2.0, 3.0]
                       for c in COLUMNS})
    summarise(df, 'test')
    out = capsys.readouterr().out
    assert 'all nan' in out


############################
# Chunking and resuming    #
############################

def test_run_family_resumes_from_existing_chunks(holdout, tmp_path, monkeypatch):
    """
    Chunks already present are skipped and the rest rebuilt, and the assembled
    frame agrees with an uninterrupted run.

    The reconstruction is stubbed. The released flows are fixed at T = 216 with
    a conditioning vector of 3 T + 3 = 651 entries, and their decoder expects
    961 coefficients, so this deliberately small fixture cannot pass through
    them. test_11_end_to_end covers the real path on a real holdout surface;
    what is tested here is the driver.

    The stub returns an unspotted photosphere, which is a legitimate if poor
    reconstruction, so every metric returns a real value and pr_auc exercises
    both its branches: zero where the truth has visible spots and nan where it
    does not.
    """
    import argparse

    import run_holdout as rh

    pairs, meta = load_holdout_index(holdout, n=N_SURF, seed=0)
    n_coeffs_L = (L + 1) ** 2

    monkeypatch.setattr(rh, 'load_vae', lambda *a, **k: (None, None, None))
    monkeypatch.setattr(rh, 'load_flow', lambda *a, **k: (None, {'ch': [0, 1, 2]}))
    monkeypatch.setattr(
        rh, 'sample_draws',
        lambda signals, betas, *a, **k: np.zeros(
            (len(signals), k.get('n_draws', 8), n_coeffs_L), dtype=np.float32))

    rendered = []

    def fake_render(v, *a, **k):
        # the production grid, since run_family renders the truth at the
        # default and the two meet in ssim_map
        v = np.atleast_2d(v)
        rendered.append(v.shape[0])
        return np.ones((v.shape[0], N_THETA, N_PHI))

    monkeypatch.setattr(rh, 'render_draws', fake_render)

    args = argparse.Namespace(draws=8, batch=4, seed=0, save_draws=False,
                              spectra=False, log_sigma_phot=-4.0,
                              log_sigma_astro=-3.5, device='cpu')

    original_chunk = rh.CHUNK
    rh.CHUNK = 3                       # two chunks from six pairs
    try:
        out = str(tmp_path / 'results')
        df1, _ = rh.run_family('phot_axay', pairs, meta, holdout,
                               str(WEIGHTS), out, args)

        fam = os.path.join(out, 'phot_axay')
        chunks = sorted(f for f in os.listdir(fam) if f.startswith('chunk_'))
        assert len(chunks) == 2
        assert len(df1) == N_SURF
        assert list(df1.columns) == COLUMNS

        kept = pd.read_csv(os.path.join(fam, chunks[0]))
        n_rendered = len(rendered)
        os.remove(os.path.join(fam, chunks[1]))

        df2, _ = rh.run_family('phot_axay', pairs, meta, holdout,
                               str(WEIGHTS), out, args)

        print(f'{n_rendered} render calls for two chunks, '
              f'{len(rendered) - n_rendered} for the one rebuilt')
        assert pd.read_csv(os.path.join(fam, chunks[0])).equals(kept)
        assert len(rendered) - n_rendered < n_rendered, 'the kept chunk was recomputed'
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))
    finally:
        rh.CHUNK = original_chunk


def test_run_family_writes_the_combined_csv(holdout, tmp_path, monkeypatch):
    """The per-chunk files are concatenated and sorted by surface index."""
    import argparse

    import run_holdout as rh

    pairs, meta = load_holdout_index(holdout, n=N_SURF, seed=0)
    monkeypatch.setattr(rh, 'load_vae', lambda *a, **k: (None, None, None))
    monkeypatch.setattr(rh, 'load_flow', lambda *a, **k: (None, {'ch': [0, 1, 2]}))
    monkeypatch.setattr(
        rh, 'sample_draws',
        lambda signals, betas, *a, **k: np.zeros(
            (len(signals), 8, (L + 1) ** 2), dtype=np.float32))
    monkeypatch.setattr(
        rh, 'render_draws',
        lambda v, *a, **k: np.ones((np.atleast_2d(v).shape[0], N_THETA, N_PHI)))

    args = argparse.Namespace(draws=8, batch=4, seed=0, save_draws=False,
                              spectra=False, log_sigma_phot=-4.0,
                              log_sigma_astro=-3.5, device='cpu')

    out = str(tmp_path / 'results')
    df, _ = rh.run_family('phot_axay', pairs, meta, holdout, str(WEIGHTS),
                          out, args)

    combined = os.path.join(out, 'full_phot_axay_lp-4.0_la-3.5.csv')
    assert os.path.exists(combined)
    on_disk = pd.read_csv(combined)
    assert len(on_disk) == N_SURF
    assert list(on_disk['idx']) == sorted(on_disk['idx'])
    assert list(on_disk.columns) == COLUMNS

############################
# Channel order            #
############################

def test_run_family_passes_the_stored_channel_order(holdout, tmp_path,
                                                    monkeypatch):
    """
    run_family must hand sample_draws the stored three-channel signal, since
    sample_draws applies to_model_order itself. A select_channels call in
    run_family permutes twice, and [2,0,1] composed with itself is [1,2,0],
    which puts the photometric series in an astrometric slot. The context then
    standardises to |z| in the tens of thousands and the flow's spline inverse
    fails its discriminant assertion.

    The channels are told apart by their means. Photometry is a relative flux
    near one; the astrometric channels are centroid offsets near zero.
    """
    import argparse

    import run_holdout as rh

    pairs, meta = load_holdout_index(holdout, n=N_SURF, seed=0)
    captured = {}

    def capture(signals, betas, family, *a, **k):
        captured['sig'] = np.asarray(signals)
        return np.zeros((len(signals), 8, (L + 1) ** 2), dtype=np.float32)

    monkeypatch.setattr(rh, 'load_vae', lambda *a, **k: (None, None, None))
    monkeypatch.setattr(rh, 'load_flow', lambda *a, **k: (None, {'ch': [0, 1, 2]}))
    monkeypatch.setattr(rh, 'sample_draws', capture)
    monkeypatch.setattr(
        rh, 'render_draws',
        lambda v, *a, **k: np.ones((np.atleast_2d(v).shape[0], N_THETA, N_PHI)))

    args = argparse.Namespace(draws=8, batch=4, seed=0, save_draws=False,
                              spectra=False, log_sigma_phot=-4.0,
                              log_sigma_astro=-3.5, device='cpu')
    rh.run_family('phot_axay', pairs, meta, holdout, str(WEIGHTS),
                  str(tmp_path / 'results'), args)

    sig = captured['sig']
    means = [float(np.mean(sig[:, j])) for j in range(3)]
    print(f'channel means as passed to sample_draws: '
          + ', '.join(f'{m:+.4f}' for m in means))
    assert sig.shape[1] == 3
    assert means[2] > 0.5, 'stored order puts photometry last'
    assert abs(means[0]) < 0.1 and abs(means[1]) < 0.1

    once = to_model_order(sig, 'phot_axay', stored=True)
    twice = to_model_order(once, 'phot_axay', stored=True)
    m_once = [float(np.mean(once[:, j])) for j in range(3)]
    m_twice = [float(np.mean(twice[:, j])) for j in range(3)]
    print(f'one permutation: ' + ', '.join(f'{m:+.4f}' for m in m_once))
    print(f'two permutations: ' + ', '.join(f'{m:+.4f}' for m in m_twice))
    assert m_once[0] > 0.5, 'model order puts photometry first'
    assert m_twice[0] < 0.1, 'the double permutation must be detectable'
    assert np.array_equal(once, select_channels(sig, 'phot_axay'))


############################
# The filling factor       #
############################

L_SFF = 20
BETA_SFF = 60.0


def _capped_surface(theta_deg, radius_deg=15.0, contrast=0.4):
    """One spot at a given colatitude, rendered on the production grid."""
    s = generate_spotted_surface(
        L_SFF, [{'theta': np.radians(theta_deg), 'phi': 0.0,
                 'radius': np.radians(radius_deg), 'contrast': contrast}],
        lanczos=True)
    return render_draws(coeffs_to_real(s)[None], N_THETA, N_PHI)[0]


def test_filling_factor_ignores_a_spot_in_the_unobservable_cap():
    """
    The cap is the colatitudes below beta, which no observer at that
    inclination ever sees. A spot there contributes nothing to sff_true, and a
    spot at the same size on the visible hemisphere contributes a positive
    fraction. An unweighted mean over the grid counts both.
    """
    print(f'cap boundary at beta = {BETA_SFF} deg: latitude '
          f'{cap_boundary_lat(np.radians(BETA_SFF)):.1f}')

    hidden = _capped_surface(10.0)
    shown = _capped_surface(110.0)
    print(f'minimum intensity: hidden {hidden.min():.4f}, shown {shown.min():.4f}')
    assert hidden.min() < SPOT_THRESHOLD, 'the spot must be deep enough to count'
    assert shown.min() < SPOT_THRESHOLD

    plain_hidden = float(np.mean(spot_mask(hidden)))
    out_hidden = score(hidden, hidden, BETA_SFF)
    out_shown = score(shown, shown, BETA_SFF)
    print(f'hidden spot: unweighted {plain_hidden:.5f}, sff_true '
          f'{out_hidden["sff_true"]:.5f}; shown spot: sff_true '
          f'{out_shown["sff_true"]:.5f}')

    assert plain_hidden > 1e-3, 'the unweighted mean must see it, or the test '\
                                'cannot tell the two definitions apart'
    assert out_hidden['sff_true'] == 0.0
    assert out_shown['sff_true'] > 1e-3


def test_filling_factor_is_area_weighted():
    """
    Area weighting by sin(theta) is what makes the number a fraction of stellar
    surface rather than of grid cells. A spot near a pole covers many more cells
    than the same solid angle at the equator, so the two definitions separate.
    """
    polar = _capped_surface(100.0)          # just inside the visible region
    equatorial = _capped_surface(90.0)

    w = weights(0.0, N_THETA, N_PHI, kind='vis')
    for name, img in (('near-polar', polar), ('equatorial', equatorial)):
        plain = float(np.mean(spot_mask(img)))
        area = wmean(spot_mask(img), w)
        print(f'{name}: unweighted {plain:.5f}, area weighted {area:.5f}, '
              f'ratio {plain / area:.4f}')
        assert abs(plain / area - 1.0) > 1e-3


############################
# Render resolution        #
############################

def test_ssim_depends_on_the_render_grid_and_rmse_does_not():
    """
    SSIM compares a fixed pixel window, so its value is a property of the grid
    as much as of the surfaces. On the released holdout the same reconstructions
    score 0.94289 at 60 x 120 and 0.96141 at 120 x 240, while RMSE moves by
    0.00006. An SSIM number is therefore quotable only with its grid, and the
    package renders at 120 x 240.
    """
    rng = np.random.default_rng(3)
    s = generate_spotted_surface(
        L_SFF, [{'theta': np.radians(70.0), 'phi': 0.4,
                 'radius': np.radians(12.0), 'contrast': 0.5},
                {'theta': np.radians(115.0), 'phi': -1.2,
                 'radius': np.radians(9.0), 'contrast': 0.6}],
        lanczos=True)
    truth = coeffs_to_real(s)

    # An error with power at the truncation scale, which is what a decoded
    # reconstruction leaves behind. A purely low-degree error would be smooth
    # on both grids and the comparison would be empty.
    ell = np.floor(np.sqrt(np.arange(truth.size))).astype(float)
    recon = truth + 3e-3 * (ell / ell.max()) * rng.normal(size=truth.size)

    beta = np.radians(45.0)
    out = {}
    for g in ((60, 120), (120, 240)):
        t = render_draws(truth[None], *g)[0]
        r = render_draws(recon[None], *g)[0]
        out[g] = (ssim_aa_vis(t, r, beta), rmse_metric(t, r, beta, 'vis'))
        print(f'{g}: ssim_vis {out[g][0]:.5f}, rmse_vis {out[g][1]:.5f}')

    d_ssim = abs(out[(60, 120)][0] - out[(120, 240)][0])
    d_rmse = abs(out[(60, 120)][1] - out[(120, 240)][1])
    rel_rmse = d_rmse / out[(120, 240)][1]
    print(f'SSIM moves by {d_ssim:.5f}, RMSE by {d_rmse:.5f} '
          f'({rel_rmse:.2%} relative)')
    assert d_ssim > 5e-3
    assert rel_rmse < 0.10