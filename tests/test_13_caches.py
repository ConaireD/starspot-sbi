"""
Tests for scripts/build_caches.py.

build_B, build_W and the kernels are covered by their own test files, so what is
tested here is the cache layer: the provenance keys, the refusal of a mismatched
cache, and the agreement of the stored array with build_design_matrix called
directly.

Everything runs at l_max = 4 and n_beta = 3, so the design array is a few
megabytes. 
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from build_caches import (                                    # noqa: E402
    epochs, epoch_hash, provenance, check_provenance,
    kernel_path, design_path, build_kernels, build_design,
    load_design, load_kernels, describe,
)
from starspot_sbi.indexing import n_coeffs                     # noqa: E402
from starspot_sbi.kernels import precompute_kernels_fast, _GL_N  # noqa: E402
from starspot_sbi.design_matrix import build_design_matrix     # noqa: E402

L = 4
N_OBS = 16
N_BETA = 3
P_ROT = 1.0


@pytest.fixture(scope='module')
def cache(tmp_path_factory):
    """One built cache, shared, since building it twice tests nothing."""
    d = tmp_path_factory.mktemp('caches')
    build_design(str(d), L, N_OBS, N_BETA, P_ROT)
    return str(d)


############################
# Epoch hashing            #
############################

def test_epoch_hash_ignores_the_period():
    """
    W uses exp(-i m omega t) over one period, so omega t_n = 2 pi n / N whatever
    the period is. This is the claim that keeps p_rot out of the cache key.
    """
    a = epoch_hash(epochs(N_OBS, 1.0), 1.0)
    b = epoch_hash(epochs(N_OBS, 27.0), 27.0)
    print(f'p_rot 1: {a}   p_rot 27: {b}')
    assert a == b


def test_epoch_hash_sees_a_different_cadence():
    """An irregular grid must not collide with the uniform one."""
    uniform = epochs(N_OBS, P_ROT)
    irregular = np.sort(np.random.default_rng(0).uniform(0, P_ROT, N_OBS))
    assert epoch_hash(uniform, P_ROT) != epoch_hash(irregular, P_ROT)


def test_epoch_hash_sees_a_different_length():
    assert epoch_hash(epochs(16, P_ROT), P_ROT) != epoch_hash(epochs(32, P_ROT), P_ROT)


def test_epoch_hash_is_stable():
    """Same input, same hash, so a cache does not expire on a rerun."""
    assert epoch_hash(epochs(N_OBS, P_ROT), P_ROT) == epoch_hash(epochs(N_OBS, P_ROT), P_ROT)


############################
# Provenance               #
############################

def test_provenance_records_the_keys():
    p = provenance(L, N_OBS, N_BETA, epochs(N_OBS, P_ROT), P_ROT)
    assert p['l_max'] == L
    assert p['n_obs'] == N_OBS
    assert p['n_beta'] == N_BETA
    assert p['n_coeffs'] == n_coeffs(L)
    assert p['gl_nodes'] == _GL_N
    assert p['channel_order'] == ['astro_x', 'astro_y', 'phot']


def test_check_provenance_accepts_a_matching_cache(cache):
    side = design_path(cache, L, N_OBS, N_BETA).replace('.npy', '.json')
    want = provenance(L, N_OBS, N_BETA, epochs(N_OBS, P_ROT), P_ROT)
    assert check_provenance(side, want) == []


@pytest.mark.parametrize('key,value', [
    ('l_max', 5),
    ('n_obs', 32),
    ('n_beta', 4),
    ('gl_nodes', 2000),
    ('epoch_hash', 'deadbeefdeadbeef'),
])
def test_check_provenance_refuses_a_mismatch(cache, key, value):
    """
    Each key on its own must be enough to refuse. gl_nodes is the important one:
    nothing in the filename records the quadrature, so a cache built at 2000
    nodes would otherwise be used silently by a package running at 500.
    """
    side = design_path(cache, L, N_OBS, N_BETA).replace('.npy', '.json')
    want = provenance(L, N_OBS, N_BETA, epochs(N_OBS, P_ROT), P_ROT)
    want[key] = value
    problems = check_provenance(side, want)
    print(f'{key} -> {value}: {problems}')
    assert len(problems) == 1
    assert key in problems[0]


def test_check_provenance_refuses_a_missing_sidecar(tmp_path):
    want = provenance(L, N_OBS, N_BETA, epochs(N_OBS, P_ROT), P_ROT)
    problems = check_provenance(str(tmp_path / 'absent.json'), want)
    assert len(problems) == 1
    assert 'sidecar' in problems[0]


############################
# Building and loading     #
############################

def test_files_are_written(cache):
    assert os.path.exists(kernel_path(cache, L))
    assert os.path.exists(design_path(cache, L, N_OBS, N_BETA))
    assert os.path.exists(design_path(cache, L, N_OBS, N_BETA).replace('.npy', '.json'))


def test_design_shape_and_dtype(cache):
    A = load_design(cache, L, N_OBS, N_BETA, P_ROT)
    print(f'shape {A.shape}, dtype {A.dtype}, {A.nbytes / 1e6:.2f} MB')
    assert A.shape == (N_BETA, 3, N_OBS, n_coeffs(L))
    assert A.dtype == np.complex128


def test_design_is_memory_mapped(cache):
    """A caller wanting one inclination should not page in the rest."""
    assert isinstance(load_design(cache, L, N_OBS, N_BETA, P_ROT), np.memmap)


def test_kernels_round_trip(cache):
    kx, ky, kp = load_kernels(cache, L)
    ax, ay, ap = precompute_kernels_fast(L)
    for name, a, b in [('kx', kx, ax), ('ky', ky, ay), ('kphot', kp, ap)]:
        assert np.array_equal(a, b), name


def test_cached_design_matches_the_analytic_operator(cache):
    """
    The cache against build_design_matrix called directly, per inclination and
    per channel. Catches a wrong channel order or a wrong inclination index,
    neither of which changes the shape or the dtype.
    """
    A = load_design(cache, L, N_OBS, N_BETA, P_ROT)
    kx, ky, kp = load_kernels(cache, L)
    omega = 2 * np.pi / P_ROT
    t_obs = epochs(N_OBS, P_ROT)

    for b in range(N_BETA):
        direct = build_design_matrix(L, np.radians(b), omega, t_obs, [kx, ky, kp])
        for j, name in enumerate(('astro_x', 'astro_y', 'phot')):
            block = direct[j * N_OBS:(j + 1) * N_OBS]
            err = np.max(np.abs(np.asarray(A[b, j]) - block))
            assert err == 0.0, f'beta {b}, channel {name}: {err:.2e}'
    print(f'{N_BETA} inclinations x 3 channels agree bitwise')


def test_channel_order_is_not_symmetric(cache):
    """
    The agreement test above would pass under a permutation only if the channels
    were interchangeable. They are not, so this states it: swapping any pair
    changes the array.
    """
    A = np.asarray(load_design(cache, L, N_OBS, N_BETA, P_ROT))
    assert not np.allclose(A[0, 0], A[0, 1])
    assert not np.allclose(A[0, 1], A[0, 2])
    assert not np.allclose(A[0, 0], A[0, 2])

def test_describe_handles_a_cache_without_a_sidecar(cache, capsys, tmp_path):
    """The kernel cache is an .npz with no sidecar, which must not be parsed as JSON."""
    describe(cache)
    out = capsys.readouterr().out
    assert 'kernels_L' in out
    assert 'design_L' in out


def test_second_build_reuses_rather_than_rebuilding(cache):
    """A rebuild returns the same bytes and leaves the sidecar's build time."""
    side = design_path(cache, L, N_OBS, N_BETA).replace('.npy', '.json')
    with open(side) as f:
        before = json.load(f)['built']
    A1 = np.asarray(load_design(cache, L, N_OBS, N_BETA, P_ROT))
    A2 = np.asarray(build_design(cache, L, N_OBS, N_BETA, P_ROT))
    with open(side) as f:
        after = json.load(f)['built']
    assert np.array_equal(A1, A2)
    assert before == after


def test_force_rebuilds(cache):
    """--force writes again and the result is identical, since nothing is random."""
    A1 = np.asarray(load_design(cache, L, N_OBS, N_BETA, P_ROT))
    A2 = np.asarray(build_design(cache, L, N_OBS, N_BETA, P_ROT, force=True))
    assert np.array_equal(A1, A2)


def test_load_design_refuses_an_absent_cache(tmp_path):
    """The error names the command that would build it."""
    with pytest.raises(FileNotFoundError, match='build_caches'):
        load_design(str(tmp_path), L, N_OBS, N_BETA, P_ROT)


def test_load_design_refuses_a_tampered_sidecar(cache, tmp_path):
    """A cache whose sidecar disagrees is refused rather than used."""
    import shutil
    d = str(tmp_path / 'copy')
    shutil.copytree(cache, d)
    side = design_path(d, L, N_OBS, N_BETA).replace('.npy', '.json')
    with open(side) as f:
        meta = json.load(f)
    meta['gl_nodes'] = 12345
    with open(side, 'w') as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match='gl_nodes'):
        load_design(d, L, N_OBS, N_BETA, P_ROT)


def test_separate_n_beta_gives_a_separate_file(cache):
    """
    A small cache built for testing must not be picked up by a later full run.
    The inclination count is in the filename for that reason.
    """
    build_design(cache, L, N_OBS, 2, P_ROT)
    assert design_path(cache, L, N_OBS, 2) != design_path(cache, L, N_OBS, N_BETA)
    assert os.path.exists(design_path(cache, L, N_OBS, 2))
    assert load_design(cache, L, N_OBS, 2, P_ROT).shape[0] == 2
    assert load_design(cache, L, N_OBS, N_BETA, P_ROT).shape[0] == N_BETA


def test_describe_runs(cache, capsys):
    describe(cache)
    out = capsys.readouterr().out
    assert 'design_L' in out
    assert 'GB' in out