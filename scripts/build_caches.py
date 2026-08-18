"""
Build the kernel and design-matrix caches.

These are properties of the forward model rather than of any dataset. A(beta)
depends on l_max, n_obs and the set of inclinations.

p_rot does not enter. build_W uses exp(-i m omega t) with the epochs spanning one
period, so omega t_n = 2 pi n / N whatever the period is, and rescaling p_rot
leaves W unchanged. 

If irregular cadence or multi-rotation baselines ever arrive, interpolate I suppose

Contents:

    kernels_L<l_max>.npz              kx, ky, kphot, complex128
    design_L<l_max>_N<n_obs>_B<n_beta>.npy    (n_beta, 3, n_obs, (l_max+1)^2)
    design_L<l_max>_N<n_obs>_B<n_beta>.json   provenance

The design array is complex128 and about 9 GB at l_max = 30 with 91
inclinations, so it is written through a memory map and read the same way.
Channel order is (astro_x, astro_y, phot), matching the stored signal files.

Usage:
    python scripts/build_caches.py                     # L=30, N=216, 91 inclinations
    python scripts/build_caches.py --n-beta 3          # a small cache for testing
    python scripts/build_caches.py --cache-dir /shared/caches
    python scripts/build_caches.py --check             # verify without building
"""

###########
# Imports #
###########

# python
import os
import json
import hashlib
import argparse
from datetime import datetime

# standard
import numpy as np
from tqdm.auto import tqdm

# self
import starspot_sbi
from starspot_sbi.indexing import n_coeffs
from starspot_sbi.kernels import precompute_kernels_fast, _GL_N
from starspot_sbi.design_matrix import build_W, build_B


#############
# Constants #
#############

L_MAX = 30
N_OBS = 216                       # 216 = 8*27, 8 obs per day for a single a-cen rotation
N_BETA = 91                       # integer degrees 0 to 90 inclusive
P_ROT = 1.0                       # normalised anyway

DEFAULT_CACHE_DIR = 'caches'


#####################
# Provenance        #
#####################

def epochs(n_obs=N_OBS, p_rot=P_ROT):
    """One uniform rotation, t[0] = 0 exactly."""
    return np.linspace(0, p_rot, n_obs, endpoint=False)


def epoch_hash(t_obs, p_rot=P_ROT):
    """
    Hash of the phase grid rather than of the times, since W depends on
    omega * t and not on the period.
    """
    phase = np.mod(2 * np.pi * np.asarray(t_obs) / p_rot, 2 * np.pi)
    return hashlib.sha256(np.round(phase, 12).tobytes()).hexdigest()[:16]


def provenance(l_max, n_obs, n_beta, t_obs, p_rot):
    return {
        'l_max': int(l_max),
        'n_obs': int(n_obs),
        'n_beta': int(n_beta),
        'beta_deg': [0, int(n_beta) - 1],
        'n_coeffs': int(n_coeffs(l_max)),
        'channel_order': ['astro_x', 'astro_y', 'phot'],
        'gl_nodes': int(_GL_N),
        'epoch_hash': epoch_hash(t_obs, p_rot),
        'p_rot': float(p_rot),
        'package_version': getattr(starspot_sbi, '__version__', 'unknown'),
        'built': datetime.now().isoformat(timespec='seconds'),
    }


def check_provenance(sidecar_path, want):
    """
    Compare a cache's sidecar against what the caller expects. Returns a list of
    disagreements, empty when the cache is usable.

    'built' and 'package_version' are reported but do not by themselves
    invalidate a cache. Everything else is a refusal.
    """
    if not os.path.exists(sidecar_path):
        return ['no sidecar; the cache cannot be verified']
    with open(sidecar_path) as f:
        have = json.load(f)

    problems = []
    for k in ('l_max', 'n_obs', 'n_beta', 'gl_nodes', 'epoch_hash', 'channel_order'):
        if have.get(k) != want[k]:
            problems.append(f'{k}: cache has {have.get(k)!r}, expected {want[k]!r}')
    return problems


#####################
# Building          #
#####################

def kernel_path(cache_dir, l_max):
    return os.path.join(cache_dir, f'kernels_L{l_max}.npz')


def design_path(cache_dir, l_max, n_obs, n_beta):
    return os.path.join(cache_dir, f'design_L{l_max}_N{n_obs}_B{n_beta}.npy')


def build_kernels(cache_dir, l_max=L_MAX, force=False):
    """
    Measurement kernels at degree l_max. Seconds to build, kilobytes to store,
    cached mostly so that the design build and the dataset generator agree by
    construction rather than by both calling the same function.
    """
    path = kernel_path(cache_dir, l_max)
    if os.path.exists(path) and not force:
        d = np.load(path)
        if int(d['gl_nodes']) != _GL_N:
            raise ValueError(f'{path}: built with {int(d["gl_nodes"])} quadrature '
                             f'nodes, package now uses {_GL_N}; rebuild with --force')
        return d['kx'], d['ky'], d['kphot']

    print(f'building kernels at L = {l_max}, {_GL_N} quadrature nodes ... ',
          end='', flush=True)
    kx, ky, kphot = precompute_kernels_fast(l_max)
    np.savez(path, kx=kx, ky=ky, kphot=kphot, l_max=l_max, gl_nodes=_GL_N)
    print('done')
    return kx, ky, kphot


def build_design(cache_dir, l_max=L_MAX, n_obs=N_OBS, n_beta=N_BETA,
                 p_rot=P_ROT, force=False):
    """
    A(beta) for beta = 0 to n_beta - 1 degrees, shape
    (n_beta, 3, n_obs, (l_max+1)^2), channel order (astro_x, astro_y, phot).

    Written through a memory map, so peak memory is one inclination rather than
    the whole 9 GB. Returned memory-mapped for the same reason: a caller wanting
    one inclination should not page in the rest.
    """
    path = design_path(cache_dir, l_max, n_obs, n_beta)
    side = path.replace('.npy', '.json')
    t_obs = epochs(n_obs, p_rot)
    want = provenance(l_max, n_obs, n_beta, t_obs, p_rot)

    if os.path.exists(path) and not force:
        problems = check_provenance(side, want)
        if problems:
            raise ValueError(f'{path}: cache disagrees with the current build:\n  '
                             + '\n  '.join(problems)
                             + '\nrebuild with --force, or point --cache-dir elsewhere')
        return np.load(path, mmap_mode='r')

    kx, ky, kphot = build_kernels(cache_dir, l_max, force=force)
    omega = 2 * np.pi / p_rot
    W = build_W(l_max, omega, t_obs)

    A = np.lib.format.open_memmap(
        path, mode='w+', dtype=np.complex128,
        shape=(n_beta, 3, n_obs, n_coeffs(l_max)))
    gb_total = A.nbytes / 1e9

    # One tick per channel rather than per inclination: each build_B loops over
    # l_max + 1 Wigner matrices, so a per-inclination bar sits still for seconds
    # at L = 30. Flushing every ten inclinations means an interrupted build
    # leaves a partial file rather than losing the page cache.
    bar = tqdm(total=n_beta * 3, unit='matrix',
               desc=f'design L{l_max} N{n_obs} B{n_beta}')
    for b in range(n_beta):
        beta = np.radians(b)
        for j, k in enumerate((kx, ky, kphot)):
            A[b, j] = W @ build_B(l_max, beta, k)
            bar.update(1)
        bar.set_postfix(beta=f'{b} deg',
                        written=f'{(b + 1) / n_beta * gb_total:.1f}/{gb_total:.1f} GB')
        if b % 10 == 9:
            A.flush()
    bar.close()

    A.flush()
    del A

    with open(side, 'w') as f:
        json.dump(want, f, indent=2)
    return np.load(path, mmap_mode='r')


def load_design(cache_dir=DEFAULT_CACHE_DIR, l_max=L_MAX, n_obs=N_OBS,
                n_beta=N_BETA, p_rot=P_ROT):
    """
    Memory-map an existing design cache, refusing one built under different
    settings. Raises rather than building, so a caller that expected a cache to
    exist finds out.
    """
    path = design_path(cache_dir, l_max, n_obs, n_beta)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path}\nbuild it with: python scripts/build_caches.py '
            f'--l-max {l_max} --n-obs {n_obs} --n-beta {n_beta} '
            f'--cache-dir {cache_dir}')
    problems = check_provenance(path.replace('.npy', '.json'),
                                provenance(l_max, n_obs, n_beta,
                                           epochs(n_obs, p_rot), p_rot))
    if problems:
        raise ValueError(f'{path}: ' + '; '.join(problems))
    return np.load(path, mmap_mode='r')


def load_kernels(cache_dir=DEFAULT_CACHE_DIR, l_max=L_MAX):
    """Load an existing kernel cache, building it if absent since it is cheap."""
    return build_kernels(cache_dir, l_max)


#####################
# Reporting         #
#####################

def describe(cache_dir):
    """List every cache present, with its size and its provenance."""
    if not os.path.isdir(cache_dir):
        print(f'{cache_dir}: does not exist')
        return
    files = sorted(os.listdir(cache_dir))
    if not files:
        print(f'{cache_dir}: empty')
        return

    print(f'{cache_dir}:')
    for f in files:
        if f.endswith('.json'):
            continue
        p = os.path.join(cache_dir, f)
        print(f'  {f:<44s} {os.path.getsize(p) / 1e9:8.3f} GB')

        if not f.endswith('.npy'):
            continue                       # only the design caches have sidecars
        side = p[:-4] + '.json'
        if not os.path.exists(side):
            print('      no sidecar; this cache cannot be verified')
            continue
        with open(side) as fh:
            d = json.load(fh)
        print(f'      L{d["l_max"]} N{d["n_obs"]} B{d["n_beta"]}, '
              f'{d["gl_nodes"]} quadrature nodes, '
              f'epochs {d["epoch_hash"]}, built {d["built"]}')


###############
# Entry point #
###############

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', default=os.environ.get('STARSPOT_CACHE_DIR',
                                                          DEFAULT_CACHE_DIR),
                    help='where the caches live; also settable through '
                         'STARSPOT_CACHE_DIR, so several checkouts can share one')
    ap.add_argument('--l-max', type=int, default=L_MAX)
    ap.add_argument('--n-obs', type=int, default=N_OBS)
    ap.add_argument('--n-beta', type=int, default=N_BETA,
                    help='integer inclinations from 0; 3 is enough to test with')
    ap.add_argument('--p-rot', type=float, default=P_ROT)
    ap.add_argument('--kernels-only', action='store_true')
    ap.add_argument('--force', action='store_true', help='rebuild even if present')
    ap.add_argument('--check', action='store_true',
                    help='report what is present and verify it, building nothing')
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    if args.check:
        describe(args.cache_dir)
        return

    gb = (args.n_beta * 3 * args.n_obs * n_coeffs(args.l_max) * 16) / 1e9
    print(f'cache dir {os.path.abspath(args.cache_dir)}')
    print(f'L{args.l_max} N{args.n_obs} B{args.n_beta}: '
          f'design matrices will occupy {gb:.2f} GB')

    build_kernels(args.cache_dir, args.l_max, force=args.force)
    print(f'kernels: {kernel_path(args.cache_dir, args.l_max)}')

    if not args.kernels_only:
        build_design(args.cache_dir, args.l_max, args.n_obs, args.n_beta,
                     args.p_rot, force=args.force)
        print(f'design:  {design_path(args.cache_dir, args.l_max, args.n_obs, args.n_beta)}')

    print()
    describe(args.cache_dir)


if __name__ == '__main__':
    main()