# Caches

Kernels and design matrices. Not tracked by git except this file.

These are properties of the forward model. `A(beta)` depends on `l_max`, `n_obs` and the set of inclinations.

    kernels_L30.npz                 kx, ky, kphot at degree 30
    design_L30_N216_B91.npy         (91, 3, 216, 961) complex128, 9.0 GB
    design_L30_N216_B91.json        provenance

Build with:

    python scripts/build_caches.py                  # production settings
    python scripts/build_caches.py --n-beta 3       # small, for testing
    python scripts/build_caches.py --check          # report what is here

Set `STARSPOT_CACHE_DIR` to share one copy between checkouts.