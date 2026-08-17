# starspot-sbi

- Simulation-based inference for stellar surface mapping
- Photometric and astrometric time series in, posterior over the surface out
  
## Accompanies

- Deagan & Montet 2026, MNRAS, arXiv:2601.11707 (information content)
- Deagan, Taaki & Montet 2026 (this pipeline) [arXiv TBD]

## Builds on

- Taaki, Corrales & Hero 2026a, ICASSP (Cramer-Rao bounds, joint identifiability)
- Taaki, Corrales & Hero 2026b, ApJ 1003, 226, arXiv:2601.11737 (three-matrix
  forward model; the design matrix here follows its construction)
- Luger et al. 2021, AJ 162, 123 (photometric rank and null space)
- Deagan & Montet 2026, MNRAS, arXiv:2601.11707 (information content)

## What is here

- `starspot_sbi/` — the package
  - `indexing` flat index and real packing
  - `kernels` photometric and astrometric measurement kernels
  - `wigner` Wigner-d and D matrices
  - `design_matrix` W, B, A(beta), forward model
  - `surfaces` spherical-cap spot model
  - `render` coefficients to a lat-lon grid
  - `metrics` weighting, SSIM, RMSE, CRPS, spot detection
  - `models` VAE, flow, beta classifier, checkpoint loaders
- `tests/` — 472 tests, numbered in dependency order
- `notebooks/tutorials/` — how the package works
- `notebooks/paper/` — one notebook per published figure
- `scripts/` — dataset generation, diagnostics
- `docs/conventions.md` — every convention with the test that checks it
- `weights/` — model checkpoints, NOT in git (see weights/README.md)
- `data/` — datasets, NOT in git (regenerate with scripts/generate_dataset.py)

## Install

- `pip install -e .`
- needs python >= 3.10, torch >= 2.0, sbi 0.26.x
- `python -m pytest` — should be 472 passing
- tests needing weights skip cleanly if weights/ is empty

## Conventions a reader must know

- L = 30, so 961 coefficients
- beta = 90 - i; beta = 0 equator-on, beta = 90 pole-on over the SOUTH pole
- sub-observer latitude -beta; unobservable cap is NORTHERN, theta < beta
- render row 0 = theta = 0 = north pole = the pole that hides
- channel indices (phot, ax, ay) = (0, 1, 2); stored .npy is (ax, ay, phot)
- F_0 = 1 + 4.2e-9 at the production quadrature
- full list in docs/conventions.md

## Reproducing the paper

- notebooks/paper/, one per figure
- needs weights/ and the holdout dataset, neither in git
- see notebooks/paper/README.md

## Citation

- [bibtex TBD]