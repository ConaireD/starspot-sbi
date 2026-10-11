# Changelog

## Unreleased

### Fixed
- Sampling now undoes the flows' latent z-score (`z * z_std + z_mu`) before
  decoding; `load_flow` reads `z_mu`/`z_std` and rejects checkpoints without
  them (#4).

### Changed
- The package's weights are now the `_filtered` set; `_temp` flow checkpoints
  can no longer be loaded.
- `GAIN_PHOT`/`GAIN_AST` updated to the filtered training set's values.
- Results in `results/holdout/` were made with the `_temp` weights and are stale.

- Default astrometric noise (`LOG10_SIGMA_ASTRO_MISSION`) is now -3.1, the
  paper's operating point (previously -3.5, the alpha Cen B mission point).
  - `noise_sweep` default grid adds -3.6 and -3.1, the operating point and the edges of the band the SNR figure shades.

### Added
- `starspot_sbi.nullspace`: null-space correlation rho, null power fraction,
  summed shrinkage; `operators.family_operator`.