# Paper map

One row per artefact in the Paper 4 draft (`main.tex`, 18 August). A row is
closed when the notebook column names a notebook under `notebooks/paper/` that
runs top to bottom from the package, the weights and the holdout, and the
number or figure it produces matches the `.tex`. Every empty cell is work not
done. Update this file in the same commit as the work that fills a cell.

Status codes: `done` the row is closed; `data` the inputs exist on disk and
only the notebook is missing; `run` needs `run_holdout.py` output first;
`fn` needs a package function and test before the notebook; `stale` the
`.tex` still quotes superseded numbers and needs the FINAL rerun values;
`vae` a VAE-section figure whose source is the older training notebooks;
`defer` not on the submission path.

Legacy source is the cell or file in `PaperFigures_v7.ipynb`, `figs_final/`,
`beta_clf/paper/` or the VAE training notebooks that currently produces the
artefact. It exists so the port has something to check against.

## Figures

| Fig | Label | Section | Legacy source | Package function(s) | Input | Notebook | Test | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | fig:headline | 1 | F1_headline_phot_axay | pipeline.sample_draws, pipeline.classify_beta, render | weights, one holdout surface | | test_12 | run |
| 2 | fig:kurtosis | 3.4 | fig_kurtosis_lm | indexing, scipy.stats.kurtosis | 55k training surfaces | | | vae |
| 3 | fig:training_sizes | 4.3 | vae_training_low_res | training logs | ladder logs | | | vae |
| 4 | fig:vae_three_reconstructions | 4.4 | three_reconstructions_low_res | pipeline.decode_ceiling, metrics.ssim | weights, holdout | | test_12 | data |
| 5 | fig:recon_hist | 4.4 | reconstruction_quality_hist_low_res | pipeline.decode_ceiling, metrics | weights, holdout | | | data |
| 6 | fig:aassim_vs_sff | 4.4 | AASSIM_vs_SFF_low_res | decode_ceiling, metrics.filling_factor | weights, holdout | | | data |
| 7 | fig:spectral_fid | 4.5 | spectral_fidelity_low_res | decode_ceiling, pipeline.power_spectrum | weights, holdout | | | data |
| 8 | fig:vae_sff_recovery | 4.6 | recovered_sff_low_res | decode_ceiling, metrics.filling_factor | weights, holdout | | | data |
| 9 | fig:pr_auc_spot_count | 4.6 | pr_auc_spot_count_low_res | decode_ceiling, metrics.pr_auc | weights, holdout, spots.csv | | | data |
| 10 | fig:pr_auc_latlong | 4.6 | pr_auc_latlong_low_res | decode_ceiling, metrics.pr_auc | weights, holdout, spots.csv | | | data |
| 11 | fig:vae_latent_samps | 4.7 | latent_sample_gallery_low_res | vae.decoder, render | weights, aggregate latent | | | data |
| 12 | fig:latent_data_hists | 4.7 | latent_vs_data_hists_low_res | vae.decoder, metrics.filling_factor, total variation, entropy | weights, training surfaces | | | fn |
| 13 | fig:budget | 5.3 | F2_budget | ladder checkpoints, run_holdout per rung | ladder weights (not released) | | | defer |
| 14 | fig:threshold | 6.3 | B3_threshold | metrics.pr_auc sweep | run_holdout output | | | run |
| 15 | fig:cap_observability | 6.7 | G2_cap_spot_observability | operator_analysis.observable_power_fraction, surfaces | design cache | | test_15 | data |
| 16 | fig:anatomy | 7.1 | F15_anatomy | sample_draws, decode_ceiling, forward model | weights, one holdout surface | | | run |
| 17 | fig:families | 7.1 | F6_families | run_holdout output, per-degree correlation, shrinkage | run_holdout | | | run |
| 18 | fig:geometry | 7.2 | F8_geometry | run_holdout output by beta | run_holdout | | | run |
| 19 | fig:harmonics | 7.2 | B5b_harmonics | design_matrix, effective harmonic count | design cache | | | data |
| 20 | fig:operator | 7.2 | B6_operator (WRONG: slot-keyed, rank 333) | operator_analysis rank_by_gap, n_eff | results/operator/ | | test_15 | data, replaces legacy |
| 21 | fig:spectra | 7.3 | F9_spectra | power_spectrum, shrinkage vs prior draws | run_holdout --save-draws | | | run |
| 22 | fig:positions | 7.5 | F11a_positions | spot matching, great-circle error | run_holdout, spots.csv | | | fn |
| 23 | fig:contrast | 7.5 | F11b_contrast | spot matching, contrast | run_holdout, spots.csv | | | fn |
| 24 | fig:beta_diagnostics | 7.6.2 | F13_beta_diagnostics | classify_beta, calibration | beta_clf/paper/ | | | data |
| 25 | fig:beta_bimodality | 7.6.4 | F14_beta_multimodal | classify_beta, mode counting | beta_clf/paper/ | | fn | data |
| 26 | fig:calibration_budget | 7.7 | B8b_calibration_vs_budget | ladder | ladder weights | | | defer |
| 27 | fig:pit | 7.7 | F10c_pit | PIT per pixel type from draws | run_holdout --save-draws | | | fn |
| 28 | fig:nullspace | 7.8 | F12_nullspace (empty in figs_paper) | operator projector, ladder | ladder weights | | | defer |
| 29 | fig:snr_recovery | 7.10 | F7_snr_recovery | sample_draws over noise grid, metrics | weights, holdout subsample | | | fn |
| A1 | fig:skewness | App A | fig_skew_lm | scipy.stats.skew | 55k training surfaces | | | vae |
| A2 | fig:LanczosTapering | App A | fig_LanczosTapering | surfaces, kernels | none | | test_07 | data |

## Tables

| Table | Label | Section | Legacy source | Package function(s) | Input | Notebook | Status |
|---|---|---|---|---|---|---|---|
| 1 | tab:notation | 2 | hand-written | | | | done |
| 2 | tab:beta_clf | 7.6.1 | beta_table.tex from beta_clf/ | classify_beta, entropy, sigma_eq | beta_clf/paper/ | | data; symmetric point (-4,-4), state once |
| A1 | tab:definitions | App C | hand-written | | | | done, may cut |
| A2 | tab:metric_register | App C | metrics doc | | | | done |

## Prose numbers, by section

| Section | Claim | Current value in .tex | FINAL value | Producer | Status |
|---|---|---|---|---|---|
| 4.4 | VAE recon SSIM_aa median | see fig 5 | | decode_ceiling + metrics | data |
| 4.6 | VAE filling factor bias | under 8 per cent | | decode_ceiling + filling_factor | data |
| 4.6 | VAE PR-AUC vs spot count | 0.993 to 0.96 | | decode_ceiling + pr_auc | data |
| 7 top | scope notice | present | delete | | stale |
| 7.1 | SSIM_vis four families | 0.970 etc | 0.8905 / 0.9282 / 0.9246 / 0.9411 | run_holdout | stale |
| 7.1 | RMSE_vis | 0.0158 | 0.0369 / 0.0272 / 0.0283 / 0.0239 | run_holdout | stale |
| 7.1 | PR-AUC | 0.911 | 0.4561 / 0.7193 / 0.6934 / 0.7864 | run_holdout | stale |
| 7.1 | CRPS | | 0.0090 / 0.0067 / 0.0069 / 0.0059 | run_holdout | stale |
| 7.1 | err-unc correlation | | 0.581 / 0.682 / 0.674 / 0.690 | run_holdout | stale |
| 7.2 | rank of A(beta) | 349 to 296 | 96 / 174 / 3 by gap; N_eff 95.8 / 173.2 / 3.0 | operator_analysis | stale; paragraph waits on reframing |
| 7.3 | decoder ceiling numbers | | | decode_ceiling | run |
| 7.4 | achieved vs ceiling info gain | | | posterior_shrinkage.py | fn |
| 7.5 | spot position error median | | 2.91 deg, dlat 1.87, dlon 1.31 | spot matching | fn |
| 7.5 | filling factor true vs recovered | | 0.0395 vs 0.0277 | run_holdout | run |
| 7.5 | relative contrast slope, closes TODO | TODO | 0.782 | F11b middle panel | fn |
| 7.6.1 | sigma_eq widths | 6.4 / 6.1 / 3.9 / 1.2 | 6.40 / 2.90 / 1.87 / 0.87 | beta_table.tex | stale; prose wrong, table right |
| 7.6.1 | Delta nats | | 1.236 / 2.027 / 2.465 / 3.233 | beta_table.tex | stale |
| 7.7 | TARP coverage | | ~3 per cent below nominal | calibration from draws | fn |
| 7.8 | null-space rho | 0.79 | 0.644 at mission point; 0.79 is noiseless top rung, both right | projector + draws | stale |
| 7.10 | PR-AUC vs noise, gain factor | | 0.856 at -6, 0.785 at -3.5; gain 2.02 to 1.10 | noise sweep | fn |

## Missing package capabilities

These are the `fn` rows. Each is a function with a test before its notebook.

| Capability | Serves | Where it lives now | Where it goes |
|---|---|---|---|
| spot matching (assignment, great-circle error, contrast pairing) | figs 22, 23; 7.5 numbers | PaperFigures_v7 F11 cells | metrics.py |
| TARP, SBC, PIT from stored draws | fig 27; 7.7 numbers | PaperFigures_v7 F10 cells | metrics.py or new calibration.py |
| noise sweep driver | fig 29; 7.10 numbers | PaperFigures_v7 F7 cell | scripts/noise_sweep.py |
| bimodality mode count | fig 25 | BetaClassifier notebook | metrics.py |
| total variation, spot entropy | fig 12 | VAE notebooks | metrics.py |
| posterior shrinkage vs prior draws | figs 17, 21; 7.4 | posterior_shrinkage.py (in review) | scripts/ |

## Deferred and why

Figs 13, 26, 28 need the ladder checkpoints, which are not the released
models. They stay as `figs_final/` output for this submission with a note in
`notebooks/paper/README.md`. Figs 2, 3, A1 are VAE-section artefacts whose
source is training-time logging; the notebook reads the saved arrays if they
exist and is otherwise deferred the same way.
