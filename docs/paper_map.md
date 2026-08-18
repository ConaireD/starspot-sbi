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

## The holdout run

`run_holdout.py` reproduces Section 7 and the `run` rows are unblocked. All
four families over the same 5000 pairs are in `results/holdout`, with draws,
from

    python scripts/translate_pairs.py --out /tmp/pairs5000.csv
    python scripts/run_holdout.py --family all --pairs /tmp/pairs5000.csv \
        --holdout ../Sydney/data/fast/L30_N216_grouped/explicit_holdout \
        --draws 256 --save-draws --out results/holdout

The `idx` column of `figdata/full_*.csv` indexes a row of
`nsf_stage3_ladder/cache/holdfull_*.npy`, not a surface. `translate_pairs.py`
recovers the permutation from each cache row's twenty inclinations, uniquely
for all 100000 surfaces, and all four families use the same 5000 pairs.

`compare_holdout.py`, 5000 paired rows per family: inclination agrees on every
row, `sff_true` agrees to the render resolution, and RMSE, CRPS, PR-AUC and
`sff_rec` agree to the third decimal with correlations 0.92 to 0.996. The
reference SSIM medians reproduce the FINAL column below to four decimals.

SSIM is the one systematic difference and it is the render grid. The window is
a fixed number of pixels, so the score depends on the resolution: the notebook
rendered at 60 x 120 and the package renders at 120 x 240, which raises SSIM by
0.019 to 0.044. The 120 x 240 values are the ones quoted. An SSIM number is
comparable only against another at the same grid, which Section 7.1 states.

Two defects were fixed in `run_holdout.py` on the way, both invisible to the
suite as it stood and now tested in `test_14`: `select_channels` was applied in
`run_family` and again inside `sample_draws`, and the filling factors were
unweighted means over the whole sphere.

## Figures

| Fig | Label | Section | Legacy source | Package function(s) | Input | Notebook | Test | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | fig:headline | 1 | F1_headline_phot_axay | pipeline.sample_draws, pipeline.classify_beta, render | weights, one holdout surface | | test_12 | run, input ready |
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
| 14 | fig:threshold | 6.3 | B3_threshold | metrics.pr_auc sweep | results/holdout | | | run, input ready |
| 15 | fig:cap_observability | 6.7 | G2_cap_spot_observability | operator_analysis.observable_power_fraction, surfaces | design cache | | test_15 | data |
| 16 | fig:anatomy | 7.1 | F15_anatomy | sample_draws, decode_ceiling, forward model | weights, one holdout surface | | | run, input ready |
| 17 | fig:families | 7.1 | F6_families | run_holdout output, per-degree correlation, shrinkage | results/holdout | | | run, input ready |
| 18 | fig:geometry | 7.2 | F8_geometry | run_holdout output by beta | results/holdout | | | run, input ready |
| 19 | fig:harmonics | 7.2 | B5b_harmonics | design_matrix, effective harmonic count | design cache | | | data |
| 20 | fig:operator | 7.2 | B6_operator (WRONG: slot-keyed, rank 333) | operator_analysis rank_by_gap, n_eff | results/operator/ | | test_15 | data, replaces legacy |
| 21 | fig:spectra | 7.3 | F9_spectra | power_spectrum, shrinkage vs prior draws | results/holdout draws | | | run, input ready |
| 22 | fig:positions | 7.5 | F11a_positions | spot matching, great-circle error | results/holdout, spots.csv | | | fn |
| 23 | fig:contrast | 7.5 | F11b_contrast | spot matching, contrast | results/holdout, spots.csv | | | fn |
| 24 | fig:beta_diagnostics | 7.6.2 | F13_beta_diagnostics | classify_beta, calibration | beta_clf/paper/ | | | data |
| 25 | fig:beta_bimodality | 7.6.4 | F14_beta_multimodal | classify_beta, mode counting | beta_clf/paper/ | | fn | data |
| 26 | fig:calibration_budget | 7.7 | B8b_calibration_vs_budget | ladder | ladder weights | | | defer |
| 27 | fig:pit | 7.7 | F10c_pit | PIT per pixel type from draws | results/holdout draws | | | fn |
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
| A2 | tab:metric_register | App C | metrics doc | | | | done, add the render grid |

## Prose numbers, by section

Section 7 values are the 5000-surface medians from `results/holdout`, in the
order phot / phot_ax / phot_ay / phot_axay. SSIM is at 120 x 240; the FINAL
column of the previous revision was at 60 x 120 and is superseded.

| Section | Claim | Current value in .tex | New value | Producer | Status |
|---|---|---|---|---|---|
| 4.4 | VAE recon SSIM_aa median | see fig 5 | | decode_ceiling + metrics | data |
| 4.6 | VAE filling factor bias | under 8 per cent | | decode_ceiling + filling_factor | data |
| 4.6 | VAE PR-AUC vs spot count | 0.993 to 0.96 | | decode_ceiling + pr_auc | data |
| 7 top | scope notice | present | delete | | stale |
| 7.1 | SSIM_vis four families | 0.970 etc | 0.9345 / 0.9552 / 0.9524 / 0.9626 | run_holdout | stale, values ready |
| 7.1 | RMSE_vis | 0.0158 | 0.0369 / 0.0273 / 0.0282 / 0.0238 | run_holdout | stale, values ready |
| 7.1 | PR-AUC | 0.911 | 0.4585 / 0.7246 / 0.6959 / 0.7920 | run_holdout | stale, values ready |
| 7.1 | CRPS | | 0.0090 / 0.0067 / 0.0070 / 0.0060 | run_holdout | stale, values ready |
| 7.1 | err-unc correlation | | 0.5938 / 0.6931 / 0.6851 / 0.7018 | run_holdout | stale, values ready |
| 7.1 | render grid of every SSIM | absent | state 120 x 240 once | | new sentence |
| 7.2 | rank of A(beta) | 349 to 296 | 96 / 174 / 3 by gap; N_eff 95.8 / 173.2 / 3.0 | operator_analysis | stale; paragraph waits on reframing |
| 7.3 | decoder ceiling numbers | | | decode_ceiling | run, input ready |
| 7.4 | achieved vs ceiling info gain | | | posterior_shrinkage.py | fn |
| 7.5 | spot position error median | | 2.91 deg, dlat 1.87, dlon 1.31 | spot matching | fn, needs recomputing |
| 7.5 | filling factor true vs recovered | | 0.0395 vs 0.0306 | run_holdout | values ready |
| 7.5 | relative contrast slope, closes TODO | TODO | 0.782 | F11b middle panel | fn |
| 7.6.1 | sigma_eq widths | 6.4 / 6.1 / 3.9 / 1.2 | 6.40 / 2.90 / 1.87 / 0.87 | beta_table.tex | stale; prose wrong, table right |
| 7.6.1 | Delta nats | | 1.236 / 2.027 / 2.465 / 3.233 | beta_table.tex | stale |
| 7.7 | TARP coverage | | ~3 per cent below nominal | calibration from draws | fn |
| 7.8 | null-space rho | 0.79 | 0.644 at mission point; 0.79 is noiseless top rung, both right | projector + draws | stale |
| 7.10 | PR-AUC vs noise, gain factor | | 0.856 at -6, 0.785 at -3.5; gain 2.02 to 1.10 | noise sweep | fn, needs recomputing |

The 7.5 spot-position and 7.10 noise-sweep values predate the channel-order and
filling-factor fixes and were produced at 60 x 120. Both need recomputing from
`results/holdout` when their functions land.

## Missing package capabilities

These are the `fn` rows. Each is a function with a test before its notebook.

| Capability | Serves | Where it lives now | Where it goes |
|---|---|---|---|
| spot matching (assignment, great-circle error, contrast pairing) | figs 22, 23; 7.5 numbers | PaperFigures_v7 F11 cells | metrics.py |
| TARP, SBC, PIT from stored draws | fig 27; 7.7 numbers | PaperFigures_v7 F10 cells | metrics.py or new calibration.py |
| noise sweep driver | fig 29; 7.10 numbers | PaperFigures_v7 F7 cell | scripts/noise_sweep.py |
| bimodality mode count | fig 25 | BetaClassifier notebook | metrics.py |
| total variation, spot entropy | fig 12 | VAE notebooks | metrics.py |
| posterior shrinkage vs prior draws | figs 17, 21; 7.4 | scripts/posterior_shrinkage.py, committed with test_16 | done |

## Deferred and why

Figs 13, 26, 28 need the ladder checkpoints, which are not the released
models. They stay as `figs_final/` output for this submission with a note in
`notebooks/paper/README.md`. Figs 2, 3, A1 are VAE-section artefacts whose
source is training-time logging; the notebook reads the saved arrays if they
exist and is otherwise deferred the same way.