# Model checkpoints

- NOT tracked by git (~131 MB); .gitignore keeps this README only
- Obtain: [TBD — GitHub Release / Zenodo / on request]

## Expected filenames

- `vae_n640000_seed101_temp.pt` — the frozen VAE, 96-d latent, L = 30
- `flow_{phot,phot_ax,phot_ay,phot_axay}_temp.pt` — noise-conditioned NSF, one per family
- `clf_{phot,phot_ax,phot_ay,phot_axay}_temp.pt` — 91-way beta classifier, one per family

## The _temp suffix

- marks these as pending verification, pre-dating the positivity filter and retrain
- to be dropped at release; `SUFFIX` in tests/test_10_models.py is the single edit

## Notes

- flows carry `ch`, `n_aux`, gains and log-sigma ranges; loaders return them
- classifiers carry `arch` and `temperature`; temperature is applied outside the
  network, so a caller that discards meta loses the calibration
- VAE carries `mu_data`, `std_data`, `dc_value`, `config`
- architecture parameters for the flows are NOT stored, they come from
  models.FLOW / models.EMB
- context lengths: flows 218/435/435/651, classifiers 217/434/434/650
- load them with starspot_sbi.models.{load_vae, load_flow, load_classifier}