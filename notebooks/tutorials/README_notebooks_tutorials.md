# Tutorials

- how the package works, not how the paper's numbers were made
- small L, few surfaces, no weights needed except 02
- readable rather than pinned; expected to track the package as it changes

- `00_forward_model.ipynb` — the derivation, with each step checked against the code
- `01_surfaces_and_signals.ipynb` — build a spotted star, generate its signal
- `02_inference.ipynb` — load the models, reconstruct one surface (needs weights)
- `03_metrics.ipynb` — weighting, SSIM, detection

## Rules

- no def or class in a notebook; import from the package
- notebooks import, run, plot, explain