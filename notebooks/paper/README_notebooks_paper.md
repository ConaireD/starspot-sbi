# Paper figures

- one notebook per figure, named fig{NN}_{shortname}.ipynb
- shared loaders, paths and plot defaults in `_setup.py`
- notebooks import from _setup, define nothing themselves

## Requirements

- weights/ populated (see weights/README.md)
- the explicit holdout dataset (scripts/generate_dataset.py, or on request)
- neither is in git, so these do not run on a bare clone

## Pinning

- fixed seeds, fixed checkpoints, fixed holdout
- a figure that does not reproduce means something changed; find out what
- record in _setup.py: mission point (-4.0, -3.5), 256 draws, holdout size

## Figure map

| Figure | Notebook | Section | Needs |
|---|---|---|---|
| 1 | fig01_headline.ipynb | 1 | weights, holdout |
| 2 | fig02_kurtosis.ipynb | 3.2 | training surfaces |
| ... | | | |

## Status

- TO BE POPULATED after the Section 7 regeneration with the corrected
  visibility mask; current PaperFigures_v7 numbers predate it