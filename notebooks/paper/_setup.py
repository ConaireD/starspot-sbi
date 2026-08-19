"""
Shared state for the paper figure notebooks.

Every notebook in this directory begins with

    from _setup import *

and defines nothing itself. What lives here: paths, the pinned run settings,
loaders for the holdout, the weights, the run output and the operator analysis,
and the plot defaults. No figure logic.

The paths outside the repository are the holdout dataset and the figure output
directory. Both are overridable through the environment, STARSPOT_HOLDOUT and
STARSPOT_FIGS, so a notebook runs on a machine where the dataset sits
elsewhere.

Run `describe()` in a fresh kernel to see what is present and what is missing.
"""

###########
# Imports #
###########

# python
import os
import sys
import json
from functools import lru_cache

# standard
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# self
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from starspot_sbi.indexing import coeffs_to_real, real_to_coeffs   # noqa: E402
from starspot_sbi.render import render, N_THETA, N_PHI             # noqa: E402
from starspot_sbi.models import (FAMILIES, load_vae, load_flow,    # noqa: E402
                                 load_classifier)
from starspot_sbi.pipeline import (sample_draws, posterior_mean,   # noqa: E402
                                   render_draws, decode_ceiling,
                                   classify_beta, power_spectrum,
                                   select_channels)
from starspot_sbi import metrics                                   # noqa: E402

from run_holdout import read_pair, load_holdout_index              # noqa: E402


#############
# Constants #
#############

# The pinned run. These reproduce results/holdout; changing one means the
# figures no longer match the .tex.
LOG_SIGMA_PHOT = -4.0
LOG_SIGMA_ASTRO = -3.5
N_DRAWS = 256
N_HOLDOUT = 5000
SEED = 0
VAE_TAG = 'n640000_seed101'
WEIGHTS_SUFFIX = '_temp'
LATENT_DIM = 96
T_OBS = 216
L_MAX = 30

# The render grid. SSIM is computed on a fixed pixel window, so its value
# depends on this: the legacy notebooks rendered at 60 x 120 and score about
# 0.02 to 0.04 lower. Quote the grid with any SSIM number.
GRID = (N_THETA, N_PHI)

FAMILY_ORDER = ('phot', 'phot_ax', 'phot_ay', 'phot_axay')
FAMILY_LABELS = {'phot': 'photometry',
                 'phot_ax': 'photometry + x',
                 'phot_ay': 'photometry + y',
                 'phot_axay': 'photometry + x + y'}
FAMILY_COLOURS = {'phot': '#4c4c4c', 'phot_ax': '#3b7ea1',
                  'phot_ay': '#c1666b', 'phot_axay': '#2a9d5c'}

#########
# Paths #
#########

WEIGHTS = os.path.join(REPO, 'weights')
CACHES = os.path.join(REPO, 'caches')
RUN = os.path.join(REPO, 'results', 'holdout')
OPERATOR = os.path.join(REPO, 'results', 'operator')
SHRINKAGE = os.path.join(REPO, 'results', 'shrinkage')

HOLDOUT = os.environ.get(
    'STARSPOT_HOLDOUT',
    os.path.join(REPO, '..', 'Sydney', 'data', 'fast',
                 'L30_N216_grouped', 'explicit_holdout'))
BETA_CLF = os.environ.get(
    'STARSPOT_BETA_CLF',
    os.path.join(REPO, '..', 'Sydney', 'FromScratch', 'beta_clf', 'paper'))

FIGS = os.environ.get('STARSPOT_FIGS',
                      os.path.join(os.path.dirname(__file__), 'figs'))
os.makedirs(FIGS, exist_ok=True)


def run_csv(family):
    """The combined per-surface metrics for one family."""
    return os.path.join(
        RUN, f'full_{family}_lp{LOG_SIGMA_PHOT}_la{LOG_SIGMA_ASTRO}.csv')


###########
# Loaders #
###########

@lru_cache(maxsize=8)
def load_metrics(family='phot_axay'):
    """Per-surface metrics for one family, 5000 rows, indexed by position."""
    return pd.read_csv(run_csv(family))


@lru_cache(maxsize=1)
def load_all_metrics():
    """The four families in one frame, with a family column."""
    return pd.concat([load_metrics(f).assign(family=f) for f in FAMILY_ORDER],
                     ignore_index=True)


@lru_cache(maxsize=1)
def load_meta():
    """The holdout metadata, deduplicated and indexed by surface_idx."""
    m = pd.read_csv(os.path.join(HOLDOUT, 'metadata.csv'))
    return m.drop_duplicates('surface_idx', keep='first').set_index('surface_idx')


@lru_cache(maxsize=1)
def load_spots():
    """The spot catalogue, for the position and contrast figures."""
    return pd.read_csv(os.path.join(HOLDOUT, 'spots.csv'))


def load_truth(surface_idx, slot=0):
    """
    Stored coefficients, signal, inclination and spot count for one pair.

    Longitudes in spots.csv are offset by pi from the stored surfaces, a known
    consequence of the place_spot fix; see PORT_NOTES.md before comparing them.
    """
    return read_pair(HOLDOUT, load_meta(), int(surface_idx), int(slot))


@lru_cache(maxsize=8)
def _draw_index(family):
    """(idx, slot) to (chunk name, row), built once per family."""
    fam_dir = os.path.join(RUN, family)
    index, chunks = {}, []
    for name in sorted(f for f in os.listdir(fam_dir)
                       if f.startswith('chunk_') and f.endswith('.csv')):
        rows = pd.read_csv(os.path.join(fam_dir, name))
        for k, r in enumerate(rows.itertuples()):
            index[(int(r.idx), int(r.slot))] = (name, k)
        chunks.append(name)
    return index, tuple(chunks)


def load_draws(family, surface_idx, slot):
    """
    Posterior coefficient draws for one pair, shape (N_DRAWS, 961),
    real-packed and un-standardised.
    """
    index, _ = _draw_index(family)
    key = (int(surface_idx), int(slot))
    if key not in index:
        raise KeyError(f'{key} is not in the {family} run')
    name, row = index[key]
    path = os.path.join(RUN, family, 'draws', name.replace('.csv', '.npy'))
    return np.load(path, mmap_mode='r')[row]


def iter_draws(family, n_max=None):
    """
    Yield (surface_idx, slot, beta, draws) over the run, one chunk in memory at
    a time. 5000 surfaces at 256 draws is 5 GB per family, so a figure that
    needs all of them reduces as it goes rather than accumulating.
    """
    fam_dir = os.path.join(RUN, family)
    _, chunks = _draw_index(family)
    seen = 0
    for name in chunks:
        rows = pd.read_csv(os.path.join(fam_dir, name))
        block = np.load(os.path.join(fam_dir, 'draws',
                                     name.replace('.csv', '.npy')),
                        mmap_mode='r')
        for k, r in enumerate(rows.itertuples()):
            if n_max is not None and seen >= n_max:
                return
            seen += 1
            yield int(r.idx), int(r.slot), float(r.beta), np.asarray(block[k])


@lru_cache(maxsize=1)
def load_run_manifest():
    with open(os.path.join(RUN, 'manifest.json')) as f:
        return json.load(f)


@lru_cache(maxsize=2)
def load_operator():
    """Rank, N_eff and projector tables from operator_analysis.py."""
    return {name: pd.read_csv(os.path.join(OPERATOR, f'{name}.csv'))
            for name in ('rank_vs_beta', 'neff_vs_beta', 'projector_vs_beta')}


def load_per_degree(beta_deg):
    """The observable fraction per degree at one inclination."""
    return pd.read_csv(os.path.join(OPERATOR, f'per_degree_beta{beta_deg:03d}.csv'))


###########
# Weights #
###########

@lru_cache(maxsize=2)
def load_autoencoder(device='cpu'):
    """(vae, stats) for the canonical checkpoint."""
    vae, _, stats = load_vae(
        os.path.join(WEIGHTS, f'vae_{VAE_TAG}{WEIGHTS_SUFFIX}.pt'),
        device=device)
    return vae, stats


@lru_cache(maxsize=8)
def load_family_flow(family, device='cpu'):
    """(estimator, meta) for one family."""
    return load_flow(
        os.path.join(WEIGHTS, f'flow_{family}{WEIGHTS_SUFFIX}.pt'),
        latent_dim=LATENT_DIM, device=device)


@lru_cache(maxsize=8)
def load_family_classifier(family, device='cpu'):
    """(classifier, meta) for the inclination posterior."""
    return load_classifier(
        os.path.join(WEIGHTS, f'clf_{family}{WEIGHTS_SUFFIX}.pt'),
        T=T_OBS, device=device)


def draw_settings(**over):
    """
    The pinned sampling arguments, for a notebook that samples rather than
    reading the run. Pass overrides to vary one thing deliberately.
    """
    out = {'log10_sigma_phot': LOG_SIGMA_PHOT,
           'log10_sigma_astro': LOG_SIGMA_ASTRO,
           'n_draws': N_DRAWS, 'seed': SEED}
    out.update(over)
    return out


#################
# Plot defaults #
#################

# MNRAS single column is 240 pt and the text width is 504 pt.
COL = 240 / 72.27
WIDE = 504 / 72.27

mpl.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.0,
    'legend.frameon': False,
    'figure.constrained_layout.use': True,
})

SURFACE_CMAP = 'magma'


def save_fig(fig, name, formats=('pdf', 'png')):
    """Write a figure to FIGS under both formats and report the paths."""
    out = []
    for ext in formats:
        p = os.path.join(FIGS, f'{name}.{ext}')
        fig.savefig(p)
        out.append(p)
    print('wrote ' + ', '.join(os.path.relpath(p, REPO) for p in out))
    return out


#############
# Inventory #
#############

def describe():
    """What is present, what is missing, and what the run was."""
    print(f'repository {REPO}')
    for label, path in (('holdout', HOLDOUT), ('weights', WEIGHTS),
                        ('run', RUN), ('operator', OPERATOR),
                        ('shrinkage', SHRINKAGE), ('beta_clf', BETA_CLF),
                        ('figures', FIGS)):
        mark = 'present' if os.path.exists(path) else 'MISSING'
        print(f'  {label:<10} {mark:<8} {path}')

    print(f'\nrender grid {GRID[0]} x {GRID[1]}, noise point '
          f'({LOG_SIGMA_PHOT}, {LOG_SIGMA_ASTRO}), {N_DRAWS} draws')
    try:
        man = load_run_manifest()
        print(f'run: {man["n_pairs"]} pairs, families '
              f'{", ".join(man["families"])}, draws saved '
              f'{man["saved_draws"]}, pairs from {man["pairs_source"]}')
        if (man['log10_sigma_phot'], man['log10_sigma_astro']) != \
                (LOG_SIGMA_PHOT, LOG_SIGMA_ASTRO):
            print('  the run noise point differs from the constants above')
        if man['draws'] != N_DRAWS:
            print(f'  the run used {man["draws"]} draws, not {N_DRAWS}')
    except FileNotFoundError:
        print('run: no manifest; see docs/paper_map.md for the command')

    for f in FAMILY_ORDER:
        if os.path.exists(run_csv(f)):
            d = load_metrics(f)
            print(f'  {f:<10} {len(d):>5} rows, SSIM_vis median '
                  f'{d["ssim_vis"].median():.4f}')
        else:
            print(f'  {f:<10} absent')