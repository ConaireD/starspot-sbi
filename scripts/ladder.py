"""
The noiseless training-budget ladder, for Figure 13.

The ladder is legacy: seventeen sqrt(2)-spaced rungs from 5,000 to 1,280,000
training surfaces, trained by NSF_Stage3_NtrainLadder_Faster.ipynb and left in
Sydney/FromScratch/nsf_stage3_ladder. Its checkpoints are bare state dicts for
an architecture that predates starspot_sbi.models, and its flows are noiseless,
so neither load_flow nor the noise-conditioned pipeline can read them. This
module is the one place that knows how to.

Everything here reproduces the training notebook's own conventions exactly.
`nll_on(load_rung('n640000'), *holdout_rows())` returns 85.0039, matching
n640000/done.json to four decimals, which is the check that the numbers below
sit on the same footing as the recorded val and holdout columns.

The rung records carry validation and holdout NLL but no train-subset NLL: it
was never cached, which is why the legacy figure's memorisation panel is empty.
`train_rows` rebuilds it, using the same pool split each rung trained on.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# The ladder's own constants, copied from the training notebook. They are not
# recorded in the checkpoints, so they cannot be recovered from them.
LADDER = os.environ.get(
    'STARSPOT_LADDER',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                 'Sydney', 'FromScratch', 'nsf_stage3_ladder'))
LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                      'Sydney', 'FromScratch')
CACHE = os.path.join(LADDER, 'cache')

SEED = 42
T = 216
N_INC = 20
LATENT_DIM = 96
N_VAL_SURF = 2000
N_NLL_EVAL = 10000
EMB = dict(embedding_dim=128, patch_size=16, d_model=64, n_heads=4,
           n_attn_layers=2)
FLOW = dict(hidden_features=128, num_transforms=20, num_bins=8)


def rungs():
    """The rung records, one row per rung, ascending in training surfaces."""
    rows = []
    for name in sorted(os.listdir(LADDER)):
        p = os.path.join(LADDER, name, 'done.json')
        if os.path.exists(p):
            with open(p) as f:
                rows.append({**json.load(f), 'tag': name})
    return pd.DataFrame(rows).sort_values('n_surf').reset_index(drop=True)


def _cache(name, mmap=True):
    return np.load(os.path.join(CACHE, f'{name}.npy'),
                   mmap_mode='r' if mmap else None)


def zscore_stats():
    """The single latent standardisation every rung was trained against."""
    d = torch.load(os.path.join(CACHE, 'npe_zstats.pt'), weights_only=False)
    return d['z_mu'], d['z_std']


def train_surfaces():
    """
    The pool split each rung drew from: rung n trained on the first n rows.

    The pool is every spotted surface plus fifty immaculate ones, shuffled at
    SEED + 7; the first N_VAL_SURF go to validation and the rest to training.
    """
    spot = np.asarray(_cache('pool_spot'))
    keep = np.sort(np.concatenate([np.where(spot)[0], np.where(~spot)[0][:50]]))
    keep = keep[np.random.default_rng(SEED + 7).permutation(len(keep))]
    return np.sort(keep[N_VAL_SURF:]), np.sort(keep[:N_VAL_SURF])


def _rows(sig, beta, mu, sd, idx, slot, eps_seed, z_mu, z_std):
    """(Z, X) for one set of (surface, slot) pairs, in the training layout."""
    s = torch.from_numpy(
        np.ascontiguousarray(sig[idx, slot])).reshape(len(idx), -1)
    b = torch.from_numpy(np.asarray(beta)[idx, slot]).unsqueeze(1)
    eps = torch.randn((len(idx), LATENT_DIM),
                      generator=torch.Generator().manual_seed(eps_seed))
    z = (torch.from_numpy(np.asarray(mu[idx]))
         + torch.from_numpy(np.asarray(sd[idx])) * eps)
    return (z - z_mu) / z_std, torch.cat([s, b], 1)


def holdout_rows():
    """
    The explicit-holdout evaluation rows, exactly as the ladder scored them.

    N_NLL_EVAL spotted surfaces at one inclination each. Reproduces the `test`
    column of every rung's done.json.
    """
    z_mu, z_std = zscore_stats()
    spot = np.asarray(_cache('hold_spot'))
    hs = np.where(spot)[0]
    idx = np.sort(np.random.default_rng(SEED + 99).choice(
        hs, min(N_NLL_EVAL, len(hs)), replace=False))
    slot = np.random.default_rng(SEED + 100).integers(0, N_INC, size=len(idx))
    return _rows(_cache('hold_sig'), _cache('hold_beta'), _cache('hold_mu'),
                 _cache('hold_sd'), idx, slot, SEED + 1, z_mu, z_std)


def train_rows(n_surf, n_eval=N_NLL_EVAL, seed=SEED + 4242):
    """
    Evaluation rows drawn from the surfaces this rung actually trained on.

    One inclination per surface, matching the holdout convention, so the two
    NLLs differ only in which surfaces they cover. Rungs below n_eval surfaces
    are scored on all of theirs, so the small-budget points average over fewer
    rows and carry a correspondingly larger standard error.
    """
    z_mu, z_std = zscore_stats()
    train, _ = train_surfaces()
    surf = train[:int(n_surf)]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(surf, min(n_eval, len(surf)), replace=False))
    slot = rng.integers(0, N_INC, size=len(idx))
    return _rows(_cache('pool_sig'), _cache('pool_beta'), _cache('pool_mu'),
                 _cache('pool_sd'), idx, slot, seed + 1, z_mu, z_std)


def load_rung(tag, device='cpu'):
    """One rung's flow, from its bare state dict."""
    if LEGACY not in sys.path:
        sys.path.insert(0, LEGACY)
    import reconstruction_utils as ru
    from sbi.neural_nets import posterior_nn

    net = ru.AttnSignalEmbeddingConditional(T=T, n_channels=3, dropout=0.0,
                                            beta_dim=8, **EMB)
    est = posterior_nn(model='nsf', embedding_net=net, **FLOW)(
        torch.randn(2, LATENT_DIM), torch.randn(2, 3 * T + 1))
    est.load_state_dict(torch.load(os.path.join(LADDER, tag, 'best_model.pt'),
                                   map_location='cpu', weights_only=False),
                        strict=True)
    est.to(device).eval()
    for p in est.parameters():
        p.requires_grad_(False)
    return est


@torch.no_grad()
def nll_on(est, Z, X, batch=4096, device='cpu'):
    """Mean negative log-likelihood in nats, and its standard error."""
    out = []
    for a in range(0, len(Z), batch):
        out.append((-est.log_prob(Z[a:a + batch].to(device),
                                  condition=X[a:a + batch].to(device))
                    ).reshape(-1).cpu())
    v = torch.cat(out).double().numpy()
    v = v[np.isfinite(v)]
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


SSIM_CSV = os.path.join(LEGACY, 'figdata', 'ladder_ssim.csv')


def ladder_ssim():
    """
    Per-surface SSIM_vis through the full pipeline, one row per (rung, surface).

    From the legacy PaperFigures pass: 400 holdout surfaces, one inclination
    each, 512 posterior draws decoded and averaged, scored against the truth
    under visibility weighting at the rung's own inclination. The same 400
    surfaces are used at every rung, so the curve is paired.

    This is SSIM_vis, not the area-adjusted SSIM^aa_vis: they are different
    numbers on the same reconstructions and Section 7.7 quotes the latter.
    """
    d = pd.read_csv(SSIM_CSV)
    sets = d.groupby('tag')['idx'].apply(frozenset)
    if len(set(sets)) != 1:
        raise ValueError('the rungs do not share one surface sample; the '
                         'paired claim in the caption would be false')
    return d
