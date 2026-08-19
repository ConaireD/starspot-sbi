"""
Main script for inference

The process:
    stored signal (astro_x, astro_y, phot)
      -> add noise
      -> self-normalise
      -> apply the gains
      -> append the auxiliaries (i.e. noise levels)
      -> sample the flow
      -> decode every draw
      -> un-standardise

which leaves per-surface coefficient draws of shape (n_draws, (L+1)^2).
Everything one cares about can be derived from these.

Sampling goes through the density estimator rather than the sbi DirectPosterior
wrapper, since the wrapper takes one observation at a time while the estimator
accepts a batch of conditioning vectors.

Input channel order is stored order by default, matching the raw signals/*.npy
files. The training and evaluation caches of the legacy tree, pool_sig.npy and
holdfull_sig.npy, hold model order instead, so a caller reading those passes
stored=False.
"""

###########
# Imports #
###########

# python
from functools import lru_cache

# standard
import numpy as np

# machine learning
import torch

# self
from starspot_sbi.indexing import lm_indices, coeffs_to_real, real_to_coeffs
from starspot_sbi.models   import (FAMILIES, LOG10_SIGMA_PHOT, LOG10_SIGMA_ASTRO,
                                   beta_norm_from_deg, make_context,
                                   clf_context, clf_predict)
from starspot_sbi.render   import N_THETA, N_PHI, build_Ylm_matrix

#############
# Constants #
#############

# Stored .npy files hold (astro_x, astro_y, phot); the models index channels as
# (phot, astro_x, astro_y). This is the one place the permutation is applied.
STORED_TO_MODEL = [2, 0, 1]

# Mission reference point, docs/conventions.md section 12.
LOG10_SIGMA_PHOT_MISSION = -4.0
LOG10_SIGMA_ASTRO_MISSION = -3.5

#####################
# Channel Selection #
#####################

def select_channels(signals, family):
    """
    Take (..., 3, T) in stored order and return (..., n_ch, T) in the order the
    family's flow expects.
    """
    rows = [STORED_TO_MODEL[c] for c in FAMILIES[family]]
    return np.asarray(signals)[..., rows, :]


def to_model_order(signals, family, stored=True):
    """
    Return (..., n_ch, T) in the order the family's flow expects.

    A three-channel array is ambiguous for phot_axay, since the stored order and
    the model order have the same shape. The stored argument resolves it and
    defaults to the stored order, which is what the raw signal files hold.
    """
    signals = np.asarray(signals)
    n_ch = len(FAMILIES[family])
    if signals.shape[-2] == 3 and (stored or n_ch != 3):
        return select_channels(signals, family)
    if signals.shape[-2] != n_ch:
        raise ValueError(f"{family} expects {n_ch} channels, "
                         f"got {signals.shape[-2]}")
    return signals


def _context_ranges(meta):
    """
    The training ranges the noise levels are normalised against, taken from the
    checkpoint where it records them.
    """
    return {'log10_sigma_phot': tuple(meta.get('log10_sigma_phot',
                                               LOG10_SIGMA_PHOT)),
            'log10_sigma_astro': tuple(meta.get('log10_sigma_astro',
                                                LOG10_SIGMA_ASTRO))}


def _reinstate_dc(vecs, stats):
    """Prepend the fixed DC coefficient when the auto-encoder omits it."""
    if stats['include_dc']:
        return np.asarray(vecs, dtype=np.float32)
    if stats['dc_value'] is None:
        raise ValueError('dc_value is None for a checkpoint trained without '
                         'the DC coefficient')
    vecs = np.asarray(vecs)
    full = np.empty(vecs.shape[:-1] + (vecs.shape[-1] + 1,), dtype=np.float32)
    full[..., 0] = stats['dc_value']
    full[..., 1:] = vecs
    return full


def _strip_dc(vecs, stats):
    """Drop the DC coefficient when the auto-encoder was trained without it."""
    return np.asarray(vecs) if stats['include_dc'] else np.asarray(vecs)[..., 1:]


#####################
# Rendering         #
#####################

@lru_cache(maxsize=2)
def _basis(L, n_theta, n_phi, device):
    """Render basis on the device, shape (n_coeffs, n_grid), complex64."""
    Y = build_Ylm_matrix(L, n_theta, n_phi)          # (n_grid, n_coeffs)
    return torch.as_tensor(np.ascontiguousarray(Y.T), dtype=torch.complex64,
                           device=device)

def render_draws(coeffs_real, n_theta=N_THETA, n_phi=N_PHI, device='cpu',
                 batch=512):
    """
    Render a stack of real-packed coefficient vectors, shape (n, (L+1)^2), to
    images of shape (n, n_theta, n_phi).
    """
    x = np.asarray(coeffs_real)
    if x.ndim == 1:
        x = x[None]
    L = int(round(np.sqrt(x.shape[-1]))) - 1
    if (L + 1) ** 2 != x.shape[-1]:
        raise ValueError(f"{x.shape[-1]} coefficients is not a square")
    Yt = _basis(L, n_theta, n_phi, device)

    out = []
    for lo in range(0, x.shape[0], batch):
        block = np.stack([real_to_coeffs(v) for v in x[lo:lo + batch]])
        c = torch.as_tensor(block, dtype=torch.complex64, device=device)
        out.append((c @ Yt).real.reshape(-1, n_theta, n_phi).cpu().numpy())
    return np.concatenate(out, axis=0)


def power_spectrum(coeffs_real, L=None):
    """
    C_l = sum_m |s_l^m|^2 per degree, for a stack of real-packed vectors, shape
    (n, L+1).
    """
    x = np.asarray(coeffs_real)
    if x.ndim == 1:
        x = x[None]
    if L is None:
        L = int(round(np.sqrt(x.shape[-1]))) - 1
    ells = np.array([l for l, m in lm_indices(L)])
    out = np.empty((x.shape[0], L + 1))
    for i, v in enumerate(x):
        out[i] = np.bincount(ells, weights=np.abs(real_to_coeffs(v)) ** 2,
                             minlength=L + 1)
    return out

#####################
# Sampling          #
#####################

def sample_draws(signals, betas_deg, family, vae, stats, est, meta,
                 log10_sigma_phot=LOG10_SIGMA_PHOT_MISSION,
                 log10_sigma_astro=LOG10_SIGMA_ASTRO_MISSION,
                 n_draws=256, seed=0, batch_size=64, device='cpu',
                 stored=True):
    """
    Posterior coefficient draws, shape (B, n_draws, (L+1)^2), real-packed and
    un-standardised, float32.

    signals    (B, 3, T) in stored order, or (B, n_ch, T) with stored=False
    betas_deg  (B,) in degrees
    seed       offset by the row position within this call, so the draws for a
               block of rows do not depend on how many rows precede or follow
               them in the same call. A resumed run reproduces an uninterrupted
               one when the caller passes seed offset by the number of rows
               already done and leaves batch_size unchanged. Changing batch_size
               moves the chunk boundaries and changes every draw.
    """
    signals = to_model_order(signals, family, stored=stored)
    betas_deg = np.asarray(betas_deg, dtype=float)
    n = signals.shape[0]
    if betas_deg.shape[0] != n:
        raise ValueError(f"{n} signals against {betas_deg.shape[0]} inclinations")
    ranges = _context_ranges(meta)

    out = []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        b = hi - lo

        gen = torch.Generator().manual_seed(seed + lo)
        ctx = make_context(
            torch.tensor(signals[lo:hi], dtype=torch.float32),
            torch.tensor(beta_norm_from_deg(betas_deg[lo:hi]), dtype=torch.float32),
            torch.full((b,), float(log10_sigma_phot)),
            torch.full((b,), float(log10_sigma_astro)),
            meta['ch'], gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'],
            gen=gen, **ranges).to(device)

        torch.manual_seed(seed + lo)
        with torch.no_grad():
            # sbi returns (*sample_shape, condition_batch, latent_dim).
            draws = est.sample((n_draws,), ctx)
            if tuple(draws.shape[:2]) != (n_draws, b):
                raise ValueError(
                    f"estimator returned {tuple(draws.shape)}, expected "
                    f"({n_draws}, {b}, latent_dim)")
            vecs = vae.decoder(draws.reshape(-1, draws.shape[-1]))
            vecs = vecs.reshape(n_draws, b, -1).permute(1, 0, 2)

        raw = vecs.cpu().numpy() * stats['std_data'] + stats['mu_data']
        out.append(_reinstate_dc(raw, stats))

    return np.concatenate(out, axis=0)

def sample_latents(signals, betas_deg, family, est, meta,
                   log10_sigma_phot=LOG10_SIGMA_PHOT_MISSION,
                   log10_sigma_astro=LOG10_SIGMA_ASTRO_MISSION,
                   n_draws=256, seed=0, batch_size=64, device='cpu',
                   stored=True):
    """
    Posterior latent draws, shape (B, n_draws, latent_dim), float32.

    The draws sample_draws decodes, before the decoder: the noise generator
    and the sampling seed follow the same per-batch construction, so with the
    same arguments decode_latents of these reproduces sample_draws' output.
    Calibration in latent space (SBC, TARP) needs these, since the saved runs
    keep only the decoded coefficients.
    """
    signals = to_model_order(signals, family, stored=stored)
    betas_deg = np.asarray(betas_deg, dtype=float)
    n = signals.shape[0]
    if betas_deg.shape[0] != n:
        raise ValueError(f"{n} signals against {betas_deg.shape[0]} inclinations")
    ranges = _context_ranges(meta)

    out = []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        b = hi - lo

        gen = torch.Generator().manual_seed(seed + lo)
        ctx = make_context(
            torch.tensor(signals[lo:hi], dtype=torch.float32),
            torch.tensor(beta_norm_from_deg(betas_deg[lo:hi]), dtype=torch.float32),
            torch.full((b,), float(log10_sigma_phot)),
            torch.full((b,), float(log10_sigma_astro)),
            meta['ch'], gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'],
            gen=gen, **ranges).to(device)

        torch.manual_seed(seed + lo)
        with torch.no_grad():
            draws = est.sample((n_draws,), ctx)
            if tuple(draws.shape[:2]) != (n_draws, b):
                raise ValueError(
                    f"estimator returned {tuple(draws.shape)}, expected "
                    f"({n_draws}, {b}, latent_dim)")
        out.append(draws.permute(1, 0, 2).float().cpu().numpy())

    return np.concatenate(out, axis=0)


def posterior_mean(draws):
    """Mean over the draw axis, (B, n_draws, n_coeffs) to (B, n_coeffs)."""
    return np.asarray(draws).mean(axis=1)

def reconstruct(signals, betas_deg, family, vae, stats, est, meta,
                n_theta=N_THETA, n_phi=N_PHI, **kwargs):
    """
    Rendered posterior means, shape (B, n_theta, n_phi).

    A convenience wrapper for callers wanting nothing else.
    """
    device = kwargs.get('device', 'cpu')
    draws = sample_draws(signals, betas_deg, family, vae, stats, est, meta, **kwargs)
    return render_draws(posterior_mean(draws), n_theta, n_phi, device=device)


#####################
# Other estimates   #
#####################

def encode_latents(coeffs, vae, stats, batch_size=128, device='cpu'):
    """
    Encoder posterior for each true surface: (mu, sigma), each of shape
    (B, latent_dim), float32. coeffs is (B, (L+1)^2) complex, as stored.

    A draw from the aggregate encoded distribution is mu + sigma * eps over a
    surface sample, which is how the VAE-as-prior figures sample it.
    """
    coeffs = np.asarray(coeffs)
    mus, sigmas = [], []
    for lo in range(0, coeffs.shape[0], batch_size):
        block = coeffs[lo:lo + batch_size]
        packed = _strip_dc(np.stack([coeffs_to_real(c) for c in block]), stats)
        normed = (packed - stats['mu_data']) / stats['std_data']
        with torch.no_grad():
            h = vae.encoder(torch.tensor(normed, dtype=torch.float32).to(device))
            _, mu, log_var = vae.latent(h)
        mus.append(mu.cpu().numpy())
        sigmas.append(np.exp(0.5 * log_var.cpu().numpy()))
    return (np.concatenate(mus, axis=0).astype(np.float32),
            np.concatenate(sigmas, axis=0).astype(np.float32))


def decode_latents(z, vae, stats, batch_size=512, device='cpu'):
    """
    Decode latent vectors to real-packed coefficient vectors, shape
    (B, (L+1)^2), un-standardised, float32.
    """
    z = np.asarray(z, dtype=np.float32)
    out = []
    for lo in range(0, z.shape[0], batch_size):
        with torch.no_grad():
            vecs = vae.decoder(
                torch.tensor(z[lo:lo + batch_size]).to(device)).cpu().numpy()
        raw = vecs * stats['std_data'] + stats['mu_data']
        out.append(_reinstate_dc(raw, stats))
    return np.concatenate(out, axis=0)


def decode_ceiling_coeffs(coeffs, vae, stats, batch_size=128, device='cpu'):
    """
    The decoded encoder mean per true surface, real-packed, (B, (L+1)^2).
    The coefficient-space form of decode_ceiling, for spectral and
    coefficient-error figures.
    """
    coeffs = np.asarray(coeffs)
    out = []
    for lo in range(0, coeffs.shape[0], batch_size):
        block = coeffs[lo:lo + batch_size]
        packed = _strip_dc(np.stack([coeffs_to_real(c) for c in block]), stats)
        normed = (packed - stats['mu_data']) / stats['std_data']
        with torch.no_grad():
            h = vae.encoder(torch.tensor(normed, dtype=torch.float32).to(device))
            _, mu, _ = vae.latent(h)
            vecs = vae.decoder(mu).cpu().numpy()
        raw = vecs * stats['std_data'] + stats['mu_data']
        out.append(_reinstate_dc(raw, stats))
    return np.concatenate(out, axis=0)


def decode_ceiling(coeffs, vae, stats, n_theta=N_THETA, n_phi=N_PHI,
                   batch_size=128, device='cpu'):
    """
    Encode each true surface, decode the encoder mean, render. The best any flow
    trained on this auto-encoder can reach, so it separates compression error
    from inference error. coeffs is (B, (L+1)^2) complex, as stored.
    """
    return render_draws(
        decode_ceiling_coeffs(coeffs, vae, stats, batch_size=batch_size,
                              device=device),
        n_theta, n_phi, device=device)


def classify_beta(signals, family, clf, meta,
                  log10_sigma_phot=LOG10_SIGMA_PHOT_MISSION,
                  log10_sigma_astro=LOG10_SIGMA_ASTRO_MISSION,
                  seed=0, batch_size=256, device='cpu', stored=True):
    """
    Posterior over the 91 integer inclinations, shape (B, 91). The temperature is
    applied here to ensure calibration.
    """
    signals = to_model_order(signals, family, stored=stored)
    ranges = _context_ranges(meta)

    out = []
    for lo in range(0, signals.shape[0], batch_size):
        block = signals[lo:lo + batch_size]
        b = block.shape[0]
        gen = torch.Generator().manual_seed(seed + lo)
        ctx = clf_context(
            torch.tensor(block, dtype=torch.float32),
            torch.full((b,), float(log10_sigma_phot)),
            torch.full((b,), float(log10_sigma_astro)),
            meta['ch'], gain_phot=meta['gain_phot'], gain_ast=meta['gain_ast'],
            gen=gen, **ranges).to(device)
        out.append(clf_predict(clf, ctx, temperature=meta['temperature']).cpu().numpy())
    return np.concatenate(out, axis=0)