"""
Network architectures and checkpoint loaders.
 
Three model types:
 
    VAE           EncoderFlex -> LatentSpace -> DecoderFlex over real-packed
                  spherical-harmonic coefficients, configured by ArchConfig
    flow          neural spline flow over the 96-dimensional latent, conditioned
                  on a signal and on (beta, log sigma) auxiliaries
    classifier    91-way softmax over integer beta, conditioned on a signal and
                  on the log sigmas alone
 
Channel indices are (phot, astro_x, astro_y) = (0, 1, 2), and a signal family is
a list of these, so photometry with y astrometry is [0, 2].
The stored .npy files hold (astro_x, astro_y, phot), so the permutation [2, 0, 1]
is applied when a signal is read; see docs/conventions.md section 9.
"""

###########
# Imports #
###########

# python?
from dataclasses import dataclass
from typing import List, Optional

# standard
import numpy as np

# machine learning
import torch
import torch.nn as nn
import torch.nn.functional as F

from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors import DirectPosterior

#####################
# Constants + notes #
#####################

# Truncation degree of the surface expansion. The released checkpoints were
# trained at L_MAX = 30, so the VAE consumes 961 coefficients. Changing it
# means retraining the VAE, and with it the flows and classifiers, which take
# the VAE's latent as their target. The dataset is regenerated from
# scripts/generate_dataset.py at the new degree first.
 
L_MAX = 30
N_COEFFS = (L_MAX + 1) ** 2

# Channel indices. The stored .npy files hold (astro_x, astro_y, phot), so the
# permutation [2, 0, 1] is applied when a signal is read. 
# Adding a channel means a new kernel in kernels.py, a new block in the design
# matrix, a regenerated dataset, and retrained flows and classifiers.

CH_PHOT, CH_AX, CH_AY = 0, 1, 2
FAMILIES = {
    'phot':      [CH_PHOT],
    'phot_ax':   [CH_PHOT, CH_AX],
    'phot_ay':   [CH_PHOT, CH_AY],
    'phot_axay': [CH_PHOT, CH_AX, CH_AY],
}

# Inclination domain and its discretisation. beta runs over the integer degrees
# 0 to 90 inclusive, giving 91 classes in the classifer. The flow conditions on 
# beta as a continuous normalised scalar and is unaffected by the class count. 
# A finer grid means retraining the classifiers and regenerating the 
# design-matrix cache, which holds one matrix per integer degree.

BETA_MIN, BETA_MAX = 0.0, 90.0
N_BETA_CLASSES = 91

# Training ranges for the noise conditioning.
# These normalise the auxiliary entries, so an inference call conditioning on
# sigma_phot = 1e-4 passes _u(-4.0, -7.0, -2.0) = +0.2 rather than -4.0.
LOG10_SIGMA_PHOT = (-7.0, -2.0)
LOG10_SIGMA_ASTRO = (-7.0, -2.0)


# Signal gains, from nsf_noisy/gains.json: the reciprocals of the median signal
# RMS over 20,000 sampled surfaces. The photometric signal has RMS about 0.0032
# and the astrometric about 0.0025, so the gains bring both to order unity
# before the network's own standardisation, which was fitted on gained inputs
# and frozen into the state dictionary. Feeding un-gained signals to a loaded
# model presents that layer with inputs some 300 times too small.
#
# These are defaults for make_context and clf_context. load_flow and
# load_classifier return the values stored in each checkpoint, and those are
# what an inference call should use.
GAIN_PHOT = 314.82056848788335
GAIN_AST = 398.15769118301773

# Flow architecture, identical across families apart from the channel count.
FLOW = dict(hidden_features=128, num_transforms=20, num_bins=8)
EMB  = dict(embedding_dim=128, patch_size=16, d_model=64, n_heads=4, n_attn_layers=2)
AUX_DIM = 16

#####################
# Packing helpers   #
#####################
 
def input_dim(l_min, l_max=L_MAX):
    """Length of the real-packed vector the VAE consumes."""
    n = (l_max + 1) ** 2
    return n if l_min == 0 else n - 1

# Reachable only through tokenizer='order'. No released checkpoint uses it, but will exist as an option
def band_index_sets(l_max, l_min):
    """
    Indices of the real-packed vector grouped by degree, one array per degree.
    Each holds the m = 0 entry, then the real parts for m = 1..l, then the
    imaginary parts.
    """
    n_m0 = l_max - l_min + 1
    re_off = n_m0
    im_off = n_m0 + sum(range(1, l_max + 1))
 
    def repos(l, m):
        return sum(range(1, l)) + (m - 1)
 
    sets = []
    for l in range(l_min, l_max + 1):
        idx = [l - l_min]
        idx += [re_off + repos(l, m) for m in range(1, l + 1)]
        idx += [im_off + repos(l, m) for m in range(1, l + 1)]
        sets.append(np.array(idx, dtype=np.int64))
    return sets

def band_index_sets(l_max, l_min):
    """
    Indices of the real-packed vector grouped by degree, one array per degree.
    Each holds the m = 0 entry, then the real parts for m = 1..l, then the
    imaginary parts.
    """
    n_m0 = l_max - l_min + 1
    re_off = n_m0
    im_off = n_m0 + sum(range(1, l_max + 1))
 
    def repos(l, m):
        return sum(range(1, l)) + (m - 1)
 
    sets = []
    for l in range(l_min, l_max + 1):
        idx = [l - l_min]
        idx += [re_off + repos(l, m) for m in range(1, l + 1)]
        idx += [im_off + repos(l, m) for m in range(1, l + 1)]
        sets.append(np.array(idx, dtype=np.int64))
    return sets


def order_index_sets(l_max, l_min):
    """As band_index_sets, grouped by order rather than degree."""
    n_m0 = l_max - l_min + 1
    re_off = n_m0
    im_off = n_m0 + sum(range(1, l_max + 1))
 
    def repos(l, m):
        return sum(range(1, l)) + (m - 1)
 
    sets = [np.arange(n_m0, dtype=np.int64)]
    for m in range(1, l_max + 1):
        idx = [re_off + repos(l, m) for l in range(max(m, 1), l_max + 1)]
        idx += [im_off + repos(l, m) for l in range(max(m, 1), l_max + 1)]
        sets.append(np.array(idx, dtype=np.int64))
    return sets

def _build_mlp(dims, dropout=0.0, final_activation=False):
    """GELU multilayer perceptron over the given layer widths."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if not is_last or final_activation:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
 
 
def _attn_stage(d_model, n_heads, n_layers, dropout):
    el = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
        dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(el, num_layers=n_layers)

#####################
# VAE architecture  #
#####################

@dataclass
class ArchConfig:
    """
    VAE architecture and training configuration, stored in the checkpoint under
    'config'. The released checkpoint uses tokenizer='band', pooling='cls',
    funnel='none', d_model=128, output_dim=128, latent_dim=96, l_min=0.
    """
    tokenizer: str = 'band'
    chunk_size: int = 32
    funnel: str = 'none'
    n_merge_stages: int = 2
    growth: int = 2
    pooling: str = 'flatten'
    d_model: int = 64
    n_heads: int = 4
    layers_per_stage: int = 2
    latent_dim: int = 96
    output_dim: int = 128
    head_hidden_dims: Optional[List[int]] = None
    attn_dropout: float = 0.1
    head_dropout: float = 0.15
    beta_final: float = 1.1
    beta_schedule_type: str = 'sigmoid'
    beta_warmup_epochs: int = 50
    beta_steepness: float = 5.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    l_min: int = 1
    n_epochs: int = 5000
    patience: int = 25
    grad_clip: float = 100.0
 
 
class Tokenizer(nn.Module):
    """Real-packed coefficients to a sequence of tokens, one per degree or order."""
 
    def __init__(self, mode, l_min, d_model, chunk_size=32, l_max=L_MAX):
        super().__init__()
        self.mode, self.d_model = mode, d_model
        D = input_dim(l_min, l_max)
        if mode in ('band', 'order'):
            sets = (band_index_sets(l_max, l_min) if mode == 'band'
                    else order_index_sets(l_max, l_min))
            for i, s in enumerate(sets):
                self.register_buffer(f'idx_{i}', torch.tensor(s))
            self.projs = nn.ModuleList([nn.Linear(len(s), d_model) for s in sets])
            self.n_tokens = len(sets)
        elif mode == 'chunk':
            self.chunk_size = chunk_size
            self.n_tokens = (D + chunk_size - 1) // chunk_size
            self.pad = self.n_tokens * chunk_size - D
            self.proj = nn.Linear(chunk_size, d_model)
            self.pos = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        else:
            raise ValueError(f"unknown tokenizer mode {mode!r}")
 
    def forward(self, x):
        b = x.shape[0]
        if self.mode == 'chunk':
            if self.pad > 0:
                x = F.pad(x, (0, self.pad))
            return self.proj(x.view(b, self.n_tokens, self.chunk_size)) + self.pos
        tokens = torch.empty(b, self.n_tokens, self.d_model,
                             device=x.device, dtype=x.dtype)
        for i in range(self.n_tokens):
            tokens[:, i] = self.projs[i](x[:, getattr(self, f'idx_{i}')])
        return tokens
 
 
class Detokenizer(nn.Module):
    """Tokens back to real-packed coefficients, inverting Tokenizer."""
 
    def __init__(self, mode, l_min, d_model, chunk_size=32, l_max=L_MAX):
        super().__init__()
        self.mode = mode
        self.D = input_dim(l_min, l_max)
        if mode in ('band', 'order'):
            sets = (band_index_sets(l_max, l_min) if mode == 'band'
                    else order_index_sets(l_max, l_min))
            for i, s in enumerate(sets):
                self.register_buffer(f'idx_{i}', torch.tensor(s))
            self.projs = nn.ModuleList([nn.Linear(d_model, len(s)) for s in sets])
            self.n_tokens = len(sets)
        elif mode == 'chunk':
            self.chunk_size = chunk_size
            self.n_tokens = (self.D + chunk_size - 1) // chunk_size
            self.proj = nn.Linear(d_model, chunk_size)
        else:
            raise ValueError(f"unknown tokenizer mode {mode!r}")
 
    def forward(self, tokens):
        b = tokens.shape[0]
        if self.mode == 'chunk':
            return self.proj(tokens).reshape(b, -1)[:, :self.D]
        out = torch.empty(b, self.D, device=tokens.device, dtype=tokens.dtype)
        for i in range(self.n_tokens):
            out[:, getattr(self, f'idx_{i}')] = self.projs[i](tokens[:, i])
        return out
 
# TokenMerge, TokenSplit, AttnPool and _token_schedule are reachable only
# through funnel='merge' or pooling in ('flatten', 'mean', 'attn'). The released
# checkpoint uses funnel='none' and pooling='cls', so none of them runs.
class TokenMerge(nn.Module):
    """Halve the token count by projecting adjacent pairs."""
 
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(2 * d_in, d_out)
 
    def forward(self, t):
        b, n, d = t.shape
        if n % 2:
            t = torch.cat([t, torch.zeros(b, 1, d, device=t.device, dtype=t.dtype)], dim=1)
        return self.proj(t.reshape(b, -1, 2 * d))
 
 
class TokenSplit(nn.Module):
    """Double the token count, inverting TokenMerge."""
 
    def __init__(self, d_in, d_out, n_target):
        super().__init__()
        self.proj = nn.Linear(d_in, 2 * d_out)
        self.n_target, self.d_out = n_target, d_out
 
    def forward(self, t):
        b = t.shape[0]
        return self.proj(t).reshape(b, -1, self.d_out)[:, :self.n_target]
 
 
class AttnPool(nn.Module):
    """Pool a token sequence to one vector with a learned query."""
 
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
 
    def forward(self, t):
        out, _ = self.mha(self.q.expand(t.shape[0], -1, -1), t, t)
        return out[:, 0]
 
 
def _token_schedule(n0, n_stages):
    ns = [n0]
    for _ in range(n_stages):
        ns.append((ns[-1] + 1) // 2)
    return ns
 
 
class EncoderFlex(nn.Module):
    """Real-packed coefficients to a fixed-width summary."""
 
    def __init__(self, cfg, l_max=L_MAX):
        super().__init__()
        self.cfg = cfg
        self.tok = Tokenizer(cfg.tokenizer, cfg.l_min, cfg.d_model, cfg.chunk_size, l_max)
        n0 = self.tok.n_tokens
        S = cfg.n_merge_stages if cfg.funnel == 'merge' else 0
        self.token_counts = _token_schedule(n0, S)
        self.dims = [cfg.d_model * (cfg.growth ** k) for k in range(S + 1)]
 
        self.use_cls = (cfg.pooling == 'cls')
        if self.use_cls:
            assert cfg.funnel == 'none', "cls pooling requires funnel='none'"
            self.cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
 
        stages, merges = [], []
        for k in range(S + 1):
            stages.append(_attn_stage(self.dims[k], cfg.n_heads,
                                      cfg.layers_per_stage, cfg.attn_dropout))
            if k < S:
                merges.append(TokenMerge(self.dims[k], self.dims[k + 1]))
        self.stages, self.merges = nn.ModuleList(stages), nn.ModuleList(merges)
 
        d_f, n_f = self.dims[-1], self.token_counts[-1]
        pooled_dim = ((n_f + (1 if self.use_cls else 0)) * d_f
                      if cfg.pooling == 'flatten' else d_f)
        if cfg.pooling == 'attn':
            self.pool = AttnPool(d_f, cfg.n_heads)
        hhd = cfg.head_hidden_dims if cfg.head_hidden_dims is not None else [cfg.output_dim * 2]
        self.head = _build_mlp([pooled_dim] + list(hhd) + [cfg.output_dim],
                               dropout=cfg.head_dropout)
 
    def forward(self, x):
        t = self.tok(x)
        if self.use_cls:
            t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        for k, stage in enumerate(self.stages):
            t = stage(t)
            if k < len(self.merges):
                t = self.merges[k](t)
        p = self.cfg.pooling
        if p == 'flatten':
            h = t.reshape(t.shape[0], -1)
        elif p == 'mean':
            h = t.mean(dim=1)
        elif p == 'cls':
            h = t[:, 0]
        elif p == 'attn':
            h = self.pool(t)
        else:
            raise ValueError(f"unknown pooling {p!r}")
        return self.head(h)
 
 
class DecoderFlex(nn.Module):
    """Latent to real-packed coefficients, inverting EncoderFlex."""
 
    def __init__(self, cfg, l_max=L_MAX):
        super().__init__()
        self.cfg = cfg
        self.detok = Detokenizer(cfg.tokenizer, cfg.l_min, cfg.d_model,
                                 cfg.chunk_size, l_max)
        n0 = self.detok.n_tokens
        S = cfg.n_merge_stages if cfg.funnel == 'merge' else 0
        self.token_counts = _token_schedule(n0, S)
        self.dims = [cfg.d_model * (cfg.growth ** k) for k in range(S + 1)]
 
        d_f, n_f = self.dims[-1], self.token_counts[-1]
        hhd = cfg.head_hidden_dims if cfg.head_hidden_dims is not None else [n_f * d_f * 2]
        self.head = _build_mlp([cfg.latent_dim] + list(hhd) + [n_f * d_f],
                               dropout=cfg.head_dropout, final_activation=True)
 
        stages, splits = [], []
        for k in range(S, -1, -1):
            stages.append(_attn_stage(self.dims[k], cfg.n_heads,
                                      cfg.layers_per_stage, cfg.attn_dropout))
            if k > 0:
                splits.append(TokenSplit(self.dims[k], self.dims[k - 1],
                                         self.token_counts[k - 1]))
        self.stages, self.splits = nn.ModuleList(stages), nn.ModuleList(splits)
        self.n_f, self.d_f = n_f, d_f
 
    def forward(self, z):
        t = self.head(z).reshape(z.shape[0], self.n_f, self.d_f)
        for k, stage in enumerate(self.stages):
            t = stage(t)
            if k < len(self.splits):
                t = self.splits[k](t)
        return self.detok(t)
 
 
class LatentSpace(nn.Module):
    """Summary to a diagonal Gaussian posterior, with the reparameterised draw."""
 
    def __init__(self, encoder_output_dim, latent_dim, init_log_var_bias=-2.0,
                 log_var_clamp=(-10., 10.)):
        super().__init__()
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_dim, latent_dim)
        nn.init.constant_(self.fc_log_var.bias, init_log_var_bias)
        self.log_var_clamp = log_var_clamp
 
    def forward(self, h):
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(*self.log_var_clamp)
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu), mu, log_var
 
 
class VAE(nn.Module):
    """Encoder, latent, decoder. forward returns (reconstruction, mu, log_var)."""
 
    def __init__(self, encoder, latent, decoder):
        super().__init__()
        self.encoder, self.latent, self.decoder = encoder, latent, decoder
 
    def forward(self, x):
        h = self.encoder(x)
        z, mu, lv = self.latent(h)
        return self.decoder(z), mu, lv
 
 
#####################
# Signal embeddings #
#####################

class AttnSignalEmbeddingAux(nn.Module):
    """
    Summarises a multi-channel time series and n_aux scalars into a fixed-width
    conditioning vector.
 
    The input is flattened, channel-major: the first n_channels * T entries are
    the signal and the remainder are the auxiliaries. The series is cut into
    patches of patch_size, each patch of each channel becomes a token, and
    self-attention mixes the tokens before pooling.
 
    A patch length of 16 over T = 216 leaves m = 30, whose period is about seven
    samples, shorter than the patch, so the highest orders may be under-resolved
    by the summary.
    """
 
    def __init__(self, T, n_channels=3, embedding_dim=128, patch_size=16,
                 d_model=64, n_heads=4, n_attn_layers=2, dropout=0.0,
                 n_aux=2, aux_dim=AUX_DIM):
        super().__init__()
        self.T, self.n_channels, self.patch_size = T, n_channels, patch_size
        self.n_patches = (T + patch_size - 1) // patch_size
        self.pad = self.n_patches * patch_size - T
        self.n_tokens = n_channels * self.n_patches
        self.n_aux = n_aux
 
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.1)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(el, num_layers=n_attn_layers)
        self.aux_proj = nn.Sequential(nn.Linear(n_aux, aux_dim), nn.GELU(),
                                      nn.Linear(aux_dim, aux_dim))
        self.head = nn.Sequential(
            nn.Linear(self.n_tokens * d_model + aux_dim, embedding_dim * 2),
            nn.GELU(), nn.Linear(embedding_dim * 2, embedding_dim))
        self.signal_len = n_channels * T
 
    def forward(self, x):
        b = x.shape[0]
        sig = x[:, :self.signal_len].view(b, self.n_channels, self.T)
        aux = x[:, self.signal_len:]
        if self.pad > 0:
            sig = F.pad(sig, (0, self.pad))
        tokens = sig.view(b, self.n_channels, self.n_patches, self.patch_size)
        tokens = self.patch_proj(tokens.reshape(b, self.n_tokens, self.patch_size)) \
            + self.pos_embed
        return self.head(torch.cat([self.attention(tokens).reshape(b, -1),
                                    self.aux_proj(aux)], dim=1))
 
 
class _ClfEmbedding(nn.Module):
    """
    As AttnSignalEmbeddingAux, with the final projection named `out` rather than
    `head` to match the classifier state dictionaries.
    """
 
    def __init__(self, T, n_channels, embedding_dim=256, patch_size=16, d_model=64,
                 n_heads=4, n_attn_layers=2, dropout=0.0, n_aux=1, aux_dim=AUX_DIM):
        super().__init__()
        self.T, self.n_channels, self.patch_size = T, n_channels, patch_size
        self.n_patches = (T + patch_size - 1) // patch_size
        self.pad = self.n_patches * patch_size - T
        self.n_tokens = n_channels * self.n_patches
 
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.1)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(el, num_layers=n_attn_layers)
        self.aux_proj = nn.Sequential(nn.Linear(n_aux, aux_dim), nn.GELU(),
                                      nn.Linear(aux_dim, aux_dim))
        self.out = nn.Sequential(
            nn.Linear(self.n_tokens * d_model + aux_dim, embedding_dim * 2),
            nn.GELU(), nn.Linear(embedding_dim * 2, embedding_dim))
        self.signal_len = n_channels * T
 
    def forward(self, x):
        b = x.shape[0]
        sig = x[:, :self.signal_len].view(b, self.n_channels, self.T)
        aux = x[:, self.signal_len:]
        if self.pad:
            sig = F.pad(sig, (0, self.pad))
        tok = sig.view(b, self.n_channels, self.n_patches, self.patch_size)
        tok = self.patch_proj(tok.reshape(b, self.n_tokens, self.patch_size)) + self.pos_embed
        return self.out(torch.cat([self.attention(tok).reshape(b, -1),
                                   self.aux_proj(aux)], 1))
 
 
class BetaClassifier(nn.Module):
    """
    91-way softmax over the integer inclinations beta = 0 to 90 degrees.
 
    The auxiliaries are the log sigmas alone. Beta is the target, so it is not an
    input, which makes the conditioning vector one entry shorter than the flow of
    the same family.
    """
 
    def __init__(self, T, n_channels, n_aux, n_beta=N_BETA_CLASSES,
                 embedding_dim=256, patch_size=16, d_model=64, n_heads=4,
                 n_attn_layers=2, aux_dim=AUX_DIM, head_hidden=256, dropout=0.0):
        super().__init__()
        self.embedding = _ClfEmbedding(
            T=T, n_channels=n_channels, embedding_dim=embedding_dim,
            patch_size=patch_size, d_model=d_model, n_heads=n_heads,
            n_attn_layers=n_attn_layers, dropout=dropout,
            n_aux=n_aux, aux_dim=aux_dim)
        layers = [nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, head_hidden),
                  nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(head_hidden, n_beta))
        self.clf_head = nn.Sequential(*layers)
        self.signal_len = self.embedding.signal_len
 
    def forward(self, x):
        """Raw logits. The temperature is applied by clf_predict, not here."""
        return self.clf_head(self.embedding(x))
 

#####################
# Conditioning      #
#####################

def _u(l10, lo, hi):
    """Map a log sigma from its training range to [-1, 1]."""
    return 2.0 * (l10 - lo) / (hi - lo) - 1.0
 
 
def beta_norm_from_deg(beta_deg):
    """Map beta from [0, 90] degrees to [-1, 1]."""
    return 2.0 * (beta_deg - BETA_MIN) / (BETA_MAX - BETA_MIN) - 1.0
 
 
def beta_deg_from_norm(beta_norm):
    """Inverse of beta_norm_from_deg."""
    return (beta_norm + 1.0) * 0.5 * (BETA_MAX - BETA_MIN) + BETA_MIN
 
 
def n_aux_for_flow(ch_sel):
    """Flow auxiliaries: beta, log sigma_phot, and log sigma_astro if astrometric."""
    return 2 + (1 if any(c != CH_PHOT for c in ch_sel) else 0)
 
 
def n_aux_for_clf(ch_sel):
    """Classifier auxiliaries: the log sigmas alone, since beta is the target."""
    return 1 + (1 if any(c != CH_PHOT for c in ch_sel) else 0)
 
 
def _preprocess(sig, l10_p, l10_a, ch_sel, gain_phot, gain_ast, gen=None):
    """
    Add noise, self-normalise per channel, apply the gains.
 
    Noise is added before the mean is taken, since on real data the baseline is
    estimated from the noisy series and is therefore itself noisy. The
    photometric channel becomes a relative flux and the astrometric channels are
    mean-subtracted, and both are then scaled by their gain.
    """
    B, nc, _ = sig.shape
    sd = torch.empty(B, nc, 1, dtype=sig.dtype)
    for j, c in enumerate(ch_sel):
        sd[:, j, 0] = torch.pow(10.0, l10_p if c == CH_PHOT else l10_a)
    y = sig + sd * torch.randn(sig.shape, generator=gen, dtype=sig.dtype)
 
    out = torch.empty_like(y)
    for j, c in enumerate(ch_sel):
        if c == CH_PHOT:
            out[:, j] = (y[:, j] / y[:, j].mean(1, keepdim=True) - 1.0) * gain_phot
        else:
            out[:, j] = (y[:, j] - y[:, j].mean(1, keepdim=True)) * gain_ast
    return out
 
 
def make_context(sig, beta_norm, l10_p, l10_a, ch_sel,
                 gain_phot=GAIN_PHOT, gain_ast=GAIN_AST, gen=None,
                 log10_sigma_phot=LOG10_SIGMA_PHOT,
                 log10_sigma_astro=LOG10_SIGMA_ASTRO):
    """
    Conditioning vector for a flow, shape (B, len(ch_sel) * T + n_aux):
 
        [ ch[0] signal (T) | ch[1] signal (T) | ... | beta_norm | u(l10_p) | u(l10_a)? ]
 
    sig has shape (B, len(ch_sel), T) in the order given by ch_sel. beta_norm is
    already normalised to [-1, 1]; use beta_norm_from_deg. l10_p and l10_a are
    raw log10 sigmas and are normalised here.
    """
    B = sig.shape[0]
    out = _preprocess(sig, l10_p, l10_a, ch_sel, gain_phot, gain_ast, gen)
    aux = [beta_norm.reshape(B, 1), _u(l10_p, *log10_sigma_phot).reshape(B, 1)]
    if any(c != CH_PHOT for c in ch_sel):
        aux.append(_u(l10_a, *log10_sigma_astro).reshape(B, 1))
    return torch.cat([out.reshape(B, -1)] + aux, 1)
 
 
def clf_context(sig, l10_p, l10_a, ch_sel,
                gain_phot=GAIN_PHOT, gain_ast=GAIN_AST, gen=None,
                log10_sigma_phot=LOG10_SIGMA_PHOT,
                log10_sigma_astro=LOG10_SIGMA_ASTRO):
    """
    Conditioning vector for a classifier, shape (B, len(ch_sel) * T + n_aux):
 
        [ ch[0] signal (T) | ... | u(l10_p) | u(l10_a)? ]
 
    Signal preprocessing is identical to make_context. Beta is absent.
    """
    B = sig.shape[0]
    out = _preprocess(sig, l10_p, l10_a, ch_sel, gain_phot, gain_ast, gen)
    aux = [_u(l10_p, *log10_sigma_phot).reshape(B, 1)]
    if any(c != CH_PHOT for c in ch_sel):
        aux.append(_u(l10_a, *log10_sigma_astro).reshape(B, 1))
    return torch.cat([out.reshape(B, -1)] + aux, 1)



#####################
# Construction      #
#####################
 
def build_flow(n_channels, n_aux, T, latent_dim, flow=None, emb=None,
               aux_dim=AUX_DIM, device='cpu'):
    """
    Build a neural spline flow with its conditional embedding, returning the
    density estimator.
 
    The estimator is primed with a zero batch to trigger the lazy shape
    initialisation in sbi, which leaves the context standardisation layer at
    mean 0 and standard deviation 1. The trained values arrive with the state
    dictionary, so the priming batch only has to be the right shape.
    """
    flow = FLOW if flow is None else flow
    emb = EMB if emb is None else emb
 
    net = AttnSignalEmbeddingAux(
        T=T, n_channels=n_channels, embedding_dim=emb['embedding_dim'],
        patch_size=emb['patch_size'], d_model=emb['d_model'],
        n_heads=emb['n_heads'], n_attn_layers=emb['n_attn_layers'],
        dropout=0.0, n_aux=n_aux, aux_dim=aux_dim)
 
    build = posterior_nn(model='nsf', embedding_net=net,
                         hidden_features=flow['hidden_features'],
                         num_transforms=flow['num_transforms'],
                         num_bins=flow['num_bins'])
    init_x = torch.zeros((8, n_channels * T + n_aux), dtype=torch.float32)
    init_z = torch.zeros((8, latent_dim), dtype=torch.float32)
    return build(init_z, init_x).to(device)
 
 
def make_prior(latent_dim, device='cpu'):
    """
    The nominal prior handed to the sbi posterior object, an isotropic Gaussian
    over the latent.
 
    The true marginal under the training joint is the standardised aggregate
    encoder distribution, which has unit variance by construction and is neither
    isotropic nor Gaussian. The support is unbounded, so no leakage correction
    applies, but log densities reported by the sampling wrapper should be read
    with the difference in mind.
    """
    return torch.distributions.MultivariateNormal(
        torch.zeros(latent_dim, device=device),
        torch.eye(latent_dim, device=device))
 
 
def make_posterior(est, latent_dim, device='cpu'):
    """Wrap a trained density estimator in an sbi DirectPosterior."""
    return DirectPosterior(posterior_estimator=est,
                           prior=make_prior(latent_dim, device),
                           device=str(device))

#####################
# Loaders           #
#####################
 
def load_vae(ckpt_path, device='cpu', l_max=L_MAX):
    """
    Load a VAE checkpoint. Returns (model, cfg, stats) with the model in eval
    mode and gradients disabled, and stats a dict of mu_data, std_data,
    dc_value and include_dc.
 
    Two key layouts exist. The grid-search checkpoints store the architecture
    under 'config' and an earlier baseline under 'hyperparameters'; only the
    first is supported here, since the released checkpoint uses it and the
    second needs a different architecture.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'config' not in ckpt:
        raise KeyError(f"{ckpt_path}: no 'config' key; found {sorted(ckpt.keys())}")
 
    cfg = ArchConfig(**{k: v for k, v in ckpt['config'].items()
                        if k in ArchConfig.__dataclass_fields__})
    model = VAE(EncoderFlex(cfg, l_max),
                LatentSpace(cfg.output_dim, cfg.latent_dim),
                DecoderFlex(cfg, l_max)).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
 
    stats = {
        'mu_data': np.asarray(ckpt['mu_data']).ravel(),
        'std_data': np.asarray(ckpt['std_data']).ravel(),
        'dc_value': ckpt.get('dc_value', None),
        'include_dc': cfg.l_min == 0,
    }
    return model, cfg, stats
 
 
def load_flow(ckpt_path, latent_dim=96, T=None, device='cpu'):
    """
    Load a noise-conditioned flow. Returns (est, meta) with meta holding ch,
    n_aux, T, gain_phot, gain_ast and the two log sigma ranges.
 
    The architecture is not recorded in the checkpoint and comes from the FLOW
    and EMB constants. T is recovered from the context standardisation layer,
    whose length is len(ch) * T + n_aux, and may be given explicitly instead.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ch, n_aux = list(ck['ch']), int(ck['n_aux'])
 
    if T is None:
        ctx_dim = ck['net']['net._embedding_net.0._mean'].numel()
        T = (ctx_dim - n_aux) // len(ch)
 
    est = build_flow(len(ch), n_aux, T, latent_dim, device=device)
    est.load_state_dict(ck['net'], strict=True)
    est.eval()
    for p in est.parameters():
        p.requires_grad_(False)
 
    meta = {'ch': ch, 'n_aux': n_aux, 'T': T,
            'gain_phot': float(ck['gain_phot']), 'gain_ast': float(ck['gain_ast']),
            'log10_sigma_phot': tuple(ck['log10_sigma_phot']),
            'log10_sigma_astro': tuple(ck['log10_sigma_astro']),
            'context_dim': len(ch) * T + n_aux}
    return est, meta
 
 
def load_classifier(ckpt_path, T, device='cpu'):
    """
    Load a beta classifier. Returns (clf, meta) with meta holding ch, n_aux,
    temperature, the gains and the log sigma ranges.
 
    The architecture is recorded in the checkpoint under 'arch'. T is not, and
    must be supplied. The temperature is applied by clf_predict rather than
    inside the network, so a caller that uses the module directly and discards
    meta loses the calibration.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ch = list(ck['ch'])
    arch = ck['arch']
    n_aux = n_aux_for_clf(ch)
 
    clf = BetaClassifier(
        T=T, n_channels=len(ch), n_aux=n_aux, n_beta=N_BETA_CLASSES,
        embedding_dim=arch['embedding_dim'], patch_size=arch['patch_size'],
        d_model=arch['d_model'], n_heads=arch['n_heads'],
        n_attn_layers=arch['n_attn_layers'], aux_dim=arch['aux_dim'],
        head_hidden=arch['head_hidden'], dropout=arch['dropout']).to(device)
    clf.load_state_dict(ck['net'], strict=True)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
 
    meta = {'ch': ch, 'n_aux': n_aux, 'T': T,
            'temperature': float(ck['temperature']),
            'gain_phot': float(ck['gain_phot']), 'gain_ast': float(ck['gain_ast']),
            'log10_sigma_phot': tuple(ck['log10_sigma_phot']),
            'log10_sigma_astro': tuple(ck['log10_sigma_astro']),
            'context_dim': len(ch) * T + n_aux}
    return clf, meta
 
 
@torch.no_grad()
def clf_predict(clf, x_clf, temperature=1.0):
    """Posterior over the 91 integer inclinations, shape (B, 91)."""
    return torch.softmax(clf(x_clf) / temperature, dim=-1)
 