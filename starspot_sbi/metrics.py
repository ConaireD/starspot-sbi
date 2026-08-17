"""
Metrics on rendered surfaces: weighting, SSIM, RMSE, CRPS, spot detection.

Every weighted metric takes its weights from lat_mu_grid

    visible at some phase    sin(theta - beta) > VIS_TOL
    mu_max(theta)            max(sin(theta - beta), 0)
    mu_mean(theta)           phase average of max(n_hat . r_hat, 0)

theta is the colatitude from the spin axis on the render grid of render.py
Under the package inclination convention (docs/conventions.md section 2) the
permanently unobservable region is the north cap theta < beta, so the visibility
mask excludes the FIRST rows of an image.

Variants: the paper reports SSIM_vis^aa as headline and SSIM_wmean^aa as
companion, with RMSE, MAE, PR-AUC and CRPS given whole-sphere and
visibility-weighted. The threshold-on-mu variant of the older tree is not
implemented here, since its threshold is a free parameter the paper does not
justify.
"""

###########
# Imports #
###########
# standard
import numpy as np

# special
from scipy.ndimage import uniform_filter1d

# self
from starspot_sbi.render import N_THETA, N_PHI, grid_coordinates


#############
# Constants #
#############

VIS_TOL = 1e-12
# A row grazing the terminator counts as invisible. Without this, beta = 0 keeps
# the theta = pi row

SPOT_THRESHOLD = 0.9
# Intensity below which a pixel counts as spotted, for detection metrics. The
# photosphere is normalised to unit intensity.

SSIM_L = 1.0
# Dynamic range for the SSIM stabilisation constants. The photosphere is normalised
# to 1, so 1 is the physical scale, and a per-image estimate makes scores incomparable
# between quiet and spotted stars.

#####################
# Geometric weights #
#####################

def lat_mu_grid(beta, n_theta=N_THETA):
    """
    Per-row geometry for a render grid of n_theta colatitudes at inclination
    beta (radians).

    Returns (lat_deg, mu_max, mu_mean), each shape (n_theta,):

        lat_deg   standard latitude, +90 at row 0, decreasing
        mu_max    max over rotation phase of the foreshortening, 0 where the row
                  is never visible
        mu_mean   phase average of the foreshortening, 0 where never visible

    mu_max = sin(theta - beta) follows from maximising n_hat . r_hat over phase.
    For mu_mean, write n_hat . r_hat = a cos(psi) + b with a = sin(theta) cos(beta)
    and b = -cos(theta) sin(beta); the phase average of max(a cos psi + b, 0) is
    b when b >= a, zero when b <= -a, and (a sin(psi0) + b psi0) / pi with
    psi0 = arccos(-b / a) in between.
    """
    colat_deg, _ = grid_coordinates(n_theta, 2)          # n_phi is irrelevant here
    theta = np.radians(colat_deg)
    lat_deg = 90.0 - colat_deg

    a = np.sin(theta) * np.cos(beta)
    b = -np.cos(theta) * np.sin(beta)

    mu_max = np.maximum(np.sin(theta - beta), 0.0)

    mu_mean = np.zeros_like(theta)
    always = b >= a                                       # visible at every phase
    never = b <= -a
    partial = ~(always | never)
    mu_mean[always] = b[always]
    psi0 = np.arccos(np.clip(-b[partial] / a[partial], -1.0, 1.0))
    mu_mean[partial] = (a[partial] * np.sin(psi0) + b[partial] * psi0) / np.pi

    return lat_deg, mu_max, mu_mean

def visibility_mask(beta, n_theta=N_THETA):
    """Boolean per row: True where the row is presented to the observer at some phase."""
    _, mu_max, _ = lat_mu_grid(beta, n_theta)
    return mu_max > VIS_TOL

def cap_boundary_lat(beta):
    """
    Latitude of the unobservable cap boundary in degrees. Rows above this
    latitude are never visible. Positive by construction, since the cap is
    northern.
    """
    return 90.0 - np.degrees(beta)

def weights(beta, n_theta=N_THETA, n_phi=N_PHI, kind='vis'):
    """
    Pixel weights on the render grid, shape (n_theta, n_phi), normalised to sum
    to one. All variants are area weighted by sin(theta).

    kind : 'full'   whole sphere
           'vis'    restricted to the ever-visible region
           'wmean'  scaled by the phase-averaged foreshortening

    'vis' is the headline choice. It scores the whole sphere at beta = 0 and
    half of it at beta = 90, so an inclination trend in a visibility-weighted
    metric mixes a change in performance with a change in the region being
    scored. Report 'full' alongside it whenever the comparison is against an
    operator-level quantity.
    """
    colat_deg, _ = grid_coordinates(n_theta, 2)
    sin_theta = np.sin(np.radians(colat_deg))
    _, mu_max, mu_mean = lat_mu_grid(beta, n_theta)

    if kind == 'full':
        w_row = sin_theta
    elif kind == 'vis':
        w_row = sin_theta * (mu_max > VIS_TOL)
    elif kind == 'wmean':
        w_row = sin_theta * mu_mean
    else:
        raise ValueError(f"unknown weighting {kind!r}; expected 'full', 'vis' or 'wmean'")

    w = np.repeat(w_row[:, None], n_phi, axis=1)
    total = w.sum()
    if total == 0:
        raise ValueError("weights sum to zero; check beta and n_theta")
    return w / total

def wmean(x, w):
    """Weighted mean of x under w, with w not required to be normalised."""
    return float(np.sum(w * x) / np.sum(w))

########
# SSIM #
########
def _box_theta_polar_fix(a, win_size):
    """
    Box filter along theta with pole-correct continuation: the virtual row
    across a pole is the reflected interior row rolled by half the phi grid,
    since phi and phi + pi meet at the pole. The pole row itself is excluded
    from the reflection, being its own image.

    Requires an even number of phi samples and an odd window.
    """
    pad = win_size // 2
    roll = a.shape[1] // 2
    top = np.roll(a[1:pad + 1][::-1], roll, axis=1)
    bot = np.roll(a[-pad - 1:-1][::-1], roll, axis=1)
    app = np.concatenate([top, a, bot], axis=0)
    out = uniform_filter1d(app, win_size, axis=0, mode='constant')
    return out[pad:-pad]


def _box(a, win_size):
    """Separable box filter: theta pole-correct, phi periodic."""
    a = _box_theta_polar_fix(a, win_size)
    a = uniform_filter1d(a, win_size, axis=1, mode='wrap')
    return a

def ssim_map(img1, img2, win_size=7, k1=0.01, k2=0.03, data_range=None):
    """Per-pixel structural similarity (Wang et al. 2004), phi-periodic windows."""
    L = SSIM_L if data_range is None else data_range
    if L < 1e-10:
        return np.ones_like(img1, dtype=float)

    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu1 = _box(img1, win_size)
    mu2 = _box(img2, win_size)
    s1 = _box(img1 * img1, win_size) - mu1 * mu1
    s2 = _box(img2 * img2, win_size) - mu2 * mu2
    s12 = _box(img1 * img2, win_size) - mu1 * mu2
    return ((2 * mu1 * mu2 + C1) * (2 * s12 + C2) /
            ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)))

def ssim_2d(img1, img2, win_size=7):
    """Unweighted mean SSIM. Not actually used in paper."""
    return float(ssim_map(img1, img2, win_size).mean())


def ssim_aa_vis(true_surf, recon_surf, beta, win_size=7):
    """Area-weighted SSIM over the ever-visible region. The main variant."""
    w = weights(beta, *true_surf.shape, kind='vis')
    return wmean(ssim_map(true_surf, recon_surf, win_size), w)


def ssim_aa_wmean(true_surf, recon_surf, beta, win_size=7):
    """Area-weighted SSIM scaled by the phase-averaged foreshortening."""
    w = weights(beta, *true_surf.shape, kind='wmean')
    return wmean(ssim_map(true_surf, recon_surf, win_size), w)


def ssim_aa_full(true_surf, recon_surf, beta=0.0, win_size=7):
    """
    Whole-sphere area-weighted SSIM. Independent of beta, which is accepted only
    so the three variants share a signature. This is the variant VAE model
    selection used.
    """
    w = weights(beta, *true_surf.shape, kind='full')
    return wmean(ssim_map(true_surf, recon_surf, win_size), w)

#####################
# Point estimates   #
#####################

def rmse(true_surf, recon_surf, beta, kind='vis'):
    """Area-weighted RMSE."""
    w = weights(beta, *true_surf.shape, kind=kind)
    return float(np.sqrt(wmean((true_surf - recon_surf) ** 2, w)))

def mae(true_surf, recon_surf, beta, kind='vis'):
    """Area-weighted mean absolute error."""
    w = weights(beta, *true_surf.shape, kind=kind)
    return wmean(np.abs(true_surf - recon_surf), w)

def err_unc_corr(true_surf, recon_mean, recon_std, beta, kind='vis'):
    """
    Weighted correlation between the absolute error map and the posterior
    standard deviation map. Tests whether the reported uncertainty localises the
    error rather than merely having the right magnitude on average.
    """
    w = weights(beta, *true_surf.shape, kind=kind)
    err = np.abs(true_surf - recon_mean)
    m_e, m_s = wmean(err, w), wmean(recon_std, w)
    cov = wmean((err - m_e) * (recon_std - m_s), w)
    v_e = wmean((err - m_e) ** 2, w)
    v_s = wmean((recon_std - m_s) ** 2, w)
    if v_e <= 0 or v_s <= 0:
        return float('nan')
    return float(cov / np.sqrt(v_e * v_s))

#####################
# CRPS              #
#####################

def crps(true_surf, samples, beta, kind='vis'):
    """
    Area-weighted continuous ranked probability score.

        CRPS = E|Y - x| - 0.5 E|Y - Y'|

    samples has shape (n_samples, n_theta, n_phi). The sharpness term uses the
    exact sorted form,

        0.5 E|Y - Y'| = (1 / n^2) sum_i (2i - n - 1) y_(i)

    """
    n = samples.shape[0]
    accuracy = np.mean(np.abs(samples - true_surf[None, :, :]), axis=0)

    ordered = np.sort(samples, axis=0)
    coef = (2 * np.arange(1, n + 1) - n - 1)[:, None, None]
    sharpness = np.sum(coef * ordered, axis=0) / (n ** 2)

    w = weights(beta, *true_surf.shape, kind=kind)
    return wmean(accuracy - sharpness, w)


#####################
# Spot detection    #
#####################

def spot_mask(surf, threshold=SPOT_THRESHOLD):
    """Boolean mask of pixels counted as spotted."""
    return surf < threshold

def pr_auc(true_surf, recon_surf, beta, kind='vis',
           threshold=SPOT_THRESHOLD, n_thresholds=200):
    """
    Area-weighted precision-recall AUC between the true spot mask and the
    reconstruction, swept over reconstruction thresholds.
    """
    w = weights(beta, *true_surf.shape, kind=kind)
    truth = spot_mask(true_surf, threshold)

    pos = np.sum(w * truth)
    if pos <= 0:
        return float('nan')

    lo = min(recon_surf.min(), threshold) - 1e-9
    hi = max(recon_surf.max(), threshold) + 1e-9

    precision, recall = [], []
    for t in np.linspace(hi, lo, n_thresholds):
        pred = recon_surf < t
        tp = np.sum(w * (pred & truth))
        fp = np.sum(w * (pred & ~truth))
        if tp + fp <= 0:
            continue
        precision.append(tp / (tp + fp))
        recall.append(tp / pos)

    if len(recall) < 2:
        return 0.0

    recall = np.asarray(recall)
    precision = np.asarray(precision)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def detection_operating_points(true_surf, recon_surf, beta, kind='vis',
                               threshold=SPOT_THRESHOLD, n_thresholds=200,
                               precision_floor=0.95):
    """
    Diagnostic thresholds for the sweep in pr_auc: the threshold maximising the
    weighted F1, and the highest-recall threshold whose weighted precision
    exceeds precision_floor. 
    """
    w = weights(beta, *true_surf.shape, kind=kind)
    truth = spot_mask(true_surf, threshold)
    pos = np.sum(w * truth)

    best_f1, best_f1_t = -1.0, float('nan')
    recall_at_floor, t_at_floor = -1.0, float('nan')

    lo = min(recon_surf.min(), threshold) - 1e-9
    hi = max(recon_surf.max(), threshold) + 1e-9
    for t in np.linspace(hi, lo, n_thresholds):
        pred = recon_surf < t
        tp = np.sum(w * (pred & truth))
        fp = np.sum(w * (pred & ~truth))
        if tp + fp <= 0 or pos <= 0:
            continue
        p, r = tp / (tp + fp), tp / pos
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, best_f1_t = f1, t
        if p > precision_floor and r > recall_at_floor:
            recall_at_floor, t_at_floor = r, t

    return {'f1_max': best_f1, 'f1_threshold': best_f1_t,
            'precision_floor_threshold': t_at_floor,
            'recall_at_precision_floor': recall_at_floor,
            'truth_threshold': threshold}

#####################
# Assembly          #
#####################

def scalar_metrics(true_surf, recon_surf, beta, recon_std=None, samples=None,
                   prefix=''):
    """
    The metrics the paper reports, as a flat dict suitable for a DataFrame row.

    recon_std enables the error-uncertainty correlation, samples the CRPS; both
    are omitted when not supplied.
    """
    out = {
        'ssim_aa_vis':   ssim_aa_vis(true_surf, recon_surf, beta),
        'ssim_aa_wmean': ssim_aa_wmean(true_surf, recon_surf, beta),
        'ssim_aa_full':  ssim_aa_full(true_surf, recon_surf),
        'rmse_vis':      rmse(true_surf, recon_surf, beta, 'vis'),
        'rmse_full':     rmse(true_surf, recon_surf, beta, 'full'),
        'mae_vis':       mae(true_surf, recon_surf, beta, 'vis'),
        'mae_full':      mae(true_surf, recon_surf, beta, 'full'),
        'pr_auc_vis':    pr_auc(true_surf, recon_surf, beta, 'vis'),
        'pr_auc_full':   pr_auc(true_surf, recon_surf, beta, 'full'),
    }
    if recon_std is not None:
        out['err_unc_corr'] = err_unc_corr(true_surf, recon_surf, recon_std, beta)
    if samples is not None:
        out['crps_vis'] = crps(true_surf, samples, beta, 'vis')
        out['crps_full'] = crps(true_surf, samples, beta, 'full')

    return {prefix + k: v for k, v in out.items()} if prefix else out