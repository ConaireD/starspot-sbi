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

    Requires an even number of phi samples, an odd window, and at least
    win_size // 2 + 1 rows, since the reflection draws win_size // 2 interior
    rows from each end.
    """
    pad = win_size // 2
    if a.shape[0] < pad + 1:
        raise ValueError(
            f"theta axis has {a.shape[0]} rows; a window of {win_size} needs at "
            f"least {pad + 1}, since the reflection draws {pad} interior rows "
            f"from each pole")
    if a.shape[1] % 2:
        raise ValueError(f"phi axis has {a.shape[1]} samples; an even number is "
                         f"required for the half-turn roll")
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

def pr_auc_weighted(true_surf, recon_surf, w,
                    threshold=SPOT_THRESHOLD, n_thresholds=200):
    """
    Precision-recall AUC under an explicit weight array, swept over
    reconstruction thresholds. nan when the truth mask carries no weight,
    0.0 when it does and the reconstruction never fires.
    """
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


def pr_auc(true_surf, recon_surf, beta, kind='vis',
           threshold=SPOT_THRESHOLD, n_thresholds=200):
    """
    Area-weighted precision-recall AUC between the true spot mask and the
    reconstruction, swept over reconstruction thresholds.
    """
    w = weights(beta, *true_surf.shape, kind=kind)
    return pr_auc_weighted(true_surf, recon_surf, w, threshold, n_thresholds)


def pr_auc_bands(true_surf, recon_surf, axis='lat', band_width=None,
                 threshold=SPOT_THRESHOLD, n_thresholds=200):
    """
    Whole-sphere area-weighted PR-AUC restricted to bands of latitude or
    longitude, for one surface. Returns (centres, values) with one entry per
    band; a band whose truth mask carries no weight is nan.

    axis 'lat' bins the render rows (default band 10 deg); 'lon' bins the
    columns (default 20 deg). Band edges follow the render grid convention:
    row 0 is latitude +90 and the columns run from -180 to +180 degrees.
    """
    n_theta, n_phi = true_surf.shape
    w = weights(0.0, n_theta, n_phi, kind='full')

    if axis == 'lat':
        band_width = 10.0 if band_width is None else float(band_width)
        edges = np.arange(-90.0, 90.0 + band_width / 2, band_width)
        coord = 90.0 - np.degrees(np.linspace(0, np.pi, n_theta))
        along_rows = True
    elif axis == 'lon':
        band_width = 20.0 if band_width is None else float(band_width)
        edges = np.arange(-180.0, 180.0 + band_width / 2, band_width)
        coord = np.degrees(np.linspace(-np.pi, np.pi, n_phi))
        along_rows = False
    else:
        raise ValueError(f"axis must be 'lat' or 'lon', got {axis!r}")

    bins = np.clip(np.digitize(coord, edges) - 1, 0, len(edges) - 2)
    centres = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(centres.size, np.nan)
    for b in range(centres.size):
        wb = w.copy()
        if along_rows:
            wb[bins != b, :] = 0.0
        else:
            wb[:, bins != b] = 0.0
        if wb.sum() <= 0:
            continue
        values[b] = pr_auc_weighted(true_surf, recon_surf, wb,
                                    threshold, n_thresholds)
    return centres, values


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
# Spot matching     #
#####################

def periodic_label(mask):
    """
    Connected components of a boolean mask, periodic in longitude.

    scipy.ndimage.label treats the first and last columns as distant; on the
    render grid they are the same meridian, so components touching both are
    one spot. Labels are relabelled 1..n after merging.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return lab, n
    parent = np.arange(n + 1)
    for a, b in zip(lab[:, 0], lab[:, -1]):
        if a and b:
            ra, rb = int(a), int(b)
            while parent[ra] != ra:
                ra = parent[ra]
            while parent[rb] != rb:
                rb = parent[rb]
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.zeros(n + 1, dtype=int)
    for k in range(1, n + 1):
        r = k
        while parent[r] != r:
            r = parent[r]
        roots[k] = r
    new = {v: i + 1 for i, v in enumerate(sorted(set(roots[1:].tolist())))}
    out = np.zeros_like(lab)
    for k in range(1, n + 1):
        out[lab == k] = new[roots[k]]
    return out, len(new)


def spot_centroids(surf, beta, kind='vis', threshold=SPOT_THRESHOLD,
                   min_area_frac=2e-4):
    """
    The connected spotted regions of a rendered surface, deepest first.

    Each entry carries the area-weighted centroid (lat, lon in degrees,
    latitude +90 at render row 0), the area fraction of the scored region's
    weight, the minimum intensity (depth), and the effective angular radius
    of a cap of the same area. Components below min_area_frac are dropped.
    Two merged true spots return one centroid between them, so these are
    descriptive quantities (draft Section 6.3).
    """
    n_theta, n_phi = surf.shape
    w = weights(beta, n_theta, n_phi, kind=kind)
    m = spot_mask(surf, threshold) & (w > 0)
    if not m.any():
        return []
    colat, lon = grid_coordinates(n_theta, n_phi)
    lat = 90.0 - colat
    lab, n = periodic_label(m)
    out = []
    for k in range(1, n + 1):
        rs, cs = np.where(lab == k)
        ww = w[rs, cs]
        af = float(ww.sum())          # w sums to one over the scored region
        if af < min_area_frac:
            continue
        circ = (ww * np.exp(1j * np.radians(lon[cs]))).sum()
        out.append({'lat': float((lat[rs] * ww).sum() / ww.sum()),
                    'lon': float(np.degrees(np.angle(circ))),
                    'area_frac': af,
                    'depth': float(surf[rs, cs].min()),
                    'r_eff': float(np.degrees(np.arccos(
                        np.clip(1.0 - 2.0 * af, -1.0, 1.0))))})
    return sorted(out, key=lambda d: d['depth'])


def gc_separation(lat1, lon1, lat2, lon2):
    """Great-circle separation between two points, all in degrees."""
    a, b = np.radians(lat1), np.radians(lat2)
    return float(np.degrees(np.arccos(np.clip(
        np.sin(a) * np.sin(b)
        + np.cos(a) * np.cos(b) * np.cos(np.radians(lon1 - lon2)), -1, 1))))


def match_spots(true_surf, recon_surf, beta, kind='vis',
                threshold=SPOT_THRESHOLD, gate_deg=25.0, res_deg=6.0,
                min_area_frac=2e-4):
    """
    Greedy nearest-neighbour match of recovered to true spots, one row per
    matched true spot.

    Each row carries the true position, the latitude and arc-converted
    longitude errors, the great-circle separation, the contrast deficits
    (1 - minimum intensity) and area fractions on both sides, and the
    isolation flag: a true spot is isolated when every other true spot sits
    further than the sum of their effective radii plus res_deg. Matches
    beyond gate_deg are dropped, so unmatched spots are absent and the
    error distribution is conditioned on successful matching (draft Section
    6.3). The greedy order follows depth, deepest true spot first.
    """
    ts = spot_centroids(true_surf, beta, kind, threshold, min_area_frac)
    if not ts:
        return []
    rs = spot_centroids(recon_surf, beta, kind, threshold, min_area_frac)
    iso = []
    for i, a in enumerate(ts):
        gaps = [gc_separation(a['lat'], a['lon'], b['lat'], b['lon'])
                - a['r_eff'] - b['r_eff']
                for j, b in enumerate(ts) if j != i]
        iso.append(bool(len(gaps) == 0 or min(gaps) > res_deg))
    rows, used = [], set()
    for i, t in enumerate(ts):
        best, bj = gate_deg, None
        for q, r in enumerate(rs):
            if q in used:
                continue
            d = gc_separation(t['lat'], t['lon'], r['lat'], r['lon'])
            if d < best:
                best, bj = d, q
        if bj is None:
            continue
        used.add(bj)
        r = rs[bj]
        dlon = ((r['lon'] - t['lon'] + 180.0) % 360.0) - 180.0
        rows.append({'lat_true': t['lat'], 'lon_true': t['lon'],
                     'dlat': r['lat'] - t['lat'],
                     'dlon_gc': dlon * np.cos(np.radians(t['lat'])),
                     'sep': gc_separation(t['lat'], t['lon'],
                                          r['lat'], r['lon']),
                     'isolated': iso[i], 'r_eff': t['r_eff'],
                     'depth_true': 1.0 - t['depth'],
                     'depth_rec': 1.0 - r['depth'],
                     'area_true': t['area_frac'],
                     'area_rec': r['area_frac']})
    return rows


#####################
# Population summaries #
#####################

def total_variation(surf, beta=0.0, kind='full'):
    """
    Area-weighted total variation on the sphere, periodic in longitude.

    High values mean sharp edges; a reconstruction below the truth is over
    smoothed and one above it carries spurious fine structure. The render
    grid's first and last columns sample the same meridian, so the periodic
    gradient is taken over the first n_phi - 1 columns with spacing
    2 pi / (n_phi - 1); the legacy implementation padded across the
    duplicated column, which mis-scales the seam gradient by one grid step.
    """
    n_theta, n_phi = surf.shape
    colat_deg, _ = grid_coordinates(n_theta, n_phi)
    theta = np.radians(colat_deg)

    d_dtheta = np.gradient(surf, theta, axis=0)

    dphi = 2 * np.pi / (n_phi - 1)
    core = surf[:, :-1]
    padded = np.concatenate([core[:, -1:], core, core[:, :1]], axis=1)
    d_dphi_core = np.gradient(padded, axis=1)[:, 1:-1] / dphi
    d_dphi = np.concatenate([d_dphi_core, d_dphi_core[:, :1]], axis=1)

    sin_colat = np.maximum(np.sin(theta)[:, None], 1e-6)
    grad = np.sqrt(d_dtheta ** 2 + (d_dphi / sin_colat) ** 2)
    # score the periodic core only: the duplicated final column would count
    # its meridian twice and break invariance under a longitude roll
    w = weights(beta, n_theta, n_phi, kind=kind)[:, :-1]
    return float(wmean(grad[:, :-1], w))


def _equal_area_hist(values_2d, n_bins=64):
    """
    Accumulate a per-pixel weight map into equal-area (sin latitude by
    longitude) bins, area weighting each pixel, normalised to a
    probability array.
    """
    n_theta, n_phi = values_2d.shape
    colat_deg, lon_deg = grid_coordinates(n_theta, n_phi)
    sin_lat = np.cos(np.radians(colat_deg))          # sin(lat) = cos(colat)
    area = np.sin(np.radians(colat_deg))

    sin_edges = np.linspace(-1, 1, n_bins + 1)
    lon_edges = np.linspace(-180.0, 180.0, n_bins + 1)
    si = np.clip(np.searchsorted(sin_edges, sin_lat, side='right') - 1,
                 0, n_bins - 1)
    li = np.clip(np.searchsorted(lon_edges, lon_deg, side='right') - 1,
                 0, n_bins - 1)

    hist = np.zeros((n_bins, n_bins))
    for i in range(n_theta):
        np.add.at(hist[si[i]], li, area[i] * values_2d[i])
    total = hist.sum()
    return hist / total if total > 0 else hist


def spot_distribution_entropy(surf, threshold=SPOT_THRESHOLD, n_bins=64):
    """
    Shannon entropy in nats of the spot pixel positions over the sphere, in
    equal-area bins. Widely spread spots score high, concentrated ones low,
    and an immaculate surface returns nan (no spot pixels, no
    distribution).
    """
    p = _equal_area_hist(spot_mask(surf, threshold).astype(float), n_bins)
    m = p > 0
    if not m.any():
        return float('nan')
    return float(-np.sum(p[m] * np.log(p[m])))


def surface_intensity_entropy(surf, n_bins=64):
    """
    Shannon entropy in nats of the intensity distribution over the sphere,
    in equal-area bins. Raises on a negative surface, where the histogram
    is not a distribution.
    """
    if np.any(surf < 0):
        raise ValueError('surface has negative values; the intensity '
                         'entropy is not defined')
    p = _equal_area_hist(np.asarray(surf, dtype=float), n_bins)
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def hist_kl_split(values_p, values_q, split=None, bins=60, eps=1e-12):
    """
    KL(P || Q) between two scalar samples on shared bins, decomposed
    additively into the contributions left and right of a split value
    (default the median of P), with the Jensen-Shannon divergence
    alongside. P is the data population and Q the generated one, so the
    decomposition localises where the generated population misses mass.
    """
    p_raw = np.asarray(values_p, dtype=float)
    p_raw = p_raw[np.isfinite(p_raw)]
    q_raw = np.asarray(values_q, dtype=float)
    q_raw = q_raw[np.isfinite(q_raw)]

    lo = min(p_raw.min(), q_raw.min())
    hi = max(p_raw.max(), q_raw.max())
    edges = np.linspace(lo, hi, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    p = np.histogram(p_raw, bins=edges)[0].astype(float)
    q = np.histogram(q_raw, bins=edges)[0].astype(float)
    p /= p.sum()
    q /= q.sum()

    ps = p + eps
    ps /= ps.sum()
    qs = q + eps
    qs /= qs.sum()

    integrand = ps * np.log(ps / qs)
    if split is None:
        split = float(np.median(p_raw))
    left = centres < split

    m = 0.5 * (ps + qs)
    js = 0.5 * np.sum(ps * np.log(ps / m)) + 0.5 * np.sum(qs * np.log(qs / m))

    return {'kl_total': float(integrand.sum()),
            'kl_left': float(integrand[left].sum()),
            'kl_right': float(integrand[~left].sum()),
            'js': float(js), 'split': split,
            'edges': edges, 'centres': centres, 'p': p, 'q': q}


#####################
# Mode counting     #
#####################

def count_modes(p, height_frac=0.1, prominence_frac=0.05):
    """
    The number of resolved modes of a discrete posterior grid.

    A mode is a peak reaching height_frac of the maximum with a prominence
    of prominence_frac of the maximum; the grid is padded with -1 so
    boundary maxima count. The thresholds are the legacy figure's and the
    draft's fig:beta_bimodality caption states them.
    """
    from scipy.signal import find_peaks
    p = np.asarray(p, dtype=float)
    pk = float(p.max())
    padded = np.concatenate([[-1.0], p, [-1.0]])
    peaks, _ = find_peaks(padded, height=height_frac * pk,
                          prominence=prominence_frac * pk)
    return int(len(peaks))


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