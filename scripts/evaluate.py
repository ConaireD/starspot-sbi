"""
Score a reconstruction against a truth with the metrics the paper reports.
This is just for verification.

Three modes:

    --truth a.npy --recon b.npy --beta 45
        two coefficient vectors or two rendered images, one inclination

    --pairs pairs.csv
        a CSV with columns truth, recon, beta_deg and optionally samples and
        recon_std, one row per surface, reporting the distribution of every
        metric across rows

    --demo
        a synthetic pair, for checking the script runs without any data

Inputs may be complex coefficient vectors of length (L+1)^2, real-packed vectors
of the same length, or rendered images of shape (n_theta, n_phi). The three are
distinguished by shape and dtype, and coefficient inputs are rendered on the
production grid before scoring.

Metrics and weightings follow docs/conventions.md section 11. Reported are the
three SSIM variants, RMSE and MAE whole-sphere and visibility-weighted, PR-AUC in
both weightings, and where the extra inputs are supplied, the CRPS and the
error-uncertainty correlation.

Usage:
    python scripts/evaluate.py --demo
    python scripts/evaluate.py --truth truth.npy --recon recon.npy --beta 45
    python scripts/evaluate.py --pairs holdout_pairs.csv --json out.json
"""

###########
# Imports #
###########

import os
import json
import argparse

import numpy as np

from starspot_sbi.indexing import n_coeffs, coeffs_to_real, real_to_coeffs
from starspot_sbi.render import N_THETA, N_PHI, render, render_coeffs
from starspot_sbi.metrics import (SPOT_THRESHOLD, scalar_metrics, spot_mask,
                                  visibility_mask, cap_boundary_lat,
                                  detection_operating_points)
from starspot_sbi.surfaces import generate_spotted_surface


#############
# Loading   #
#############

def as_image(x, n_theta=N_THETA, n_phi=N_PHI):
    """
    Interpret an array as a rendered image, rendering it first if it is a
    coefficient vector.

    A two-dimensional array is taken as already rendered. A one-dimensional
    complex array is a coefficient vector, a one-dimensional real array is the
    real packing, and both are rendered on the production grid.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x.astype(np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a vector or an image, got shape {x.shape}")

    L = int(round(np.sqrt(x.shape[0]))) - 1
    if (L + 1) ** 2 != x.shape[0]:
        raise ValueError(f"length {x.shape[0]} is not (L+1)^2 for integer L")

    if np.iscomplexobj(x):
        return render_coeffs(x, n_theta, n_phi, L=L)
    return render(x, n_theta, n_phi, L=L)


def load(path):
    """Load a .npy file, reporting the path if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


#############
# Reporting #
#############

def describe_geometry(beta_deg, n_theta=N_THETA):
    """The scoring region at this inclination, for the header of a report."""
    beta = np.radians(beta_deg)
    mask = visibility_mask(beta, n_theta)
    return {
        'beta_deg': float(beta_deg),
        'inclination_i_deg': float(90.0 - beta_deg),
        'cap_boundary_lat_deg': float(cap_boundary_lat(beta)),
        'visible_rows': int(mask.sum()),
        'n_theta': int(n_theta),
    }


def score_one(truth, recon, beta_deg, samples=None, recon_std=None,
              threshold=SPOT_THRESHOLD):
    """Every reported metric for one pair, as a flat dict."""
    beta = np.radians(beta_deg)
    out = scalar_metrics(truth, recon, beta, recon_std=recon_std,
                         samples=samples)
    out['spot_fraction_true'] = float(np.mean(spot_mask(truth, threshold)))
    out['spot_fraction_recon'] = float(np.mean(spot_mask(recon, threshold)))
    return out


def print_one(geom, scores, ops=None):
    print(f"inclination   beta = {geom['beta_deg']:.1f} deg, "
          f"i = {geom['inclination_i_deg']:.1f} deg")
    print(f"scoring region  cap boundary at latitude "
          f"{geom['cap_boundary_lat_deg']:+.1f} deg, "
          f"{geom['visible_rows']} of {geom['n_theta']} rows visible")
    print()
    width = max(len(k) for k in scores)
    for k, v in scores.items():
        print(f"  {k:<{width}}  {v:.6f}" if np.isfinite(v)
              else f"  {k:<{width}}  {v}")
    if ops is not None:
        print()
        print(f"  detection thresholds: F1 max {ops['f1_max']:.4f} at "
              f"{ops['f1_threshold']:.4f}, truth mask at "
              f"{ops['truth_threshold']:.2f}")


def print_distribution(rows):
    """Median and quartiles of every metric across a set of scored pairs."""
    keys = [k for k in rows[0] if all(k in r for r in rows)]
    print(f"{len(rows)} pairs")
    print()
    width = max(len(k) for k in keys)
    print(f"  {'metric':<{width}}  {'min':>10} {'q25':>10} {'median':>10} "
          f"{'q75':>10} {'max':>10} {'n':>6}")
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        good = vals[np.isfinite(vals)]
        if good.size == 0:
            print(f"  {k:<{width}}  {'all nan':>10}")
            continue
        q = np.percentile(good, [0, 25, 50, 75, 100])
        print(f"  {k:<{width}}  {q[0]:10.5f} {q[1]:10.5f} {q[2]:10.5f} "
              f"{q[3]:10.5f} {q[4]:10.5f} {good.size:6d}")


#############
# Modes     #
#############

def run_single(args):
    truth = as_image(load(args.truth))
    recon = as_image(load(args.recon))
    if truth.shape != recon.shape:
        raise ValueError(f"truth {truth.shape} and recon {recon.shape} differ")

    samples = load(args.samples) if args.samples else None
    recon_std = as_image(load(args.recon_std)) if args.recon_std else None

    geom = describe_geometry(args.beta, truth.shape[0])
    scores = score_one(truth, recon, args.beta, samples, recon_std,
                       args.threshold)
    ops = detection_operating_points(truth, recon, np.radians(args.beta),
                                     threshold=args.threshold)
    print_one(geom, scores, ops)
    return {'geometry': geom, 'metrics': scores, 'operating_points': ops}


def run_pairs(args):
    import csv

    rows, geoms = [], []
    with open(args.pairs) as f:
        for r in csv.DictReader(f):
            truth = as_image(load(r['truth']))
            recon = as_image(load(r['recon']))
            beta_deg = float(r['beta_deg'])
            samples = load(r['samples']) if r.get('samples') else None
            std = as_image(load(r['recon_std'])) if r.get('recon_std') else None
            rows.append(score_one(truth, recon, beta_deg, samples, std,
                                  args.threshold))
            geoms.append(beta_deg)

    if not rows:
        raise ValueError(f"{args.pairs}: no rows")
    print(f"inclinations {min(geoms):.0f} to {max(geoms):.0f} deg")
    print()
    print_distribution(rows)
    return {'n_pairs': len(rows), 'beta_deg': geoms, 'metrics': rows}


def run_demo(args):
    """
    A synthetic pair: a three-spot surface and a noisy copy of it. Checks the
    script runs end to end without any data, and gives a reader a worked example
    of the output format.
    """
    L = 20
    spots = [{'theta': np.radians(50), 'phi': 0.4, 'radius': np.radians(12),
              'contrast': 0.5},
             {'theta': np.radians(105), 'phi': -1.4, 'radius': np.radians(9),
              'contrast': 0.6},
             {'theta': np.radians(75), 'phi': 2.2, 'radius': np.radians(10),
              'contrast': 0.7}]
    s = generate_spotted_surface(L, spots, lanczos=True)
    truth = render_coeffs(s, L=L)

    rng = np.random.default_rng(0)
    recon = truth + 0.01 * rng.normal(size=truth.shape)
    samples = recon[None] + 0.01 * rng.normal(size=(64,) + truth.shape)

    geom = describe_geometry(args.beta, truth.shape[0])
    scores = score_one(truth, recon, args.beta, samples, samples.std(axis=0),
                       args.threshold)
    ops = detection_operating_points(truth, recon, np.radians(args.beta),
                                     threshold=args.threshold)
    print("demonstration: a three-spot surface at L = 20 against a noisy copy")
    print()
    print_one(geom, scores, ops)
    return {'geometry': geom, 'metrics': scores, 'operating_points': ops}


###############
# Entry point #
###############

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--truth', help='.npy: coefficients or a rendered image')
    ap.add_argument('--recon', help='.npy: coefficients or a rendered image')
    ap.add_argument('--samples', help='.npy of shape (n, n_theta, n_phi), for CRPS')
    ap.add_argument('--recon-std', help='.npy: posterior standard deviation map')
    ap.add_argument('--beta', type=float, default=45.0,
                    help='inclination in degrees, 0 equator-on, 90 pole-on')
    ap.add_argument('--pairs', help='CSV with columns truth, recon, beta_deg')
    ap.add_argument('--threshold', type=float, default=SPOT_THRESHOLD,
                    help=f'intensity below which a pixel is spotted '
                         f'(default {SPOT_THRESHOLD})')
    ap.add_argument('--json', help='write the results to this path')
    ap.add_argument('--demo', action='store_true',
                    help='score a synthetic pair, no data needed')
    args = ap.parse_args()

    if args.demo:
        result = run_demo(args)
    elif args.pairs:
        result = run_pairs(args)
    elif args.truth and args.recon:
        result = run_single(args)
    else:
        ap.error('give --truth and --recon, or --pairs, or --demo')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(result, f, indent=2, default=float)
        print()
        print(f"written to {args.json}")


if __name__ == '__main__':
    main()