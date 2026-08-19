"""
Tests for pr_auc_weighted and pr_auc_bands.

Assumptions these tests cannot check:
- The render grid convention (row 0 at latitude +90, columns from -180 to
  +180, both endpoint-inclusive) matches starspot_sbi.render. The band
  edges are derived from linspace here exactly as in the function, so a
  shared error in that convention would pass.
- The default band widths (10 deg latitude, 20 deg longitude) are the
  legacy figure's; nothing here checks them against the legacy notebook.

The surfaces are built by placing spots with starspot_sbi.surfaces and
rendering at the production grid, so the tests exercise the same path the
figure notebook uses.
"""

import numpy as np

from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.render import render
from starspot_sbi.surfaces import generate_spotted_surface
from starspot_sbi.metrics import (pr_auc, pr_auc_weighted, pr_auc_bands,
                                  weights, spot_mask)

L = 30


def _surface(theta_deg, phi_deg, radius_deg=12.0, contrast=0.6):
    spots = [{'theta': np.radians(theta_deg), 'phi': np.radians(phi_deg),
              'radius': np.radians(radius_deg), 'contrast': contrast}]
    return render(coeffs_to_real(generate_spotted_surface(L, spots,
                                                          lanczos=True)))


def test_pr_auc_weighted_reproduces_pr_auc():
    truth = _surface(90.0, 0.0)
    recon = truth + 0.02 * np.cos(np.linspace(0, 4, truth.size)
                                  ).reshape(truth.shape)
    w = weights(0.0, *truth.shape, kind='full')
    a = pr_auc(truth, recon, 0.0, 'full')
    b = pr_auc_weighted(truth, recon, w)
    print(f'pr_auc {a:.6f}, pr_auc_weighted {b:.6f}')
    assert np.isclose(a, b, rtol=1e-12)


def test_bands_localise_an_equatorial_spot_in_latitude():
    truth = _surface(90.0, 0.0)
    centres, vals = pr_auc_bands(truth, truth, axis='lat')
    hit = np.isfinite(vals)
    print(f'finite latitude bands: {centres[hit]}')
    assert hit.any()
    # every band holding truth pixels straddles the spot's latitude range
    assert np.abs(centres[hit]).max() <= 25.0
    # Self-comparison is not exactly one: the sweep cuts a trapezoid corner
    # near recall 1 (conventions.md section 11), and a band that clips the
    # spot at its edge holds mostly near-threshold pixels, where the corner
    # is largest. Measured: 0.870 at the edge bands, 0.981 at the centre.
    assert vals[hit].min() > 0.85
    centre_bands = np.abs(centres[hit]) < 10.0
    assert vals[hit][centre_bands].min() > 0.97


def test_bands_localise_a_spot_in_longitude():
    truth = _surface(90.0, 90.0)
    centres, vals = pr_auc_bands(truth, truth, axis='lon')
    hit = np.isfinite(vals)
    print(f'finite longitude bands: {centres[hit]}, values {vals[hit]}')
    assert hit.any()
    assert np.abs(centres[hit] - 90.0).max() <= 35.0
    # See the latitude test for why edge bands sit below one.
    assert vals[hit].min() > 0.85
    centre_bands = np.abs(centres[hit] - 90.0) < 15.0
    assert vals[hit][centre_bands].min() > 0.97


def test_an_empty_truth_band_is_nan_and_a_missed_spot_scores_low():
    truth = _surface(30.0, 0.0)           # colatitude 30, latitude +60
    flat = np.ones_like(truth)
    centres, vals = pr_auc_bands(truth, flat, axis='lat')
    southern = centres < -30.0
    print(f'southern-band values: {vals[southern]}')
    assert np.all(np.isnan(vals[southern]))
    hit = np.isfinite(vals)
    # the reconstruction never fires, so scored bands return 0.0
    assert np.allclose(vals[hit], 0.0)


def test_band_weights_are_area_weights():
    truth = _surface(90.0, 0.0)
    w = weights(0.0, *truth.shape, kind='full')
    # one all-sphere latitude band reproduces the full-sphere pr_auc
    recon = np.roll(truth, 3, axis=1)
    centres, vals = pr_auc_bands(truth, recon, axis='lat', band_width=180.0)
    full = pr_auc_weighted(truth, recon, w)
    print(f'single-band {vals[0]:.6f}, full-sphere {full:.6f}')
    assert np.isclose(vals[0], full, rtol=1e-12)
