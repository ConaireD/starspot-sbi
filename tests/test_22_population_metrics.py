"""
Tests for total_variation, the two entropies, and hist_kl_split.

Assumptions these tests cannot check:
- The published figure's values used the legacy NewMetrics implementations
  with a median-relative spot threshold; the package uses the absolute 0.9
  threshold of every other detection metric, and the seam gradient uses
  the package grid spacing 2 pi / (n_phi - 1). Both departures are stated
  in the metric docstrings; nothing here compares against legacy caches.
- The equal-area binning at 64 bins is the legacy resolution; the entropy
  values depend on it and it is asserted only through limiting cases.
"""

import numpy as np

from starspot_sbi.indexing import coeffs_to_real
from starspot_sbi.render import render
from starspot_sbi.surfaces import generate_spotted_surface
from starspot_sbi.metrics import (total_variation, spot_distribution_entropy,
                                  surface_intensity_entropy, hist_kl_split)

L = 30


def _surface(spots):
    return render(coeffs_to_real(generate_spotted_surface(
        L, [{'theta': np.radians(90.0 - la), 'phi': np.radians(lo),
             'radius': np.radians(r), 'contrast': c}
            for la, lo, r, c in spots], lanczos=True)))


def test_total_variation_orders_smooth_and_structured_surfaces():
    flat = np.ones((120, 240))
    one = _surface([(10.0, 30.0, 10.0, 0.6)])
    three = _surface([(10.0, 30.0, 10.0, 0.6), (-40.0, -100.0, 9.0, 0.55),
                      (60.0, 150.0, 8.0, 0.65)])
    tv0, tv1, tv3 = (total_variation(s) for s in (flat, one, three))
    print(f'tv flat {tv0:.2e}, one spot {tv1:.4f}, three spots {tv3:.4f}')
    assert tv0 < 1e-12
    assert tv0 < tv1 < tv3


def test_total_variation_is_invariant_under_longitude_roll():
    surf = _surface([(20.0, 170.0, 10.0, 0.6)])
    rolled = np.roll(surf[:, :-1], 60, axis=1)
    rolled = np.concatenate([rolled, rolled[:, :1]], axis=1)
    a, b = total_variation(surf), total_variation(rolled)
    print(f'tv {a:.5f} against rolled {b:.5f}')
    assert abs(a - b) / a < 1e-6


def test_spot_entropy_orders_concentrated_and_spread_spots():
    tight = _surface([(0.0, 0.0, 8.0, 0.6), (5.0, 12.0, 8.0, 0.6)])
    spread = _surface([(0.0, 0.0, 8.0, 0.6), (-50.0, 140.0, 8.0, 0.6)])
    e_t = spot_distribution_entropy(tight)
    e_s = spot_distribution_entropy(spread)
    print(f'entropy tight {e_t:.3f}, spread {e_s:.3f}')
    assert e_t < e_s
    assert np.isnan(spot_distribution_entropy(np.ones((120, 240))))


def test_intensity_entropy_raises_on_a_negative_surface():
    import pytest
    with pytest.raises(ValueError):
        surface_intensity_entropy(np.full((120, 240), -0.1))
    e = surface_intensity_entropy(np.ones((120, 240)))
    # the 120-row grid cannot fill 64 sin-latitude bins evenly, so the
    # uniform surface sits a little below the log(64^2) = 8.318 maximum
    print(f'uniform-surface entropy {e:.3f} against the maximum '
          f'{np.log(64 ** 2):.3f}')
    assert 8.2 < e <= np.log(64 ** 2) + 1e-9


def test_hist_kl_split_vanishes_for_identical_samples_and_localises_a_deficit():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    r = hist_kl_split(a, a.copy())
    print(f"identical: kl {r['kl_total']:.2e}, js {r['js']:.2e}")
    assert r['kl_total'] < 1e-6
    assert abs(r['kl_total'] - r['kl_left'] - r['kl_right']) < 1e-12
    # q missing the right tail: the deficit lands on the right of the split
    b = a[a < 1.0]
    r2 = hist_kl_split(a, b, split=0.0)
    print(f"truncated: kl {r2['kl_total']:.3f}, left {r2['kl_left']:.3f}, "
          f"right {r2['kl_right']:.3f}")
    assert r2['kl_right'] > 5 * max(r2['kl_left'], 1e-9)
