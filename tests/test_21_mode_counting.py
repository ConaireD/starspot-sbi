"""
Tests for count_modes.

Assumptions these tests cannot check:
- The 10 per cent height and 5 per cent prominence thresholds match the
  legacy figure cell; the values here exercise the rule, not its
  provenance.
- The -1 padding convention (boundary maxima count as modes) matches how
  the classifier grids behave at beta = 0 and 90; asserted on constructed
  grids only.
"""

import numpy as np

from starspot_sbi.metrics import count_modes


def _gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def test_a_single_peak_counts_one():
    x = np.arange(91.0)
    assert count_modes(_gauss(x, 45, 4)) == 1


def test_two_separated_peaks_count_two():
    x = np.arange(91.0)
    p = _gauss(x, 25, 3) + 0.8 * _gauss(x, 65, 3)
    assert count_modes(p) == 2


def test_a_boundary_maximum_counts():
    x = np.arange(91.0)
    p = _gauss(x, 0, 4)
    assert count_modes(p) == 1
    p2 = _gauss(x, 90, 4) + 0.5 * _gauss(x, 30, 3)
    assert count_modes(p2) == 2


def test_a_shoulder_below_the_prominence_threshold_does_not_count():
    x = np.arange(91.0)
    main = _gauss(x, 45, 4)
    bump = 0.02 * _gauss(x, 70, 2)          # 2 per cent of the peak
    n = count_modes(main + bump)
    print(f'modes with a 2 per cent shoulder: {n}')
    assert n == 1


def test_a_peak_below_the_height_threshold_does_not_count():
    x = np.arange(91.0)
    p = _gauss(x, 45, 3) + 0.05 * _gauss(x, 80, 2)
    assert count_modes(p) == 1
    p2 = _gauss(x, 45, 3) + 0.2 * _gauss(x, 80, 2)
    assert count_modes(p2) == 2
