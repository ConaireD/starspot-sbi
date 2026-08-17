"""
Tests for render.py.

The grid convention is what these guard. Row 0 is theta = 0, the +z pole, which
under the package inclination convention is the pole that hides. A mask built on
the wrong end of the theta axis is silent in every aggregate statistic and wrong
in every latitude-resolved one, which is how three files in the older tree came
to carry a mirrored visibility mask.

Tolerances are set with headroom and printed where the value is informative.
"""

import numpy as np
import pytest
from scipy.special import sph_harm_y

from starspot_sbi.indexing import (lm_indices, lm_to_idx, n_coeffs,
                                   coeffs_to_real, real_to_coeffs)
from starspot_sbi.surfaces import place_spot, generate_spotted_surface
from starspot_sbi.render import (
    N_THETA, N_PHI,
    build_Ylm_matrix,
    get_Ylm,
    render,
    render_coeffs,
    render_normed,
    grid_coordinates,
)

L = 8
NT, NP = 60, 120        # small grid for speed; production is N_THETA, N_PHI


############
# The grid #
############

def test_production_grid_is_120_by_240():
    assert (N_THETA, N_PHI) == (120, 240)


def test_basis_shape_and_columns():
    Y = build_Ylm_matrix(L, NT, NP)
    assert Y.shape == (NT * NP, n_coeffs(L))

    theta = np.linspace(0, np.pi, NT)
    phi = np.linspace(-np.pi, np.pi, NP)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    for l, m in [(0, 0), (1, -1), (3, 2), (8, -5), (8, 8)]:
        col = Y[:, lm_to_idx(l, m)]
        assert np.allclose(col, sph_harm_y(l, m, TH.ravel(), PH.ravel()))


def test_grid_coordinates_endpoints():
    """
    Row 0 is the north pole and row -1 the south, both axes endpoint-inclusive.
    """
    colat, lon = grid_coordinates(NT, NP)
    assert colat.shape == (NT,)
    assert lon.shape == (NP,)
    assert colat[0] == pytest.approx(0.0)
    assert colat[-1] == pytest.approx(180.0)
    assert lon[0] == pytest.approx(-180.0)
    assert lon[-1] == pytest.approx(180.0)
    print(f"row 0 colatitude {colat[0]:.1f} deg = latitude {90 - colat[0]:+.1f}; "
          f"row -1 = latitude {90 - colat[-1]:+.1f}")


def test_grid_coordinates_defaults_match_module():
    colat, lon = grid_coordinates()
    assert (colat.size, lon.size) == (N_THETA, N_PHI)


def test_pole_rows_are_constant_in_longitude():
    """theta = 0 and theta = pi are single physical points, sampled n_phi times."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=n_coeffs(L))
    img = render(v, NT, NP)
    assert np.ptp(img[0]) < 1e-10
    assert np.ptp(img[-1]) < 1e-10


def test_cache_returns_the_same_object():
    a = get_Ylm(L, NT, NP)
    b = get_Ylm(L, NT, NP)
    assert a is b


def test_cached_basis_is_read_only():
    """The cache shares one array between callers, so writes must raise."""
    Y = get_Ylm(L, NT, NP)
    assert not Y.flags.writeable
    with pytest.raises(ValueError):
        Y[0, 0] = 0.0


def test_cache_distinguishes_grid_sizes():
    assert get_Ylm(L, NT, NP) is not get_Ylm(L, NT // 2, NP)
    assert get_Ylm(L, NT, NP).shape != get_Ylm(L - 1, NT, NP).shape


##############
# Rendering  #
##############

def test_uniform_map_renders_flat():
    """s_0^0 = 2 sqrt(pi) I gives a constant image at intensity I."""
    for I in [0.5, 1.0, 1.7]:
        s = np.zeros(n_coeffs(L), dtype=complex)
        s[0] = 2 * np.sqrt(np.pi) * I
        img = render(coeffs_to_real(s), NT, NP)
        assert np.max(np.abs(img - I)) < 1e-12


def test_render_matches_direct_sum():
    rng = np.random.default_rng(1)
    v = rng.normal(size=n_coeffs(L))
    s = real_to_coeffs(v)
    img = render(v, 20, 40)

    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(-np.pi, np.pi, 40)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    direct = np.real(sum(s[lm_to_idx(l, m)] * sph_harm_y(l, m, TH, PH)
                         for l, m in lm_indices(L)))
    err = np.max(np.abs(img - direct))
    print(f"render vs direct sum: {err:.2e}")
    assert err < 1e-12


def test_render_coeffs_agrees_with_render():
    """The complex-input path and the real-packed path give the same image."""
    rng = np.random.default_rng(2)
    v = rng.normal(size=n_coeffs(L))
    s = real_to_coeffs(v)
    assert np.max(np.abs(render(v, NT, NP) - render_coeffs(s, NT, NP))) < 1e-12


def test_render_infers_L_from_length():
    for LL in [4, 6, 8]:
        v = np.zeros(n_coeffs(LL))
        v[0] = 2 * np.sqrt(np.pi)
        assert render(v, 20, 40).shape == (20, 40)


def test_render_is_real_for_a_real_map():
    """The imaginary part discarded by render is machine noise."""
    rng = np.random.default_rng(3)
    s = real_to_coeffs(rng.normal(size=n_coeffs(L)))
    Y = get_Ylm(L, NT, NP)
    full = Y @ s
    ratio = np.max(np.abs(full.imag)) / np.max(np.abs(full.real))
    print(f"max|Im| / max|Re| = {ratio:.2e}")
    assert ratio < 1e-12


def test_render_is_linear():
    rng = np.random.default_rng(4)
    a = rng.normal(size=n_coeffs(L))
    b = rng.normal(size=n_coeffs(L))
    lhs = render(2.3 * a - 0.4 * b, NT, NP)
    rhs = 2.3 * render(a, NT, NP) - 0.4 * render(b, NT, NP)
    assert np.max(np.abs(lhs - rhs)) < 1e-12


##########################
# Spot position on grid  #
##########################

@pytest.mark.parametrize('lat_deg,lon_deg', [
    (60, 45), (-60, 45), (0, 0), (30, -120), (-45, 170),
])
def test_spot_renders_at_its_latitude_and_longitude(lat_deg, lon_deg):
    """
    A dark spot at (lat, lon) puts the image minimum at that pixel. This ties
    the render grid to the placement convention: a flipped theta axis or a
    shifted longitude origin fails by a sign or by pi.
    """
    LL = 20
    nth, nph = 180, 360
    s = place_spot(LL, np.deg2rad(12), -0.5,
                   np.radians(90 - lat_deg), np.radians(lon_deg))
    img = render_coeffs(s, nth, nph, L=LL)

    i, j = np.unravel_index(np.argmin(img), img.shape)
    colat, lon = grid_coordinates(nth, nph)
    lat_found = 90 - colat[i]
    dlon = (lon[j] - lon_deg + 180) % 360 - 180
    print(f"injected ({lat_deg:+.0f}, {lon_deg:+.0f}) -> "
          f"found ({lat_found:+.1f}, {lon[j]:+.1f}), dlon {dlon:+.1f}")
    assert abs(lat_found - lat_deg) < 2.0
    assert abs(dlon) < 2.0


def test_north_and_south_spots_land_on_opposite_halves():
    """
    A spot at +60 lands in the first half of the theta axis, one at -60 in the
    second. Guards against a reversed theta axis, which every statistic
    symmetric in latitude would miss.
    """
    LL = 20
    nth, nph = 120, 240
    for lat_deg, expect_first_half in [(60, True), (-60, False)]:
        s = place_spot(LL, np.deg2rad(12), -0.5, np.radians(90 - lat_deg), 0.0)
        img = render_coeffs(s, nth, nph, L=LL)
        i, _ = np.unravel_index(np.argmin(img), img.shape)
        print(f"spot at latitude {lat_deg:+d} -> row {i} of {nth}")
        assert (i < nth // 2) == expect_first_half


def test_spot_at_the_north_pole_darkens_row_zero():
    """
    The most direct statement of the row-0 convention: a spot centred on the
    +z pole darkens the first rows and leaves the last ones alone.

    The row-mean minimum sits a few rows in rather than at row 0, because a
    hard-edged cap truncated at finite L overshoots just inside its edge. The
    taper removes that, so this is tested both ways.
    """
    LL = 20
    for lanczos in (False, True):
        s = place_spot(LL, np.deg2rad(15), -0.5, 0.0, 0.0, lanczos=lanczos)
        prof = render_coeffs(s, 120, 240, L=LL).mean(axis=1)
        print(f"lanczos={lanczos}: row 0 {prof[0]:+.4f}, "
              f"row -1 {prof[-1]:+.4f}, argmin row {np.argmin(prof)}")
        assert prof[0] < -0.4            # north pole is inside the spot
        assert abs(prof[-1]) < 0.05      # south pole is untouched
        assert np.argmin(prof) < 10      # the dark region is at the top

####################
# render_normed    #
####################

def test_render_normed_include_dc_inverts_standardisation():
    """
    include_dc=True: the vector holds all (L+1)^2 entries and dc_value is
    ignored. This is the setting the canonical checkpoint was trained under.
    """
    rng = np.random.default_rng(5)
    spots = [{'theta': 1.0, 'phi': 0.4, 'radius': np.deg2rad(10), 'contrast': 0.6}]
    s = generate_spotted_surface(L, spots)
    vec = coeffs_to_real(s)

    mu_data = rng.normal(size=n_coeffs(L)) * 0.1
    std_data = 1.0 + rng.uniform(size=n_coeffs(L)) * 0.5
    vec_std = (vec - mu_data) / std_data

    direct = render(vec, NT, NP)
    through = render_normed(vec_std, mu_data, std_data, None,
                            include_dc=True, n_theta=NT, n_phi=NP)
    err = np.max(np.abs(direct - through))
    print(f"include_dc=True round trip: {err:.2e}")
    assert err < 1e-10


def test_render_normed_dc_value_ignored_when_include_dc():
    rng = np.random.default_rng(6)
    vec = rng.normal(size=n_coeffs(L))
    mu, sd = np.zeros(n_coeffs(L)), np.ones(n_coeffs(L))
    a = render_normed(vec, mu, sd, None, include_dc=True, n_theta=NT, n_phi=NP)
    b = render_normed(vec, mu, sd, 999.0, include_dc=True, n_theta=NT, n_phi=NP)
    assert np.array_equal(a, b)


def test_render_normed_excluding_dc_inverts_standardisation():
    """
    include_dc=False: the vector holds (L+1)^2 - 1 entries with the DC term
    dropped, and dc_value supplies entry 0. mu_data and std_data exclude it too.
    """
    rng = np.random.default_rng(7)
    spots = [{'theta': 1.4, 'phi': -0.7, 'radius': np.deg2rad(9), 'contrast': 0.7}]
    s = generate_spotted_surface(L, spots)
    vec = coeffs_to_real(s)

    dc_value = vec[0]
    mu_data = rng.normal(size=n_coeffs(L) - 1) * 0.1
    std_data = 1.0 + rng.uniform(size=n_coeffs(L) - 1) * 0.5
    vec_std = (vec[1:] - mu_data) / std_data

    direct = render(vec, NT, NP)
    through = render_normed(vec_std, mu_data, std_data, dc_value,
                            include_dc=False, n_theta=NT, n_phi=NP)
    err = np.max(np.abs(direct - through))
    print(f"include_dc=False round trip: {err:.2e}")
    assert err < 1e-10


def test_render_normed_dc_shifts_the_image_by_a_constant():
    """
    The DC term is the only coefficient with no angular structure, so changing
    it moves the whole image together. One unit of intensity is 2 sqrt(pi).
    """
    rng = np.random.default_rng(8)
    n = n_coeffs(L) - 1
    vec_std = rng.normal(size=n)
    mu, sd = np.zeros(n), np.ones(n)

    base = render_normed(vec_std, mu, sd, 0.0, include_dc=False,
                         n_theta=NT, n_phi=NP)
    shifted = render_normed(vec_std, mu, sd, 2 * np.sqrt(np.pi),
                            include_dc=False, n_theta=NT, n_phi=NP)
    assert np.max(np.abs((shifted - base) - 1.0)) < 1e-12


def test_render_normed_does_not_mutate_input():
    n = n_coeffs(L) - 1
    vec = np.ones(n)
    original = vec.copy()
    render_normed(vec, np.zeros(n), np.ones(n), 5.0, include_dc=False,
                  n_theta=NT, n_phi=NP)
    assert np.array_equal(vec, original)


def test_render_normed_output_is_float64():
    """
    The original assembled the include_dc=False branch in float32. Nothing else
    in the pipeline drops to single precision there, so this port keeps float64.
    """
    n = n_coeffs(L) - 1
    img = render_normed(np.ones(n), np.zeros(n), np.ones(n), 1.0,
                        include_dc=False, n_theta=NT, n_phi=NP)
    assert img.dtype == np.float64