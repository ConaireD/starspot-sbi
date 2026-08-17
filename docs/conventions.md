# Conventions
Conventions this package relies on, and the test that checks it. 
The derivations are in `notebooks/Mathematics.ipynb`; tags of the form S3, R4,
K7, D5, C4 refer to its equations. 

---

## 1. Body frame and angles

The star's spin axis defines `+z`, pointing along the angular momentum vector by
the right-hand rule, so `+z` is the north rotational pole. 

| Quantity | Definition |
|---|---|
| colatitude `theta` | measured from `+z`, range `[0, pi]` |
| azimuth `phi` | measured from `+x` towards `+y`, range `(-pi, pi]` |
| latitude `lambda` | `90 deg - theta`, so latitude is `+90 deg` at the north pole |
| direction | `r_hat = (sin theta cos phi, sin theta sin phi, cos theta)` |

The stellar inclination `i` is the angle between the spin axis and the line of
sight, so `i = 90 deg` is equator-on and `i = 0 deg` is pole-on. This package
parameterises geometry by

    beta = 90 deg - i

so that `beta = 0` is equator-on and `beta = 90 deg` is pole-on. `beta` ranges
over `[0, 90 deg]`. The isotropic prior over orientations is uniform in `cos i`,
which is `p(beta) ~ cos beta`.

---

## 2. Inclination

The observer sits at latitude `-beta`. The permanently unobservable region is the
north cap `theta < beta`. At `beta = 90 deg` the observer sees the south pole.

The observer direction and sky axes in body coordinates are (D5, D7)

    n_hat(t) = ( cos beta cos wt, -cos beta sin wt, -sin beta)
    e_y(t)   = ( sin beta cos wt, -sin beta sin wt,  cos beta)     projected spin axis, north up
    e_x(t)   = ( sin wt,           cos wt,           0        ) = e_y x n_hat

A point at colatitude `theta` satisfies `max over phase of n_hat . r_hat =
sin(theta - beta)`, hence it is visible at some phase if and only if
`theta > beta`. The south cap `theta > pi - beta` is always visible, and the cap
boundary lies at latitude `+(90 deg - beta)`.

Checked by `test_design_matrix.py::test_pole_on_flux_is_constant` and
`::test_pole_on_centroid_rotates_rigidly`, and in `Mathematics.ipynb` section 4
against a pixel-space operator built from `n_hat` alone. 

---

## 3. Spherical harmonics

Orthonormal complex harmonics with the Condon-Shortley phase, as implemented by
`scipy.special.sph_harm_y`, `lpmv` and `sympy.assoc_legendre` (S2).

    Y_l^m = N_l^m P_l^m(cos theta) e^{i m phi},   m >= 0
    Y_l^-m = (-1)^m conj(Y_l^m)
    N_l^m = sqrt( (2l+1)/(4 pi) * (l-m)!/(l+m)! )

In this basis `Y_1^1(pi/2, 0) = -sqrt(3/(8 pi)) = -0.345494`. A basis without the
Condon-Shortley phase gives `+0.345494`.

On the meridian `phi = 0` the harmonics are real, and for negative order (S6)

    Y_l^m(theta, 0) = (-1)^{|m|} N_l^{|m|} P_l^{|m|}(cos theta),   m < 0

which is the factor `precompute_kernels_exact` carries as `cs_phase` and
`place_spot` carries in its `sign` array.

A uniform map of intensity `I` has (S5)

    s_0^0 = 2 sqrt(pi) I,   every other coefficient zero

---

## 4. Indexing and packing

Flat index (S3)

    idx(l, m) = l^2 + l + m

so a degree-L map is a vector of length `(L+1)^2`, ordered by degree and, within
each degree, by order from `-l` to `+l`.

Reality of the surface requires (S4)

    s_l^{-m} = (-1)^m conj(s_l^m),   hence s_l^0 real

The stored real vector drops the redundant half (S5)

    s_real = [ s_l^0 | Re s_l^{m>0} | Im s_l^{m>0} ]

each block ordered by `l` then `m`, again `(L+1)^2` real numbers. The first `L+1`
entries are therefore the `m = 0` coefficients in ascending `l`. 

Note that `coeffs_to_real` never reads the `m < 0` half of the array, so a round
trip through both functions cannot detect a wrong sign there.

`L` is never a module-level global. It is inferred from array length where the
length determines it, and passed explicitly otherwise.

---

## 5. Measurement kernels

Computed once in a reference geometry: observer along `+x`, equator-on, `t = 0`,
sky axes `e_x = y_hat` and `e_y = z_hat`. Visible hemisphere
`phi in (-pi/2, pi/2)`.

    V        = (1/pi) sin theta cos phi = (1/pi) max(x_hat . r_hat, 0)
    x_obs    = sin theta sin phi
    y_obs    = cos theta

`V` carries foreshortening and no limb darkening. The `1/pi` normalises a uniform
unit-intensity star to unit flux.

    k^h_{lm} = integral rho^h V Y_l^m dOmega,    rho^phot = 1, rho^x = x_obs, rho^y = y_obs

with no conjugate on `Y`, so the forward model contracts with a plain dot product
rather than a sesquilinear inner product (K2). This matters when moving a
rotation onto the kernel; see section 7.

The azimuthal integral for the `x` channel is signed at `m = +-2` (K4):

    I_phi_x(+2) = +i pi/4
    I_phi_x(-2) = -i pi/4

### Selection rules (K7)

| channel | non-zero `(l, m)` |
|---|---|
| x    | `l` odd and `m` odd, or `(l, abs(m)) = (2, 2)` |
| y    | `l` odd and `m` even, or `(l, abs(m)) = (2, 1)` |
| phot | `l` even and `m` even, or `(l, abs(m)) = (1, 1)` |

Photometry is blind to every odd `l >= 3` (Russell 1906); astrometry to every
even `l >= 4`. Inclination mixes orders within a degree and never changes which
degrees are visible.

These rules assume no limb darkening and do not survive it. Only used as a test for integral correctness.

---

## 6. Quadrature and the accuracy floor

`_GL_N = 500` Gauss-Legendre nodes. This value is the default for the paper,
so changing it produces a new dataset.

Gauss-Legendre with `N` nodes integrates polynomials of degree `<= 2N-1` exactly.
The `x`-channel integrand is a polynomial, so `kx` is exact to roundoff. The `y`
and photometric integrands carry `sqrt(1-u^2)`, which is not analytic at
`u = +-1`, hence the quadrature converges algebraically as `N^-3`.

Measured against the sympy path at `L = 8`, relative:

| kernel | error |
|---|---|
| kx    | 2.7e-11 |
| ky    | 4.3e-07 |
| kphot | 8.6e-07 |

Those are maxima over `(l, m)`, dominated by high degree. At `l = 0`, where the
normalisation is set,

    F_0 = 1 + 4.2e-9

which bounds the accuracy of the whole forward model at the production setting,
five orders below the mission photometric noise of 1e-4. Raising to 2000 nodes
reduces it by a factor of 63.97 against the predicted `4^3 = 64`.

Tolerances around `F_0` are written relative to
the signal scale rather than as absolute bounds. Kernel entries that vanish
analytically are ~1e-16 at `N = 500` and ~1e-15 at `N = 2000`, because the sum
accumulates four times as many terms, so selection-rule bounds are relative to
the largest entry of each kernel.

Raising `N` improves the quadrature error and worsens the roundoff. Substituting
back to `theta`, where the square root becomes `sin theta` and the integrand is
smooth on `[0, pi]`, or using a rule built for endpoint singularities such as
tanh-sinh, is the route to better accuracy.

---

## 7. Rotations

Active z-y-z Euler angles, `R(alpha, beta, gamma) = Rz(alpha) Ry(beta) Rz(gamma)`,
applied about the body axes. A rotation acts on a function by
`(U(R) f)(r_hat) = f(R^-1 r_hat)`, and on coefficients by `s_l -> D^l(R) s_l`
(R1). The operator is block diagonal: rotation moves power between orders within
a degree and never between degrees.

    D^l_{m m'}(alpha, beta, gamma) = e^{-i m alpha} d^l_{m m'}(beta) e^{-i m' gamma}

The gamma phase attaches to the column index `m'`. Attaching it to the row index
is wrong by O(1), and the notebook measures both.

At `l = 1`, `d^1_{m=1, m'=0}(beta) = -sin(beta)/sqrt(2)`, which identifies the
row-column order and the sense: the construction describes an active rotation by
`+beta` about `+y`, carrying a point on `+z` towards `+x`.

Note that some references write the same matrix with rows `m'`, columns `m`, and
the sign `(-1)^{m'-m+s}`. Relabelling shows the two agree.

### Moving a rotation onto the kernel

The contraction in the forward model is bilinear, `a . b = sum_m a_m b_m`, hence

    (D s) . k = s . (D^T k),   with D^T(alpha, beta, gamma) = D(gamma, -beta, alpha)

The sesquilinear identity `<D s, k> = <s, D^H k>` with
`D^H(alpha, beta, gamma) = D(-gamma, -beta, -alpha)` conjugates `k`, which the
kernel definition does not, and taken literally it produces `d(-beta)` where
`d(beta)` belongs.

### Accuracy

Fast and sympy paths agree to machine precision through `l = 12`. Orthogonality
of the fast path degrades by cancellation as `beta -> pi/2`:

| `(l, beta)` | `max abs(d^T d - I)` |
|---|---|
| (30, 0.4)   | 2.2e-11 |
| (30, pi/2)  | 5.5e-07 |

`l = 30` is the production degree and `beta = pi/2` a sampled edge of the
inclination grid, so 5.5e-7 is the working accuracy of the rotation. It remains
four orders below `sigma_phot ~ 1e-4`.

The log-factorial evaluation adds `+1e-300` to the logirthms to prevent nans, so the
off-diagonal entries of `d(0)` come back as ~1e-300 rather than exactly zero.

---

## 8. The design matrix

    mu^h = Re( W_omega B^h_beta s )

    W_omega[t, m]        = exp(-i m omega t)                shape (N_obs, 2L+1)
    B^h_beta[m, l^2 + i] = (d^l(beta) k^h_l)_i              shape (2L+1, (L+1)^2)

Column `idx(l, m)` of `B` has at most one non-zero entry, in row `m`. At
`beta = 0`, `B` is the kernel laid out by `(l, m)`.

    A(beta) = vstack over channels of (W B^h_beta)          shape (K N_obs, (L+1)^2)
    y = Re(A(beta) s) + n

The imaginary part is identically zero for any `s` obeying the reality condition,
because the weights and the surface are both real. The measured residual ratio is
1e-19 to 3e-15. A large imaginary residual means a convention has changed, so the
ratio is a diagnostic.

Rotational equivariance: `s_l^m -> e^{-i m D} s_l^m` is equivalent to
`t -> t + D/omega`.

---

## 9. Channel order

| context | order |
|---|---|
| stored `.npy` signal files, per `manifest.json` | `(astro_x, astro_y, phot)` |
| model code | `(phot, astro_x, astro_y)` |

The permutation `[2, 0, 1]` converts stored to model order. Applying it incorrectly divides the astrometric channel by its own mean and scales it by the photometric gain, which produces finite and plausible-looking numbers. The permutation is applied in one function, at the single boundary between storage and model, and nowhere else.

`build_design_matrix` stacks channels in the order of the kernel list it receives, so the permutation does not appear in A. The dataset generator passes `[kx, ky, kphot]`.

---

## 10. Spot model

A circular spot is a spherical cap. Centred at the pole it is axisymmetric, so
only `m = 0` coefficients are non-zero (C1):

    s_l^0 = delta * 2 pi * sqrt((2l+1)/(4 pi)) * integral_{cos rho}^{1} P_l(u) du

with the Legendre identity
`integral_x^1 P_l du = [P_{l-1}(x) - P_{l+1}(x)]/(2l+1)` and `P_{-1} = 1`. At
`l = 0` this is `delta sqrt(pi) (1 - cos rho)`.

Moved to `(theta_s, phi_s)` by the single-column Wigner rotation (C4)

    s_l^m = e^{-i m phi_s} d^l_{m0}(theta_s) s_l^0

with (C5)

    d^l_{m0}(theta) = sqrt(4 pi / (2l+1)) Y_l^m(theta, 0)

and no `(-1)^m` for `m > 0`. Check at `l = 1`:
`d^1_{10} = -sin(theta)/sqrt(2) = sqrt(4 pi/3) Y_1^1(theta, 0)`. A spot placed at
longitude `phi_s` renders at `phi_s`.

### Lanczos taper

    s_l^0 -> s_l^0 * sinc(l / (L+1)),   with np.sinc's convention sin(pi x)/(pi x)

This suppresses Gibbs ringing from the hard cap edge, at the cost of a slightly
broadened spot.

### Positivity

TODO: CHANGE THIS SECTION WHEN MODELS ARE RERUN DURING CO-AUTHOR FEEDBACK

Nothing in the coefficient basis enforces `S >= 0`, and there is no closed-form
positivity condition on the coefficients. Overlapping spots add their deficits
and can drive the intensity negative.

Measured on the 1.28M-surface training set, rendered on a `(L+1) x (2L+2)` grid:

| statistic | value |
|---|---|
| median minimum intensity | 0.536 |
| 1st percentile | 0.190 |
| deepest minimum | -0.590 |
| surfaces below 0 | 700 (0.055 per cent) |
| surfaces below -0.05 | 373 (0.029 per cent) |

`scripts/generate_dataset.py` rejects surfaces below a configurable threshold,
default 0.0.

---

## 11. Rendering and metrics

### The render grid

    theta = linspace(0, pi, n_theta)        row 0 is theta = 0, the +z pole
    phi   = linspace(-pi, pi, n_phi)

Both axes are endpoint-inclusive, so the poles and the `phi = +-pi` meridian are
each sampled and the grid is not area-uniform. Area weighting belongs to the
metrics. Production grid `(n_theta, n_phi) = (120, 240)`. Images display with
`origin='upper'` so that row 0 appears at the top.

Row 0 is the north pole, which is the pole that hides, so a visibility mask
excludes the first rows of a rendered image.

### The visibility mask

    visible at some phase    sin(theta - beta) > VIS_TOL
    mu_max(theta)            max(sin(theta - beta), 0)
    mu_mean(theta)           phase average of max(n_hat . r_hat, 0)

`VIS_TOL = 1e-12`. Without it, at `beta = 0` the `theta = pi` row survives
because `sin(pi)` evaluates to 1.2e-16, and the visible range becomes `[1, 59]`
rather than the geometric `[1, 58]` at `n_theta = 60`.

`mu_mean` has a closed form. Writing `n_hat . r_hat = a cos(psi) + b` with
`a = sin(theta) cos(beta)` and `b = -cos(theta) sin(beta)`, the phase average of
`max(a cos psi + b, 0)` is `b` when `b >= a`, zero when `b <= -a`, and
`(a sin(psi0) + b psi0) / pi` with `psi0 = arccos(-b / a)` between. This agrees
with numerical phase averaging to 3e-5, which is the integration error.

`test_metrics.py` asserts the mask by render row index against the forward-model
geometry rather than by latitude label, so that a sign error in the latitude
convention cannot cancel against a sign error in the mask.

### Weightings

TODO: THE BELOW SECTIONS ARE TOO VERBOSE, SIMPLIFY

All are area weighted by `sin(theta)`.

| kind | weight |
|---|---|
| `full`  | `sin(theta)`, whole sphere |
| `vis`   | restricted to the ever-visible region |
| `wmean` | scaled by the phase-averaged foreshortening |

`vis` is the main metric. It scores the whole sphere at `beta = 0` and half of it at
`beta = 90`, so an inclination trend in a visibility-weighted metric mixes a
change in performance with a change in the region being scored. `full`
is used when talking about the VAE

### SSIM

`L_SSIM = 1`, rather than the dynamic range estimated per image: the photosphere
is normalised to unit intensity so 1 is the physical scale, and a per-image
estimate makes scores incomparable between quiet and spotted stars. Window 7,
`k1 = 0.01`, `k2 = 0.03`.

The window filter is separable, pole-correct along theta and periodic along phi.
The virtual row across a pole is the reflected interior row rolled by half the
phi grid, since `phi` and `phi + pi` meet at the pole; the pole row itself is
excluded from the reflection, being its own image.

### Spot detection

The truth mask is intensity below 0.9. The reconstruction is swept over
thresholds rather than thresholded once, so a systematic contrast bias is not
charged twice. IoU and recovered spot counts are not reported for that reason: a
reconstruction whose spots are correctly placed but too shallow shrinks under a
fixed threshold and is scored as a detection failure.

`pr_auc` returns nan when the truth mask is empty over the weighted region, and
0.0 when the truth mask is non-empty but the reconstruction never fires. The
distinction matters for medians over a holdout, where nan means nothing to detect
and zero means detected nothing.

A perfect reconstruction scores about 0.996 rather than 1 at the default 200
thresholds, because the truth threshold falls between two swept values and the
trapezoid cuts a corner near recall 1. Denser sweeps approach 1.

### CRPS

    CRPS = E|Y - x| - 0.5 E|Y - Y'|

The sharpness term uses the exact sorted form,
`0.5 E|Y - Y'| = (1/n^2) sum_i (2i - n - 1) y_(i)`, which is exact for the sample
and deterministic.

---

## 12. Dataset

| quantity | value |
|---|---|
| `L_MAX` | 30, so 961 coefficients |
| `N_OBS` | 216 epochs per rotation |
| `P_ROT` | 1.0, `omega = 2 pi` |
| `t_obs` | `linspace(0, P_ROT, 216, endpoint=False)`, so `t[0] = 0` exactly |
| inclinations | 91 integer degrees, 0 to 90 inclusive |
| per surface | 20 inclinations drawn without replacement |
| signal dtype | float32 |
| signal shape | `(n_inc, 3, N_OBS)` |

Spot prior:

| parameter | distribution |
|---|---|
| number of spots | integer, 1 to 11 inclusive |
| latitude | sin-uniform on (-90, 90) deg, isotropic |
| longitude | uniform on (0, 360) deg |
| radius | uniform on (6, 12) deg |
| contrast | uniform on (0.5, 0.9), background is 1 |
| taper | Lanczos, always on |

Contrast is the spot intensity rather than the deficit, so spots are always
darker than the background and never black.

Seeding is `SeedSequence([split_seed, surface_idx])`, so surface `i` is
reproducible at any chunk size and after any resume. Training seed 20260101,
holdout seed 20270707. A holdout seed is never reused by a training split.

Both `metadata.csv` files hold more rows than surfaces. The training split has
2,084,000 rows over 1,280,000 distinct `surface_idx`, and the holdout has
170,000 rows over 100,000. Because a surface depends only on its index, repeated
rows are byte-identical and the `.npy` files they name are complete and correct.
A reader must call `drop_duplicates('surface_idx', keep='first')` before counting
rows or aggregating over them, or the repeated surfaces will carry twice the
weight of the rest.

---

## 13. Numbers that identify each convention

| Convention | Number | Checked in |
|---|---|---|
| Condon-Shortley basis | `Y_1^1(pi/2, 0) = -0.345494` | `Mathematics.ipynb` section 1 |
| coefficient normalisation | `s_0^0 = 2 sqrt(pi)` for unit intensity | section 1 |
| real packing | first `L+1` real entries are `s_l^0` | `test_indexing.py` |
| Wigner-d sense and index order | `d^1_{10}(beta) = -sin(beta)/sqrt(2)` | `test_wigner.py`, section 2 |
| Euler phase attachment | `D_{mm'}` carries `e^{-i m' gamma}` on the column | `test_wigner.py`, section 2 |
| `I_phi_x` sign | `+i pi/4` at `m = +2`, `-i pi/4` at `m = -2` | `test_phi_integrals.py`, section 3 |
| flux normalisation | `F_0 = 1 + 4.2e-9` | `test_design_matrix.py`, section 4 |
| inclination | pole-on flux constant; centroid track circular | `test_design_matrix.py`, section 4 |
| time phase | `W[t, m] = exp(-i m omega t)` | `test_design_matrix.py` |
| spot longitude | a spot at `phi_s` renders at `phi_s` | `test_surfaces.py` |
| `d_{m0}` identity | no `(-1)^m` for `m > 0` | `test_surfaces.py` |
| render row 0 | a spot at the `+z` pole darkens row 0 | `test_render.py` |
| visibility mask | visible rows [30, 59] at `beta = 90`, `n_theta = 60` | `test_metrics.py` |
| SSIM pole continuation | a longitude-constant field scores 1 at the pole rows | `test_metrics.py` |