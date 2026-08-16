# Conventions

Definitions the code assumes everywhere. Anything reimplementing this
pipeline has to match these exactly.

## Coefficient ordering

A degree-L map is a length (L+1)^2 vector. The flat index is

    idx(l, m) = l^2 + l + m

so l ascends and, within each l, m runs from -l to +l. l=0 is index 0.
`starspot_sbi.indexing.lm_to_idx` and `idx_to_lm` are the canonical
implementation; nothing should reimplement this arithmetic inline.

## Real and complex representations

Surfaces are stored on disk as complex vectors satisfying the reality
condition

    s_l^{-m} = (-1)^m conj(s_l^m)

with s_l^0 real. The (-1)^m factor is the Condon-Shortley phase.

The VAE consumes the equivalent real vector, three contiguous blocks:

    [ s_l^0 for l = 0..L | Re s_l^m for m>0 | Im s_l^m for m>0 ]

ordered by l then ascending m within each block. This has
(L+1) + 2 * L(L+1)/2 = (L+1)^2 entries, so the length is unchanged.

`coeffs_to_real` drops the m<0 half; `real_to_coeffs` rebuilds it from the
reality condition. real -> complex -> real is exact for any real vector.
complex -> real -> complex is exact only for inputs that already satisfy the
reality condition, which every stored surface does (verified: max residual
0.0 on the L30_N216 dataset).

## Spherical harmonic normalisation

Orthonormal, with the Condon-Shortley phase, i.e.

    Y_l^m(theta, phi) = N_l^m P_l^{|m|}(cos theta) exp(i m phi),
    N_l^m = sqrt( (2l+1)/(4 pi) * (l-|m|)! / (l+|m|)! )

The practical consequence, and the fastest way to check an external
implementation matches: for a map of uniform intensity I,

    s_0^0 = 2 sqrt(pi) I

Confirmed on the stored dataset, where s_0^0 = 3.507 for a surface of mean
intensity 0.989.

## Angles and geometry

TODO — write when design.py lands. Needs: definition of beta (inclination),
theta/phi convention (colatitude vs latitude), rotation sense and phase
origin, plane-of-sky x/y axis definitions.

## Observables

TODO — write when kernels.py lands. Needs: the three kernels and their
geometric weights, the parity selection rules, and the fact that F_0 = 1 so
sigma_phot is a fractional flux precision.

## Channel and array layout

TODO — write when the dataset code lands. Needs: channel order in the
(n_inc, 3, N_OBS) arrays, and the beta grid.