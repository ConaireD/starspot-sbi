"""
Repository for constants used in the StarSpot-SBI package.
"""

##################
# Harmonic Basis #
##################

L_MAX = 30                  # Maximum degree of spherical harmonics
N_COEFFS = (L_MAX + 1) ** 2 # = 961

################
# Observations #
################

T_OBS = 216                 # Epochs per rotation
N_INC = 20                  # inclinations per training surface

# Channel indices in moder order. Stored .npy files hold (ax, ay, phot)
PHOT, AX, AY = 0, 1, 2
FAMILIES = {
    'phot'      : [PHOT],
    'phot_ax'   : [PHOT, AX],
    'phot_ay'   : [PHOT, AY],
    'phot_axay' : [PHOT, AX, AY],
}

# Inclination grid: beta = 90 - i, integer degrees
BETA_MIN, BETA_MAX = 0.0, 90.0
N_BETA = 91

#########
# Noise #
#########

# Training range of the noise-conditioned flows and classifiers
LOG10_SIGMA_PHOT_RANGE  = (-7.0, -2.0)
LOG10_SIGMA_ASTRO_RANGE = (-7.0, -2.0)

# Reference noise for paper
LOG10_SIGMA_PHOT_REF  = -4.0
LOG10_SIGMA_ASTRO_REF = -3.1

# Levels for the SNR sweep (paper figure)
SNR_GRID = (-6.0, -5.0, -4.5, -4.0, -3.6, -3.5, -3.1, -3.0, -2.5, -2.0)

##########
# Models #
##########

LATENT_DIM = 96
VAE_TAG = 'n640000_seed101'
WEIGHTS_SUFFIX = '_filtered'

########
# Data #
########

N_TRAIN = 640_000
N_VAL   =   2_000

MIN_INTENSITY = 0.3                 # surface filter threshold

SEED_TRAIN = 20260101
SEED_HOLDOUT = 20270707
SEED_SELECTION = 42                  # default_rng permutation of the pool

##############
# Evaluation #
##############

N_DRAWS = 256
N_PAIRS = 5000

