"""
Interpolation module for ssEinterpolator.
"""

import numpy as np
from scipy.interpolate import splprep, splev

def interpolate_to_latent(sr, state, t):
    """Interpolate sr and state into a latent space."""
    # Normalize data
    pass

def inverse_interpolation(tck, u, sr_min, sr_max, state_min, state_max):
    """Transform latent space back to sr, state, and t."""
    pass