"""
ssEinterpolator package for interpolating numerical Slow Slip simulations.
"""

__version__ = "0.1.0"

from .interpolation import interpolate_to_latent, inverse_interpolation
from .io import load_simulation_data
from .visualization import plot_results