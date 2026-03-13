"""Visualization module for ssEinterpolator."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.interpolate import splev


def plot_depth(depth: float, axs: list[Axes]) -> None:
    """Plot original vs. spline-reconstructed slip rate and state at a given depth.

    Compares raw simulation data against the B-spline latent representation for
    a single along-fault depth, drawing three subplots: state-vs-SR phase space,
    state vs. time, and log slip rate vs. time.

    Args:
        depth: Along-fault depth value to plot. The nearest index in ``lf`` is used.
        axs: List of three Matplotlib Axes objects for the three subplots:
            [phase space, state vs. time, slip rate vs. time].
    """
    idx = np.argmin(np.abs(lf - depth))
    mask = (data['t'] > 80) & (data['t'] < 110)
    sr = np.log10(np.abs(data['sr'][idx][mask]))
    state = np.copy(data['state'][idx][mask])
    t = np.copy(data['t'][mask])
    tck, u = interpolate_to_latent_single_along_stk(sr, state)
    p_interp, s_interp = splev(u, tck)
    state_interp = p_interp * (state.max() - state.min()) + state.min()
    sr_interp = s_interp * (sr.max() - sr.min()) + sr.min()
    ax = axs[0]
    ax.scatter(state, sr, label='data', s=3)
    ax.scatter(state_interp, sr_interp, s=1, label='spline', zorder=10)
    ax.set_ylabel('log10(slip rate)')
    ax.set_xlabel('state')
    ax.legend(loc='best')
    ax = axs[1]
    ax.plot(t, state)
    ax.plot(t, state_interp)
    ax.set_ylabel('state')
    ax.set_xlabel('time')
    ax = axs[2]
    ax.plot(t, sr)
    ax.plot(t, sr_interp)
    ax.set_ylabel('log10(slip rate)')
    ax.set_xlabel('time')
