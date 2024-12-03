"""
Interpolation module for ssEinterpolator.
"""

import numpy as np
from scipy.interpolate import splprep, splrep, splev

def interpolate_to_latent(sr, state, t):
    """Interpolate sr and state into a latent space."""
    # Normalize data
    pass

def interpolate_to_latent_single_along_stk(sr, state, num_of_knots=500, smothhness=0.1):
    p = np.copy(state)
    s = np.copy(sr)
    p_min = np.min(p)
    p_max = np.max(p)
    s_min = np.min(s)
    s_max = np.max(s)

    p = (p  - p_min) / (p_max - p_min)
    s = (s  - s_min) / (s_max - s_min)
    tck, u = splprep([p, s], s=smothhness, nest=num_of_knots)
    return tck, u

def interpolate_time_parametric_space(t, u):
    return splrep(t, u, s=0.1)




def inverse_interpolation(tck, u, sr_min, sr_max, state_min, state_max):
    """Transform latent space back to sr, state, and t."""
    pass

def inverse_interpolate_to_latent_single_along_stk(t_interp, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min):
    u_interp = splev(t_interp, t_to_u)
    mask = (u_interp > 0) & (u_interp < 1)
    t_interp = t_interp[mask]
    u_interp = u_interp[mask]
    return inverse_1d(u_interp, u_to_pars, state_max, state_min, sr_max, sr_min)

def inverse_1d(u, u_to_pars, par1_max, par1_min, par2_max, par2_min):
    p_interp, s_interp = splev(u, u_to_pars)
    par1 = p_interp * (par1_max - par1_min) + par1_min
    par2 = s_interp * (par2_max - par2_min) + par2_min
    return par1, par2

def build_t_interp(t_start, t_end, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min):
    t_interp = np.linspace(t_start, t_end, 1_000_000):
    

def find_slip_events(sr_interp, threshold=-7, sse_threshold=60*60):
    """Find significant slip events in the data."""
    # Find events where slip rate exceeds threshold

    sses = sr_interp > threshold
    diff = np.diff(sses.astype(int))

    starts = np.where(diff == 1)[0] + 1  # +1 to shift to the start of the sequence
    ends = np.where(diff == -1)[0] + 1
    if sses[0]:
        starts = np.insert(starts, 0, 0)

    # If sses ends with True, add n as the last end
    if sses[-1]:
        ends = np.append(ends, len(sses))

    A = np.column_stack((starts, ends))
    sses = []
    for i in range(A.shape[0]):
        # sses.append(t[A[i, 0] + np.argmax(np.abs(d[idx, A[i, 0]:A[i, 1]]))])
        sses.append(np.mean(t_interp[A[i, 0]: A[i, 1]]))
    
    return np.array(sses)