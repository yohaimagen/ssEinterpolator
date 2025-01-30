"""
Interpolation module for ssEinterpolator.
"""

import numpy as np
from scipy.interpolate import splprep, splrep, splev, make_lsq_spline

def interpolate_to_latent(sr, state, slip, t, num_of_knots=1000, num_of_t_knots=300, t_knots_placment='linspace', ratio=None):
    """Interpolate sr and state into a latent space."""
    if t_knots_placment != 'quantile' and t_knots_placment != 'linspace' and t_knots_placment != 'both':
        raise ValueError("t_knots_placment must be 'quantile' or 'linspace' or 'both'.")
    if t_knots_placment == 'both' and ratio is None:
        raise ValueError("ratio must be provided when t_knots_placment is 'both'.")
    # Normalize data
    latent = []
    
    for idx in np.arange(sr.shape[0]):
        sr_i = np.log10(np.abs(sr[idx]))
        state_i = state[idx]
        slip_i = slip[idx]
        u_to_pars, u = interpolate_to_latent_single_along_stk(sr=sr_i, state=state_i, slip=slip_i, num_of_knots=num_of_knots)
        t_to_u = interpolate_time_parametric_space(t, u, num_knots=num_of_t_knots, knots_placment=t_knots_placment, ratio=ratio)
        state_min, state_max = np.min(state_i), np.max(state_i)
        sr_min, sr_max = np.min(sr_i), np.max(sr_i)
        slip_min, slip_max = np.min(slip_i), np.max(slip_i)
        lat = np.concatenate((t_to_u.t, t_to_u.c, u_to_pars[0], u_to_pars[1][0], u_to_pars[1][1], u_to_pars[1][2], [state_min, state_max, sr_min, sr_max, slip_min, slip_max]))
        latent.append(lat)
    latent.append([t[0], t[-1]])
    return np.concatenate(latent)



def interpolate_to_latent_single_along_stk(sr, state, slip, num_of_knots=500, degree=3, iterations=5):
    """
    Interpolate sr and state into a latent space, using a single interpolation
    for all values of t.

    Parameters
    ----------
    sr : array_like
        Slip rate values.
    state : array_like
        State values.
    num_of_knots : int, optional
        Number of knots in the interpolation. Defaults to 500.


    Returns
    -------
    tck : tuple
        A tuple containing the knots and coefficients.
    u : array_like
        The values of the parameterization.
    """
    p = np.copy(state)
    s = np.copy(sr)
    c = np.copy(slip)
    p_min = np.min(p)
    p_max = np.max(p)
    s_min = np.min(s)
    s_max = np.max(s)
    c_min = np.min(c)
    c_max = np.max(c)
    p = (p  - p_min) / (p_max - p_min)
    s = (s  - s_min) / (s_max - s_min)
    if c_max == c_min:
        c = np.zeros(c.shape)
    else:
        c = (c  - c_min) / (c_max - c_min)
    
    v = ((p[1:] - p[:-1]) ** 2 + (s[1:] - s[:-1]) ** 2) ** 0.5
    v = np.concatenate(([0], v))
    v = np.cumsum(v)
    u = v / v[-1]
    
    # Initial quantile-based knots
    try:
        qn = int((num_of_knots - degree - 1) * 0.8)
        ln = num_of_knots - degree - 1 - qn
        quantiles = np.percentile(u, np.linspace(0, 100, qn))
        linspace = np.linspace(u[0], u[-1], ln)
        knots = np.sort(np.unique(np.concatenate((quantiles, linspace))))
        knots = np.concatenate(([u[0]] * degree, knots, [u[-1]] * degree))
        u_to_p = make_lsq_spline(u, p, knots, degree)
        u_to_s = make_lsq_spline(u, s, knots, degree)
        u_to_c = make_lsq_spline(u, c, knots, degree)
    except:
        knots = np.percentile(u, np.linspace(0, 100, num_of_knots - degree - 3))
        knots = np.concatenate(([u[0]] * degree, knots, [u[-1]] * degree))
        u_to_p = make_lsq_spline(u, p, knots, degree)
        u_to_s = make_lsq_spline(u, s, knots, degree)
        u_to_c = make_lsq_spline(u, c, knots, degree)

    tck = [u_to_p.t, [u_to_p.c, u_to_s.c, u_to_c.c], degree]
    return tck, u



def interpolate_time_parametric_space(t, u, num_knots, degree=3, knots_placment='linspace', ratio=None):
    """
    Interpolate time into a latent space.

    Parameters
    ----------
    t : array_like
        Time values.
    u : array_like
        Parameterization of the latent space.
    num_knots : int
        Total number of knots to use.
    degree : int, optional
        Degree of the spline. Default is 3.
    knots_placment : str, optional
        Method for knot placement: 'quantile', 'linspace', or 'both'. Default is 'linspace'.
    ratio : float, optional
        Ratio of linspace to quantile knots when knots_placment is 'both'. Required if knots_placment is 'both'.

    Returns
    -------
    tck : tuple
        A tuple containing the knots and coefficients.
    """
    if knots_placment != 'quantile' and knots_placment != 'linspace' and knots_placment != 'both':
        raise ValueError("knots_placment must be 'quantile' or 'linspace' or 'both'.")
    if knots_placment == 'both' and ratio is None:
        raise ValueError("ratio must be provided when knots_placment is 'both'.")
    t_min = t[0]
    t_max = t[-1]
    t = (t - t_min) / (t_max - t_min)
    tl = 0
    tr = 1
    if knots_placment == 'quantile':
        knots = np.percentile(t, np.linspace(0, 100, num_knots - degree - 3))
    elif knots_placment == 'linspace':
        knots = np.linspace(tl, tr, num_knots - degree - 3)
    elif knots_placment == 'both':
        num_knots_linspace = int(num_knots * ratio)
        num_knots_quantile = num_knots - num_knots_linspace
        knots_linspace = np.linspace(tl, tr, num_knots_linspace - degree - 3)
        knots_quantile = np.percentile(t, np.linspace(0, 100, num_knots_quantile ))
        knots = np.sort(np.unique(np.concatenate((knots_linspace, knots_quantile))))
        #if knots.shape[0] != num_knots - degree - 3 add knots
        # if knots.shape[0] < num_knots - degree - 3:
        #     raise ValueError("knots.shape[0] < num_knots - degree - 3")
        #     # add_knots = np.linspace(tl, tr, num_knots - knots.shape[0] - degree - 3)
        #     # knots = np.sort(np.unique(np.concatenate((knots, add_knots))))
    while knots.shape[0]  < num_knots - degree - 3:
        additional_knots_needed = num_knots - degree - 3 - knots.shape[0]
        
        # Calculate midpoints of existing knots for sparse areas
        midpoints = (knots[:-1] + knots[1:]) / 2
        
        # Ensure uniqueness and sorting
        knots = np.sort(np.unique(np.concatenate((knots, midpoints[:additional_knots_needed]))))
   
    
    assert np.all(np.diff(t) > 0), "t must be strictly increasing"
    assert np.all(np.diff(u) > 0), "u must be strictly increasing"
    assert degree < len(t) - num_knots, "Spline degree too high for the given data"

    knots = np.concatenate(([tl] * degree, knots, [tr] * degree))
    res = make_lsq_spline(t, u, knots, degree)
    return res

def inverse_interpolate_time_parametric_space(t_interp, t_to_u):
    u_interp = splev(t_interp, t_to_u)
    # mask = (u_interp > 0) & (u_interp < 1)
    # t_interp = t_interp[mask]
    # u_interp = u_interp[mask]
    return t_interp, u_interp




def inverse_interpolation(latent_vec, lf, t_to_u_knot_l, t_to_u_cof_l, u_to_par_knot_l, u_to_par_cof_l, t_interp):
    """Transform latent space back to sr, state."""

    l = t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l * 3 + 6
    reconstructed_sr = []
    reconstructed_state = []
    reconstructed_slip = []
    t_interps = []
    for idx in range(lf.shape[0]):
        sig_latent = latent_vec[l * idx: (idx + 1) * l]
        t_to_u_knots = sig_latent[:t_to_u_knot_l]
        t_to_u_cofs = sig_latent[t_to_u_knot_l:t_to_u_knot_l + t_to_u_cof_l]
        sig_t_to_u = [t_to_u_knots, t_to_u_cofs, 3]
        u_to_pars_knots = sig_latent[t_to_u_knot_l + t_to_u_cof_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l]
        u_to_state_cofs = sig_latent[t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l]
        u_to_sr_cofs = sig_latent[t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l + u_to_par_cof_l]
        u_to_slip_cofs = sig_latent[t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l + u_to_par_cof_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l + u_to_par_cof_l + u_to_par_cof_l]
        sig_u_to_pars = [u_to_pars_knots, [u_to_state_cofs, u_to_sr_cofs, u_to_slip_cofs], 3]
        state_min = sig_latent[-6]
        state_max = sig_latent[-5]
        sr_min = sig_latent[-4]
        sr_max = sig_latent[-3]
        slip_min = sig_latent[-2]
        slip_max = sig_latent[-1]
        
        t_interp1, state_interp, sr_interp, slip_interp = inverse_interpolate_to_latent_single_along_stk(t_interp, sig_t_to_u, sig_u_to_pars, state_max, state_min, sr_max, sr_min, slip_max, slip_min)
        reconstructed_sr.append(sr_interp)
        reconstructed_state.append(state_interp)
        reconstructed_slip.append(slip_interp)
        t_interps.append(t_interp1)
    reconstructed_sr = np.array(reconstructed_sr)
    reconstructed_state = np.array(reconstructed_state)
    reconstructed_slip = np.array(reconstructed_slip)
    t_min = latent_vec[-2]
    t_max = latent_vec[-1]
    t_interp = (t_interp * (t_max - t_min)) + t_min
    return reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp

def inverse_interpolate_to_latent_single_along_stk(t_interp, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min, slip_max, slip_min):
    """
    Perform an inverse interpolation from time values to slip rate and state parameters.

    Parameters
    ----------
    t_interp : array_like
        Interpolated time values.
    t_to_u : tuple
        Spline parameters for mapping time to parameter space.
    u_to_pars : tuple
        Spline parameters for mapping latent space back to parameters.
    state_max : float
        Maximum state value for scaling.
    state_min : float
        Minimum state value for scaling.
    sr_max : float
        Maximum slip rate value for scaling.
    sr_min : float
        Minimum slip rate value for scaling.

    Returns
    -------
    array_like
        Interpolated state and slip rate values.
    """
    t_interp, u_interp = inverse_interpolate_time_parametric_space(t_interp, t_to_u)
    u_to_state = [u_to_pars[0], u_to_pars[1][0], u_to_pars[2]]
    u_to_sr = [u_to_pars[0], u_to_pars[1][1], u_to_pars[2]]
    u_to_slip = [u_to_pars[0], u_to_pars[1][2], u_to_pars[2]]
    state_interp = inverse_1d(u_interp, u_to_state, state_max, state_min)
    sr_interp = inverse_1d(u_interp, u_to_sr, sr_max, sr_min)
    slip_interp = inverse_1d(u_interp, u_to_slip, slip_max, slip_min)
    return t_interp, state_interp, sr_interp, slip_interp

def inverse_1d(u, u_to_par, par_max, par_min):
    """
    Perform an inverse interpolation from parameter space to parameters.

    Parameters
    ----------
    u : array_like
        Parameter values in the latent space.
    u_to_pars : tuple
        Spline parameters for mapping latent space back to parameters.
    par1_max : float
        Maximum value of parameter 1.
    par1_min : float
        Minimum value of parameter 1.
    par2_max : float
        Maximum value of parameter 2.
    par2_min : float
        Minimum value of parameter 2.

    Returns
    -------
    array_like, array_like
        Interpolated parameter values.
    """
    p_interp  = splev(u, u_to_par)
    par = p_interp * (par_max - par_min) + par_min
    return par
