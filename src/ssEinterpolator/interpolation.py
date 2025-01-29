"""
Interpolation module for ssEinterpolator.
"""

import numpy as np
from scipy.interpolate import splprep, splrep, splev, make_lsq_spline

def interpolate_to_latent(sr, state, t, num_of_knots=1000, num_of_t_knots=300, t_knots_placment='linspace', ratio=None):
    """Interpolate sr and state into a latent space."""
    if t_knots_placment != 'quantile' and t_knots_placment != 'linspace' and t_knots_placment != 'both':
        raise ValueError("t_knots_placment must be 'quantile' or 'linspace' or 'both'.")
    if t_knots_placment == 'both' and ratio is None:
        raise ValueError("ratio must be provided when t_knots_placment is 'both'.")
    # Normalize data
    latent = []
    
    for idx in np.arange(sr.shape[0]):
        sr_i = np.copy(np.log10(np.abs(sr[idx])))
        state_i = np.copy(state[idx])
        u_to_pars, u = interpolate_to_latent_single_along_stk_lsq(sr=sr_i, state=state_i, num_of_knots=num_of_knots)
        t_to_u = interpolate_time_parametric_space_lsq(t, u, num_knots=num_of_t_knots, knots_placment=t_knots_placment, ratio=ratio)
        state_min, state_max = np.min(state_i), np.max(state_i)
        sr_min, sr_max = np.min(sr_i), np.max(sr_i)
        lat = np.concatenate((t_to_u.t, t_to_u.c, u_to_pars[0], u_to_pars[1][0], u_to_pars[1][1], [state_min, state_max, sr_min, sr_max]))
        latent.append(lat)
    latent.append([t[0], t[-1]])
    return np.concatenate(latent)

def interpolate_to_latent_single_along_stk(sr, state, num_of_knots=500, smothhness=0.1):
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
    smothhness : float, optional
        Smoothness of the interpolation. Defaults to 0.1.

    Returns
    -------
    tck : tuple
        A tuple containing the knots and coefficients.
    u : array_like
        The values of the parameterization.
    """
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

def interpolate_to_latent_single_along_stk_lsq(sr, state, num_of_knots=500, degree=3, iterations=5):
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
    p_min = np.min(p)
    p_max = np.max(p)
    s_min = np.min(s)
    s_max = np.max(s)
    p = (p  - p_min) / (p_max - p_min)
    s = (s  - s_min) / (s_max - s_min)
    
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
    except:
        knots = np.percentile(u, np.linspace(0, 100, num_of_knots - degree - 3))
        knots = np.concatenate(([u[0]] * degree, knots, [u[-1]] * degree))
        u_to_p = make_lsq_spline(u, p, knots, degree)
        u_to_s = make_lsq_spline(u, s, knots, degree)

    tck = [u_to_p.t, [u_to_p.c, u_to_s.c], degree]
    return tck, u

def interpolate_time_parametric_space(t, u):
    """
    Interpolate time into a latent space.

    Parameters
    ----------
    t : array_like
        Time values.
    u : array_like
        Parameterization of the latent space.

    Returns
    -------
    tck : tuple
        A tuple containing the knots and coefficients.
    """
    return splrep(t, u, s=0.1)

def interpolate_time_parametric_space_lsq(t, u, num_knots, degree=3, knots_placment='linspace', ratio=None):
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




def inverse_interpolation(latent_vec, lf, t_to_u_knot_l, t_to_u_cof_l, u_to_par_knot_l, u_to_par_cof_l, build_t_interp_f=False, build_t_interp_args=None, t_interp=None):
    """Transform latent space back to sr, state."""
    if not build_t_interp_f and t_interp is None:
        raise ValueError("t_interp must be provided if build_t_interp is False.")
    if build_t_interp_f:
        if 't_min' not in build_t_interp_args or 't_max' not in build_t_interp_args or 'lf' not in build_t_interp_args or 'dt_max' not in build_t_interp_args:
            raise ValueError("t_min, t_max, dt_max and lf must be provided if build_t_interp is True.")
        
        
    l = t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l * 2 + 4
    print(l)
    if build_t_interp_f:
        idx = np.argmin(np.abs(lf - build_t_interp_args['lf']))
        sig_latent = latent_vec[l * idx: (idx + 1) * l]
        sig_t_to_u = [sig_latent[:t_to_u_knot_l], sig_latent[t_to_u_knot_l:t_to_u_knot_l + t_to_u_cof_l], 3]
        sig_u_to_pars = [sig_latent[t_to_u_knot_l + t_to_u_cof_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l], [sig_latent[t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l], sig_latent[t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l + u_to_par_cof_l]], 3]
        state_min = sig_latent[-4]
        state_max = sig_latent[-3]
        sr_min = sig_latent[-2]
        sr_max = sig_latent[-1]
        # return build_t_interp_args['t_min'], build_t_interp_args['t_max'], sig_t_to_u, sig_u_to_pars, state_max, state_min, sr_max, sr_min, build_t_interp_args['dt_max']
        t_interp = build_t_interp(build_t_interp_args['t_min'], build_t_interp_args['t_max'], sig_t_to_u, sig_u_to_pars, state_max, state_min, sr_max, sr_min, max_dt=build_t_interp_args['dt_max'])
        
    reconstructed_sr = []
    reconstructed_state = []
    t_interps = []
    for idx in range(lf.shape[0]):
        sig_latent = latent_vec[l * idx: (idx + 1) * l]
        sig_t_to_u = [sig_latent[:t_to_u_knot_l], sig_latent[t_to_u_knot_l:t_to_u_knot_l + t_to_u_cof_l], 3]
        sig_u_to_pars = [sig_latent[t_to_u_knot_l + t_to_u_cof_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l], [sig_latent[t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l], sig_latent[t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l: t_to_u_knot_l + t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l + u_to_par_cof_l]], 3]
        state_min = sig_latent[-4]
        state_max = sig_latent[-3]
        sr_min = sig_latent[-2]
        sr_max = sig_latent[-1]
        
        t_interp1, state_interp, sr_interp = inverse_interpolate_to_latent_single_along_stk(t_interp, sig_t_to_u, sig_u_to_pars, state_max, state_min, sr_max, sr_min)
        reconstructed_sr.append(sr_interp)
        reconstructed_state.append(state_interp)
        t_interps.append(t_interp1)
    # min_len = min([len(i) for i in reconstructed_sr])

    # reconstructed_sr = np.array([i[:min_len] for i in reconstructed_sr])
    # reconstructed_state = np.array([i[:min_len] for i in reconstructed_state])
    # t_interp = t_interps[0][:min_len]
    reconstructed_sr = np.array(reconstructed_sr)
    reconstructed_state = np.array(reconstructed_state)
    t_min = latent_vec[-2]
    t_max = latent_vec[-1]
    t_interp = (t_interp * (t_max - t_min)) + t_min
    return reconstructed_sr, reconstructed_state, t_interp

def inverse_interpolate_to_latent_single_along_stk(t_interp, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min):
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
    state_interp, sr_interp = inverse_1d(u_interp, u_to_pars, state_max, state_min, sr_max, sr_min)
    return t_interp, state_interp, sr_interp

def inverse_1d(u, u_to_pars, par1_max, par1_min, par2_max, par2_min):
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
    p_interp, s_interp = splev(u, u_to_pars)
    par1 = p_interp * (par1_max - par1_min) + par1_min
    par2 = s_interp * (par2_max - par2_min) + par2_min
    return par1, par2
def find_slip_events(t_interp, sr_interp, threshold=-7, sse_threshold=0.3):
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
    sses = np.array(sses)
    diffs = np.diff(sses)
    mask = diffs >= sse_threshold
    mask = np.concatenate(([True], mask))
    
    return sses[mask]

def fine_tune_sse(sse, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min):
   
    t_interp = np.linspace(sse-0.001, sse+0.001, 1_000_000)
    t_interp, u_interp = inverse_interpolate_time_parametric_space(t_interp, t_to_u)

    state_interp, sr_interp = inverse_1d(u_interp, u_to_pars, state_max, state_min, sr_max, sr_min) 

    return t_interp[np.argmax(sr_interp)]

def create_sses_t(intervals_depth=4):
    """Create refined time points around a reference point."""
    intervals = [[1e5, 300], [1e4, 30], [1e3, 3], [1e2, 0.3], [1e1, 0.03], [1e0, 0.003]]
    intervals = intervals[:intervals_depth]
    
    sses_t = np.arange(-5e5, 5e5, 3000)
    if intervals_depth > 0:
        intervals[-1][0] *= 6
        # intervals[-2][0] *= 6
        for interval, time_step in intervals:
            interval = interval / 2
            sses_t_inner = np.arange(-interval, interval, time_step)
            sses_t = np.concatenate((sses_t[sses_t < - interval], sses_t_inner, sses_t[sses_t > interval] ))
    return sses_t

def create_interpolation_time(t_start, t_end, sses_with_depths, base_step=30000):
    """Create interpolation time points with refinement around slip events.
    
    Args:
        t_start: Start time
        t_end: End time
        sses_with_depths: List of tuples (sse_time, intervals_depth)
        base_step: Base time step for interpolation
    """
    t_interpolate = np.arange(t_start, t_end, base_step)
    
    for sses, depth in sses_with_depths:
        print(t_interpolate.shape)
        sses_t = create_sses_t(depth)
        for sse in sses:
            t_interpolate = np.concatenate((
                t_interpolate[t_interpolate < sse - 5e5],
            sses_t + sse,
            t_interpolate[t_interpolate > sse + 5e5]
        ))
        print(t_interpolate.shape)
    
    t_interpolate = t_interpolate[t_interpolate < t_end]
    return t_interpolate

def build_t_interp(t_start, t_end, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min, max_dt=0.3):
    years_to_seconds = 365 * 24 * 60 * 60
    seconds_to_years = 1 / years_to_seconds
    t_interp = np.linspace(t_start, t_end, 10_000_000)
    
    t_interp, state_interp, sr_interp  = inverse_interpolate_to_latent_single_along_stk(t_interp, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min)
    threshold = -7#np.max(sr_interp) - 2
    sses = find_slip_events(t_interp, sr_interp, threshold=threshold)
    sses = np.sort(sses)
    while True:
        diffs = np.diff(sses)  # Compute differences between consecutive elements
        mask = diffs >= max_dt    # Find pairs satisfying the difference condition
        if mask.all():        # If all differences are >= dt, we're done
            break
        keep_indices = np.where(np.append(mask, True))[0]  # Keep the last element too
        sses = sses[keep_indices]
    print(sses.shape)
    sses_new = []
    for sse in sses:
        sses_new.append(fine_tune_sse(sse, t_to_u, u_to_pars, state_max, state_min, sr_max, sr_min))
    sses_new = np.array(sses_new)
    return create_interpolation_time(t_start * years_to_seconds , t_end * years_to_seconds, [(sses_new * years_to_seconds, 4)]) * seconds_to_years

