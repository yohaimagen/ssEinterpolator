"""Interpolation module for ssEinterpolator."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.interpolate import splev, make_lsq_spline, interp1d
from scipy.interpolate import BSpline
from joblib import Parallel, delayed


def process_single_series(
    args: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, str, float | None, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-depth latent block and shared t→u knots for one depth.

    This is the per-depth worker function called in parallel by
    :func:`interpolate_to_latent`. It applies the log transform to slip rate,
    builds the intrinsic-parameter and time splines, and returns the t→u knot
    vector separately from the per-depth coefficient block so that callers can
    store knots once per SSE rather than once per depth.

    Args:
        args: Tuple of
            (idx, sr_i, state_i, slip_i, t, num_of_knots, num_of_t_knots,
            t_knots_placment, ratio, method) where:

            - idx: Depth index (used only for debug printing).
            - sr_i: Slip rate array at this depth, shape [n_time].
            - state_i: State variable at this depth, shape [n_time].
            - slip_i: Cumulative slip at this depth, shape [n_time].
            - t: Normalized time array, shape [n_time].
            - num_of_knots: Number of knots for the u→pars spline.
            - num_of_t_knots: Number of knots for the t→u spline.
            - t_knots_placment: Knot placement strategy ('quantile', 'linspace',
              or 'both').
            - ratio: Fraction of linspace knots when t_knots_placment is 'both'.
            - method: Solver for make_lsq_spline ('qr' or 'norm-eq').

    Returns:
        Tuple (t_to_u_knots, depth_block) where:

        - t_to_u_knots: 1-D array of t→u spline knots, shape [t_to_u_knot_l].
          Identical across all depths for a given SSE (same time array).
        - depth_block: 1-D array concatenating t→u coefficients, u→pars spline
          knots/coefficients for (state, sr, slip), and six normalization
          constants (state_min, state_max, sr_min, sr_max, slip_min, slip_max).
          Does NOT include t→u knots.
    """
    idx, sr_i, state_i, slip_i, t, num_of_knots, num_of_t_knots, t_knots_placment, ratio, method = args
    sr_i = np.log10(np.abs(sr_i))
    u_to_pars, u = interpolate_to_latent_single_along_stk(
        sr=sr_i, state=state_i, slip=slip_i, num_of_knots=num_of_knots, method=method,
    )
    if np.sum(np.isnan(u)) > 0:
        print(f'NaN in u for series {idx}')
    t_to_u = interpolate_time_parametric_space(
        t, u, num_knots=num_of_t_knots, knots_placment=t_knots_placment, ratio=ratio, method=method,
    )
    state_min, state_max = np.min(state_i), np.max(state_i)
    sr_min, sr_max = np.min(sr_i), np.max(sr_i)
    slip_min, slip_max = np.min(slip_i), np.max(slip_i)
    depth_block = np.concatenate((
        t_to_u.c,
        u_to_pars[0], u_to_pars[1][0], u_to_pars[1][1], u_to_pars[1][2],
        [state_min, state_max, sr_min, sr_max, slip_min, slip_max],
    ))
    return t_to_u.t, depth_block


def interpolate_to_latent(
    sr: np.ndarray,
    state: np.ndarray,
    slip: np.ndarray,
    t: np.ndarray,
    num_of_knots: int = 1000,
    num_of_t_knots: int = 300,
    t_knots_placment: str = 'linspace',
    ratio: float | None = None,
    n_workers: int = 30,
    method: Literal['qr', 'norm-eq'] = 'qr',
) -> np.ndarray:
    """Convert a full SSE simulation snapshot into a single latent vector.

    Processes all depth indices in parallel. Each depth is encoded by two
    B-splines (u→pars and t→u). The t→u knot vector is identical across all
    depths (all share the same time axis), so it is stored **once** at the
    beginning of the output vector rather than being replicated per depth.

    Output format::

        [t→u knots | depth_0_block | depth_1_block | … | t_min | t_max]

    where each ``depth_i_block`` contains t→u coefficients, u→pars knots and
    coefficients for (state, sr, slip), and six normalization constants.

    Args:
        sr: Slip rate array, shape [n_depths, n_time].
        state: State variable array, shape [n_depths, n_time].
        slip: Cumulative slip array, shape [n_depths, n_time].
        t: Normalized time array, shape [n_time], values in [0, 1].
        num_of_knots: Number of interior knots for the u→pars spline.
        num_of_t_knots: Number of interior knots for the t→u spline.
        t_knots_placment: Strategy for placing t→u knots. One of 'quantile',
            'linspace', or 'both'.
        ratio: Fraction of linspace knots when t_knots_placment is 'both'.
            Required if t_knots_placment == 'both'.
        n_workers: Maximum number of parallel joblib workers.
        method: Linear solver for make_lsq_spline. 'qr' (default) is more
            numerically stable; 'norm-eq' is faster.

    Returns:
        1-D latent vector of length
        ``t_to_u_knot_l + n_depths * dp_laten_vec_length + 2``, where the
        first ``t_to_u_knot_l`` elements are the shared t→u knots and the last
        two elements are t[0] and t[-1].

    Raises:
        ValueError: If t_knots_placment is not one of the allowed values, or if
            t_knots_placment is 'both' and ratio is not provided.
    """
    if t_knots_placment not in {'quantile', 'linspace', 'both'}:
        raise ValueError("t_knots_placment must be 'quantile', 'linspace', or 'both'.")
    if t_knots_placment == 'both' and ratio is None:
        raise ValueError("ratio must be provided when t_knots_placment is 'both'.")

    args_list = [
        (idx, sr[idx], state[idx], slip[idx], t, num_of_knots, num_of_t_knots, t_knots_placment, ratio, method)
        for idx in range(sr.shape[0])
    ]

    n = min(n_workers, len(args_list))
    results: list[tuple[np.ndarray, np.ndarray]] = Parallel(n_jobs=n, backend="loky")(
        delayed(process_single_series)(a) for a in args_list
    )

    # t→u knots are identical across all depths (same time array); store once.
    t_to_u_knots = results[0][0]
    depth_blocks = [r[1] for r in results]

    return np.concatenate([t_to_u_knots] + depth_blocks + [[t[0], t[-1]]])


def insert_dense_u(u: np.ndarray, thrsh: float) -> np.ndarray:
    """Densify an arc-length parameter array so no gap exceeds a threshold.

    For each consecutive pair (u[i], u[i+1]) whose gap exceeds ``thrsh``,
    inserts uniformly spaced interior points so that all gaps are ≤ ``thrsh``.
    Implemented without a Python loop for efficiency.

    Args:
        u: Monotonically increasing parameter array, shape [n].
        thrsh: Maximum allowed gap between consecutive values.

    Returns:
        Densified parameter array with all gaps ≤ thrsh, shape [m] where m ≥ n.
    """
    du = np.diff(u)
    counts = np.maximum(np.ceil(du / thrsh).astype(int), 1)

    starts = np.repeat(u[:-1], counts)
    steps = np.repeat(du / counts, counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.concatenate(([0], np.cumsum(counts[:-1]))), counts)
    result = np.empty(counts.sum() + 1)
    result[:-1] = starts + offsets * steps
    result[-1] = u[-1]
    return result


def interpolate_to_latent_single_along_stk(
    sr: np.ndarray,
    state: np.ndarray,
    slip: np.ndarray,
    num_of_knots: int = 500,
    degree: int = 3,
    iterations: int = 5,
    thrsh: float = 0.01,
    method: Literal['qr', 'norm-eq'] = 'qr',
) -> tuple[list[Any], np.ndarray]:
    """Fit a B-spline mapping u → (state, sr, slip) for a single depth.

    Computes the arc-length parameter u from the normalized (state, log-SR)
    trajectory, densifies u via :func:`insert_dense_u`, and fits three cubic
    least-squares B-splines (one per output variable) with a hybrid
    quantile/linspace knot placement.

    Args:
        sr: Log10 slip rate at a single depth, shape [n_time]. Already
            log-transformed by the caller.
        state: Friction state variable at a single depth, shape [n_time].
        slip: Cumulative slip at a single depth, shape [n_time].
        num_of_knots: Total number of knots (including boundary repetitions)
            for the u→pars spline.
        degree: B-spline degree. Default is 3 (cubic).
        iterations: Unused; reserved for future iterative refinement.
        thrsh: Maximum gap in u before densification is triggered.

    Returns:
        Tuple (tck, u) where:

        - tck: List [knots, [state_coeffs, sr_coeffs, slip_coeffs], degree].
        - u: Arc-length parameter array, shape [n_time], values in [0, 1].
    """
    # method is passed through to make_lsq_spline
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

    try:
        qn = int((num_of_knots - degree - 1) * 0.8)
        ln = num_of_knots - degree - 1 - qn
        quantiles = np.percentile(u, np.linspace(0, 100, qn))
        linspace = np.linspace(u[0], u[-1], ln)
        knots = np.sort(np.unique(np.concatenate((quantiles, linspace))))
        knots = np.concatenate(([u[0]] * degree, knots, [u[-1]] * degree))
    except Exception:
        knots = np.percentile(u, np.linspace(0, 100, num_of_knots - degree - 3))
        knots = np.concatenate(([u[0]] * degree, knots, [u[-1]] * degree))

    u_new = insert_dense_u(u, thrsh)

    p_interp = interp1d(u, p, kind="linear")
    s_interp = interp1d(u, s, kind="linear")
    c_interp = interp1d(u, c, kind="linear")
    p_dense = p_interp(u_new)
    s_dense = s_interp(u_new)
    c_dense = c_interp(u_new)

    u_to_p = make_lsq_spline(u_new, p_dense, knots, degree, method=method)
    u_to_s = make_lsq_spline(u_new, s_dense, knots, degree, method=method)
    u_to_c = make_lsq_spline(u_new, c_dense, knots, degree, method=method)

    tck = [u_to_p.t, [u_to_p.c, u_to_s.c, u_to_c.c], degree]
    return tck, u


def interpolate_time_parametric_space(
    t: np.ndarray,
    u: np.ndarray,
    num_knots: int,
    degree: int = 3,
    knots_placment: str = 'linspace',
    ratio: float | None = None,
    method: Literal['qr', 'norm-eq'] = 'qr',
) -> BSpline:
    """Fit a B-spline mapping t → u for a single depth's SSE cycle.

    Normalizes time to [0, 1], places knots according to the chosen strategy,
    densifies the time grid, and fits a cubic least-squares B-spline using the
    norm-eq solver.

    Args:
        t: Time array at a single depth, shape [n_time]. Need not be normalized;
            normalization is applied internally.
        u: Arc-length parameter array, shape [n_time], values in [0, 1].
            Must be strictly increasing.
        num_knots: Total number of knots (including boundary repetitions).
        degree: B-spline degree. Default is 3 (cubic).
        knots_placment: Knot placement strategy. One of 'quantile', 'linspace',
            or 'both'.
        ratio: Fraction of linspace knots when knots_placment is 'both'. Required
            if knots_placment == 'both'.

    Returns:
        Fitted BSpline object representing the t→u mapping.

    Raises:
        ValueError: If knots_placment is invalid or ratio is missing for 'both'.
        AssertionError: If t or u are not strictly increasing.
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
        knots_quantile = np.percentile(t, np.linspace(0, 100, num_knots_quantile))
        knots = np.sort(np.unique(np.concatenate((knots_linspace, knots_quantile))))
    while knots.shape[0] < num_knots - degree - 3:
        additional_knots_needed = num_knots - degree - 3 - knots.shape[0]
        midpoints = (knots[:-1] + knots[1:]) / 2
        knots = np.sort(np.unique(np.concatenate((knots, midpoints[:additional_knots_needed]))))

    assert np.all(np.diff(t) > 0), "t must be strictly increasing"
    assert np.all(np.diff(u) > 0), "u must be strictly increasing"

    knots = np.concatenate(([tl] * degree, knots, [tr] * degree))

    t_dense = np.linspace(tl, tr, num_knots * 5)
    t_dense = np.unique(np.sort(np.concatenate((t_dense, t))))
    u_interp = interp1d(t, u, kind="linear")
    u_dense = u_interp(t_dense)

    res = make_lsq_spline(t_dense, u_dense, knots, degree, method=method)
    return res


def inverse_interpolate_time_parametric_space(
    t_interp: np.ndarray,
    t_to_u: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the t→u spline at interpolation time points.

    Args:
        t_interp: Normalized time points at which to evaluate the spline,
            shape [n_interp], values in [0, 1].
        t_to_u: Scipy spline tuple (knots, coefficients, degree) for the
            t→u mapping.

    Returns:
        Tuple (t_interp, u_interp) where u_interp has shape [n_interp].
    """
    u_interp = splev(t_interp, t_to_u)
    return t_interp, u_interp


def inverse_interpolation(
    latent_vec: np.ndarray,
    lf: np.ndarray,
    t_to_u_knot_l: int,
    t_to_u_cof_l: int,
    u_to_par_knot_l: int,
    u_to_par_cof_l: int,
    t_interp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct sr, state, and slip time series from a latent vector.

    Unpacks per-depth spline parameters from the latent vector, evaluates each
    spline at the interpolation grid, and denormalizes the outputs.

    Applies one vectorization optimization over a naive per-depth splev loop:

    - **Step 3a** — t→u batch across all depths: t→u knots are shared (stored
      once in the latent vector, not per-depth), so all n_depths coefficient
      vectors are stacked into a ``[t_to_u_cof_l, n_depths]`` matrix and
      evaluated in a single ``BSpline`` call → ``u_all`` shape
      ``[n_interp, n_depths]``.  Reduces 346 splev calls to 1 BSpline call
      per SSE.

    - **Step 3b** — multi-output u→pars per depth: within each depth, state, sr,
      and slip share the same ``u_to_pars_knots`` slice from the latent vector.
      Their coefficients are stacked ``[u_cof_l, 3]`` and evaluated in one
      ``BSpline`` call per depth, replacing 3 ``splev`` calls with 1.

    Args:
        latent_vec: Flattened latent vector encoding all depths, shape
            [t_to_u_knot_l + n_depths * dp_laten_vec_length + 2]. The first
            t_to_u_knot_l elements are the shared t→u knots; the last two are
            the original time bounds (t_min, t_max).
        lf: Along-fault depth array, shape [n_depths].
        t_to_u_knot_l: Number of knots in the t→u spline.
        t_to_u_cof_l: Number of coefficients in the t→u spline (= knot_l - 4).
        u_to_par_knot_l: Number of knots in the u→pars spline.
        u_to_par_cof_l: Number of coefficients per variable in the u→pars spline.
        t_interp: Normalized interpolation time grid, shape [n_interp],
            values in [0, 1].

    Returns:
        Tuple (reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp)
        where each array has shape [n_depths, n_interp] for the physical fields
        and shape [n_interp] for the time axis (denormalized to physical time).
    """
    # Format: [t→u knots | depth_0_block | depth_1_block | ... | t_min | t_max]
    # Per-depth block: t→u coeffs + u→pars knots + 3×u→pars coeffs + 6 norm constants.
    l = t_to_u_cof_l + u_to_par_knot_l + u_to_par_cof_l * 3 + 6

    n_depths = lf.shape[0]
    n_t = t_interp.shape[0]

    # Step 3a: batch t→u evaluation across all depths.
    # Shared knots stored once at the start; sort once to satisfy BSpline.
    t_to_u_knots = np.sort(latent_vec[:t_to_u_knot_l])
    # Stack per-depth coefficient vectors: shape [t_to_u_cof_l, n_depths].
    all_t_to_u_cofs = np.empty((t_to_u_cof_l, n_depths))
    for idx in range(n_depths):
        all_t_to_u_cofs[:, idx] = latent_vec[t_to_u_knot_l + idx * l: t_to_u_knot_l + idx * l + t_to_u_cof_l]
    # One BSpline call → u values for every depth: shape [n_t, n_depths].
    u_all = BSpline(t_to_u_knots, all_t_to_u_cofs, k=3)(t_interp)

    reconstructed_sr    = np.empty((n_depths, n_t))
    reconstructed_state = np.empty((n_depths, n_t))
    reconstructed_slip  = np.empty((n_depths, n_t))

    for idx in range(n_depths):
        start = t_to_u_knot_l + idx * l
        sig_latent = latent_vec[start: start + l]

        u_interp = u_all[:, idx]  # [n_t] — from the batched Step 3a call

        # u→pars: one knot vector shared by all three output variables at this
        # depth (same slice in the latent block). Stack coefficients into
        # [u_to_par_cof_l, 3] and evaluate all three in one BSpline call,
        # exactly analogous to Step 3a.
        # POD reconstruction can perturb knots out of order → sort.
        # If sort introduces NaN-at-end (latent had NaN), fall back to splev
        # which tolerates both non-monotone knots and NaN silently.
        u_to_pars_knots = np.sort(sig_latent[t_to_u_cof_l: t_to_u_cof_l + u_to_par_knot_l])

        base = t_to_u_cof_l + u_to_par_knot_l
        u_to_state_cofs = sig_latent[base:                       base + u_to_par_cof_l]
        u_to_sr_cofs    = sig_latent[base + u_to_par_cof_l:      base + 2 * u_to_par_cof_l]
        u_to_slip_cofs  = sig_latent[base + 2 * u_to_par_cof_l:  base + 3 * u_to_par_cof_l]

        state_min = sig_latent[-6]
        state_max = sig_latent[-5]
        sr_min    = sig_latent[-4]
        sr_max    = sig_latent[-3]
        slip_min  = sig_latent[-2]
        slip_max  = sig_latent[-1]

        # Step 3b: one BSpline call → shape [n_t, 3].
        # POD can corrupt knots into degenerate geometry; fall back to splev
        # (tolerant of corrupt knots) only for those rare depths.
        cofs_3 = np.stack([u_to_state_cofs, u_to_sr_cofs, u_to_slip_cofs], axis=1)
        try:
            out = BSpline(u_to_pars_knots, cofs_3, k=3)(u_interp)  # [n_t, 3]
            reconstructed_state[idx] = out[:, 0] * (state_max - state_min) + state_min
            reconstructed_sr[idx]    = out[:, 1] * (sr_max    - sr_min)    + sr_min
            reconstructed_slip[idx]  = out[:, 2] * (slip_max  - slip_min)  + slip_min
        except ValueError:
            u_to_pars_knots_raw = sig_latent[t_to_u_cof_l: t_to_u_cof_l + u_to_par_knot_l]
            reconstructed_state[idx] = splev(u_interp, [u_to_pars_knots_raw, u_to_state_cofs, 3]) * (state_max - state_min) + state_min
            reconstructed_sr[idx]    = splev(u_interp, [u_to_pars_knots_raw, u_to_sr_cofs,    3]) * (sr_max    - sr_min)    + sr_min
            reconstructed_slip[idx]  = splev(u_interp, [u_to_pars_knots_raw, u_to_slip_cofs,  3]) * (slip_max  - slip_min)  + slip_min

    t_min = latent_vec[-2]
    t_max = latent_vec[-1]
    t_interp = (t_interp * (t_max - t_min)) + t_min
    return reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp


def inverse_interpolate_to_latent_single_along_stk(
    t_interp: np.ndarray,
    t_to_u: list[Any],
    u_to_pars: list[Any],
    state_max: float,
    state_min: float,
    sr_max: float,
    sr_min: float,
    slip_max: float,
    slip_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct physical variables for a single depth from spline parameters.

    Evaluates the t→u spline to get the intrinsic parameter, then evaluates
    the u→pars splines and denormalizes each variable.

    Args:
        t_interp: Normalized time grid, shape [n_interp], values in [0, 1].
        t_to_u: Scipy spline tuple (knots, coefficients, degree) for t→u.
        u_to_pars: Spline descriptor [knots, [state_coeffs, sr_coeffs, slip_coeffs],
            degree] for the u→pars mapping.
        state_max: Maximum state value used for denormalization.
        state_min: Minimum state value used for denormalization.
        sr_max: Maximum log10 slip rate used for denormalization.
        sr_min: Minimum log10 slip rate used for denormalization.
        slip_max: Maximum slip value used for denormalization.
        slip_min: Minimum slip value used for denormalization.

    Returns:
        Tuple (t_interp, state_interp, sr_interp, slip_interp), each shape
        [n_interp]. t_interp is returned unchanged.
    """
    t_interp, u_interp = inverse_interpolate_time_parametric_space(t_interp, t_to_u)
    u_to_state = [u_to_pars[0], u_to_pars[1][0], u_to_pars[2]]
    u_to_sr = [u_to_pars[0], u_to_pars[1][1], u_to_pars[2]]
    u_to_slip = [u_to_pars[0], u_to_pars[1][2], u_to_pars[2]]
    state_interp = inverse_1d(u_interp, u_to_state, state_max, state_min)
    sr_interp = inverse_1d(u_interp, u_to_sr, sr_max, sr_min)
    slip_interp = inverse_1d(u_interp, u_to_slip, slip_max, slip_min)
    return t_interp, state_interp, sr_interp, slip_interp


def inverse_1d(
    u: np.ndarray,
    u_to_par: list[Any],
    par_max: float,
    par_min: float,
) -> np.ndarray:
    """Evaluate a normalized spline and denormalize the result.

    Args:
        u: Intrinsic parameter values at which to evaluate, shape [n_interp].
        u_to_par: Scipy spline tuple (knots, coefficients, degree).
        par_max: Maximum physical value for denormalization.
        par_min: Minimum physical value for denormalization.

    Returns:
        Denormalized physical values, shape [n_interp].
    """
    p_interp = splev(u, u_to_par)
    par = p_interp * (par_max - par_min) + par_min
    return par
