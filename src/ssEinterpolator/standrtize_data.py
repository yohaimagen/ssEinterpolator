from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.interpolate import interp1d

from utils import find_slip_events


def read_data(data_dir: str, prefix: str) -> dict[str, np.ndarray]:
    """Load raw simulation .npy files for a given parameter prefix.

    Expects files named ``{prefix}{key}.npy`` for each key in
    ['sr', 'ts', 'state', 'slip', 'n', 't'].

    Args:
        data_dir: Directory containing the .npy files.
        prefix: Filename prefix identifying the simulation.

    Returns:
        Dict mapping each key ('sr', 'ts', 'state', 'slip', 'n', 't') to its
        loaded numpy array.
    """
    data = {}
    for key in ['sr', 'ts', 'state', 'slip', 'n', 't']:
        file_path = os.path.join(data_dir, f'{prefix}{key}.npy')
        data[key] = np.load(file_path)
    return data


def find_new_sses(
    sses1: np.ndarray,
    sses2: np.ndarray,
    interval: float = 1e6,
) -> np.ndarray:
    """Find SSE times in sses2 that have no match within ±interval in sses1.

    Args:
        sses1: Reference SSE time array, shape [n1].
        sses2: Candidate SSE time array to filter, shape [n2].
        interval: Half-width of the matching window. An SSE in sses2 is
            considered present in sses1 if it falls within ±interval of any
            element in sses1.

    Returns:
        Array of SSE times from sses2 not matched in sses1, shape [m] where
        m ≤ n2.
    """
    new_sses = []
    for sse2 in sses2:
        matches = np.any((sses1 - interval <= sse2) & (sse2 <= sses1 + interval))
        if not matches:
            new_sses.append(sse2)
    return np.array(new_sses)


def create_sses_t(intervals_depth: int = 4) -> np.ndarray:
    """Build a time offset array with refinement around t=0 for a single SSE.

    Generates a coarse background grid over [-5e5, 5e5] seconds and
    progressively replaces the inner region with finer grids up to
    intervals_depth levels.

    Args:
        intervals_depth: Number of refinement levels. Each level replaces
            the innermost interval with a 10× finer grid.

    Returns:
        Non-uniform time offset array centered at 0, shape [n_points].
    """
    intervals = [[1e5, 300], [1e4, 30], [1e3, 3], [1e2, 0.3], [1e1, 0.03]]
    intervals = intervals[:intervals_depth]

    sses_t = np.arange(-5e5, 5e5, 3000)
    if intervals_depth > 0:
        intervals[-1][0] *= 6
        for interval, time_step in intervals:
            interval = interval / 2
            sses_t_inner = np.arange(-interval, interval, time_step)
            sses_t = np.concatenate((sses_t[sses_t < - interval], sses_t_inner, sses_t[sses_t > interval]))
    return sses_t


def create_interpolation_time(
    t_start: float,
    t_end: float,
    sses_with_depths: list[tuple[np.ndarray, int]],
    base_step: float = 30000,
) -> np.ndarray:
    """Create a non-uniform time grid with refinement around each SSE onset.

    Starts from a uniform grid with step base_step and replaces the ±5e5 s
    neighbourhood around each SSE with the offset array from
    :func:`create_sses_t`.

    Args:
        t_start: Start time in seconds.
        t_end: End time in seconds.
        sses_with_depths: List of (sses, depth) pairs where sses is an array
            of SSE onset times in seconds and depth is the refinement level
            passed to :func:`create_sses_t`.
        base_step: Uniform time step for the background grid in seconds.

    Returns:
        Non-uniform time grid clipped to [t_start, t_end), shape [n_points].
    """
    t_interpolate = np.arange(t_start, t_end, base_step)

    for sses, depth in sses_with_depths:
        sses_t = create_sses_t(depth)
        for sse in sses:
            t_interpolate = np.concatenate((
                t_interpolate[t_interpolate < sse - 5e5],
                sses_t + sse,
                t_interpolate[t_interpolate > sse + 5e5],
            ))

    t_interpolate = t_interpolate[t_interpolate < t_end]
    return t_interpolate


def interpolate_data(
    data: np.ndarray,
    t_original: np.ndarray,
    t_new: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate 2-D simulation data to a new time grid.

    Args:
        data: Input array, shape [n_depths, n_time_original].
        t_original: Original time axis, shape [n_time_original].
        t_new: Target time axis, shape [n_time_new].

    Returns:
        Interpolated array, shape [n_depths, n_time_new].
    """
    interpolated = np.zeros((data.shape[0], t_new.shape[0]))
    for i in range(data.shape[0]):
        f = interp1d(t_original, data[i])
        interpolated[i] = f(t_new)
    return interpolated


def main() -> None:
    """CLI entry point: interpolate simulation data to a refined time grid.

    Reads raw .npy simulation files, detects SSE onsets, builds a refined time
    grid around each event, interpolates all data arrays to that grid, and saves
    the results.
    """
    parser = argparse.ArgumentParser(description='Interpolate slip rate data with refined time points around events.')
    parser.add_argument('input_dir', type=str, help='Directory containing input .npy files')
    parser.add_argument('output_dir', type=str, help='Directory to save interpolated data')
    parser.add_argument('prefix', type=str, help='prefix of the data files')
    parser.add_argument('sr_idx', type=int, help='slip rate index to detect on slip events')
    parser.add_argument('time_range', type=float, nargs=2, help='Time range to process (start end)')

    parser.add_argument('--threshold', type=float, default=3e-2, help='Threshold for slip event detection')
    parser.add_argument('--T_threshold', type=float, default=60*60, help='Threshold for minimum time difference between slip events')

    parser.add_argument('--base_dt', type=float, default=30000, help='Base time step for interpolation')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = read_data(args.input_dir, args.prefix)
    t_start, t_end = args.time_range
    mask = (data['t'] > t_start) & (data['t'] < t_end)
    t = np.copy(data['t'])[mask] * 365 * 24 * 60 * 60
    sr_masked = data['sr'][:, mask]

    Slip_events = [[find_slip_events(t, np.log10(np.abs(sr_masked[args.sr_idx])), args.threshold), args.T_threshold]]

    t_interpolate = create_interpolation_time(t[0], t[-1], Slip_events, base_step=args.base_dt)

    t_interpolate = t_interpolate[(t_interpolate <= t[-1]) & (t_interpolate >= t[0])]

    interpolated_data = {}
    for key in ['sr', 'ts', 'state', 'slip', 'n']:
        if args.time_range:
            data_masked = data[key][:, mask]
        else:
            data_masked = data[key]
        interpolated_data[key] = interpolate_data(data_masked, t, t_interpolate)

    for key, value in interpolated_data.items():
        output_path = os.path.join(args.output_dir, f'{args.prefix}{key}.npy')
        np.save(output_path, value)

    np.save(os.path.join(args.output_dir, f'{args.prefix}t.npy'), t_interpolate)

    np.save(os.path.join(args.output_dir, f'{args.prefix}_sses.npy'), np.concatenate([sses[0] for sses in Slip_events]))


if __name__ == '__main__':
    main()
