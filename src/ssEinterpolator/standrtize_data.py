import numpy as np
from scipy.interpolate import interp1d
import argparse
import os
from pathlib import Path

def read_data(data_dir, w):
    """Read data files for a given w value from the specified directory."""
    data = {}
    for key in ['sr', 'ts', 'state', 'slip', 'n', 't']:
        file_path = os.path.join(data_dir, f'w{w}_{key}.npy')
        data[key] = np.load(file_path)
    return data

def find_slip_events(t, threshold=3e-2, sse_threshold=60*60):
    """Find significant slip events in the data."""
    # Find events where slip rate exceeds threshold
    dt = np.diff(t)
    sses_steps = t[1:][dt < threshold]

    sses = np.diff(sses_steps) < sse_threshold
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
        sses.append(np.mean(sses_steps[A[i, 0]: A[i, 1]]))
    
    return np.array(sses)

def find_new_sses(sses1, sses2, interval=1e6):
    """
    Find SSEs in set 2 that are not present in set 1.
    An SSE from set 2 is considered present in set 1 if it falls within ±interval of any SSE in set 1.
    
    Args:
        sses1: Array of SSEs times from first set
        sses2: Array of SSEs times from second set
        interval: Time interval to consider SSEs as equivalent
        
    Returns:
        Array of SSE times from set 2 that are not present in set 1
    """
    new_sses = []
    for sse2 in sses2:
        # Check if sse2 is within interval of any sse1
        matches = np.any((sses1 - interval <= sse2) & (sse2 <= sses1 + interval))
        if not matches:
            new_sses.append(sse2)
    return np.array(new_sses)


def create_sses_t(intervals_depth=4):
    """Create refined time points around a reference point."""
    intervals = [[1e5, 300], [1e4, 30], [1e3, 3], [1e2, 0.3], [1e1, 0.03]]
    intervals = intervals[:intervals_depth]
    
    sses_t = np.arange(-5e5, 5e5, 3000)
    if intervals_depth > 0:
        intervals[-1][0] *= 6
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
        sses_t = create_sses_t(depth)
        for sse in sses:
            t_interpolate = np.concatenate((
                t_interpolate[t_interpolate < sse - 5e5],
            sses_t + sse,
            t_interpolate[t_interpolate > sse + 5e5]
        ))
    
    t_interpolate = t_interpolate[t_interpolate < t_end]
    return t_interpolate

def interpolate_data(data, t_original, t_new):
    """Interpolate data arrays to new time points."""
    interpolated = np.zeros((data.shape[0], t_new.shape[0]))
    for i in range(data.shape[0]):
        f = interp1d(t_original, data[i])
        interpolated[i] = f(t_new)
    return interpolated

def main():
    parser = argparse.ArgumentParser(description='Interpolate slip rate data with refined time points around events.')
    parser.add_argument('input_dir', type=str, help='Directory containing input .npy files')
    parser.add_argument('output_dir', type=str, help='Directory to save interpolated data')
    parser.add_argument('w', type=str, help='width of W')
    parser.add_argument('time_range', type=float, nargs=2, help='Time range to process (start end)')

    parser.add_argument('--threshold', type=float, default=3e-2, help='Threshold for slip event detection')
    
    parser.add_argument('--base_dt', type=float, default=30000, help='Base time step for interpolation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(args.w)
    # Read input data
    data = read_data(args.input_dir, args.w)
    # Apply time range mask if specified
    t_start, t_end = args.time_range
    mask = (data['t'] > t_start) & (data['t'] < t_end)
    t = np.copy(data['t'])[mask] * 365 * 24 * 60 * 60  # Convert to seconds
    sr_masked = data['sr'][:, mask]
   
    
    # Find slip events
    SSEs = [[find_slip_events(t, args.threshold), 5]]
    for k in range(1, 6):
        sses_t = find_new_sses(np.concatenate([sses[0] for sses in SSEs]), find_slip_events(t, args.threshold * 10**k, sse_threshold=1e5), interval=5e6)
        if len(sses_t) > 0:
            SSEs.append([sses_t, 5 - k])
    
    # Create interpolation time points
    t_interpolate = create_interpolation_time(t[0], t[-1], SSEs, base_step=args.base_dt)
    
    t_interpolate = t_interpolate[(t_interpolate <= t[-1]) & (t_interpolate >= t[0])]
    
    
    # Interpolate all data arrays
    interpolated_data = {}
    for key in ['sr', 'ts', 'state', 'slip', 'n']:
        if args.time_range:
            data_masked = data[key][:, mask]
        else:
            data_masked = data[key]
        interpolated_data[key] = interpolate_data(data_masked, t, t_interpolate)
    
    # Save interpolated data
    for key, value in interpolated_data.items():
        output_path = os.path.join(args.output_dir, f'w{args.w}_{key}_interpolated.npy')
        np.save(output_path, value)
    
    # Save interpolated time points
    np.save(os.path.join(args.output_dir, f'w{args.w}_t_interpolated.npy'), t_interpolate)

    # Save sses
    np.save(os.path.join(args.output_dir, f'w{args.w}_sses.npy'), np.concatenate([sses[0] for sses in SSEs]))

if __name__ == '__main__':
    main()