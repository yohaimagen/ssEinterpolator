import numpy as np

def find_slip_events(t, sr, threshold=-7, sse_threshold=0.3):
    """Find significant slip events in the data."""
    # Find events where slip rate exceeds threshold

    sses = sr > threshold
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
        sses.append(np.mean(t[A[i, 0]: A[i, 1]]))
    sses = np.array(sses)
    diffs = np.diff(sses)
    mask = diffs >= sse_threshold
    mask = np.concatenate(([True], mask))
    
    return sses[mask]

