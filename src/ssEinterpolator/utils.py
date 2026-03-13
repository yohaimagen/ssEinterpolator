from __future__ import annotations

import numpy as np


def find_slip_events(
    t: np.ndarray,
    sr: np.ndarray,
    threshold: float = -7,
    sse_threshold: float = 0.3,
) -> np.ndarray:
    """Detect slow slip event onset times from a log slip-rate time series.

    Identifies contiguous intervals where the log slip rate exceeds ``threshold``,
    computes the mean time of each interval as its representative onset, and
    then filters out events that are too close together (separated by less than
    ``sse_threshold`` in time).

    Args:
        t: Time array in years, shape [n_time]. Must be monotonically increasing.
        sr: Log10 slip rate array, shape [n_time].
        threshold: Log10 slip rate value above which a time step is considered
            part of a slip event.
        sse_threshold: Minimum time separation (in years) between consecutive
            events. Events closer than this are merged by keeping only the first.

    Returns:
        Array of SSE onset times (mean time of each qualifying interval),
        shape [n_events].
    """
    sses = sr > threshold
    diff = np.diff(sses.astype(int))

    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if sses[0]:
        starts = np.insert(starts, 0, 0)

    if sses[-1]:
        ends = np.append(ends, len(sses))

    A = np.column_stack((starts, ends))
    sses = []
    for i in range(A.shape[0]):
        sses.append(np.mean(t[A[i, 0]: A[i, 1]]))
    sses = np.array(sses)
    diffs = np.diff(sses)
    mask = diffs >= sse_threshold
    mask = np.concatenate(([True], mask))

    return sses[mask]
