from __future__ import annotations

from typing import Callable

import numpy as np


class Data:
    """Container for slow slip event simulation data at a single parameter point.

    Attributes:
        sr: Slip rate array, shape [n_depths, n_time].
        slip: Cumulative slip array, shape [n_depths, n_time].
        state: Rate-and-state friction state variable, shape [n_depths, n_time].
        t: Time in years, shape [n_time].
        params: Forward model parameter vector, shape [n_features].
    """

    def __init__(
        self,
        params: np.ndarray,
        load_f: Callable[[list[str]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
        str_params: list[str] | None = None,
        sr: np.ndarray | None = None,
        slip: np.ndarray | None = None,
        state: np.ndarray | None = None,
        t: np.ndarray | None = None,
        prefix: str = '',
        supix: str = '',
    ) -> None:
        """Initialize Data from either a loader callback or direct arrays.

        Args:
            params: Forward model parameter vector, shape [n_features].
            load_f: Callback accepting str_params and returning (sr, slip, state, t).
                Required when str_params is provided.
            str_params: String identifiers for the parameter combination used to
                locate simulation files via load_f.
            sr: Slip rate array, shape [n_depths, n_time]. Required if load_f is None.
            slip: Cumulative slip array, shape [n_depths, n_time]. Required if
                load_f is None.
            state: Friction state variable, shape [n_depths, n_time]. Required if
                load_f is None.
            t: Time in seconds, shape [n_time]. Required if load_f is None.
            prefix: Unused prefix string (legacy).
            supix: Unused suffix string (legacy).

        Raises:
            ValueError: If neither load_f/str_params nor direct arrays are provided.
        """
        if load_f is None and params is None and sr is None and slip is None and state is None and t is None:
            raise ValueError('Either load_f, params,  or sr, slip, state, t must be provided')
        if load_f is None and str_params is None:
            if sr is None or slip is None or state is None or t is None:
                raise ValueError('if load_f, str_params not provided sr, slip, state, t must be provided')
            self.sr = sr
            self.slip = slip
            self.state = state
            self.t = t / (365*24*60*60)
        else:
            self.sr, self.slip, self.state, self.t = load_f(str_params)
            self.t /= (365*24*60*60)
        self.params = params

    def mask_data(self, tl: float, tr: float) -> Data:
        """Extract a time-windowed subset of the data.

        Args:
            tl: Left (inclusive) time bound in years.
            tr: Right (exclusive) time bound in years.

        Returns:
            A new Data object containing only timesteps in [tl, tr).
        """
        mask = (self.t >= tl) & (self.t < tr)
        masked_data = Data(
            params=self.params,
            load_f=None,
            str_params=None,
            sr=np.copy(self.sr[:, mask]),
            slip=np.copy(self.slip[:, mask]),
            state=np.copy(self.state[:, mask]),
            t=np.copy(self.t[mask]) * (365*24*60*60),
        )
        return masked_data
