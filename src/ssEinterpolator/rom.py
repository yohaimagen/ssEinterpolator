from __future__ import annotations

import os
import logging
import datetime
import sys
import warnings
from typing import Callable

import numpy as np
from scipy.interpolate import RBFInterpolator
from multiprocessing import Pool
from joblib import parallel_backend

from .interpolation import interpolate_to_latent, inverse_interpolation
from .data import Data
from .utils import find_slip_events


def process_data(
    args: tuple[ROM, np.ndarray, list[str]],
) -> tuple[str, list[Data]]:
    """Load and split a single simulation's data by SSE, for use with multiprocessing.

    This module-level function is the worker target for :class:`ROM`'s data
    loading pool. It loads the simulation, detects SSEs, and splits the data
    into per-SSE windows.

    Args:
        args: Tuple of (rom_instance, f_param, str_param) where:

            - rom_instance: The calling ROM instance (carries config).
            - f_param: Forward model parameter vector for this simulation.
            - str_param: String identifiers used to locate files via rom.load_f.

    Returns:
        Tuple (key, split_data) where key is the underscore-joined str_param
        and split_data is a list of Data objects, one per SSE interval.

    Raises:
        ValueError: If fewer than sses_starts + sses_num SSEs are detected.
    """
    self, f_param, str_param = args
    self.logger.info(f'Processing data for {args[1]}')
    print(args)
    print(f'processing {str_param}')
    key = '_'.join(str_param)
    data = Data(load_f=self.load_f, str_params=str_param, params=f_param)
    data = data.mask_data(self.t_start, self.t_end)
    print(f'loading data for key={key} with shape {data.sr.shape}')
    idx = np.argmin(np.abs(self.lf - self.along_dp_sses_depth_detector))
    sses = find_slip_events(data.t, np.log10(np.abs(data.sr[idx])), threshold=self.sses_detector_threshold)
    print(f'found {sses.shape[0]} sses for key={key}')
    if sses.shape[0] < self.sses_starts + self.sses_num:
        raise ValueError(f'Not enough slip events detected for key={key}, {sses.shape[0]} found, {self.sses_starts + self.sses_num} required')
    split_data = self.split_data_by_sse(data, sses)
    split_data[self.sses_starts: self.sses_starts + self.sses_num + 1]
    self.logger.info(f'Finished processing data for {key} found {sses.shape[0]} sses')
    return key, split_data


def calc_latent_helper(args: tuple[ROM, int]) -> int:
    """Placeholder worker for parallel latent matrix construction.

    Args:
        args: Tuple of (rom_instance, sse_idx).

    Returns:
        Placeholder integer 1 (not yet implemented).
    """
    self, idx = args
    print(f'creating latent matrix for sse {idx}')
    latent = 1
    return latent


class ROM:
    """Reduced Order Model for slow slip event simulation data.

    Orchestrates the full pipeline: data loading → latent encoding → POD →
    RBF training → prediction. Each SSE index is handled independently,
    yielding one RBF interpolator per SSE.

    Attributes:
        D: Dict of Data objects keyed by underscore-joined str_param, holding
            full simulation time series.
        D_sses: Dict mapping each key to a list of per-SSE Data windows.
        latent: List (one per SSE index) of latent matrices, shape
            [n_sims, latent_dim].
        A: List (one per SSE) of POD amplitude matrices, shape [n_sims, r].
        U: List (one per SSE) of POD basis matrices, shape [latent_dim, r].
        S: List (one per SSE) of singular value arrays, shape [r].
        V: List (one per SSE) of right singular vector matrices, shape [r, n_sims].
        RBFs: List of trained RBFInterpolator objects, one per SSE.
        f_params: Forward model parameter matrix, shape [n_sims, n_features].
        f_params_normalized: Z-score normalized f_params, shape [n_sims, n_features].
        f_mean: Per-feature mean of f_params, shape [n_features].
        f_std: Per-feature std of f_params, shape [n_features].
        lf: Along-fault depth array, shape [n_depths].
        t_interpolate: Normalized interpolation time grid used during prediction.
        r: Number of POD modes retained after build_pod.
    """

    def __init__(
        self,
        f_params: np.ndarray,
        str_params: np.ndarray,
        load_f: Callable[[list[str]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        lf_path: str,
        sses_num: int,
        t_to_u_knot_l: int,
        u_to_par_knot_l: int,
        along_dp_sses_depth_detector: int = 195,
        sses_detector_threshold: float = -4,
        sses_starts: int = 0,
        t_start: float = 0,
        t_end: float = 1e99,
        base_step_t_interpolate: float = 1e-4,
        depths_t_interpolate: int = 9,
        load_data: bool = True,
        log: bool = False,
        rbf_kernal: str = 'linear',
        n_workers: int = 30,
        load_workers: int = 1,
    ) -> None:
        """Initialize the ROM, load simulations, and split data by SSE.

        Args:
            f_params: Forward model parameter matrix, shape [n_sims, n_features].
            str_params: String identifier array, shape [n_sims, n_str_parts].
                Each row is joined with '_' to form a dict key.
            load_f: Callback that accepts a str_param list and returns
                (sr, slip, state, t) arrays for that simulation.
            lf_path: Path to a .npy file containing the along-fault depth array.
            sses_num: Number of consecutive SSE windows to extract per simulation.
            t_to_u_knot_l: Total knots for the t→u spline (controls temporal
                resolution of the latent encoding).
            u_to_par_knot_l: Total knots for the u→pars spline (controls spatial
                resolution of the latent encoding).
            along_dp_sses_depth_detector: Depth index used to detect SSE onsets.
            sses_detector_threshold: Log10 SR threshold for SSE detection.
            sses_starts: Index of the first SSE to include (skip earlier transients).
            t_start: Left time bound (years) for masking raw simulation data.
            t_end: Right time bound (years) for masking raw simulation data.
            base_step_t_interpolate: Base time step for the normalized
                interpolation grid.
            depths_t_interpolate: Number of refinement levels around SSE onset
                in the interpolation time grid.
            load_data: If True, load and split simulation data on construction.
                Set False to defer loading (e.g. when loading latent from files).
            log: If True, enable file and stream logging.
            rbf_kernal: RBF kernel type passed to RBFInterpolator (e.g. 'linear').
            n_workers: Number of parallel workers for latent matrix construction.
            load_workers: Number of parallel workers for data loading.
        """
        self.log = log
        self.n_workers = n_workers
        self.logger = self.setup_logger(log)
        self.rbf_kernal = rbf_kernal
        self.lf = np.load(lf_path)
        self.f_params = f_params
        self.f_mean = np.mean(self.f_params, axis=0)
        self.f_std = np.std(self.f_params, axis=0)

        self.f_std[self.f_std == 0] = 1.0

        self.f_params_normalized = (self.f_params - self.f_mean) / self.f_std

        self.str_params = str_params
        self.load_f = load_f
        self.t_start = t_start
        self.t_end = t_end
        self.along_dp_sses_depth_detector = along_dp_sses_depth_detector
        self.sses_detector_threshold = sses_detector_threshold
        self.sses_starts = sses_starts
        self.base_step_t_interpolate = base_step_t_interpolate
        self.depths_t_interpolate = depths_t_interpolate
        self.D: dict[str, Data] = {}
        self.D_sses: dict[str, list[Data]] = {}
        self.sses_num = sses_num
        self.update_spline_length(t_to_u_knot_l, u_to_par_knot_l)
        self.t_interpolate = self.create_interpolation_time(base_step=base_step_t_interpolate, depths=depths_t_interpolate)

        self.logger.info(f'Creating {sses_num} sses for {len(f_params)} params')
        if load_data:
            with Pool(processes=min(load_workers, len(self.f_params))) as pool:
                results = pool.map(process_data, [(self, self.f_params[k], self.str_params[k]) for k in range(len(self.f_params))])

            for key, split_data in results:
                self.D_sses[key] = split_data[self.sses_starts: self.sses_starts + self.sses_num + 1]
        else:
            for f_param, str_param in zip(f_params, str_params):
                key = '_'.join(str_param)
                self.D_sses[key] = []

    def setup_logger(self, log: bool) -> logging.Logger:
        """Configure and return a logger for this ROM instance.

        Args:
            log: If True, attach file and stream handlers at DEBUG level.
                If False, disable the logger entirely.

        Returns:
            Configured Logger instance.
        """
        logger = logging.getLogger(f"multiprocessing_logger_{id(self)}")
        if log:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"
                )
                log_file = f"rom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                logger.addHandler(stream_handler)
        else:
            logger.disabled = True
            logger.addHandler(logging.NullHandler())
        return logger

    def update_spline_length(self, t_to_u_knot_l: int, u_to_par_knot_l: int) -> None:
        """Update spline length parameters and recompute derived latent dimensions.

        Args:
            t_to_u_knot_l: Total number of knots for the t→u spline.
            u_to_par_knot_l: Total number of knots for the u→pars spline.
        """
        self.t_to_u_knot_l = t_to_u_knot_l
        self.u_to_par_knot_l = u_to_par_knot_l
        self.t_to_u_cof_l = self.t_to_u_knot_l - 4
        self.u_to_par_cof_l = self.u_to_par_knot_l - 4
        self.dp_laten_vec_length = self.t_to_u_knot_l + self.t_to_u_cof_l + self.u_to_par_knot_l + self.u_to_par_cof_l * 2 + 4

    def split_data_by_sse(self, data: Data, sses: np.ndarray) -> list[Data]:
        """Split a continuous Data object into per-SSE windows.

        Each window spans from one SSE onset time to the next.

        Args:
            data: Full simulation Data object with continuous time series.
            sses: Array of SSE onset times in years, shape [n_sses].

        Returns:
            List of Data objects of length len(sses) - 1, one per inter-SSE interval.
        """
        split_data = []
        for i in range(len(sses) - 1):
            split_data.append(data.mask_data(sses[i], sses[i + 1]))
        return split_data

    def create_one_sse_latent_matrix(
        self,
        idx: int,
        save_path: str | None = None,
    ) -> np.ndarray:
        """Build the latent matrix for a single SSE index across all simulations.

        Args:
            idx: SSE index (0-based, must be in [0, sses_num)).
            save_path: If provided, save the latent matrix as
                ``{save_path}/latent_{idx}.npy``.

        Returns:
            Latent matrix of shape [n_sims, latent_dim].

        Raises:
            ValueError: If idx is out of range.
        """
        self.logger.info(f'Creating latent matrix for sse {idx}')
        if idx >= self.sses_num or idx < 0:
            raise ValueError('idx should be in range of sses_num')
        print(f'creating latent matrix for sse {idx}')
        latent = []
        for k in self.D_sses.keys():
            self.logger.info(f'Creating latent matrix for sse {idx} and key {k}')
            print(f'Creating latent matrix for sse {idx} and key {k}')
            data = self.D_sses[k][idx]

            latent_vec = interpolate_to_latent(data.sr, data.state, data.slip, data.t, num_of_knots=self.u_to_par_knot_l, num_of_t_knots=self.t_to_u_knot_l, t_knots_placment='both', ratio=0.67, n_workers=self.n_workers)
            if np.sum(np.isnan(latent_vec)) > 0:
                warnings.warn(f'Latent vector for sse {idx} and key {k} contains NaN values')
                self.logger.warning(f'Latent vector for sse {idx} and key {k} contains NaN values')
            latent.append(latent_vec)
        latent = np.stack(latent)
        if save_path is not None:
            np.save(f'{save_path}/latent_{idx}.npy', latent)
        return latent

    def build_latent_matrices(self, save_path: str | None = None) -> None:
        """Build latent matrices for all SSE indices and store in self.latent.

        Args:
            save_path: If provided, save each latent matrix to disk under this
                directory as ``latent_{idx}.npy``.
        """
        self.latent: list[np.ndarray] = []
        with parallel_backend("loky", n_jobs=self.n_workers):
            for i in range(self.sses_num):
                self.latent.append(self.create_one_sse_latent_matrix(i, save_path))

    def save_latent_matrices(self, path: str) -> None:
        """Save all latent matrices to .npy files.

        Args:
            path: Directory in which to save ``latent_0.npy``, ``latent_1.npy``, etc.
        """
        for k, latent in enumerate(self.latent):
            np.save(f'{path}/latent_{k}.npy', latent)

    def load_latent_matrices(self, path: str) -> None:
        """Load latent matrices from .npy files into self.latent.

        Args:
            path: Directory containing ``latent_0.npy``, ``latent_1.npy``, etc.
                Files are loaded in order from 0 to sses_num - 1.
        """
        self.latent = []
        for i in range(self.sses_num):
            self.latent.append(np.load(f'{path}/latent_{i}.npy'))

    def create_interpolation_time(
        self,
        base_step: float = 1e-4,
        depths: int = 9,
    ) -> np.ndarray:
        """Create a normalized interpolation time grid refined near SSE onsets.

        Builds a grid over [0, 1] with a uniform base spacing, then adds
        progressively finer sub-grids near t=0 and t=1 to capture the rapid
        dynamics at SSE onset and termination.

        Args:
            base_step: Uniform time step for the background grid.
            depths: Number of refinement levels. Each level adds a finer
                sub-grid of width 10^(-(level-3)) at each end.

        Returns:
            Sorted, non-uniform time grid, shape [n_points].
        """
        t_interpolate = np.arange(0, 1, base_step)

        for power in range(5, depths + 1, 1):
            interval_length = 10 ** -(power - 3)
            if power == depths:
                interval_length *= 4
            nt = np.arange(0, interval_length, 10 ** -power)
            t_interpolate = np.concatenate([nt, t_interpolate[(t_interpolate > interval_length) & (t_interpolate < 1 - interval_length)], (1 - interval_length) + nt])

        return t_interpolate

    def build_pod(self, r: int | None = None) -> None:
        """Compute POD decomposition via SVD for each SSE latent matrix.

        Decomposes each latent matrix as Latent = A @ U.T and stores the
        truncated components. Results are stored in self.A, self.U, self.S,
        self.V.

        Args:
            r: Number of POD modes to retain. Defaults to the full latent
                dimension (no truncation).
        """
        if r is None:
            r = self.latent[0].shape[1]
        self.r = r
        self.A: list[np.ndarray] = []
        self.U: list[np.ndarray] = []
        self.S: list[np.ndarray] = []
        self.V: list[np.ndarray] = []
        for k, latent in enumerate(self.latent):
            print(k)
            u, s, vh = np.linalg.svd(latent.T, full_matrices=False)
            u = u[:, :r]
            s = s[:r]
            vh = vh[:r, :]
            self.U.append(u)
            self.S.append(s)
            self.V.append(vh)
            a = latent @ u
            self.A.append(a)

    def save_pod_to_files(self, src_dir: str) -> None:
        """Save POD components A, U, S, V to .npy files.

        Args:
            src_dir: Directory to save files. Created if it does not exist.
        """
        os.makedirs(src_dir, exist_ok=True)
        for k in range(len(self.A)):
            np.save(os.path.join(src_dir, f"A_{k}.npy"), self.A[k])
            np.save(os.path.join(src_dir, f"U_{k}.npy"), self.U[k])
            np.save(os.path.join(src_dir, f"S_{k}.npy"), self.S[k])
            np.save(os.path.join(src_dir, f"V_{k}.npy"), self.V[k])
        np.save(os.path.join(src_dir, "r.npy"), self.r)
        print(f"POD components saved to {src_dir}")

    def load_pod_from_files(self, src_dir: str) -> None:
        """Load POD components A, U, S, V from .npy files.

        Args:
            src_dir: Directory containing ``A_k.npy``, ``U_k.npy``, ``S_k.npy``,
                ``V_k.npy`` for k = 0, 1, … and ``r.npy``.
        """
        self.A = []
        self.U = []
        self.S = []
        self.V = []

        k = 0
        while True:
            a_path = os.path.join(src_dir, f"A_{k}.npy")
            if not os.path.exists(a_path):
                break
            self.A.append(np.load(a_path))
            self.U.append(np.load(os.path.join(src_dir, f"U_{k}.npy")))
            self.S.append(np.load(os.path.join(src_dir, f"S_{k}.npy")))
            self.V.append(np.load(os.path.join(src_dir, f"V_{k}.npy")))
            k += 1

        self.r = int(np.load(os.path.join(src_dir, "r.npy")))
        print(f"Loaded {k} POD components from {src_dir}")

    def save_orig_pod(self) -> None:
        """Snapshot current POD components before any slicing.

        Saves copies of U, S, V, A to ``*_orig`` attributes so that
        :meth:`slice_pod` can be called repeatedly without recomputing POD.
        """
        self.U_orig = [u for u in self.U]
        self.S_orig = [s for s in self.S]
        self.V_orig = [vh for vh in self.V]
        self.A_orig = [a for a in self.A]

    def slice_pod(self, r: np.ndarray | list[int] | None = None) -> None:
        """Re-slice POD components to a subset of modes from the saved originals.

        Requires a prior call to :meth:`save_orig_pod`. Recomputes self.A from
        the sliced basis.

        Args:
            r: Index array or slice selecting which modes to keep. Applied to
                columns of U_orig and elements of S_orig/V_orig.
        """
        self.U = [u[:, r] for u in self.U_orig]
        self.S = [s[r] for s in self.S_orig]
        self.V = [vh[r, :] for vh in self.V_orig]
        self.A = []
        for k, latent in enumerate(self.latent):
            a = latent @ self.U[k]
            self.A.append(a)

    def build_rom(self) -> None:
        """Train one RBFInterpolator per SSE mapping parameters → POD amplitudes.

        Fits self.RBFs using the normalized forward model parameters and the
        POD amplitude matrices from self.A.
        """
        self.RBFs: list[RBFInterpolator] = []
        for a in self.A:
            rbf = RBFInterpolator(self.f_params_normalized, a, kernel=self.rbf_kernal)
            self.RBFs.append(rbf)

    def predict(self, w: np.ndarray) -> list[Data]:
        """Predict reconstructed SSE time series for a new parameter vector.

        Args:
            w: Forward model parameter vector, shape [n_features]. Need not
                be normalized; normalization is applied internally.

        Returns:
            List of Data objects of length sses_num, one per SSE. Time is in
            physical seconds.
        """
        w_norm = (w - self.f_mean) / self.f_std
        recostracted_sses = []
        for rbf, u in zip(self.RBFs, self.U):
            apred = rbf(w_norm)
            ypred = (u @ apred.T).reshape(1, -1)
            reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp = inverse_interpolation(ypred[0], self.lf, self.t_to_u_knot_l, self.t_to_u_cof_l, self.u_to_par_knot_l, self.u_to_par_cof_l, t_interp=self.t_interpolate)
            recostracted_sses.append(Data(params=w, sr=reconstructed_sr, state=reconstructed_state, slip=reconstructed_slip, t=t_interp * (365*24*60*60)))
        return recostracted_sses

    def predict_latent(self, w: np.ndarray) -> list[np.ndarray]:
        """Predict latent vectors for a new parameter vector without inversion.

        Args:
            w: Forward model parameter vector, shape [n_features].

        Returns:
            List of latent vectors, shape [1, latent_dim], one per SSE.
        """
        w_norm = (w - self.f_mean) / self.f_std
        latent_sses = []
        for rbf, u in zip(self.RBFs, self.U):
            apred = rbf(w_norm)
            ypred = (u @ apred.T).reshape(1, -1)
            latent_sses.append(ypred)
        return latent_sses

    def leave_one_out(self, leave_one_out: int) -> list[Data]:
        """Perform leave-one-out cross-validation and return reconstructed SSEs.

        Removes simulation at index leave_one_out from training, trains a
        temporary RBF, and reconstructs the held-out simulation's time series.

        Args:
            leave_one_out: Index of the simulation to hold out (0-based).

        Returns:
            List of reconstructed Data objects, one per SSE, for the held-out
            simulation. Time is in physical seconds.
        """
        recostracted_sses = []

        for a, u, s, vh in zip(self.A, self.U, self.S, self.V):
            a_test = a[leave_one_out]
            a_train = np.delete(a, leave_one_out, axis=0)
            y_test = self.f_params_normalized[leave_one_out]
            y_train = np.delete(self.f_params_normalized, leave_one_out, axis=0)
            rbf = RBFInterpolator(y_train, a_train, kernel=self.rbf_kernal)
            apred = rbf(y_test.reshape(1, -1))
            ypred = (u @ apred.T).reshape(1, -1)
            reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp = inverse_interpolation(ypred[0], self.lf, self.t_to_u_knot_l, self.t_to_u_cof_l, self.u_to_par_knot_l, self.u_to_par_cof_l, t_interp=self.t_interpolate)
            recostracted_sses.append(Data(self.f_params[leave_one_out], sr=reconstructed_sr, state=reconstructed_state, slip=reconstructed_slip, t=t_interp * (365*24*60*60)))
        return recostracted_sses

    def leave_one_out_minimal(
        self,
        leave_one_out: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Leave-one-out cross-validation returning only POD amplitude arrays.

        Cheaper than :meth:`leave_one_out` because it skips the inverse spline
        reconstruction.

        Args:
            leave_one_out: Index of the simulation to hold out (0-based).

        Returns:
            Tuple (a_tests, a_preds) where each is a list of arrays (one per SSE).
            a_tests contains the true held-out amplitudes; a_preds the predicted.
        """
        a_tests = []
        a_preds = []

        for latent, a, u, s, vh in zip(self.latent, self.A, self.U, self.S, self.V):
            a_test = a[leave_one_out]
            a_train = np.delete(a, leave_one_out, axis=0)
            y_test = self.f_params_normalized[leave_one_out]
            y_train = np.delete(self.f_params_normalized, leave_one_out, axis=0)
            rbf = RBFInterpolator(y_train, a_train, kernel='linear')
            apred = rbf(y_test.reshape(1, -1))
            a_tests.append(a_test)
            a_preds.append(apred)
        return a_tests, a_preds

    def leave_one_out_latent(self, leave_one_out: int) -> list[np.ndarray]:
        """Leave-one-out cross-validation returning reconstructed latent vectors.

        Args:
            leave_one_out: Index of the simulation to hold out (0-based).

        Returns:
            List of reconstructed latent vectors, shape [1, latent_dim], one
            per SSE.
        """
        latents = []

        for latent, a, u, s, vh in zip(self.latent, self.A, self.U, self.S, self.V):
            a_test = a[leave_one_out]
            a_train = np.delete(a, leave_one_out, axis=0)
            y_test = self.f_params_normalized[leave_one_out]
            y_train = np.delete(self.f_params_normalized, leave_one_out, axis=0)
            rbf = RBFInterpolator(y_train, a_train, kernel='linear')
            apred = rbf(y_test.reshape(1, -1))
            ypred = (u @ apred.T).reshape(1, -1)
            latents.append(ypred)
        return latents

    def build_reconstructed_time_series(self, reconstructed_sses: list[Data]) -> Data:
        """Concatenate per-SSE Data objects into a single continuous time series.

        Stitches SSE windows together by discarding any overlap (keeping points
        where t < next SSE's t_min) and concatenating in chronological order.

        Args:
            reconstructed_sses: List of Data objects in chronological order,
                one per SSE.

        Returns:
            Single Data object spanning all SSEs. Time is in physical seconds.
        """
        sr = np.copy(reconstructed_sses[0].sr)
        state = np.copy(reconstructed_sses[0].state)
        slip = np.copy(reconstructed_sses[0].slip)
        t = np.copy(reconstructed_sses[0].t)
        for sse in reconstructed_sses[1:]:
            mask = t < sse.t.min()
            sr = np.concatenate([sr[:, mask], sse.sr], axis=1)
            state = np.concatenate([state[:, mask], sse.state], axis=1)
            slip = np.concatenate([slip[:, mask], sse.slip], axis=1)
            t = np.concatenate([t[mask], sse.t])
        return Data(params=reconstructed_sses[0].params, sr=sr, state=state, slip=slip, t=t * (365*24*60*60))
