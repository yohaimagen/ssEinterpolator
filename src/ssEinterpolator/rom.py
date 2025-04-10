import numpy as np
from scipy.interpolate import RBFInterpolator
from multiprocessing import Pool
from .interpolation import *
from .data import Data
from .utils import find_slip_events
import warnings


import logging
import datetime
import sys
log_file = f"rom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("multiprocessing_logger")
logger.setLevel(logging.DEBUG)
# Prevent adding handlers multiple times
if not logger.handlers:
    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(processName)s - %(message)s"
    )
    
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create a stream handler (optional, for console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def process_data(args):
    logger.info(f'Processing data for {args[1]}')
    print(args)
    self, f_param, str_param = args  # Include 'self' to access instance variables
    print(f'processing {str_param}')
    key = '_'.join(str_param)
    data = Data(load_f=self.load_f, str_params=str_param, params=f_param)
    data = data.mask_data(self.t_start, self.t_end)
    print(f'loading data for key={key} with shape {data.sr.shape}')
    idx = np.argmin(np.abs(self.lf - self.along_dp_sses_depth_detector))
    sses = find_slip_events(data.t, np.log10(np.abs(data.sr[idx])), threshold=self.sses_detector_threshold)
    print(f'found {sses.shape[0]} sses for key={key}')
    if sses.shape[0] < self.sses_starts + self.sses_num:
        raise ValueError(f'Not enough slip events detected for key={key}')
    split_data = self.split_data_by_sse(data, sses)
    split_data[self.sses_starts: self.sses_starts + self.sses_num + 1]
    # data = data.mask_data(split_data[self.sses_starts].t.min(), split_data[:self.sses_starts + self.sses_num][-1].t.max())
    logger.info(f'Finished processing data for {key} found {sses.shape[0]} sses')
    return key, split_data

def calc_latent_helper(args):
    self, idx = args
    print(f'creating latent matrix for sse {idx}')
    # latent = self.create_one_sse_latent_matrix(idx)
    latent = 1
    return latent

class ROM:
    def __init__(self, f_params, str_params, load_f, lf_path, sses_num, t_to_u_knot_l, u_to_par_knot_l, along_dp_sses_depth_detector = 195, sses_detector_threshold=-4, sses_starts=0, t_start=0, t_end=1e99, base_step_t_interpolate=1e-4, depths_t_interpolate=9):
        self.lf = np.load(lf_path)
        self.f_params = f_params
        self.str_params = str_params
        self.load_f = load_f
        self.t_start = t_start
        self.t_end = t_end
        self.along_dp_sses_depth_detector = along_dp_sses_depth_detector
        self.sses_detector_threshold = sses_detector_threshold
        self.sses_starts = sses_starts
        self.base_step_t_interpolate = base_step_t_interpolate
        self.depths_t_interpolate = depths_t_interpolate
        self.D = {}
        self.D_sses = {}
        self.sses_num = sses_num
        self.update_spline_length(t_to_u_knot_l, u_to_par_knot_l)
        self.t_interpolate = self.create_interpolation_time(base_step=base_step_t_interpolate, depths=depths_t_interpolate)
        # for f_param, str_param in zip(f_params, str_params):
        #     key = '_'.join(str_param)
        #     data = Data(load_f=load_f, str_params=str_param, params=f_param)
        #     data = data.mask_data(t_start, t_end)
        #     print(f'loading data for key={key} with shape {data.sr.shape}')
        #     idx = np.argmin(np.abs(self.lf - along_dp_sses_depth_detector))
        #     sses = find_slip_events(data.t, np.log10(np.abs(data.sr[idx])), threshold=sses_detector_threshold)
        #     if sses.shape[0] < sses_starts + sses_num:
        #         raise ValueError(f'Not enough slip events detected for key={key}')
        #     split_data = self.split_data_by_sse(data, sses)
        #     self.D_sses[key] = split_data[sses_starts: sses_starts + sses_num + 1]
        #     self.D[key] = data.mask_data(split_data[sses_starts].t.min(), split_data[:sses_starts + sses_num][-1].t.max())
        
        # print(self.str_params.shape)
        # print(str_params.shape)
        logger.info(f'Creating {sses_num} sses for {len(f_params)} params')
        with Pool(processes=30) as pool:
            results = pool.map(process_data, [(self, self.f_params[k], self.str_params[k]) for k in range(len(self.f_params))])

        for key, split_data in results:
            self.D_sses[key] = split_data[self.sses_starts: self.sses_starts + self.sses_num + 1]
            # self.D[key] = data
        
    
    def update_spline_length(self, t_to_u_knot_l, u_to_par_knot_l):
        self.t_to_u_knot_l = t_to_u_knot_l
        self.u_to_par_knot_l = u_to_par_knot_l
        self.t_to_u_cof_l = self.t_to_u_knot_l - 4
        self.u_to_par_cof_l = self.u_to_par_knot_l - 4
        self.dp_laten_vec_length = self.t_to_u_knot_l + self.t_to_u_cof_l + self.u_to_par_knot_l +self. u_to_par_cof_l * 2 + 4
        
    def split_data_by_sse(self, data, sses):
        split_data = []
        for i in range(len(sses) - 1):
            split_data.append(data.mask_data(sses[i], sses[i + 1]))
        return split_data
    
    def create_one_sse_latent_matrix(self, idx, save_path=None):
        logger.info(f'Creating latent matrix for sse {idx}')
        if idx >= self.sses_num or idx < 0:
            raise ValueError('idx should be in range of sses_num')
        print(f'creating latent matrix for sse {idx}')
        latent = []
        for k in self.D_sses.keys():
            logger.info(f'Creating latent matrix for sse {idx} and key {k}')
            data = self.D_sses[k][idx]

            latent_vec = interpolate_to_latent(data.sr, data.state, data.slip, data.t, num_of_knots=self.u_to_par_knot_l, num_of_t_knots=self.t_to_u_knot_l, t_knots_placment='both', ratio=0.67)
            if np.sum(np.isnan(latent_vec)) > 0:
                warnings.warn(f'Latent vector for sse {idx} and key {k} contains NaN values')
                logger.warning(f'Latent vector for sse {idx} and key {k} contains NaN values')
            latent.append(latent_vec)
        latent = np.stack(latent)
        if save_path is not None:
            np.save(f'{save_path}/latent_{idx}.npy', latent)
        return latent

    
    def build_latent_matrices(self, save_path=None):
        self.latent = []
        for i in range(self.sses_num):
            self.latent.append(self.create_one_sse_latent_matrix(i, save_path))
            
    
   
            
    def save_latent_matrices(self, path):
        for k, latent in enumerate(self.latent):
            np.save(f'{path}/latent_{k}.npy', latent)
            
    def load_latent_matrices(self, path):
        self.latent = []
        for i in range(self.sses_num):
            self.latent.append(np.load(f'{path}/latent_{i}.npy'))
            
    def create_interpolation_time(self, base_step=1e-4, depths=9):
        """Create interpolation time points with refinement around slip events.
        
        Args:
            t_start: Start time
            t_end: End time
            sses_with_depths: List of tuples (sse_time, intervals_depth)
            base_step: Base time step for interpolation
        """
        t_interpolate = np.arange(0, 1, base_step)
        
        for power in range(5, depths + 1, 1):
            interval_length = 10 ** -(power - 3)
            if power == depths:
                interval_length *= 4
            nt = np.arange(0, interval_length, 10 ** -power)
            t_interpolate = np.concatenate([nt, t_interpolate[(t_interpolate > interval_length) & (t_interpolate < 1 - interval_length)], (1 - interval_length) + nt])
        
        return t_interpolate
    
    def build_pod(self):
        self.A = []
        self.U = []
        self.S = []
        self.V = []
        for k, latent in enumerate(self.latent):
            print(k)
            u, s, vh = np.linalg.svd(latent.T, full_matrices=False)
            self.U.append(u)
            self.S.append(s)
            self.V.append(vh)
            a = latent @ u
            self.A.append(a)
            
    def build_rom(self):
        self.RBFs = []
        for a in self.A:
            rbf = RBFInterpolator(self.f_params, a, kernel='linear')
            self.RBFs.append(rbf)
    
    def predict(self, w):
        recostracted_sses = []
        for rbf, u in zip(self.RBFs, self.U):
            apred = rbf(w)
            ypred = (u @ apred.T).reshape(1, -1)
            reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp = inverse_interpolation(ypred[0], self.lf, self.t_to_u_knot_l, self.t_to_u_cof_l, self.u_to_par_knot_l, self.u_to_par_cof_l, t_interp=self.t_interpolate)
            recostracted_sses.append(Data(sr=reconstructed_sr, state=reconstructed_state, slip=reconstructed_slip, t=t_interp * (365*24*60*60)))
        return recostracted_sses
            
    def leave_one_out(self, leave_one_out):
        recostracted_sses = []
        # Mean and standard deviation per dimension
        mean = np.mean(self.f_params, axis=0)
        std = np.std(self.f_params, axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

        # Normalized coordinates
        f_params_normalized = (self.f_params - mean) / std
        
        for latent, a, u, s, vh in zip(self.latent, self.A, self.U, self.S, self.V):
            a_test = a[leave_one_out]
            a_train = np.delete(a, leave_one_out, axis=0)
            y_test = f_params_normalized[leave_one_out]
            y_train = np.delete(f_params_normalized, leave_one_out, axis=0)
            rbf = RBFInterpolator(y_train, a_train, kernel='linear')
            apred = rbf(y_test.reshape(1, -1))
            ypred = (u @ apred.T).reshape(1, -1)
            reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp = inverse_interpolation(ypred[0], self.lf, self.t_to_u_knot_l, self.t_to_u_cof_l, self.u_to_par_knot_l, self.u_to_par_cof_l, t_interp=self.t_interpolate)
            recostracted_sses.append(Data(self.f_params[leave_one_out] ,sr=reconstructed_sr, state=reconstructed_state, slip=reconstructed_slip, t=t_interp * (365*24*60*60)))
        return recostracted_sses
    
    
    def build_reconstructed_time_series(self, reconstructed_sses):
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
            
    
        
        


