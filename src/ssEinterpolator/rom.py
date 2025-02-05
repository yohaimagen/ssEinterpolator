import numpy as np
from scipy.interpolate import RBFInterpolator
from .interpolation import *
from .data import Data
from .utils import find_slip_events

class ROM:
    def __init__(self, prefixes, numerals, dir, lf_path, sses_num, t_to_u_knot_l, u_to_par_knot_l, supix = '', along_dp_sses_depth_detector = 195, sses_detector_threshold=-4, sses_starts=0, t_start=0, t_end=1e99, base_step_t_interpolate=1e-4, depths_t_interpolate=9):
        self.lf = np.load(lf_path)
        self.D = {}
        self.sses_num = sses_num
        self.t_to_u_knot_l = t_to_u_knot_l
        self.u_to_par_knot_l = u_to_par_knot_l
        self.t_to_u_cof_l = self.t_to_u_knot_l - 4
        self.u_to_par_cof_l = self.u_to_par_knot_l - 4
        self.dp_laten_vec_length = self.t_to_u_knot_l + self.t_to_u_cof_l + self.u_to_par_knot_l +self. u_to_par_cof_l * 2 + 4
        self.t_interpolate = self.create_interpolation_time(base_step=base_step_t_interpolate, depths=depths_t_interpolate)
        self.numerals = np.array(numerals).reshape(-1, 1)
        for prefix in prefixes:
            data = Data(dir=dir, supix=supix, prefix=prefix)
            self.D[prefix] = data.mask_data(t_start, t_end)
        self.sample_data_by_sses(sses_num, sses_starts, along_dp_sses_depth_detector, sses_detector_threshold)
        
    

    def split_data_by_sse(self, data, sses):
        split_data = []
        for i in range(len(sses) - 1):
            split_data.append(data.mask_data(sses[i], sses[i + 1]))
        return split_data
    
    def sample_data_by_sses(self, sses_num, sses_starts, along_dp_sses_depth_detector=195, sses_detector_threshold=-4):
        self.D_sses = {}
        self.Dm = {}
        self.sses_num = sses_num
        for w in self.D:
            data = self.D[w]
            print(data.sr.shape)
            idx = np.argmin(np.abs(self.lf - along_dp_sses_depth_detector))
            sses = find_slip_events(data.t, np.log10(np.abs(data.sr[idx])), threshold=sses_detector_threshold)
            if sses.shape[0] < sses_starts + sses_num:
                raise ValueError(f'Not enough slip events detected for w={w}')
            split_data = self.split_data_by_sse(data, sses)
            self.D_sses[w] = split_data[sses_starts: sses_starts + sses_num + 1]
            self.Dm[w] = data.mask_data(split_data[sses_starts].t.min(), split_data[:sses_starts + sses_num][-1].t.max())
    
    def create_one_sse_latent_matrix(self, idx):
        if idx >= self.sses_num or idx < 0:
            raise ValueError('idx should be in range of sses_num')
        latent = []
        for k in self.D_sses.keys():
            data = self.D_sses[k][idx]

            latent_vec = interpolate_to_latent(data.sr, data.state, data.slip, data.t, num_of_knots=self.u_to_par_knot_l, num_of_t_knots=self.t_to_u_knot_l, t_knots_placment='both', ratio=0.8)
            latent.append(latent_vec)
        latent = np.stack(latent)
        return latent
    
    def build_latent_matrices(self):
        self.latent = []
        for i in range(self.sses_num):
            self.latent.append(self.create_one_sse_latent_matrix(i))
            
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
        for latent in self.latent:
            u, s, vh = np.linalg.svd(latent.T, full_matrices=False)
            self.U.append(u)
            self.S.append(s)
            self.V.append(vh)
            a = latent @ u
            self.A.append(a)
            
    def build_rom(self):
        self.RBFs = []
        for a in self.A:
            rbf = RBFInterpolator(self.numerals, a, kernel='linear')
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
        
        for latent, a, u, s, vh in zip(self.latent, self.A, self.U, self.S, self.V):
            a_test = a[leave_one_out]
            a_train = np.delete(a, leave_one_out, axis=0)
            y_test = self.numerals[leave_one_out]
            y_train = np.delete(self.numerals, leave_one_out, axis=0)
            rbf = RBFInterpolator(y_train, a_train, kernel='linear')
            apred = rbf(y_test.reshape(1, -1))
            ypred = (u @ apred.T).reshape(1, -1)
            reconstructed_sr, reconstructed_state, reconstructed_slip, t_interp = inverse_interpolation(ypred[0], self.lf, self.t_to_u_knot_l, self.t_to_u_cof_l, self.u_to_par_knot_l, self.u_to_par_cof_l, t_interp=self.t_interpolate)
            recostracted_sses.append(Data(sr=reconstructed_sr, state=reconstructed_state, slip=reconstructed_slip, t=t_interp * (365*24*60*60)))
        return recostracted_sses
    
    # def build_reconstructed_time_series(self, reconstructed_sses):
    #     sr = [reconstructed_sse.sr for reconstructed_sse in reconstructed_sses]
    #     state = [reconstructed_sse.state for reconstructed_sse in reconstructed_sses]
    #     slip = [reconstructed_sse.slip for reconstructed_sse in reconstructed_sses]
    #     t = [reconstructed_sse.t for reconstructed_sse in reconstructed_sses]
    #     sr = np.concatenate(sr, axis=1)
    #     state = np.concatenate(state, axis=1)
    #     slip = np.concatenate(slip, axis=1)
    #     t = np.concatenate(t)
    #     return Data(sr=sr, state=state, slip=slip, t=t * (365*24*60*60))
    
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
        return Data(sr=sr, state=state, slip=slip, t=t * (365*24*60*60))
            
    
        
        


