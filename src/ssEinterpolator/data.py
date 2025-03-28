import numpy as np

class Data:
    def __init__(self, params, load_f=None, str_params=None, sr=None, slip=None, state=None, t=None, prefix='', supix = ''):
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
            self.t -= self.t[0]
        self.params = params
    
    def mask_data(self, tl, tr):
        mask = (self.t >= tl) & (self.t < tr)
        masked_data = Data(params=self.params, load_f=None, str_params=None, sr=np.copy(self.sr[:, mask]), slip=np.copy(self.slip[:, mask]), state=np.copy(self.state[:, mask]), t=np.copy(self.t[mask]) * (365*24*60*60))
        return masked_data