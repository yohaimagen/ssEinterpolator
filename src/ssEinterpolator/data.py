import numpy as np

class Data:
    def __init__(self, dir=None, sr=None, slip=None, state=None, t=None, prefix='', supix = ''):
        if dir is None and sr is None and slip is None and state is None and t is None:
            raise ValueError('Either dir or sr, slip, state, t must be provided')
        if dir is None:
            if sr is None or slip is None or state is None or t is None:
                raise ValueError('if dir not provided sr, slip, state, t must be provided')
            self.sr = sr
            self.slip = slip
            self.state = state
            self.t = t / (365*24*60*60)
        else:
            def load_data(file):
                # print(f'loading {file}')
                data = np.load(file)
                print(f'loaded {file} of shape {data.shape}')
                return np.load(file)
            self.sr = load_data(f'{dir}/{prefix}sr{supix}.npy')
            self.state = load_data(f'{dir}/{prefix}state{supix}.npy')
            self.slip = load_data(f'{dir}/{prefix}slip{supix}.npy')
            self.t = load_data(f'{dir}/{prefix}t{supix}.npy') / (365*24*60*60)
            self.t -= self.t[0]
    
    def mask_data(self, tl, tr):
        mask = (self.t >= tl) & (self.t < tr)
        masked_data = Data(None, sr=np.copy(self.sr[:, mask]), slip=np.copy(self.slip[:, mask]), state=np.copy(self.state[:, mask]), t=np.copy(self.t[mask]) * (365*24*60*60))
        return masked_data