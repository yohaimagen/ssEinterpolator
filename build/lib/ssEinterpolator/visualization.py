"""
Visualization module for ssEinterpolator.
"""

import matplotlib.pyplot as plt

def plot_depth(depth, axs):
    idx = np.argmin(np.abs(lf - depth))
    mask = (data['t'] > 80) & (data['t'] < 110)
    sr = np.log10(np.abs(data['sr'][idx][mask]))
    state = np.copy(data['state'][idx][mask])
    t = np.copy(data['t'][mask])
    tck, u = interpolate_to_latent_single_along_stk(sr, state)
    p_interp, s_interp = splev(u, tck)
    state_interp = p_interp * (state.max() - state.min()) + state.min()
    sr_interp = s_interp * (sr.max() - sr.min()) + sr.min()
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.scatter(state, sr, label='data', s=3)
    ax.scatter(state_interp, sr_interp, s=1, label='spline', zorder=10)
    ax.set_ylabel('log10(slip rate)')
    ax.set_xlabel('state')
    ax.legend(loc='best')
    ax = axs[1]
    ax.plot(t, state)
    ax.plot(t, state_interp)
    ax.set_ylabel('state')
    ax.set_xlabel('time')
    ax = axs[2]
    ax.plot(t, sr)
    ax.plot(t, sr_interp)
    ax.set_ylabel('log10(slip rate)')
    ax.set_xlabel('time')