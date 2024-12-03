"""
Visualization module for ssEinterpolator.
"""

import matplotlib.pyplot as plt

def plot_results(t, sr, state):
    """Plot the results of the interpolation."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(t, sr)
    axs[0].set_title('Slip Rate')
    axs[1].plot(t, state)
    axs[1].set_title('State')
    plt.show()