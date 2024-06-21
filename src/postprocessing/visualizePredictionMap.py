import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Colormap


def VisualiseMap(data, cmap, use_log_norm=False, vmin=None, vmax=None, cbar_label = None):
    """
    Visualize a map using matplotlib.

    Parameters:
        data (numpy.ndarray): Array containing the data to be visualized.
        cmap (str or matplotlib.colors.Colormap): Colormap to be used for visualization.
        use_log_norm (bool): Whether to use logarithmic normalization for color scaling.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.
    """
    plt.figure(figsize=(20, 10), dpi=300)
    
    if use_log_norm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        plt.imshow(data, cmap=cmap, norm=norm, extent=[-180, 180, -90, 90])
    else:
        norm = None
        plt.imshow(data, cmap=cmap, vmax = vmax, vmin = vmin, extent=[-180, 180, -90, 90])
    
    
    
    cbar_label = cbar_label
    
    cbar = plt.colorbar(label=cbar_label, aspect=40, pad=0.01, shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(cbar_label, fontsize=20)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Longitude", fontsize=20)
    plt.ylabel("Latitude", fontsize=20)
    plt.tight_layout()
    
    plt.show()