# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import sys

sys.path.append(os.path.abspath('..'))

plt.style.use('../paper_mpl_style.mplstyle')

parent_dir = Path(__file__).parent.parent
parent_str = str(parent_dir)

if parent_str not in sys.path:
    sys.path.insert(0, parent_str)
    added = True
else:
    added = False

try:
    import helper_functions as hf
finally:
    if added:
        sys.path.pop(0)  # Safe cleanup only if we added it

save_fig = True

colour_diff = 'C0'
colour_sum = 'C1'
colour_ell = 'C2'


# =============================================================================
# Data Import
# =============================================================================

signal_0 = np.load('../exp_data/Contrast_Envelope/signal_0.npy')
fit_peac = np.load('../exp_eval/exp_contrast_envelope_gauss_fits.npy')

delta_T = np.linspace(-150,150,31)

def gauss(x,amp,x0,sig,off):
    return amp * np.exp(-(x-x0)**2/(2*sig**2)) + off

fig, axs = plt.subplots(1,1,
                        figsize=(3.54, 3.54*2/(1+np.sqrt(5))))

for i, t in enumerate(delta_T):
    y_all = signal_0[:,i].ravel()                         # Flatten 2D data array for histogram computation
    v_max_all = y_all.max()                          # Determine maximum value for scaling reference
    bins_all = int(np.sqrt(len(y_all)) * (1 / abs(y_all).max()))  # Dynamically set bin count based on data size

    # Compute normalized histogram of the population distribution within [-1, 1]
    hist_all, x_all = np.histogram(y_all, bins=bins_all, range=(-1, 1), density=True)

    # Construct rectangular grid coordinates for the pseudocolor mesh
    X = np.array([t - np.diff(delta_T)[0] / 2, t + np.diff(delta_T)[0] / 2])  # Horizontal (time) bounds per slice
    Y = x_all                                                             # Vertical data range (signal axis)
    Z = hist_all[:, np.newaxis]                                           # Convert histogram to column vector

    # Render the 2D color mesh (rasterized for efficient export)
    axs.pcolormesh(X, Y, Z, rasterized=True,cmap='inferno_r')
    
axs.plot(delta_T,gauss(delta_T,*fit_peac.mean(axis=0))+0.036 ,c='w')
axs.plot(delta_T,-gauss(delta_T,*fit_peac.mean(axis=0))+0.036 ,c='w')

plt.xlabel(r'$\delta T$ (µs)',fontsize=8)
plt.ylabel('signal',fontsize=8,labelpad=-6)

plt.yticks([1,0.5,0,-0.5,-1],[1,0.5,0,-0.5,-1],fontsize=8)
plt.xticks(fontsize=8)
plt.subplots_adjust(left=0.1, 
                    right=0.985, 
                    top = 0.97, 
                    bottom = 0.17 
                    )

plt.savefig('fig2.pdf')