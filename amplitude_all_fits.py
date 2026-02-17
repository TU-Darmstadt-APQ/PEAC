# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as c 

from helper_functions import Amp_all

plt.style.use('paper_mpl_style.mplstyle')

"""
Data import, processing, and visualisation for interferometer analysis.
This code loads experimental datasets, processes them, applies masks, 
and visualises the interferometer sequence with custom colour maps 
and annotated insets representing atomic states.
"""

# =============================================================================
# Global Style and Setup
# =============================================================================

# Define interferometer times (ms)
times = np.linspace(1, 3, 21)
times_fine = np.linspace(0.95, 3.05, 201)

# =============================================================================
# Data Import
# =============================================================================
# Histogram results
histogram_results_all = np.load('exp_eval/exp_coarse_S_all_PEAC_results.npz')['results_histogram']
histogram_results_mf = np.load('exp_eval/exp_coarse_S_mF_PEAC_results.npz')['results_histogram']

# =============================================================================
# Fit routine for amplituden A_all
# =============================================================================

fit_params = [] #A0, a, p0, dp

for i in range(histogram_results_all.shape[1]):
    params,_ = c(Amp_all, 
                 times*1e-3, 
                 histogram_results_all[:,i,0],
                 p0=(np.mean((histogram_results_mf[:, :, 0], histogram_results_mf[:, :, 3], histogram_results_mf[:, :, 6])), 
                     31.1e-3, 
                     0.391, 
                     0.193, 
                     ))
    fit_params.append(params)
    
plt.plot(times,histogram_results_all.mean(axis=1)[:,0])
plt.plot(times_fine,Amp_all(times_fine*1e-3,*np.mean(fit_params,axis=0)))

plt.show()

print("A0, a, p0, dp")
print("mean:", np.mean(fit_params,axis=0))
print("std: ", np.std(fit_params,axis=0,ddof=1))

# np.save('exp_eval/amplitude_all_results.npy', fit_params)