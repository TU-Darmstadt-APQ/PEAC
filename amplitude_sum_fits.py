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


from helper_functions import Amp_sum

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
histogram_results_sum = np.load('exp_eval/exp_coarse_S_alpha_PEAC_and_ellipse_results.npz')['results_histogram']

# =============================================================================
# Fit routine for amplitude A_sum
# =============================================================================

fit_params = [] #A0, a

for i in range(histogram_results_sum.shape[1]):
    params,_ = c(Amp_sum, 
                 times*1e-3, 
                 histogram_results_sum[:,i,6], 
                 p0=(np.mean((histogram_results_sum[:, :, 0], histogram_results_sum[:, :, 3])), 
                     31.1e-3, 
                     ))
    fit_params.append(params)
    
plt.plot(times,histogram_results_sum.mean(axis=1)[:,6])
plt.plot(times_fine,Amp_sum(times_fine*1e-3,*np.mean(fit_params,axis=0)))

plt.show()

print("A0, a")
print("mean:", np.mean(fit_params,axis=0))
print("std: ", np.std(fit_params,axis=0,ddof=1))

# np.save('exp_eval/amplitude_sum_results.npy', fit_params)