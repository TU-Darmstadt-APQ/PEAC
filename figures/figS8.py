# -*- coding: utf-8 -*-
"""
@author: Dominik Pfeiffer, Daniel Derr & Ludwig Lind
"""

"""
    Plot of sum of bias and uncertainty
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

# Results of numerical MLE evaluation
num_eval_theta_scan_MLE     = np.load('../num_eval/num_eval_res_theta_S_sum_MLE.npz')

thetas_set_MLE              = num_eval_theta_scan_MLE['thetas_set']

thetas_rec_sum_num_MLE      = num_eval_theta_scan_MLE['thetas_rec_sum']
# thetas_rec_diff_num_MLE     = num_eval_theta_scan_MLE['thetas_rec_diff']

thetas_rec_sum_std_num_MLE  = num_eval_theta_scan_MLE['thetas_rec_sum_std']
# thetas_rec_diff_std_num_MLE = num_eval_theta_scan_MLE['thetas_rec_diff_std']

# A0_set_MLE                  = num_eval_theta_scan_MLE['A0_set']
# A_sum_set_MLE               = num_eval_theta_scan_MLE['A_sum_set']
# A_sum_rec_num_MLE           = num_eval_theta_scan_MLE['A_sum_rec']
# A_diff_set_MLE              = num_eval_theta_scan_MLE['A_diff_set']
# A_diff_rec_num_MLE          = num_eval_theta_scan_MLE['A_diff_rec']

# Results of numerical PEAC ##
num_eval_theta_scan_PEAC     = np.load('../num_eval/num_eval_res_theta_S_sum_PEAC.npz')

thetas_set_PEAC              = num_eval_theta_scan_PEAC['thetas_set']

thetas_rec_sum_num_PEAC      = num_eval_theta_scan_PEAC['thetas_rec_sum']
# thetas_rec_diff_num_PEAC     = num_eval_theta_scan_PEAC['thetas_rec_diff']

thetas_rec_sum_std_num_PEAC  = num_eval_theta_scan_PEAC['thetas_rec_sum_std']
# thetas_rec_diff_std_num_PEAC = num_eval_theta_scan_PEAC['thetas_rec_diff_std']


# Results of geometric ellipse fits 
num_eval_theta_scan_geom = np.load('../num_eval/num_eval_res_theta_S_sum_geo_ell.npz')

thetas_set_geom              = num_eval_theta_scan_geom['thetas_set']

thetas_rec_ell_num_geom      = num_eval_theta_scan_geom['thetas_rec_ell']

thetas_rec_ell_std_num_geom  = num_eval_theta_scan_geom['thetas_rec_ell_std']

A0_set_geom                  = num_eval_theta_scan_geom['A0_set']
sigma_set_geom               = num_eval_theta_scan_geom['sigma_set']

theta_change = hf.theta_change(A0_set_geom, sigma_set_geom, np.pi/4)

#############
### plots ###
#############
inch_to_cm = 2.54
phi_golden = (1 + np.sqrt(5)) / 2
width_inch = 17/inch_to_cm
height_inch = width_inch / (2*phi_golden)

start = 166
stop = 501

"""
    Scatter plot of Bias vs. Uncertainty
"""
markersize = 2

fig, axs = plt.subplots(1,1,
                        figsize=(3.54, 3.54/phi_golden))

axs.scatter((thetas_rec_sum_num_PEAC[start:stop]-thetas_set_PEAC[start:stop])/np.pi * 1e3, 
            thetas_rec_sum_std_num_PEAC[start:stop]/np.pi * 1e3,
            label='LSQ', 
            color=colour_sum, 
            s = markersize)

axs.scatter((thetas_rec_sum_num_MLE[start:stop]-thetas_set_MLE[start:stop])/np.pi * 1e3, 
            thetas_rec_sum_std_num_MLE[start:stop]/np.pi * 1e3,
            label='MLE', 
            color='tab:green', 
            s = markersize,
            marker = 'v')

axs.scatter((thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop])/np.pi * 1e3, 
            thetas_rec_ell_std_num_geom[start:stop]/np.pi * 1e3,
            label='geom. ellipses', 
            color=colour_ell, 
            s = markersize,
            marker = 's')

axs.plot([0,max((thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop])/np.pi) * 1e3], 
         [0,max((thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop])/np.pi) * 1e3],
         'k-',
         lw=1,
         zorder=-1)
axs.plot([0,-max((thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop])/np.pi) * 1e3], 
         [0,max((thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop])/np.pi) * 1e3],
         'k-',
         lw=1,
         zorder=-1)


plt.xlabel(r'$\theta_\text{bias}/\pi \times 10^{-3}$')
plt.ylabel(r'$\Delta\theta/ \pi \times 10^{-3}$', labelpad=2)

plt.legend(loc='upper center',
           # bbox_to_anchor = (0,0.5),
           handletextpad=0.1,
           borderpad=0.2,
           ncols=3,
           columnspacing = 1)

plt.subplots_adjust(left=0.12,
                    top=0.98,
                    bottom=0.18,
                    right=0.99)

plt.savefig('figS8.pdf')
plt.show()