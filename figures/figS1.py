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

# Results of numerical PEAC ##
num_eval_theta_scan_PEAC     = np.load('../num_eval/num_eval_res_theta_S_sum_PEAC.npz')

thetas_set_PEAC              = num_eval_theta_scan_PEAC['thetas_set']

thetas_rec_ell_num      = num_eval_theta_scan_PEAC['thetas_rec_ell']

thetas_rec_ell_std_num  = num_eval_theta_scan_PEAC['thetas_rec_ell_std']


# Results of geometric ellipse fits 
num_eval_theta_scan_geom = np.load('../num_eval/num_eval_res_theta_S_sum_geo_ell.npz')

thetas_set_geom              = num_eval_theta_scan_geom['thetas_set']

thetas_rec_ell_num_geom      = num_eval_theta_scan_geom['thetas_rec_ell']

thetas_rec_ell_std_num_geom  = num_eval_theta_scan_geom['thetas_rec_ell_std']

A0_set_geom                  = num_eval_theta_scan_geom['A0_set']
sigma_set_geom               = num_eval_theta_scan_geom['sigma_set']

#############
### plots ###
#############
inch_to_cm = 2.54
phi_golden = (1 + np.sqrt(5)) / 2
width_inch = 17/inch_to_cm
height_inch = width_inch / (2*phi_golden)

start = 166
stop = 501


### theta bias plot ####
fig_theta_bias, axs = plt.subplots(1,2,
                                   figsize = (width_inch,height_inch))
ax_theta_bias = axs[0]

# ax_theta_bias.grid(True)
ax_theta_bias.minorticks_on()
# ax_theta_bias.grid(which='minor', linestyle=':', linewidth=0.6)

ax_theta_bias.text(
    0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'A',
    # fontsize=9,
    fontweight='bold',
    transform=ax_theta_bias.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)
## bias sum histogram ##

ax_theta_bias.plot(thetas_set_PEAC[start:stop]/np.pi, (thetas_rec_ell_num[start:stop]-thetas_set_PEAC[start:stop])/np.pi*1e3, color=colour_sum,
                    linewidth=1, label=r'alg. ellipses')
ax_theta_bias.fill_between(thetas_set_PEAC[start:stop]/np.pi,
                            (thetas_rec_ell_num[start:stop]-thetas_set_PEAC[start:stop] - 1*thetas_rec_ell_std_num[start:stop])/np.pi*1e3,
                            (thetas_rec_ell_num[start:stop]-thetas_set_PEAC[start:stop] + 1*thetas_rec_ell_std_num[start:stop])/np.pi*1e3,
                            color=colour_sum, alpha=0.3)

ax_theta_bias.plot(thetas_set_geom[start:stop]/np.pi, (thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop])/np.pi*1e3, color=colour_ell,
                    linewidth=1, label=r'geom. ellipses',zorder=-1)
ax_theta_bias.fill_between(thetas_set_geom[start:stop]/np.pi,
                            (thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop] - 1*thetas_rec_ell_std_num_geom[start:stop])/np.pi*1e3,
                            (thetas_rec_ell_num_geom[start:stop]-thetas_set_geom[start:stop] + 1*thetas_rec_ell_std_num_geom[start:stop])/np.pi*1e3,
                            color=colour_ell, alpha=0.3)

theta_change = hf.theta_change(A0_set_geom, sigma_set_geom, np.pi/4)
ax_theta_bias.axvline(theta_change/np.pi, color="black", linewidth=1, ls="--")
ax_theta_bias.axvline(2-theta_change/np.pi, color="black", linewidth=1, ls="--")

ax_theta_bias.set_xlabel(r'$\theta_\text{set}/\pi$')
ax_theta_bias.set_ylabel(r'$\theta_\text{bias}/\pi\times 10^{-3}$')
ax_theta_bias.set_xlim(0.5, 1.5)
ax_theta_bias.set_ylim(-0.25*1e3/np.pi, 0.25*1e3/np.pi)
ax_theta_bias.legend(loc='lower right')

## uncertainty sum histogram ##

ax_theta_uncert = axs[1]
ax_theta_uncert.minorticks_on()

ax_theta_uncert.text(
    0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'B',
    # fontsize=9,
    fontweight='bold',
    transform=ax_theta_uncert.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)
ax_theta_uncert.plot(thetas_set_PEAC[start:stop]/np.pi, thetas_rec_ell_std_num[start:stop]/np.pi*1e3, color=colour_sum,
                    linewidth=1, label=r'PEAC')

ax_theta_uncert.plot(thetas_set_geom[start:stop]/np.pi, thetas_rec_ell_std_num_geom[start:stop]/np.pi*1e3, color=colour_ell,
                    linewidth=1, label=r'Geom. ellipses')


theta_change = 2* np.arccos(sigma_set_geom/(np.sqrt(2)*0.5625*A0_set_geom))
ax_theta_uncert.axvline(theta_change/np.pi, color="black", linewidth=1, ls="--")
ax_theta_uncert.axvline(2-theta_change/np.pi, color="black", linewidth=1, ls="--")

ax_theta_uncert.set_xlabel(r'$\theta_\text{set}/\pi$')
ax_theta_uncert.set_ylabel(r'$\Delta\theta/\pi\times 10^{-3}$')
ax_theta_uncert.set_xlim(0.5, 1.5)
ax_theta_uncert.set_ylim(1, 0.005*1e3)
# ax_theta_uncert.legend(loc='lower right')

plt.subplots_adjust(left=0.09,
                    top=0.98,
                    bottom=0.18,
                    right=0.99)

plt.savefig('figS1.pdf')