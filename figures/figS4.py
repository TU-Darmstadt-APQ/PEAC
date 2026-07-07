# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""
from matplotlib.markers import MarkerStyle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # for the manual option
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from mpl_toolkits.axes_grid1 import Divider
from mpl_toolkits.axes_grid1.axes_size import Fixed
from curlyBrace import curlyBrace

from scipy.ndimage import gaussian_filter


import os
import sys

sys.path.append(os.path.abspath('..'))

save_fig = True

import helper_functions as hf

plt.style.use('../paper_mpl_style.mplstyle')

#################
#### colours ####
#################

colour_hist_diff = 'C0'
colour_hist_sum = 'C1'
colour_hist_alpha = 'C4'
colour_hist_beta = 'C5'
colour_ell = 'C2'

# --- Size ---
phi_golden = (1 + np.sqrt(5)) / 2
pt_to_in = 1.0 / 72.0

full_width = 522
half_width = 255

full_height_golden = half_width / phi_golden # golden aspect

######## Grid ##########
one_pt_rel_w = pt_to_in / full_width

fig_width_in = full_width * pt_to_in
fig_height_in = full_height_golden * pt_to_in

height_factor = fig_height_in / fig_width_in

frac_h_0 = 0.00
frac_h_1 = 0.06
frac_h_2 = 0.24
frac_h_3 = 0.00
frac_h_4 = 0.00
frac_h_5 = 0.06
frac_h_6 = 0.28
frac_h_7 = 0.06
frac_h_8 = 0.28
frac_h_9 = 1-(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5+frac_h_6+frac_h_7+frac_h_8)

# must be positive
print(frac_h_9)

print(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5+frac_h_6+frac_h_7+frac_h_8+frac_h_9)

frac_v_0 = frac_h_0/height_factor
frac_v_1 = frac_h_1/height_factor
frac_v_2 = frac_h_2/height_factor
frac_v_3 = frac_h_3/height_factor
frac_v_4 = 1-(frac_v_0+frac_v_1+frac_v_2+frac_v_3)

# must be positive
print(frac_v_4)

print(frac_v_0+frac_v_1+frac_v_2+frac_v_3+frac_v_4)


# Exact-size figure: no automatic layout engines
fig_S_alpha_with_cut = plt.figure(figsize=(fig_width_in, fig_height_in), layout=None)

h = [Fixed(frac_h_0*fig_width_in), Fixed(frac_h_1*fig_width_in), Fixed(frac_h_2*fig_width_in),
     Fixed(frac_h_3*fig_width_in), Fixed(frac_h_4*fig_width_in), Fixed(frac_h_5*fig_width_in),
     Fixed(frac_h_6*fig_width_in), Fixed(frac_h_7*fig_width_in), Fixed(frac_h_8*fig_width_in), Fixed(frac_h_9*fig_width_in)]

v = [Fixed(frac_v_0*fig_height_in), Fixed(frac_v_1*fig_height_in), Fixed(frac_v_2*fig_height_in),
     Fixed(frac_v_3*fig_height_in), Fixed(frac_v_4*fig_height_in)]


div = Divider(fig_S_alpha_with_cut, (0, 0, 1, 1), h, v, aspect=False)

#########################################################################
#################### start: show grid via grey lines ####################
#########################################################################
# h_fracs = [frac_h_0, frac_h_1, frac_h_2, frac_h_3, frac_h_4, frac_h_5, frac_h_6, frac_h_7, frac_h_8, frac_h_9]
# v_fracs = [frac_v_0, frac_v_1, frac_v_2, frac_v_3, frac_v_4]

# h_edge_positions = np.cumsum([0]+h_fracs)
# v_edge_positions = np.cumsum([0]+v_fracs)

# # Draw vertical grid lines (column boundaries)
# for hx in h_edge_positions:
#     line = plt.Line2D([hx, hx], [0, 1], lw=0.5, color='grey', linestyle='--', transform=fig_S_alpha_with_cut.transFigure, zorder=100)
#     fig_S_alpha_with_cut.add_artist(line)
# # Draw horizontal grid lines (row boundaries)
# for vy in v_edge_positions:
#     line = plt.Line2D([0, 1], [vy, vy], lw=0.5, color='grey', linestyle='--', transform=fig_S_alpha_with_cut.transFigure, zorder=100)
#     fig_S_alpha_with_cut.add_artist(line)
#######################################################################
#################### end: show grid via grey lines ####################
#######################################################################


# ax_ellipse_hist_beta = fig_S_alpha_with_cut.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=2))
# ax_ellipse_hist_alpha = fig_S_alpha_with_cut.add_axes(div.get_position(), axes_locator=div.new_locator(nx=2, ny=3))
ax_ellipse = fig_S_alpha_with_cut.add_axes(div.get_position(), axes_locator=div.new_locator(nx=2, ny=2))

# ax_bias_hist_diff = fig_S_alpha_with_cut.add_axes(div.get_position(), axes_locator=div.new_locator(nx=7, ny=2))
# ax_bias_hist_sum = fig_S_alpha_with_cut.add_axes(div.get_position(), axes_locator=div.new_locator(nx=6, ny=3))
ax_bias = fig_S_alpha_with_cut.add_axes(div.get_position(), axes_locator=div.new_locator(nx=6, ny=2))

ax_unct = fig_S_alpha_with_cut.add_axes(div.get_position(), axes_locator=div.new_locator(nx=8, ny=2))

fig_S_alpha_with_cut.canvas.draw()

##################################
### start: load numerical data ###
##################################

# coarse alpha scan
num_eval_alpha_scan     = np.load('../num_eval/num_eval_res_alpha_S_alpha_PEAC.npz')
alphas                  =num_eval_alpha_scan['alphas']

A_alphas_set            =num_eval_alpha_scan['A_alphas_set']
A_alphas_rec            =num_eval_alpha_scan['A_alphas_rec']
A_betas_set             =num_eval_alpha_scan['A_betas_set']
A_betas_rec             =num_eval_alpha_scan['A_betas_rec']
#
sigma_set               =num_eval_alpha_scan['sigma_set']
#
thetas_set              =num_eval_alpha_scan['thetas_set']
#
thetas_rec_alphas       =num_eval_alpha_scan['thetas_rec_alphas']
thetas_rec_betas        =num_eval_alpha_scan['thetas_rec_betas']
#
thetas_rec_alphas_std   =num_eval_alpha_scan['thetas_rec_alphas_std']
thetas_rec_betas_std    =num_eval_alpha_scan['thetas_rec_betas_std']

n_alphas = len(alphas)

theta_bias_alphas_abs = np.abs(thetas_rec_alphas - np.tile(thetas_set, (n_alphas, 1)))
theta_bias_betas_abs = np.abs(thetas_rec_betas - np.tile(thetas_set, (n_alphas, 1)))


# fine alpha scan
num_eval_alpha_scan_fine     = np.load('../num_eval/num_eval_res_alpha_S_alpha_fine_PEAC.npz')
alphas_fine                  =num_eval_alpha_scan_fine['alphas']

A_alphas_set_fine            =num_eval_alpha_scan_fine['A_alphas_set']
A_alphas_rec_fine            =num_eval_alpha_scan_fine['A_alphas_rec']
A_betas_set_fine             =num_eval_alpha_scan_fine['A_betas_set']
A_betas_rec_fine             =num_eval_alpha_scan_fine['A_betas_rec']
#
sigma_set_fine               =num_eval_alpha_scan_fine['sigma_set']
#
thetas_set_fine              =num_eval_alpha_scan_fine['thetas_set']
#
thetas_rec_alphas_fine       =num_eval_alpha_scan_fine['thetas_rec_alphas']
thetas_rec_betas_fine        =num_eval_alpha_scan_fine['thetas_rec_betas']
#
thetas_rec_alphas_std_fine   =num_eval_alpha_scan_fine['thetas_rec_alphas_std']
thetas_rec_betas_std_fine    =num_eval_alpha_scan_fine['thetas_rec_betas_std']

n_alphas_fine = len(alphas_fine)

theta_bias_alphas_abs_fine = np.abs(thetas_rec_alphas_fine - np.tile(thetas_set_fine, (n_alphas_fine, 1)))
theta_bias_betas_abs_fine = np.abs(thetas_rec_betas_fine - np.tile(thetas_set_fine, (n_alphas_fine, 1)))

alpha_eps_index = -11
alpha_big_index = 10

num_eval_alpha_slice     = np.load('../num_eval/num_eval_res_alpha_S_alpha_slice_PEAC.npz')
alphas_slice                  =num_eval_alpha_slice['alphas']

A_alphas_set_slice            =num_eval_alpha_slice['A_alphas_set']
A_alphas_rec_slice            =num_eval_alpha_slice['A_alphas_rec']
A_betas_set_slice             =num_eval_alpha_slice['A_betas_set']
A_betas_rec_slice             =num_eval_alpha_slice['A_betas_rec']
#
sigma_set_slice               =num_eval_alpha_slice['sigma_set']
#
thetas_set_slice              =num_eval_alpha_slice['thetas_set']
#
thetas_rec_alphas_slice       =num_eval_alpha_slice['thetas_rec_alphas']
thetas_rec_betas_slice        =num_eval_alpha_slice['thetas_rec_betas']
#
thetas_rec_alphas_std_slice   =num_eval_alpha_slice['thetas_rec_alphas_std']
thetas_rec_betas_std_slice    =num_eval_alpha_slice['thetas_rec_betas_std']

n_alphas_slice = len(alphas_slice)

theta_bias_alphas_abs_slice = np.abs(thetas_rec_alphas_slice - np.tile(thetas_set_slice, (n_alphas_slice, 1)))
theta_bias_betas_abs_slice = np.abs(thetas_rec_betas_slice - np.tile(thetas_set_slice, (n_alphas_slice, 1)))

########## Parameters ##########
alpha = alphas[alpha_eps_index]
beta    = alpha + np.pi/2

A0          = 0.824
sigma       = 0.063
theta_set   = np.pi/4*0
seed        = 104
n_phis      = 300
###############################

ax_ellipse.set_ylabel(r"$S_{+1}$", labelpad=0)
ax_bias.set_ylabel(r"$|\theta_{\text{bias}}|/\pi \times 10^{-3}$", labelpad=0)
ax_unct.set_ylabel(r"$\Delta \theta/\pi \times 10^{-3}$", labelpad=0)

ax_ellipse.set_xlabel(r"$S_{-1}$")
ax_bias.set_xlabel(r"$\theta_\text{set}/\pi$")
ax_unct.set_xlabel(r"$\theta_\text{set}/\pi$")

ax_ellipse.tick_params(axis='both', direction='in')
ax_bias.tick_params(axis='both', direction='in')
ax_unct.tick_params(axis='both', direction='in')

# ax_ellipse_hist_beta.set_yticklabels([])
# ax_ellipse_hist_beta.tick_params(axis='both', direction='in')

# ax_bias_hist_diff.set_yticklabels([])
# ax_bias_hist_diff.tick_params(axis='both', direction='in')

# ax_ellipse_hist_alpha.set_xticklabels([])
# ax_ellipse_hist_alpha.tick_params(axis='both', direction='in')

# ax_bias_hist_sum.set_xticklabels([])
# ax_bias_hist_sum.tick_params(axis='both', direction='in')



########## Insets ################
fig_S_alpha_with_cut.canvas.draw()
ax_ellipse_width_in, ax_ellipse_height_in = hf.get_ax_size_inches(ax_ellipse, fig_S_alpha_with_cut)
aspect_ratio = ax_ellipse_width_in / ax_ellipse_height_in  # z.B. 1.5 wenn breiter als hoch
pt_rel = pt_to_in / ax_ellipse_width_in

desired_width_relative = 0.30 
desired_height_relative = desired_width_relative / aspect_ratio

# ax_inset_left = ax_ellipse.inset_axes([-desired_width_relative/2, 1-desired_height_relative/2, desired_width_relative, desired_height_relative])
# ax_inset_left = ax_ellipse.inset_axes([1-desired_width_relative-1.5*pt_rel, 1-desired_height_relative-1.5*pt_rel, desired_width_relative, desired_height_relative])

# ax_inset_right = ax_bias.inset_axes([-desired_width_relative/2, 1-desired_height_relative/2, desired_width_relative, desired_height_relative])
# ax_inset_right = ax_bias.inset_axes([1-desired_width_relative-1.5*pt_rel, 1-desired_height_relative-1.5*pt_rel, desired_width_relative, desired_height_relative])

##########################################################################
####################### START LEFT ELLIPSE SUBPLOT #######################
##########################################################################

phases = np.linspace(0, 2*np.pi, 1000)
theta = np.pi/8

S_plus = np.cos(phases+theta/2)
S_minus = np.cos(phases-theta/2)

scatter_plot_bounds = 1.1
ax_ellipse.set_xlim(-scatter_plot_bounds, scatter_plot_bounds)
ax_ellipse.set_ylim(-scatter_plot_bounds, scatter_plot_bounds)

ax_ellipse.plot(S_minus, S_plus, color=colour_ell)

from matplotlib.patches import Arc

# angle to show
alpha = (0.25+0.156)*np.pi

# vertex of the angle
x0, y0 = 0.0, 0.0

# 1) full horizontal baseline = x-axis
ax_ellipse.axhline(y=y0, color="black", lw=1)

# 2) full rotated line through the same point
ax_ellipse.axline(
    (x0, y0),
    slope=np.tan(alpha),
    color='forestgreen',
    lw=1
)

# 3) arc for the angle
r_arc = 0.30
arc = Arc(
    (x0, y0),
    2*r_arc, 2*r_arc,
    angle=0,
    theta1=0,
    theta2=np.rad2deg(alpha),
    color="black",
    lw=1
)
ax_ellipse.add_patch(arc)

# 4) small counterclockwise arrow along the arc
phi1 = np.deg2rad(28)
phi2 = np.deg2rad(38)
ax_ellipse.annotate(
    "",
    xy=(x0 + r_arc*np.cos(phi2), y0 + r_arc*np.sin(phi2)),
    xytext=(x0 + r_arc*np.cos(phi1), y0 + r_arc*np.sin(phi1)),
    arrowprops=dict(arrowstyle="->", color="black", lw=1)
)

# 5) label
phi_text = alpha / 2
ax_ellipse.text(
    x0 + 0.18*np.cos(phi_text),
    y0 + 0.18*np.sin(phi_text),
    r'$\alpha$',
    color="black",
    fontsize=10,
    ha="center",
    va="center"
)

ax_ellipse.text(
    0.05, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'A',
    # fontsize=9,
    fontweight='bold',
    transform=ax_ellipse.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)
##########################################################################
######################## END LEFT ELLIPSE SUBPLOT ########################
##########################################################################

##########################################################################
####################### START MIDDLE BIAS SUBPLOT #######################
##########################################################################


# cut plots

# ax_bias.plot(thetas_set_alg/np.pi, theta_bias_ell_abs_alg/thetas_rec_ell_std_num_alg, label="alg. ell.", color="magenta")

# alpha=0.156 for theta in 0.5 to 1.0 pi
ax_bias.plot(thetas_set/np.pi, 
             theta_bias_alphas_abs[alpha_big_index]/np.pi*1e3, 
             label=fr"$\alpha={alphas[alpha_big_index]/np.pi:.3f} \pi$", 
             color='forestgreen', ls=(0,(1,1)))

# alpha=0.249 for theta in 0.9 to 1.0 pi
# ax_bias.plot(thetas_set_fine/np.pi, theta_bias_alphas_abs_fine[alpha_eps_index]/np.pi*1e3, label=fr"$\alpha={alphas_fine[alpha_eps_index]/np.pi:.3f} \pi$", color='tab:blue')

# alpha=0.249 for theta in 0.5 to 1.0 pi
ax_bias.plot(thetas_set_slice/np.pi, 
             theta_bias_alphas_abs_slice[0]/np.pi*1e3, 
             label=fr"$\alpha={alphas_slice[0]/np.pi:.3f} \pi$", 
             color='darkred',alpha=1)

# alpha=0.250 for theta in 0.5 to 1.0 pi
ax_bias.plot(thetas_set/np.pi, 
             theta_bias_alphas_abs[-1]/np.pi*1e3, 
             label=fr"$\alpha={alphas[-1]/np.pi:.3f} \pi$",
             color=colour_hist_sum)

# ax_bias.plot(thetas_set/np.pi, theta_bias_alphas_abs[alpha_eps_index]/np.pi*1e3, label=fr"$\alpha={alphas[alpha_eps_index]/np.pi:.3f} \pi$", color=colour_hist_alpha)



ax_bias.set_xlim(0.5, 1)
# ax_bias.set_yscale("log")
ax_bias.set_ylim(0, 21)

ax_bias.legend(loc='upper left',
               bbox_to_anchor=(0.05,0.90))

##########################################
### start: inset ###
start_index = 134
height = 0.5

bias_inset = ax_bias.inset_axes([0.05,0.9-height,height*1.5,height],
                                xlim = (0.9,1),
                                ylim = (0, 40),
                                xticklabels=[], 
                                yticklabels=[],
                                xticks=[],
                                yticks=[])

# alpha=0.156 for theta in 0.5 to 1.0 pi
bias_inset.plot(thetas_set[start_index:]/np.pi, 
                theta_bias_alphas_abs[alpha_big_index,start_index:]/np.pi*1e3,
                color='forestgreen', ls=(0,(1,1)))

# alpha=0.249 for theta in 0.9 to 1.0 pi
# bias_inset.plot(thetas_set_fine/np.pi, theta_bias_alphas_abs_fine[alpha_eps_index]/np.pi*1e3, color='tab:blue')

# alpha=0.249 for theta in 0.5 to 1.0 pi
bias_inset.plot(thetas_set_slice/np.pi, 
                theta_bias_alphas_abs_slice[0]/np.pi*1e3, 
                label=fr"$\alpha={alphas_slice[0]/np.pi:.3f} \pi$", 
                color='darkred',alpha=1)

# alpha=0.250 for theta in 0.5 to 1.0 pi
bias_inset.plot(thetas_set[start_index:]/np.pi, 
                theta_bias_alphas_abs[-1,start_index:]/np.pi*1e3, 
                color=colour_hist_sum)


bias_inset.set_xticks([0.9,0.95,1],[0.9,0.95,1])
bias_inset.set_yticks([0,20,40],[0,20,40])
bias_inset.tick_params(axis='y',right=True,labelright=True,left=False,labelleft=False)

ax_bias.text(
    0.05, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'B',
    # fontsize=9,
    fontweight='bold',
    transform=ax_bias.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)

##########################################################################
####################### END MIDDLE BIAS SUBPLOT #######################
##########################################################################

##########################################################################
####################### START RIGHT UNCERT SUBPLOT #######################
##########################################################################
# alpha=0.156 for theta in 0.5 to 1.0 pi
############################################################################################
ax_unct.plot(thetas_set/np.pi, 
             thetas_rec_alphas_std[alpha_big_index]/np.pi*1e3, 
             label=fr"$\alpha={alphas[alpha_big_index]/np.pi:.3f} \pi$", 
             color='forestgreen', ls=(0,(1,1)))

# alpha=0.249 for theta in 0.9 to 1.0 pi
# ax_unct.plot(thetas_set_fine/np.pi, thetas_rec_alphas_std_fine[alpha_eps_index]/np.pi*1e3, label=fr"$\alpha={alphas_fine[alpha_eps_index]/np.pi:.3f} \pi$", color='tab:blue')

# alpha=0.249 for theta in 0.5 to 1.0 pi
ax_unct.plot(thetas_set_slice/np.pi, 
             thetas_rec_alphas_std_slice[0]/np.pi*1e3, 
             label=fr"$\alpha={alphas_slice[0]/np.pi:.3f} \pi$", 
             color='darkred',alpha=1)

# alpha=0.250 for theta in 0.5 to 1.0 pi
ax_unct.plot(thetas_set/np.pi, 
             thetas_rec_alphas_std[-1]/np.pi*1e3, 
             label=fr"$\alpha={alphas[-1]/np.pi:.3f} \pi$", 
             color=colour_hist_sum)

# ax_unct.plot(thetas_set/np.pi, thetas_rec_alphas_std[alpha_eps_index]/np.pi*1e3, label=fr"$\alpha={alphas[alpha_eps_index]/np.pi:.3f} \pi$", color=colour_hist_alpha)


ax_unct.set_xlim(0.5, 1)
ax_unct.set_ylim(0, 21)

ax_unct.text(
    0.05, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'C',
    # fontsize=9,
    fontweight='bold',
    transform=ax_unct.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)

##########################################################################
######################## END RIGHT UNCERT SUBPLOT #######################
##########################################################################

if save_fig:
    fig_S_alpha_with_cut.savefig("figS4.pdf")

plt.show()
