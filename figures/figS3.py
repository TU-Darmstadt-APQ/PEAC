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
import matplotlib.ticker as mticker
import numpy as np
from mpl_toolkits.axes_grid1 import Divider
from mpl_toolkits.axes_grid1.axes_size import Fixed
from curlyBrace import curlyBrace
from scipy.optimize import curve_fit
from pathlib import Path
import sys

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

plt.style.use('../paper_mpl_style.mplstyle')

save_fig = True
draw_grid = False

#################
#### colours ####
#################

colour_hist_diff = 'C0'
colour_hist_sum = 'C1'
colour_ell = 'C2'

# --- Size ---
phi_golden = (1 + np.sqrt(5)) / 2
pt_to_in = 1.0 / 72.0

full_width = 522
half_width = 255

full_height_golden = full_width / phi_golden # golden aspect
half_height_golden = half_width / phi_golden # golden aspect

######## Grid ##########
fig_width_in = full_width * pt_to_in
fig_height_in = half_height_golden * pt_to_in

one_pt_rel_w = pt_to_in / fig_width_in
one_pt_rel_h = pt_to_in / fig_height_in

frac_h_0 = 0.075
frac_h_1 = 0.325
frac_h_2 = 0.03
frac_h_3 = 0.02
frac_h_4 = 0.05
frac_h_5 = frac_h_0
frac_h_6 = frac_h_1
frac_h_7 = frac_h_2
frac_h_8 = frac_h_3
frac_h_9 = 1-(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5+frac_h_6+frac_h_7+frac_h_8)

# must be positive
print(frac_h_9)

print(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5+frac_h_6+frac_h_7+frac_h_8+frac_h_9)

frac_v_0 = 0.00
frac_v_1 = 0.15
frac_v_2 = 0.80
frac_v_3 = 1-(frac_v_0+frac_v_1+frac_v_2)

# must be positive
print(frac_v_3)

print(frac_v_0+frac_v_1+frac_v_2+frac_v_3)

# Exact-size figure: no automatic layout engines
fig_sigma = plt.figure(figsize=(fig_width_in, fig_height_in), layout=None)

h = [Fixed(frac_h_0*fig_width_in), Fixed(frac_h_1*fig_width_in), Fixed(frac_h_2*fig_width_in), Fixed(frac_h_3*fig_width_in),
     Fixed(frac_h_4*fig_width_in), Fixed(frac_h_5*fig_width_in), Fixed(frac_h_6*fig_width_in), Fixed(frac_h_7*fig_width_in), Fixed(frac_h_8*fig_width_in), Fixed(frac_h_9*fig_width_in)]

v = [Fixed(frac_v_0*fig_height_in), Fixed(frac_v_1*fig_height_in), Fixed(frac_v_2*fig_height_in), Fixed(frac_v_3*fig_height_in)]


div = Divider(fig_sigma, (0, 0, 1, 1), h, v, aspect=False)

#########################################################################
#################### start: show grid via grey lines ####################
#########################################################################
h_fracs = [frac_h_0, frac_h_1, frac_h_2, frac_h_3, frac_h_4, frac_h_5, frac_h_6, frac_h_7, frac_h_8, frac_h_9]
v_fracs = [frac_v_0, frac_v_1, frac_v_2, frac_v_3]

h_edge_positions = np.cumsum([0]+h_fracs)
v_edge_positions = np.cumsum([0]+v_fracs)

if draw_grid:
    # Draw vertical grid lines (column boundaries)
    for hx in h_edge_positions:
        line = plt.Line2D([hx, hx], [0, 1], lw=0.5, color='grey', linestyle='--', transform=fig_sigma.transFigure, zorder=100)
        fig_sigma.add_artist(line)
    # Draw horizontal grid lines (row boundaries)
    for vy in v_edge_positions:
        line = plt.Line2D([0, 1], [vy, vy], lw=0.5, color='grey', linestyle='--', transform=fig_sigma.transFigure, zorder=100)
        fig_sigma.add_artist(line)
#######################################################################
#################### end: show grid via grey lines ####################
#######################################################################


ax_abs_bias_bar = fig_sigma.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=2))
ax_abs_bias = fig_sigma.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=2))

ax_std_bias_bar = fig_sigma.add_axes(div.get_position(), axes_locator=div.new_locator(nx=8, ny=2))
ax_std_bias = fig_sigma.add_axes(div.get_position(), axes_locator=div.new_locator(nx=6, ny=2))

fig_sigma.canvas.draw()

ax_abs_bias.set_ylabel(r"$\sigma$", labelpad=0)
ax_std_bias.set_ylabel(r"$\sigma$", labelpad=0)

ax_abs_bias.set_xlabel(r"$\theta_\text{set}/\pi$")
ax_std_bias.set_xlabel(r"$\theta_\text{set}/\pi$")

ax_abs_bias.tick_params(axis='both', direction='in')
ax_std_bias.tick_params(axis='both', direction='in')

ax_abs_bias_bar.set_yticklabels([])
ax_abs_bias_bar.tick_params(axis='both', direction='in')

ax_std_bias_bar.set_yticklabels([])
ax_std_bias_bar.tick_params(axis='both', direction='in')

##################################
### start: load numerical data ###
##################################

num_eval_sigma_scan     = np.load('../num_eval/num_eval_res_sigma_S_sum_PEAC_and_geo_ell.npz')

sigmas_set              =num_eval_sigma_scan['sigmas_set']
sigmas_set_fine = np.linspace(sigmas_set.min(),sigmas_set.max(),1000)

A_sum_sigmas_set            =num_eval_sigma_scan['A_sum_sigmas_set']
A_sum_sigmas_rec            =num_eval_sigma_scan['A_sum_sigmas_rec']
A_diff_sigmas_set             =num_eval_sigma_scan['A_diff_sigmas_set']
A_diff_sigmas_rec             =num_eval_sigma_scan['A_diff_sigmas_rec']
#
thetas_set              =num_eval_sigma_scan['thetas_set']
#
thetas_rec_sum_sigmas       =num_eval_sigma_scan['thetas_rec_sum_sigmas']
thetas_rec_ell_sigmas        =num_eval_sigma_scan['thetas_rec_ell_sigmas']
#
thetas_rec_sum_sigmas_std   =num_eval_sigma_scan['thetas_rec_sum_sigmas_std']
thetas_rec_ell_sigmas_std    =num_eval_sigma_scan['thetas_rec_ell_sigmas_std']

n_sigmas = len(sigmas_set)

theta_bias_sum_sigmas_abs = np.abs(thetas_rec_sum_sigmas - np.tile(thetas_set, (n_sigmas, 1)))
theta_bias_ell_sigmas_abs = np.abs(thetas_rec_ell_sigmas - np.tile(thetas_set, (n_sigmas, 1)))

thetas_change = hf.theta_change(0.824, sigmas_set, np.pi/4)

vmax = np.max([theta_bias_sum_sigmas_abs,theta_bias_ell_sigmas_abs,thetas_rec_sum_sigmas_std,thetas_rec_ell_sigmas_std])/np.pi

####################################################
### Find minimum in filtered standard deveations ###
####################################################
def lin_approx(sigma,m,b):
    return m * sigma + b
minimal_std_thetas = thetas_set[thetas_rec_sum_sigmas_std.argmin(axis=1)]

params,cov = curve_fit(lin_approx , sigmas_set ,minimal_std_thetas)

################################
### end: load numerical data ###
################################


##########################################################################
####################### START LEFT ELLIPSE SUBPLOT #######################
##########################################################################

mesh_left = ax_abs_bias.pcolormesh(
        thetas_set/np.pi,
        sigmas_set,
        theta_bias_ell_sigmas_abs/np.pi,
        cmap="jet",
        shading="auto",
        # norm="log"
        rasterized=True,
        vmin=0,
        vmax=vmax
    )

ax_abs_bias.plot(lin_approx(sigmas_set_fine,*params) /np.pi,sigmas_set_fine,'w',lw=1)
ax_abs_bias.plot(thetas_change/np.pi ,sigmas_set,'orange',lw=1)
ax_abs_bias.hlines(0.063,thetas_set.min()/np.pi,thetas_set.max()/np.pi,lw=0.75,color='r',ls='--')

cbar_left = fig_sigma.colorbar(mesh_left, cax=ax_abs_bias_bar)

cbar_left.set_label(r"$|\theta_{\text{bias, ell}}|/\pi$", labelpad=3)
cbar_left.ax.yaxis.set_label_position("left")

# cbar_left.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())

cbar_left.ax.tick_params(
    axis='y',
    direction='in',
    length=3,
    width=0.8,
    labelsize=8
)

sigma_edges = np.empty(len(sigmas_set) + 1)
sigma_edges[1:-1] = 0.5 * (sigmas_set[:-1] + sigmas_set[1:])
sigma_edges[0] = sigmas_set[0] - (sigma_edges[1] - sigmas_set[0])
sigma_edges[-1] = sigmas_set[-1] + (sigmas_set[-1] - sigma_edges[-2])

# for a, th, y0, y1 in zip(sigmas_set, thetas_change, sigma_edges[:-1], sigma_edges[1:]):
#     if np.isfinite(th):
#         ax_abs_bias.vlines(th / np.pi, y0, y1, color="orange", linewidth=1)

ax_abs_bias.text(
        0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
        'A',
        # fontsize=9,
        fontweight='bold',
        transform=ax_abs_bias.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='w'
    )

# tick labels of ax_abs_bias white
ax_abs_bias.tick_params(color='w')

##########################################################################
######################## END LEFT ELLIPSE SUBPLOT ########################
##########################################################################

##########################################################################
####################### START RIGHT ELLIPSE SUBPLOT #######################
##########################################################################


mesh_right = ax_std_bias.pcolormesh(
        thetas_set/np.pi,
        sigmas_set,
        thetas_rec_ell_sigmas_std / np.pi,
        cmap="jet",
        shading="auto",
        # norm="log"
        rasterized=True,
        vmin=0,
        vmax=vmax
    )

ax_std_bias.plot(lin_approx(sigmas_set_fine,*params) /np.pi,sigmas_set_fine,'w',lw=1)
ax_std_bias.plot(thetas_change/np.pi ,sigmas_set,'orange',lw=1)
ax_std_bias.hlines(0.063,thetas_set.min()/np.pi,thetas_set.max()/np.pi,lw=0.75,color='r',ls='--')

fig_sigma.colorbar(mesh_right, cax=ax_std_bias_bar)

cbar_right = fig_sigma.colorbar(mesh_right, cax=ax_std_bias_bar)

cbar_right.set_label(r"$\Delta \theta_\text{ell} / \pi$", labelpad=3)
cbar_right.ax.yaxis.set_label_position("left")

# cbar_right.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
# cbar_right.ax.yaxis.set_minor_formatter(mticker.NullFormatter())

cbar_right.ax.tick_params(
    axis='y',
    direction='in',
    length=3,
    width=0.8,
    labelsize=8
)

# for a, th, y0, y1 in zip(sigmas_set, thetas_change, sigma_edges[:-1], sigma_edges[1:]):
#     if np.isfinite(th):
#         ax_std_bias.vlines(th / np.pi, y0, y1, color="orange", linewidth=1)

ax_std_bias.text(
        0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
        'B',
        # fontsize=9,
        fontweight='bold',
        transform=ax_std_bias.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='w'
    )

# tick labels of ax_std_bias white
ax_std_bias.tick_params(color='w')

##########################################################################
######################## END RIGHT ELLIPSE SUBPLOT ########################
##########################################################################

if save_fig:
    fig_sigma.savefig("figS3.pdf")

plt.show()
