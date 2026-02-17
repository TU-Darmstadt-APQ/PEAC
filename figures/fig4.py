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

import os
import sys

sys.path.append(os.path.abspath('..'))

import helper_functions as hf

plt.style.use('../paper_mpl_style.mplstyle')

save_fig = False

#################
#### colours ####
#################
colour_diff = 'C0'
colour_sum = 'C1'
colour_ell = 'C2'
first_inset_colour = 'black'
second_inset_colour = 'black'

# --- Size ---
phi_golden = (1 + np.sqrt(5)) / 2
pt_to_in = 1.0 / 72.0
full_width = 522
half_width = 255

full_height_golden = full_width / phi_golden  # golden ratio
full_height = full_height_golden  # 255

######## Grid ##########
one_pt_rel_w = pt_to_in / full_width

frac_h_0 = 0.00
frac_h_1 = 0.075

frac_h_2 = 0.40

frac_h_3 = 0.00
frac_h_4 = frac_h_1

frac_h_5 = frac_h_2

frac_h_6 = 1-(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5)

print(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5+frac_h_6)

one_pt_rel_h = pt_to_in / full_height


frac_v_0 = 0.00
frac_v_1 = 0.10

frac_v_2 = 0.40

frac_v_3 = 0.00
frac_v_4 = 0.00

frac_v_5 = frac_v_2

frac_v_6 = frac_v_1
frac_v_7 = frac_v_0

print(frac_v_0+frac_v_1+frac_v_2+frac_v_3+frac_v_4+frac_v_5+frac_v_6+frac_v_7)


fig_width_in = full_width * pt_to_in
fig_height_in = full_height * pt_to_in

# Exact-size figure: no automatic layout engines
fig_4 = plt.figure(figsize=(fig_width_in, fig_height_in), layout=None)

h = [Fixed(frac_h_0*fig_width_in), Fixed(frac_h_1*fig_width_in), Fixed(frac_h_2*fig_width_in), Fixed(frac_h_3*fig_width_in),
     Fixed(frac_h_4*fig_width_in), Fixed(frac_h_5*fig_width_in), Fixed(frac_h_6*fig_width_in)]

v = [Fixed(frac_v_0*fig_height_in), Fixed(frac_v_1*fig_height_in), Fixed(frac_v_2*fig_height_in), Fixed(frac_v_3*fig_height_in),
     Fixed(frac_v_4*fig_height_in), Fixed(frac_v_5*fig_height_in), Fixed(frac_v_6*fig_height_in), Fixed(frac_v_7*fig_height_in)]


div = Divider(fig_4, (0, 0, 1, 1), h, v, aspect=False)

#########################################################################
#################### start: show grid via grey lines ####################
#########################################################################
# h_fracs = [frac_h_0, frac_h_1, frac_h_2, frac_h_3, frac_h_4, frac_h_5, frac_h_6]
# v_fracs = [frac_v_0, frac_v_1, frac_v_2, frac_v_3, frac_v_4, frac_v_5, frac_v_6, frac_v_7]

# h_edge_positions = np.cumsum([0]+h_fracs)
# v_edge_positions = np.cumsum([0]+v_fracs)

# # Draw vertical grid lines (column boundaries)
# for hx in h_edge_positions:
#     line = plt.Line2D([hx, hx], [0, 1], lw=0.5, color='grey', linestyle='--', transform=fig_4.transFigure, zorder=100)
#     fig_4.add_artist(line)
# # Draw horizontal grid lines (row boundaries)
# for vy in v_edge_positions:
#     line = plt.Line2D([0, 1], [vy, vy], lw=0.5, color='grey', linestyle='--', transform=fig_4.transFigure, zorder=100)
#     fig_4.add_artist(line)
#######################################################################
#################### end: show grid via grey lines ####################
#######################################################################

ax_amplitude = fig_4.add_axes(
    div.get_position(), axes_locator=div.new_locator(nx=5, ny=5))
ax_theta_uncertainty = fig_4.add_axes(
    div.get_position(), axes_locator=div.new_locator(nx=5, ny=2))
ax_theta_reconstructed = fig_4.add_axes(
    div.get_position(), axes_locator=div.new_locator(nx=2, ny=5))
ax_theta_bias = fig_4.add_axes(
    div.get_position(), axes_locator=div.new_locator(nx=2, ny=2))
# important! -> This forces matplotlib to render the figure and properly initialise all transformation matrices.
fig_4.canvas.draw()

##################################################################
####################### START LOADING DATA #######################
##################################################################

####################################
### start: load numerical phases ###
####################################

num_eval_theta_scan     = np.load('../num_eval/num_eval_results.npz')

thetas_set              = num_eval_theta_scan['thetas_set']

thetas_rec_ell_num      = num_eval_theta_scan['thetas_rec_ell']
thetas_rec_sum_num      = num_eval_theta_scan['thetas_rec_sum']
thetas_rec_diff_num     = num_eval_theta_scan['thetas_rec_diff']

thetas_rec_ell_std_num  = num_eval_theta_scan['thetas_rec_ell_std']
thetas_rec_sum_std_num  = num_eval_theta_scan['thetas_rec_sum_std']
thetas_rec_diff_std_num = num_eval_theta_scan['thetas_rec_diff_std']

A0_set                  = num_eval_theta_scan['A0_set']
A_sum_set               = num_eval_theta_scan['A_sum_set']
A_sum_rec_num           = num_eval_theta_scan['A_sum_rec']
A_diff_set              = num_eval_theta_scan['A_diff_set']
A_diff_rec_num          = num_eval_theta_scan['A_diff_rec']


##################################
### end: load numerical phases ###
##################################

#######################################
### start: load exp. coarse results ###
#######################################
exp_coarse_eval                 = np.load(f'../exp_eval/exp_coarse_eval_results.npz')

thetas_calibrated_exp_coarse    = exp_coarse_eval['thetas_calibrated']

# hier auch das rekonstruierte theta...
thetas_bias_ell_exp_coarse      = exp_coarse_eval['theta_bias_ell']
thetas_bias_sum_exp_coarse      = exp_coarse_eval['theta_bias_hist_sum']
thetas_bias_diff_exp_coarse     = exp_coarse_eval['theta_bias_hist_diff']

thetas_rec_ell_std_exp_coarse   = exp_coarse_eval['theta_ell_std']
thetas_rec_sum_std_exp_coarse   = exp_coarse_eval['theta_hist_sum_std']
thetas_rec_diff_std_exp_coarse  = exp_coarse_eval['theta_hist_diff_std']

A_sum_rec_exp_coarse            = exp_coarse_eval['A_sum_hist_mean']
A_sum_rec_std_exp_coarse        = exp_coarse_eval['A_sum_hist_std']
A_diff_rec_exp_coarse           = exp_coarse_eval['A_diff_hist_mean']
A_diff_rec_std_exp_coarse       = exp_coarse_eval['A_diff_hist_std']

# load calibration from MZI full new
a_calib                         = exp_coarse_eval['a_calib']
a_calib_unct                    = exp_coarse_eval['a_calib_unct']

#####################################
### end: load exp. coarse results ###
#####################################

##########################################
### start: load fine experimental data ###
##########################################

### EXP DATA ###
exp_fine_eval                   = np.load(f'../exp_eval/exp_fine_eval_results.npz')

thetas_calibrated_exp_fine      = exp_fine_eval['thetas_calibrated']

# hier auch das rekonstruierte theta...
thetas_bias_ell_exp_fine        = exp_fine_eval['theta_bias_ell']
thetas_bias_sum_exp_fine        = exp_fine_eval['theta_bias_hist_sum']
thetas_bias_diff_exp_fine       = exp_fine_eval['theta_bias_hist_diff']

thetas_rec_ell_std_exp_fine     = exp_fine_eval['theta_ell_std']
thetas_rec_sum_std_exp_fine     = exp_fine_eval['theta_hist_sum_std']
thetas_rec_diff_std_exp_fine    = exp_fine_eval['theta_hist_diff_std']

A_sum_rec_exp_fine              = exp_fine_eval['A_sum_hist_mean']
A_sum_rec_std_exp_fine          = exp_fine_eval['A_sum_hist_std']

########################################
### end: load fine experimental data ###
########################################

################################################################
####################### END LOADING DATA #######################
################################################################

###################################################################
####################### START AXES SETTINGS #######################
###################################################################

################
# amplitude axis
ax_amplitude.set_ylabel(r"$A_\text{sum}, A_\text{diff}$")

ax_amplitude.set_xlim(0.5, 2.5)
ax_amplitude.set_ylim(0, 1.45*A0_set)

# ax_amplitude.grid(True, linewidth=0.6, alpha=0.5)
ax_amplitude.tick_params(axis='both', direction='in', which='both')
ax_amplitude.tick_params(axis='x', labelbottom=False)

ax_amplitude.text(
    0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'B',
    # fontsize=9,
    fontweight='bold',
    transform=ax_amplitude.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)

################
# second x-axis on top left


def forward_transform(x_bottom):
    return hf.T_calibrated(x_bottom * np.pi, a_calib) * 1e3


def inverse_transform(x_top):
    x_normalised = x_top / 1e3
    x_radians = hf.parabola_with_linear(x_normalised, a_calib)
    return x_radians / np.pi


ax_amplitude_top = ax_amplitude.secondary_xaxis(
    'top', functions=(forward_transform, inverse_transform))
ax_amplitude_top.tick_params(axis='both', direction='in', which='both')
ax_amplitude_top.set_xlabel(r"$T$ (ms)")

################
# theta uncertainty axis
ax_theta_uncertainty.set_xlabel(r"$\theta_{\text{set}} / \pi$")
ax_theta_uncertainty.set_ylabel(r"$\Delta \theta / \pi$")
# ax_theta_uncertainty.grid(True, linewidth=0.6, alpha=0.5)

ax_theta_uncertainty.set_xlim(0.5, 2.5)
ax_theta_uncertainty.set_ylim(0.0, 0.22)

ax_theta_uncertainty.tick_params(axis='both', direction='in', which='both')

ax_theta_uncertainty.text(
    0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'D',
    # fontsize=9,
    fontweight='bold',
    transform=ax_theta_uncertainty.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)


################
# theta reconstructed axis
ax_theta_reconstructed.set_ylabel(
    r"$\theta_{\text{rec}} / \pi$")
ax_theta_reconstructed.tick_params(axis='x', labelbottom=False)
# ax_theta_reconstructed.grid(True, linewidth=0.6, alpha=0.5)


ax_theta_reconstructed.set_xlim(0.5, 2.5)
ax_theta_reconstructed.set_ylim(0.5, 2.5)
ax_theta_reconstructed.tick_params(axis='both', direction='in', which='both')

ax_theta_reconstructed.text(
    0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'A',
    # fontsize=9,
    fontweight='bold',
    transform=ax_theta_reconstructed.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)

################
# second x-axis on top right
ax_theta_reconstructed_top = ax_theta_reconstructed.secondary_xaxis(
    'top', functions=(forward_transform, inverse_transform))
ax_theta_reconstructed_top.tick_params(
    axis='both', direction='in', which='both')
ax_theta_reconstructed_top.set_xlabel(r"$T$ (ms)")

################
# theta bias axis
ax_theta_bias.set_xlabel(r"$\theta_{\text{set}} / \pi$")
ax_theta_bias.set_ylabel(r"$\theta_\text{bias} / \pi$ ")
# ax_theta_bias.grid(True, linewidth=0.6, alpha=0.5)
ax_theta_bias.tick_params(axis='both', direction='in', which='both')
ax_theta_bias.set_xlim(0.5, 2.5)

ax_theta_bias.text(
    0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
    'C',
    # fontsize=9,
    fontweight='bold',
    transform=ax_theta_bias.transAxes,
    verticalalignment='top',
    horizontalalignment='left'
)

#################################################################
####################### END AXES SETTINGS #######################
#################################################################

######################################################################
####################### START AMPLITUDE SUBPLOT #######################
######################################################################
# real amplitude
ax_amplitude.plot(thetas_set/np.pi, A_sum_set,
                 color='black', ls="--", linewidth=1)
ax_amplitude.plot(thetas_set/np.pi, A_diff_set,
                 color='black', ls="--", linewidth=1)

# amplitude from fits to numerics
ax_amplitude.plot(thetas_set/np.pi, np.mean(A_sum_rec_num,
                 axis=1), color=colour_sum, linewidth=1)
ax_amplitude.plot(thetas_set/np.pi, np.mean(A_diff_rec_num,
                 axis=1), color=colour_diff, linewidth=1)

# amplitude from fits to experiment
hf.plot_line_with_wide_err(ax_amplitude, thetas_calibrated_exp_coarse / np.pi,
                           A_diff_rec_exp_coarse, 0, A_diff_rec_std_exp_coarse, colour_diff, ls="")
hf.plot_line_with_wide_err(ax_amplitude, thetas_calibrated_exp_coarse / np.pi,
                           A_sum_rec_exp_coarse, 0, A_sum_rec_std_exp_coarse, colour_sum, ls="")

ax_amplitude.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])

##########################################
### start: inset at vanishing amplitude ###
# size, etc.
ax_amplitude_width_in, ax_amplitude_height_in = hf.get_ax_size_inches(
    ax_amplitude, fig_4)
ax_amplitude_aspect_ratio = ax_amplitude_width_in / ax_amplitude_height_in

desired_width_relative_ax_amplitude = 0.20
desired_height_relative_ax_amplitude = 1.45 * \
    desired_width_relative_ax_amplitude * ax_amplitude_aspect_ratio

axins_vanishing_amplitude = ax_amplitude.inset_axes(
    [0.15, 0.38, desired_width_relative_ax_amplitude, desired_height_relative_ax_amplitude])

axins_vanishing_amplitude.spines[:].set_color(first_inset_colour)

# real amplitude
axins_vanishing_amplitude.plot(thetas_set/np.pi,
                      A_sum_set, color='black', ls="--", linewidth=1)
axins_vanishing_amplitude.plot(thetas_set/np.pi,
                      A_diff_set, color='black', ls="--", linewidth=1)

# amplitude from fits to experiment
axins_vanishing_amplitude.plot(thetas_set/np.pi, np.mean(
    A_sum_rec_num, axis=1), color=colour_sum, linewidth=1)
axins_vanishing_amplitude.plot(thetas_set/np.pi, np.mean(
    A_diff_rec_num, axis=1), color=colour_diff, linewidth=1)

# amplitude from fits to experiment
# hf.plot_line_with_wide_err(axins_vanishing_amplitude, thetas_calibrated_exp_coarse /
#                            np.pi, A_diff_rec_exp_coarse, 0, A_diff_rec_std_exp_coarse, colour_diff, ls="")
# hf.plot_line_with_wide_err(axins_vanishing_amplitude, thetas_calibrated_exp_coarse /
#                            np.pi, A_sum_rec_exp_coarse, 0, A_sum_rec_std_exp_coarse, colour_sum, ls="")

hf.plot_line_with_wide_err(axins_vanishing_amplitude, thetas_calibrated_exp_fine/np.pi,
                           A_sum_rec_exp_fine , 0, A_sum_rec_std_exp_fine ,
                           colour_sum, marker="", ls="")

# axis settings
axins_vanishing_amplitude.set_xlim(0.88, 1.12)
axins_vanishing_amplitude.set_ylim(0, 0.18)
axins_vanishing_amplitude.tick_params(
    axis='both', direction='in', which='both', labelsize=6)

axins_vanishing_amplitude.set_xticks([1.0])
axins_vanishing_amplitude.set_yticks([])

# inset highlight lines
patch_axins_vanishing_amplitude, connector1_axins_vanishing_amplitude, connector2_axins_vanishing_amplitude = mark_inset(
    ax_amplitude,
    axins_vanishing_amplitude,
    loc1=1,
    loc2=2,
    fc='none',
    ec=first_inset_colour
)

# 2 (upper left)     1 (upper right)
# 3 (lower left)     4 (lower right)

connector1_axins_vanishing_amplitude.loc1 = 3  # Inset: lower left
connector1_axins_vanishing_amplitude.loc2 = 2  # Parent: upper left

connector2_axins_vanishing_amplitude.loc1 = 4  # Inset: lower right
connector2_axins_vanishing_amplitude.loc2 = 1  # Parent: upper right

connector1_axins_vanishing_amplitude.remove()
connector2_axins_vanishing_amplitude.remove()
### end: inset at vanishing amplitude ###
########################################

#########################################
### start: inset at maxmimal amplitude ###
# size, etc.
axins_maximal_amplitude = ax_amplitude.inset_axes(
    [0.65, 0.38, desired_width_relative_ax_amplitude, desired_height_relative_ax_amplitude])

# real amplitude
axins_maximal_amplitude.plot(thetas_set/np.pi,
                   A_sum_set, color='black', ls="--", linewidth=1)
axins_maximal_amplitude.plot(thetas_set/np.pi,
                   A_diff_set, color='black', ls="--", linewidth=1)

axins_maximal_amplitude.spines[:].set_color(second_inset_colour)

# amplitude from fits to experiment
axins_maximal_amplitude.plot(thetas_set/np.pi, np.mean(A_sum_rec_num,
                   axis=1), color=colour_sum, linewidth=1)
axins_maximal_amplitude.plot(thetas_set/np.pi,
                   np.mean(A_diff_rec_num, axis=1), color=colour_diff, linewidth=1)

# amplitude from fits to experiment
hf.plot_line_with_wide_err(axins_maximal_amplitude, thetas_calibrated_exp_coarse /
                           np.pi, A_diff_rec_exp_coarse, 0, A_diff_rec_std_exp_coarse, colour_diff, ls="")
hf.plot_line_with_wide_err(axins_maximal_amplitude, thetas_calibrated_exp_coarse /
                           np.pi, A_sum_rec_exp_coarse, 0, A_sum_rec_std_exp_coarse, colour_sum, ls="")

# axis settings
axins_maximal_amplitude.set_xlim(1.88, 2.12)
axins_maximal_amplitude.set_ylim(1.45*A0_set-0.11, 1.45*A0_set)
axins_maximal_amplitude.tick_params(axis='both', direction='in',
                          which='both', labelsize=6)

axins_maximal_amplitude.set_xticks([2])
axins_maximal_amplitude.set_yticks([])

# inset highlight lines
patch_axins_maximal_amplitude, connector1_axins_maximal_amplitude, connector2_axins_maximal_amplitude = mark_inset(
    ax_amplitude,
    axins_maximal_amplitude,
    loc1=1,
    loc2=2,
    fc='none',
    ec=second_inset_colour
)

# 2 (upper left)     1 (upper right)
# 3 (lower left)     4 (lower right)

connector1_axins_maximal_amplitude.loc1 = 2  # Inset: upper left
connector1_axins_maximal_amplitude.loc2 = 3  # Parent: lower left

connector2_axins_maximal_amplitude.loc1 = 1  # Inset: upper right
connector2_axins_maximal_amplitude.loc2 = 4  # Parent: lower right

connector1_axins_maximal_amplitude.remove()
connector2_axins_maximal_amplitude.remove()
### end: inset at maximal amplitude ###
######################################

######################################################################
######################## END AMPLITUDE SUBPLOT ########################
######################################################################


###############################################################################
####################### START THETA UNCERTAINTY SUBPLOT #######################
###############################################################################
# numerical data
ax_theta_uncertainty.plot(thetas_set/np.pi, thetas_rec_ell_std_num,
                          color=colour_ell, linewidth=1)
ax_theta_uncertainty.plot(thetas_set/np.pi, thetas_rec_diff_std_num,
                          color=colour_diff, linewidth=1)
ax_theta_uncertainty.plot(thetas_set/np.pi, thetas_rec_sum_std_num,
                          color=colour_sum, linewidth=1)

# experimental data
hf.plot_line_with_wide_err(ax_theta_uncertainty, thetas_calibrated_exp_coarse/np.pi,
                           thetas_rec_ell_std_exp_coarse, 0.0, 0, colour_ell, ls="")

hf.plot_line_with_wide_err(ax_theta_uncertainty, thetas_calibrated_exp_coarse/np.pi,
                           thetas_rec_diff_std_exp_coarse, 0.0, 0, colour_diff, ls="")

hf.plot_line_with_wide_err(ax_theta_uncertainty, thetas_calibrated_exp_coarse/np.pi,
                           thetas_rec_sum_std_exp_coarse, 0.0, 0, colour_sum, ls="")

ax_theta_uncertainty.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
################################
######### start inset ##########
################################
# calculate dimensions of inset
ax_theta_uncertainty_width_in, ax_theta_uncertainty_height_in = hf.get_ax_size_inches(
    ax_theta_uncertainty, fig_4)
ax_theta_uncertainty_aspect_ratio = ax_theta_uncertainty_width_in / \
    ax_theta_uncertainty_height_in

desired_width_relative_ax_theta_uncertainty = 0.28
desired_height_relative_ax_theta_uncertainty = desired_width_relative_ax_theta_uncertainty * \
    ax_theta_uncertainty_aspect_ratio

axins_theta_uncertainty = ax_theta_uncertainty.inset_axes(
    [0.51 - desired_width_relative_ax_theta_uncertainty/2, 0.5, desired_width_relative_ax_theta_uncertainty, desired_height_relative_ax_theta_uncertainty])  # xrel, yrel, wrel hrel

axins_theta_uncertainty.spines[:].set_color(first_inset_colour)

# plot the same content inside the inset
axins_theta_uncertainty.plot(thetas_set/np.pi, thetas_rec_ell_std_num,
                                color=colour_ell, linewidth=1)
# axins_theta_uncertainty.plot(thetas_set/np.pi, thetas_rec_diff_std_num,
#                                 color=colour_diff, linewidth=1)
axins_theta_uncertainty.plot(thetas_set/np.pi, thetas_rec_sum_std_num,
                                color=colour_sum, linewidth=1)

Ts = np.linspace(1.6e-3, 1.8e-3, 21)

hf.plot_line_with_wide_err(axins_theta_uncertainty, thetas_calibrated_exp_fine/np.pi, thetas_rec_ell_std_exp_fine,
                           0, 0, colour_ell, marker="|", ls="")

# hf.plot_line_with_wide_err(axins_theta_uncertainty, thetas_calibrated_exp_fine/np.pi,
#                            thetas_rec_diff_std_exp_fine, 0, 0, colour_diff, marker="|", ls="")

hf.plot_line_with_wide_err(axins_theta_uncertainty, thetas_calibrated_exp_fine/np.pi,
                           thetas_rec_sum_std_exp_fine, 0, 0, colour_sum, marker="|", ls="")


# limit inset to the highlighted region
x1_inset_theta_unct, x2_inset_theta_unct = 0.84, 1.14   # theta_set / pi
y1_inset_theta_unct, y2_inset_theta_unct = 0, 0.09  # theta_reconstructed / pi
axins_theta_uncertainty.set_xlim(x1_inset_theta_unct, x2_inset_theta_unct)
axins_theta_uncertainty.set_ylim(y1_inset_theta_unct, y2_inset_theta_unct)

# Optional: declutter ticks inside the inset
axins_theta_uncertainty.tick_params(axis='both', which='both', labelsize=6, direction="in")
axins_theta_uncertainty.set_xticks([0.9 ,1.0, 1.1])
axins_theta_uncertainty.set_yticks([0.04, 0.08],['0.04', '0.08'])
# axins_theta_uncertainty.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))



patch_axins_theta_uncertainty, connector1_axins_theta_uncertainty, connector2_axins_theta_uncertainty = mark_inset(
    ax_theta_uncertainty,
    axins_theta_uncertainty,
    loc1=1,
    loc2=2,
    fc='none',
    ec=first_inset_colour
)

# 2 (upper left)     1 (upper right)
# 3 (lower left)     4 (lower right)

connector1_axins_theta_uncertainty.loc1 = 2  # Inset: lower left
connector1_axins_theta_uncertainty.loc2 = 2  # Parent: lower left

connector2_axins_theta_uncertainty.loc1 = 4  # Inset: lower right
connector2_axins_theta_uncertainty.loc2 = 4  # Parent: lower right

connector1_axins_theta_uncertainty.remove()
connector2_axins_theta_uncertainty.remove()
##############################
######### end inset ##########
##############################

###############################################################################
######################## END THETA UNCERTAINTY SUBPLOT ########################
###############################################################################


################################################################################
####################### START THETA RECONSTRUCED SUBPLOT #######################
################################################################################
# numerical data
ax_theta_reconstructed.plot(thetas_set/np.pi, thetas_set/np.pi, color="black",
                            linewidth=0.5, linestyle="--")
ax_theta_reconstructed.plot(thetas_set/np.pi, thetas_rec_ell_num/np.pi,
                            color=colour_ell, linewidth=1)

# experimental data
hf.plot_line_with_wide_err(ax_theta_reconstructed, thetas_calibrated_exp_coarse/np.pi,
                           (thetas_calibrated_exp_coarse+thetas_bias_ell_exp_coarse) /
                           np.pi, 0, (thetas_rec_ell_std_exp_coarse)/np.pi,
                           colour_ell, marker="+", ls="")
hf.plot_line_with_wide_err(ax_theta_reconstructed, thetas_calibrated_exp_coarse/np.pi,
                           (thetas_calibrated_exp_coarse+thetas_bias_sum_exp_coarse) /
                           np.pi, 0, (thetas_rec_sum_std_exp_coarse)/np.pi,
                           colour_sum, marker="+", ls="")

ax_theta_reconstructed.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
ax_theta_reconstructed.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
#########################################
############# INSET 1 START #############
#########################################
ax_theta_reconstructed_width_in, ax_theta_reconstructed_height_in = hf.get_ax_size_inches(
    ax_theta_reconstructed, fig_4)
ax_theta_reconstructed_aspect_ratio = ax_theta_reconstructed_width_in / \
    ax_theta_reconstructed_height_in

# --- Inset limits ---
x1_inset_theta_reconstructed_1, x2_inset_theta_reconstructed_1 = 0.9, 1.1   # theta_set / pi
y1_inset_theta_reconstructed_1, y2_inset_theta_reconstructed_1 = 0.87, 1.13  # theta_reconstructed / pi

# --- Create the inset axes (axes-fraction coords) ---
axins_theta_reconstructed_1 = ax_theta_reconstructed.inset_axes(
    [0.125, 0.50, 0.25, 0.25*ax_theta_reconstructed_aspect_ratio])

axins_theta_reconstructed_1.spines[:].set_color(first_inset_colour)

# Move y-axis + ticks to the right
# axins_theta_reconstructed_1.yaxis.tick_right()      # put ticks/labels on the right
# axins_theta_reconstructed_1.yaxis.set_label_position("right")

# Move x-axis + ticks to the top
# axins_theta_reconstructed_1.xaxis.tick_top()      # put ticks/labels on the top
# axins_theta_reconstructed_1.xaxis.set_label_position("bottom")  

# plot the same content inside the inset
axins_theta_reconstructed_1.plot(thetas_set/np.pi, thetas_rec_ell_num /
           np.pi, color=colour_ell, linewidth=1)
axins_theta_reconstructed_1.plot(thetas_set/np.pi, thetas_set/np.pi, color="black",
           linewidth=0.5, linestyle="--")

# endpoints in data coordinates
index_of_pi = np.argmin(np.abs(thetas_set-np.pi))+1
shift = 0.005
x_brace_1 = thetas_set[index_of_pi]/np.pi - shift
y_brace_1_top = thetas_rec_ell_num[index_of_pi]/np.pi
y_brace_1_bottom = 1

p_top_1 = (x_brace_1, y_brace_1_top)  # top
p_bot_1 = (x_brace_1, y_brace_1_bottom)  # bottom

# Draw the brace on the INSET axes
theta_1, summit_1, *artists_1 = curlyBrace(
    fig=fig_4,
    ax=axins_theta_reconstructed_1,             # <-- draw on inset
    p1=p_bot_1,             # top first so the summit is near the top endpoint
    p2=p_top_1,
    k_r=0.1,
    bool_auto=True,       # auto scale conversion for non-equal aspect
    str_text="",          # suppress built-in rotated text; we’ll add our own
    color="black", #darkorange
    lw=0.8
)

# Place a horizontal label to the LEFT of the summit (no rotation)
axins_theta_reconstructed_1.text(
    summit_1[0], summit_1[1], r"$\theta_{\text{bias}}$",
    ha="right", va="center",
    fontsize=8, color="black", #darkorange
    rotation=0
)

# Limit inset to the highlighted region
axins_theta_reconstructed_1.set_xlim(x1_inset_theta_reconstructed_1, x2_inset_theta_reconstructed_1)
axins_theta_reconstructed_1.set_ylim(y1_inset_theta_reconstructed_1, y2_inset_theta_reconstructed_1)

# Optional: declutter ticks inside the inset
axins_theta_reconstructed_1.tick_params(axis='both', which='both', labelsize=6, direction="in")
axins_theta_reconstructed_1.set_xticks([0.9, 1.0, 1.1])
axins_theta_reconstructed_1.set_yticks([0.9, 1.0, 1.1])

# Or hide labels entirely:
# axins_theta_reconstructed_1.set_xticklabels([])
# axins_theta_reconstructed_1.set_yticklabels([])


hf.plot_line_with_wide_err(axins_theta_reconstructed_1, thetas_calibrated_exp_fine/np.pi,
                           (thetas_calibrated_exp_fine+thetas_bias_ell_exp_fine) /
                           np.pi, 0, (thetas_rec_ell_std_exp_fine)/np.pi,
                           colour_ell, marker="", ls="")
hf.plot_line_with_wide_err(axins_theta_reconstructed_1, thetas_calibrated_exp_fine/np.pi,
                           (thetas_calibrated_exp_fine+thetas_bias_sum_exp_fine) /
                           np.pi, 0, (thetas_rec_sum_std_exp_fine)/np.pi,
                           colour_sum, marker="", ls="")

patch_axins_theta_reconstructed_1, connector1_axins_theta_reconstructed_1, connector2_axins_theta_reconstructed_1 = mark_inset(
    ax_theta_reconstructed,
    axins_theta_reconstructed_1,
    loc1=1,
    loc2=2,
    fc='none',
    ec=first_inset_colour
)

# 2 (upper left)     1 (upper right)
# 3 (lower left)     4 (lower right)

connector1_axins_theta_reconstructed_1.loc1 = 3  # Inset: lower left
connector1_axins_theta_reconstructed_1.loc2 = 2  # Parent: lower left

connector2_axins_theta_reconstructed_1.loc1 = 4  # Inset: lower right
connector2_axins_theta_reconstructed_1.loc2 = 1  # Parent: lower right

connector1_axins_theta_reconstructed_1.remove()
connector2_axins_theta_reconstructed_1.remove()

#######################################
############# INSET 1 END #############
#######################################

#########################################
############# INSET 2 START #############
#########################################

# --- Inset limits ---
x1_inset_theta_reconstructed_2, x2_inset_theta_reconstructed_2 = 1.9, 2.1   # theta_set / pi
y1_inset_theta_reconstructed_2, y2_inset_theta_reconstructed_2 = 1.87, 2.13  # theta_reconstructed / pi

# --- Create the inset axes (axes-fraction coords) ---
axins_theta_reconstructed_2 = ax_theta_reconstructed.inset_axes(
    [0.625, 0.15, 0.25, 0.25*ax_theta_reconstructed_aspect_ratio])  # xrel, yrel, wrel hrel


axins_theta_reconstructed_2.spines[:].set_color(second_inset_colour)

# Move y-axis + ticks to the right
axins_theta_reconstructed_2.yaxis.tick_right()      # put ticks/labels on the right
axins_theta_reconstructed_2.yaxis.set_label_position("right")  # if you also have a ylabel

# # Move x-axis + ticks to the top
# axins_theta_reconstructed_2.xaxis.tick_top()      # put ticks/labels on the top
# axins_theta_reconstructed_2.xaxis.set_label_position("top")  

# Plot the same content inside the inset
axins_theta_reconstructed_2.plot(thetas_set/np.pi, thetas_rec_ell_num /
           np.pi, color=colour_ell, linewidth=1)
axins_theta_reconstructed_2.plot(thetas_set/np.pi, thetas_set/np.pi, color="black",
           linewidth=0.5, linestyle="--")

# Endpoints in data coordinates
index_of_2pi = np.argmin(np.abs(thetas_set-2*np.pi))
x_brace_2 = thetas_set[index_of_2pi]/np.pi + shift
y_brace_2_top = thetas_rec_ell_num[index_of_2pi]/np.pi
y_brace_2_bottom = 2.0

p_top_2 = (x_brace_2, y_brace_2_top)  # top
p_bot_2 = (x_brace_2, y_brace_2_bottom)  # bottom

# Draw the brace on the INSET axes
theta_2, summit_2, *artists_2 = curlyBrace(
    fig=fig_4,
    ax=axins_theta_reconstructed_2,             # <-- draw on inset
    p1=p_bot_2,             # top first so the summit is near the top endpoint
    p2=p_top_2,
    k_r=0.1,
    bool_auto=True,       # auto scale conversion for non-equal aspect
    str_text="",          # suppress built-in rotated text; we’ll add our own
    color="black", # darkorange
    lw=0.8
)

# Place a horizontal label to the LEFT of the summit (no rotation)
axins_theta_reconstructed_2.text(
    summit_2[0]+0.002, summit_2[1], r"$\theta_{\text{bias}}$",
    ha="left", va="center",
    fontsize=8, color="black", #darkorange
    rotation=0
)

# Limit inset to the highlighted region
axins_theta_reconstructed_2.set_xlim(x1_inset_theta_reconstructed_2, x2_inset_theta_reconstructed_2)
axins_theta_reconstructed_2.set_ylim(y1_inset_theta_reconstructed_2, y2_inset_theta_reconstructed_2)

# Optional: declutter ticks inside the inset
axins_theta_reconstructed_2.tick_params(axis='both', which='both', labelsize=6, direction="in")
axins_theta_reconstructed_2.set_xticks([1.9, 2.0, 2.1])
axins_theta_reconstructed_2.set_yticks([1.9, 2.0, 2.1])

# Or hide labels entirely:
# axins_theta_reconstructed_2.set_xticklabels([])
# axins_theta_reconstructed_2.set_yticklabels([])

patch_axins_theta_reconstructed_2, connector1_axins_theta_reconstructed_2, connector2_axins_theta_reconstructed_2 = mark_inset(
    ax_theta_reconstructed,
    axins_theta_reconstructed_2,
    loc1=1,
    loc2=2,
    fc='none',
    ec=second_inset_colour
)

# 2 (upper left)     1 (upper right)
# 3 (lower left)     4 (lower right)

connector1_axins_theta_reconstructed_2.loc1 = 2  # Inset: lower left
connector1_axins_theta_reconstructed_2.loc2 = 3  # Parent: upper left

connector2_axins_theta_reconstructed_2.loc1 = 1  # Parent: upper right
connector2_axins_theta_reconstructed_2.loc2 = 4  # Inset: upper right

connector1_axins_theta_reconstructed_2.remove()
connector2_axins_theta_reconstructed_2.remove()
#######################################
############# INSET 2 END #############
#######################################

################################################################################
######################## END THETA RECONSTRUCED SUBPLOT ########################
################################################################################


########################################################################
####################### START THETA BIAS SUBPLOT #######################
########################################################################
ax_theta_bias.plot(thetas_set/np.pi, np.full_like(thetas_set, 0), color="black",
                   linewidth=0.5, linestyle="--")

ax_theta_bias.plot(thetas_set/np.pi, thetas_rec_ell_num-thetas_set,
                   color=colour_ell, linewidth=1)
ax_theta_bias.fill_between(thetas_set/np.pi,
                           thetas_rec_ell_num-thetas_set - 1*thetas_rec_ell_std_num,
                           thetas_rec_ell_num-thetas_set + 1*thetas_rec_ell_std_num,
                           color=colour_ell, alpha=0.3)

ax_theta_bias.plot(thetas_set/np.pi, thetas_rec_diff_num-thetas_set,
                   color=colour_diff, linewidth=1)
ax_theta_bias.fill_between(thetas_set/np.pi,
                           thetas_rec_diff_num-thetas_set -
                           1*thetas_rec_diff_std_num,
                           thetas_rec_diff_num-thetas_set +
                           1*thetas_rec_diff_std_num,
                           color=colour_diff, alpha=0.3)

ax_theta_bias.plot(thetas_set/np.pi, thetas_rec_sum_num-thetas_set,
                   color=colour_sum, linewidth=1)
ax_theta_bias.fill_between(thetas_set/np.pi,
                           thetas_rec_sum_num-thetas_set - 1*thetas_rec_sum_std_num,
                           thetas_rec_sum_num-thetas_set + 1*thetas_rec_sum_std_num,
                           color=colour_sum, alpha=0.3)

ax_theta_bias.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
########################################################################
######################## END THETA BIAS SUBPLOT ########################
########################################################################

if save_fig:
    fig_4.savefig("fig4.pdf")

plt.show()
