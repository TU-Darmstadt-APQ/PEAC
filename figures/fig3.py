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

save_fig = False

import helper_functions as hf

plt.style.use('../paper_mpl_style.mplstyle')

#################
#### colours ####
#################

colour_hist_diff = 'C0'
colour_hist_sum = 'C1'
colour_ell = 'C2'

# --- Size ---
phi_golden = (1 + np.sqrt(5)) / 2
pt_to_in = 1.0 / 72.0

# full_width = 522
half_width = 255

full_height_golden = half_width / phi_golden # golden aspect
full_height = 0.7*full_height_golden # 255

######## Grid ##########
one_pt_rel_w = pt_to_in / half_width

frac_h_0 = 0.00
frac_h_1 = 0.10
frac_h_2 = 0.325
frac_h_3 = 0.075
frac_h_4 = 0.00
frac_h_5 = frac_h_1
frac_h_6 = frac_h_2
frac_h_7 = frac_h_3
frac_h_8 = 1-(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5+frac_h_6+frac_h_7)

print(frac_h_0+frac_h_1+frac_h_2+frac_h_3+frac_h_4+frac_h_5+frac_h_6+frac_h_7+frac_h_8)

fig_width_in = half_width * pt_to_in
height_factor = frac_h_0 + frac_h_1 + frac_h_2 + frac_h_3 + one_pt_rel_w*0.8
fig_height_in = height_factor * fig_width_in  

frac_v_0 = frac_h_0/height_factor
frac_v_1 = frac_h_1/height_factor
frac_v_2 = frac_h_2/height_factor
frac_v_3 = frac_h_3/height_factor
frac_v_4 = one_pt_rel_w*0.8/height_factor

print(frac_v_0+frac_v_1+frac_v_2+frac_v_3+frac_v_4)
      

# Exact-size figure: no automatic layout engines
fig_3 = plt.figure(figsize=(fig_width_in, fig_height_in), layout=None)

h = [Fixed(frac_h_0*fig_width_in), Fixed(frac_h_1*fig_width_in), Fixed(frac_h_2*fig_width_in), Fixed(frac_h_3*fig_width_in),
     Fixed(frac_h_4*fig_width_in), Fixed(frac_h_5*fig_width_in), Fixed(frac_h_6*fig_width_in), Fixed(frac_h_7*fig_width_in), Fixed(frac_h_8*fig_width_in)]

v = [Fixed(frac_v_0*fig_height_in), Fixed(frac_v_1*fig_height_in), Fixed(frac_v_2*fig_height_in), Fixed(frac_v_3*fig_height_in),
     Fixed(frac_v_4*fig_height_in)]


div = Divider(fig_3, (0, 0, 1, 1), h, v, aspect=False)

#########################################################################
#################### start: show grid via grey lines ####################
#########################################################################
# h_fracs = [frac_h_0, frac_h_1, frac_h_2, frac_h_3, frac_h_4, frac_h_5, frac_h_6, frac_h_7, frac_h_8]
# v_fracs = [frac_v_0, frac_v_1, frac_v_2, frac_v_3, frac_v_4]

# h_edge_positions = np.cumsum([0]+h_fracs)
# v_edge_positions = np.cumsum([0]+v_fracs)

# # Draw vertical grid lines (column boundaries)
# for hx in h_edge_positions:
#     line = plt.Line2D([hx, hx], [0, 1], lw=0.5, color='grey', linestyle='--', transform=fig_3.transFigure, zorder=100)
#     fig_3.add_artist(line)
# # Draw horizontal grid lines (row boundaries)
# for vy in v_edge_positions:
#     line = plt.Line2D([0, 1], [vy, vy], lw=0.5, color='grey', linestyle='--', transform=fig_3.transFigure, zorder=100)
#     fig_3.add_artist(line)
#######################################################################
#################### end: show grid via grey lines ####################
#######################################################################


ax_ellipse_left_hist_diff = fig_3.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=2))
ax_ellipse_left_hist_sum = fig_3.add_axes(div.get_position(), axes_locator=div.new_locator(nx=2, ny=3))
ax_ellipse_left = fig_3.add_axes(div.get_position(), axes_locator=div.new_locator(nx=2, ny=2))

ax_ellipse_right_hist_diff = fig_3.add_axes(div.get_position(), axes_locator=div.new_locator(nx=7, ny=2))
ax_ellipse_right_hist_sum = fig_3.add_axes(div.get_position(), axes_locator=div.new_locator(nx=6, ny=3))
ax_ellipse_right = fig_3.add_axes(div.get_position(), axes_locator=div.new_locator(nx=6, ny=2))

fig_3.canvas.draw()

ax_ellipse_left.set_ylabel(r"$S_\text{diff}$", labelpad=-2)
ax_ellipse_right.set_ylabel(r"$S_\text{diff}$", labelpad=-2)

ax_ellipse_left.set_xlabel(r"$S_\text{sum}$")
ax_ellipse_right.set_xlabel(r"$S_\text{sum}$")

ax_ellipse_left.tick_params(axis='both', direction='in')
ax_ellipse_right.tick_params(axis='both', direction='in')

ax_ellipse_left_hist_diff.set_yticklabels([])
ax_ellipse_left_hist_diff.tick_params(axis='both', direction='in')

ax_ellipse_right_hist_diff.set_yticklabels([])
ax_ellipse_right_hist_diff.tick_params(axis='both', direction='in')

ax_ellipse_left_hist_sum.set_xticklabels([])
ax_ellipse_left_hist_sum.tick_params(axis='both', direction='in')

ax_ellipse_right_hist_sum.set_xticklabels([])
ax_ellipse_right_hist_sum.tick_params(axis='both', direction='in')



########## Insets ################
fig_3.canvas.draw()
ax_ellipse_left_width_in, ax_ellipse_left_height_in = hf.get_ax_size_inches(ax_ellipse_left, fig_3)
aspect_ratio = ax_ellipse_left_width_in / ax_ellipse_left_height_in  # z.B. 1.5 wenn breiter als hoch
pt_rel = pt_to_in / ax_ellipse_left_width_in

desired_width_relative = 0.30 
desired_height_relative = desired_width_relative / aspect_ratio

# ax_inset_left = ax_ellipse_left.inset_axes([-desired_width_relative/2, 1-desired_height_relative/2, desired_width_relative, desired_height_relative])
ax_inset_left = ax_ellipse_left.inset_axes([1-desired_width_relative-1.5*pt_rel, 1-desired_height_relative-1.5*pt_rel, desired_width_relative, desired_height_relative])

# ax_inset_right = ax_ellipse_right.inset_axes([-desired_width_relative/2, 1-desired_height_relative/2, desired_width_relative, desired_height_relative])
ax_inset_right = ax_ellipse_right.inset_axes([1-desired_width_relative-1.5*pt_rel, 1-desired_height_relative-1.5*pt_rel, desired_width_relative, desired_height_relative])


##########################################################################
####################### START LEFT ELLIPSE SUBPLOT #######################
##########################################################################


#####################################
### start: load experimental data ###
#####################################


folder = "Coarse"
Ts = np.linspace(1e-3, 3e-3, 21)
index_left = 7 # 4 ,6
S_plus_left = np.load(f'../exp_data/{folder}/signal_p1.npy')[:, index_left,:].ravel()
S_minus_left = np.load(f'../exp_data/{folder}/signal_m1.npy')[:, index_left,:].ravel()

S_sum_left = (S_plus_left + S_minus_left)/np.sqrt(2)
S_diff_left = (S_plus_left - S_minus_left)/np.sqrt(2)

###################################
### end: load experimental data ###
###################################

marker_plus = MarkerStyle((4, 2, 0), capstyle='butt')
marker_cross = MarkerStyle((4, 2, 45), capstyle='butt')

scatter_plot_bounds = np.sqrt(2) * 1.45*0.824
ax_ellipse_left.set_xlim(-scatter_plot_bounds, scatter_plot_bounds)
ax_ellipse_left.set_ylim(-scatter_plot_bounds, scatter_plot_bounds)

ax_ellipse_left.plot(S_sum_left, S_diff_left, color=colour_ell, marker=marker_cross, markersize=4, linestyle='none', markeredgewidth=0.4)

phase = np.linspace(0, 1, 20, endpoint=False) * 2 * np.pi
phases = (np.ones((15, len(phase))) * phase).ravel()

ax_inset_left.scatter(
        S_minus_left, S_plus_left,
        c=phases / np.pi, cmap='jet', s=0.001
    )
ax_inset_left.set_facecolor('#bfbfbf')
ax_inset_left.set_xlim(-1.2, 1.2)
ax_inset_left.set_ylim(-1.2, 1.2)
ax_inset_left.set_yticks([])
ax_inset_left.set_xticks([])

ax_inset_left.set_ylabel(r'$S_{+1}$', labelpad=2)
ax_inset_left.set_xlabel(r'$S_{-1}$', labelpad=2)

ax_inset_left.tick_params(
    axis='both',      # x and y
    which='both',     # major and minor ticks
    labelsize=6,      # tick label font size
    direction='in'    # ticks pointing inwards
)
ax_inset_left.tick_params(axis='x', pad=2)
ax_inset_left.tick_params(axis='y', pad=2)

ax_inset_left.xaxis.label.set_size(6)
ax_inset_left.yaxis.label.set_size(6)

######### histogram sum
bin_centres_sum_left, hist_data_sum_left, bin_edges_sum_left = hf.create_hist_not_normalised(S_sum_left)
bin_width_sum_left = bin_edges_sum_left[1:] - bin_edges_sum_left[:-1]

ax_ellipse_left_hist_sum.bar(bin_centres_sum_left, hist_data_sum_left, width=bin_width_sum_left,
          align='center', color=colour_hist_sum, linewidth=0)
ax_ellipse_left_hist_sum.set_xlim(-scatter_plot_bounds, scatter_plot_bounds)
ax_ellipse_left_hist_sum.set_xticks([])
ax_ellipse_left_hist_sum.set_yticks([])
ax_ellipse_left_hist_sum.tick_params(axis='both', direction='in')

######### histogram diff
bin_centres_diff_left, hist_data_diff_left, bin_edges_diff_left = hf.create_hist_not_normalised(S_diff_left)
bin_width_diff_left = bin_edges_diff_left[1:] - bin_edges_diff_left[:-1]

ax_ellipse_left_hist_diff.barh(bin_centres_diff_left, hist_data_diff_left, height=bin_width_diff_left,
                  align='center', color=colour_hist_diff, linewidth=0)
ax_ellipse_left_hist_diff.set_ylim(-scatter_plot_bounds, scatter_plot_bounds)
# ax_ellipse_left_hist_diff.set_xticklabels([])
# ax_ellipse_left_hist_diff.set_yticklabels([])
ax_ellipse_left_hist_diff.set_xticks([])
ax_ellipse_left_hist_diff.set_yticks([])
ax_ellipse_left_hist_diff.tick_params(axis='both', direction='in')




##########################################################################
######################## END LEFT ELLIPSE SUBPLOT ########################
##########################################################################

##########################################################################
####################### START RIGHT ELLIPSE SUBPLOT #######################
##########################################################################


#####################################
### start: load experimental data ###
#####################################

folder = "Coarse"
Ts = np.linspace(1e-3, 3e-3, 21)
index_right = 15 # 0
S_plus_right = np.load(f'../exp_data/{folder}/signal_p1.npy')[:, index_right,:].ravel()
S_minus_right = np.load(f'../exp_data/{folder}/signal_m1.npy')[:, index_right,:].ravel()

S_sum_right = (S_plus_right + S_minus_right)/np.sqrt(2)
S_diff_right = (S_plus_right - S_minus_right)/np.sqrt(2)

###################################
### end: load experimental data ###
###################################

marker_plus = MarkerStyle((4, 2, 0), capstyle='butt')
marker_cross = MarkerStyle((4, 2, 45), capstyle='butt')

ax_ellipse_right.set_xlim(-scatter_plot_bounds, scatter_plot_bounds)
ax_ellipse_right.set_ylim(-scatter_plot_bounds, scatter_plot_bounds)

ax_ellipse_right.plot(S_sum_right, S_diff_right, color=colour_ell, marker=marker_cross, markersize=4, linestyle='none', markeredgewidth=0.4)

ax_inset_right.scatter(
        S_minus_right, S_plus_right,
        c=phases / np.pi, cmap='jet', s=0.001
    )
ax_inset_right.set_facecolor('#bfbfbf')
ax_inset_right.set_xlim(-1.2, 1.2)
ax_inset_right.set_ylim(-1.2, 1.2)
ax_inset_right.set_yticks([])
ax_inset_right.set_xticks([])

ax_inset_right.set_ylabel(r'$S_{+1}$', labelpad=2)
ax_inset_right.set_xlabel(r'$S_{-1}$', labelpad=2)

ax_inset_right.tick_params(
    axis='both',      # x and y
    which='both',     # major and minor ticks
    labelsize=6,      # tick label font size
    direction='in'    # ticks pointing inwards
)
ax_inset_right.tick_params(axis='x', pad=2)
ax_inset_right.tick_params(axis='y', pad=2)

ax_inset_right.xaxis.label.set_size(6)
ax_inset_right.yaxis.label.set_size(6)

######### histogram sum
bin_centres_sum_right, hist_data_sum_right, bin_edges_sum_right = hf.create_hist_not_normalised(S_sum_right)
bin_width_sum_right = bin_edges_sum_right[1:] - bin_edges_sum_right[:-1]

ax_ellipse_right_hist_sum.bar(bin_centres_sum_right, hist_data_sum_right, width=bin_width_sum_right,
          align='center', color=colour_hist_sum, linewidth=0)
ax_ellipse_right_hist_sum.set_xlim(-scatter_plot_bounds, scatter_plot_bounds)
ax_ellipse_right_hist_sum.set_xticks([])
ax_ellipse_right_hist_sum.set_yticks([])
ax_ellipse_right_hist_sum.tick_params(axis='both', direction='in')

######### histogram diff
bin_centres_diff_right, hist_data_diff_right, bin_edges_diff_right = hf.create_hist_not_normalised(S_diff_right)
bin_width_diff_right = bin_edges_diff_right[1:] - bin_edges_diff_right[:-1]

ax_ellipse_right_hist_diff.barh(bin_centres_diff_right, hist_data_diff_right, height=bin_width_diff_right,
                  align='center', color=colour_hist_diff, linewidth=0)
ax_ellipse_right_hist_diff.set_ylim(-scatter_plot_bounds, scatter_plot_bounds)
ax_ellipse_right_hist_diff.set_xticks([])
ax_ellipse_right_hist_diff.set_yticks([])
ax_ellipse_right_hist_diff.tick_params(axis='both', direction='in')


##########################################################################
######################## END LEFT ELLIPSE SUBPLOT ########################
##########################################################################

if save_fig:
    fig_3.savefig("fig3_wip.pdf")

plt.show()
