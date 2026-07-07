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
one_pt_rel_w = pt_to_in / full_width

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

fig_width_in = full_width * pt_to_in
fig_height_in = half_height_golden * pt_to_in

frac_v_0 = 0.02
frac_v_1 = 0.15
frac_v_2 = 0.80
frac_v_3 = 1-(frac_v_0+frac_v_1+frac_v_2)

# must be positive
print(frac_v_3)

print(frac_v_0+frac_v_1+frac_v_2+frac_v_3)

# Exact-size figure: no automatic layout engines
fig_S_alpha = plt.figure(figsize=(fig_width_in, fig_height_in), layout=None)

h = [Fixed(frac_h_0*fig_width_in), Fixed(frac_h_1*fig_width_in), Fixed(frac_h_2*fig_width_in), Fixed(frac_h_3*fig_width_in),
     Fixed(frac_h_4*fig_width_in), Fixed(frac_h_5*fig_width_in), Fixed(frac_h_6*fig_width_in), Fixed(frac_h_7*fig_width_in), Fixed(frac_h_8*fig_width_in), Fixed(frac_h_9*fig_width_in)]

v = [Fixed(frac_v_0*fig_height_in), Fixed(frac_v_1*fig_height_in), Fixed(frac_v_2*fig_height_in), Fixed(frac_v_3*fig_height_in)]


div = Divider(fig_S_alpha, (0, 0, 1, 1), h, v, aspect=False)

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
        line = plt.Line2D([hx, hx], [0, 1], lw=0.5, color='grey', linestyle='--', transform=fig_S_alpha.transFigure, zorder=100)
        fig_S_alpha.add_artist(line)
    # Draw horizontal grid lines (row boundaries)
    for vy in v_edge_positions:
        line = plt.Line2D([0, 1], [vy, vy], lw=0.5, color='grey', linestyle='--', transform=fig_S_alpha.transFigure, zorder=100)
        fig_S_alpha.add_artist(line)
#######################################################################
#################### end: show grid via grey lines ####################
#######################################################################


ax_quotient_bar = fig_S_alpha.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=2))
ax_quotient = fig_S_alpha.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=2))

# ax_cut_bar = fig_S_alpha.add_axes(div.get_position(), axes_locator=div.new_locator(nx=8, ny=2))
ax_cut = fig_S_alpha.add_axes(div.get_position(), axes_locator=div.new_locator(nx=6, ny=2))

fig_S_alpha.canvas.draw()

ax_quotient.set_ylabel(r"$\alpha/\pi$", labelpad=0)
ax_cut.set_ylabel(r"$|\theta_{\text{bias}}|/\Delta \theta$", labelpad=0)

ax_quotient.set_xlabel(r"$\theta_\text{set}/\pi$")
ax_cut.set_xlabel(r"$\theta_\text{set}/\pi$")

ax_quotient.tick_params(axis='both', direction='in')
ax_cut.tick_params(axis='both', direction='in')

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

thetas_change = hf.theta_change(0.824, 0.063, alphas)

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


################################
### end: load numerical data ###
################################


##################################################################
####################### START LEFT SUBPLOT #######################
##################################################################

levels = np.linspace(0, 1.0, 11)

mesh_left = ax_quotient.contourf(
        thetas_set /np.pi,
        alphas/np.pi,
        theta_bias_alphas_abs/ thetas_rec_alphas_std,
        levels=levels,
        cmap="jet",
        extend="max"
    )

ax_quotient.contourf(
        thetas_set_fine /np.pi,
        alphas_fine/np.pi,
        theta_bias_alphas_abs_fine/ thetas_rec_alphas_std_fine,
        levels=levels,
        cmap="jet",
        extend="max"
    )
cbar_left = fig_S_alpha.colorbar(mesh_left, cax=ax_quotient_bar)

cbar_left.set_label(r"$|\theta_{\text{bias}}|/\Delta \theta$", labelpad=4)
cbar_left.ax.yaxis.set_label_position("left")
cbar_left.set_ticks(levels)


cbar_left.ax.tick_params(
    axis='y',
    direction='in',
    length=3,
    width=0.8,
    labelsize=8
)

alpha_edges = np.empty(len(alphas) + 1)
alpha_edges[1:-1] = 0.5 * (alphas[:-1] + alphas[1:])
alpha_edges[0] = alphas[0] - (alpha_edges[1] - alphas[0])
alpha_edges[-1] = alphas[-1] + (alphas[-1] - alpha_edges[-2])

# for a, th, y0, y1 in zip(alphas, thetas_change, alpha_edges[:-1], alpha_edges[1:]):
#     if np.isfinite(th):
#         ax_quotient.ax_quotient.vlines(th / np.pi, y0/np.pi, y1/np.pi, color="w", linewidth=1)
        
ax_quotient.plot(hf.theta_change(0.824, 0.063,np.linspace(alphas.min(),alphas.max(),10000))/np.pi, 
                 np.linspace(alphas.min(),alphas.max(),10000)/np.pi, 
                 'w-', 
                 lw=1)

# Optional: add thin contour lines for clarity
# cs = ax_quotient.contour(
#         thetas_set /np.pi,
#         alphas/np.pi,
#         theta_bias_alphas_abs/ thetas_rec_alphas_std,
#         levels=[1.0],
#         colors="g",
#         linewidths=1,
# )

# ax_quotient.clabel(cs, inline=True, fmt=r"$1$", fontsize=8,colors='w')
ax_quotient.set_ylim(top=0.25)

ax_quotient.text(
        0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
        'A',
        # fontsize=9,
        fontweight='bold',
        transform=ax_quotient.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='w'
    )

##########################################
### start: inset ###

height = 0.45

quotient_inset = ax_quotient.inset_axes([0.25,0.75-height,height*1.0,height],
                                xlim = (0.9,1),
                                xticklabels=[0.9,0.95,1], 
                                yticklabels=[0.245,'0.250'],
                                xticks=[0.9,0.95,1],
                                yticks=[0.245,0.25])

quotient_inset.contourf(
        thetas_set_fine /np.pi,
        alphas_fine/np.pi,
        theta_bias_alphas_abs_fine/ thetas_rec_alphas_std_fine,
        levels=levels,
        cmap="jet",
        extend="max"
    )
# quotient_inset.contour(
#         thetas_set_fine /np.pi,
#         alphas_fine/np.pi,
#         theta_bias_alphas_abs_fine/ thetas_rec_alphas_std_fine,
#         levels=[1.0],
#         colors="black",
#         linewidths=1,
# )

# frame of inset white
for spine in quotient_inset.spines.values():
    spine.set_edgecolor('w')   # or spine.set_color('w')
# ticks and their labels in inset white
quotient_inset.tick_params(color='w',labelcolor='w')
# tick labels of ax_quotient white
ax_quotient.tick_params(color='w')
###################################################################
######################## END LEFT SUBPLOT #########################
###################################################################

###################################################################
####################### START RIGHT SUBPLOT #######################
###################################################################



# Results of geometric ellipse fits
num_eval_theta_scan_geom = np.load('../num_eval/num_eval_res_theta_S_sum_geo_ell.npz')
thetas_set_geom              = num_eval_theta_scan_geom['thetas_set']
thetas_rec_ell_num_geom      = num_eval_theta_scan_geom['thetas_rec_ell']
thetas_rec_ell_std_num_geom  = num_eval_theta_scan_geom['thetas_rec_ell_std']
A0_set_geom                  = num_eval_theta_scan_geom['A0_set']
sigma_set_geom               = num_eval_theta_scan_geom['sigma_set']
theta_bias_ell_abs_geom = np.abs(thetas_rec_ell_num_geom - thetas_set_geom)

# Results of algebraic ellipse fits
num_eval_theta_scan_alg = np.load('../num_eval/num_eval_res_theta_S_sum_PEAC.npz')
thetas_set_alg              = num_eval_theta_scan_alg['thetas_set']
thetas_rec_ell_num_alg      = num_eval_theta_scan_alg['thetas_rec_ell']
thetas_rec_ell_std_num_alg  = num_eval_theta_scan_alg['thetas_rec_ell_std']
theta_bias_ell_abs_alg = np.abs(thetas_rec_ell_num_alg - thetas_set_alg)

# cut plots
# geom ellipse
ax_cut.plot(thetas_set_geom/np.pi, 
            theta_bias_ell_abs_geom/thetas_rec_ell_std_num_geom, 
            label="geom. ell.",
            color=colour_ell)#,ls=(0,(1.5,1)))

# alg ellipse
# ax_cut.plot(thetas_set_alg/np.pi, theta_bias_ell_abs_alg/thetas_rec_ell_std_num_alg, label="alg. ell.", color="magenta")

# alpha = 0.156 pi
# ax_cut.plot(thetas_set/np.pi, theta_bias_alphas_abs[alpha_big_index]/thetas_rec_alphas_std[alpha_big_index], label=fr"$\alpha={alphas[alpha_big_index]/np.pi:.3f} \pi$", color="darkred")

# alpha=0.249 for theta in 0.5 to 1.0 pi
ax_cut.plot(thetas_set_slice/np.pi, 
            theta_bias_alphas_abs_slice[0]/thetas_rec_alphas_std_slice[0], 
            label=fr"$\alpha={alphas_slice[0]/np.pi:.3f} \pi$", 
            color='darkred', ls=(0, (5, 1)))

# alpha = pi/4
ax_cut.plot(thetas_set/np.pi, theta_bias_alphas_abs[-1]/thetas_rec_alphas_std[-1], label=fr"$\alpha={alphas[-1]/np.pi:.3f} \pi$", color=colour_hist_sum)

# alpha=pi/4-eps for theta in 0.9 to 1.0 pi
# test_index = -2
# ax_cut.plot(thetas_set_fine/np.pi, theta_bias_alphas_abs_fine[test_index]/thetas_rec_alphas_std_fine[test_index], label=fr"$\alpha={alphas_fine[test_index]/np.pi:.4f} \pi$", color='magenta')

ax_cut.hlines(1, 0.54, 1, color="black", linestyle="--",lw=1)


ax_cut.set_xlim(0.5, 1)
ax_cut.set_ylim(0.0, 1.05)

ax_cut.legend(bbox_to_anchor=(0.05,0.95),loc='upper left')

ax_cut.text(
        0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
        'B',
        # fontsize=9,
        fontweight='bold',
        transform=ax_cut.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='k'
    )

###################################################################
######################## END RIGHT SUBPLOT ########################
###################################################################

if save_fig:
    fig_S_alpha.savefig("figS6.pdf")

plt.show()
