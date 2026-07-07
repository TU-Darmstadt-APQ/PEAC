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
fig_S_alpha_bias_and_unct = plt.figure(figsize=(fig_width_in, fig_height_in), layout=None)

h = [Fixed(frac_h_0*fig_width_in), Fixed(frac_h_1*fig_width_in), Fixed(frac_h_2*fig_width_in), Fixed(frac_h_3*fig_width_in),
     Fixed(frac_h_4*fig_width_in), Fixed(frac_h_5*fig_width_in), Fixed(frac_h_6*fig_width_in), Fixed(frac_h_7*fig_width_in), Fixed(frac_h_8*fig_width_in), Fixed(frac_h_9*fig_width_in)]

v = [Fixed(frac_v_0*fig_height_in), Fixed(frac_v_1*fig_height_in), Fixed(frac_v_2*fig_height_in), Fixed(frac_v_3*fig_height_in)]


div = Divider(fig_S_alpha_bias_and_unct, (0, 0, 1, 1), h, v, aspect=False)

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
        line = plt.Line2D([hx, hx], [0, 1], lw=0.5, color='grey', linestyle='--', transform=fig_S_alpha_bias_and_unct.transFigure, zorder=100)
        fig_S_alpha_bias_and_unct.add_artist(line)
    # Draw horizontal grid lines (row boundaries)
    for vy in v_edge_positions:
        line = plt.Line2D([0, 1], [vy, vy], lw=0.5, color='grey', linestyle='--', transform=fig_S_alpha_bias_and_unct.transFigure, zorder=100)
        fig_S_alpha_bias_and_unct.add_artist(line)
#######################################################################
#################### end: show grid via grey lines ####################
#######################################################################


ax_abs_bias_bar = fig_S_alpha_bias_and_unct.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=2))
ax_abs_bias = fig_S_alpha_bias_and_unct.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=2))

ax_std_bias_bar = fig_S_alpha_bias_and_unct.add_axes(div.get_position(), axes_locator=div.new_locator(nx=8, ny=2))
ax_std_bias = fig_S_alpha_bias_and_unct.add_axes(div.get_position(), axes_locator=div.new_locator(nx=6, ny=2))

fig_S_alpha_bias_and_unct.canvas.draw()

ax_abs_bias.set_ylabel(r"$\alpha/\pi$", labelpad=0)
ax_std_bias.set_ylabel(r"$\alpha/\pi$", labelpad=0)

ax_abs_bias.set_xlabel(r"$\theta_\text{set}/\pi$")
ax_std_bias.set_xlabel(r"$\theta_\text{set}/\pi$")

ax_abs_bias.tick_params(axis='both', direction='in')
ax_std_bias.tick_params(axis='both', direction='in')

##################################
### start: load numerical data ###
##################################

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

### Fine Scan of Alpha & Theta

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

thetas_change_fine = hf.theta_change(0.824, 0.063, alphas_fine)

# vmax = np.max([theta_bias_alphas_abs,thetas_rec_alphas_std])/np.pi
# vmin = np.max([theta_bias_alphas_abs,thetas_rec_alphas_std])/np.pi

################################
### end: load numerical data ###
################################


##################################################################
####################### START LEFT SUBPLOT #######################
##################################################################

mesh_left = ax_abs_bias.pcolormesh(
        thetas_set/np.pi,
        alphas/np.pi,
        theta_bias_alphas_abs/np.pi,
        cmap="jet",
        shading="auto",
        norm="log",
        rasterized=True,
        # vmin=vmin,
        # vmax=vmax
    )

ax_abs_bias.pcolormesh(
        thetas_set_fine/np.pi,
        alphas_fine/np.pi,
        theta_bias_alphas_abs_fine/np.pi,
        cmap="jet",
        shading="auto",
        norm="log",
        rasterized=True,
        vmin=theta_bias_alphas_abs.min()/np.pi,
        vmax=theta_bias_alphas_abs.max()/np.pi,
    )

##########################################
### start: inset ###

height = 0.45

bias_inset = ax_abs_bias.inset_axes([0.25,0.75-height,height*1.0,height],
                                xlim = (0.9,1),
                                xticklabels=[0.9,0.95,1], 
                                yticklabels=[0.245,0.249, '0.250'],
                                xticks=[0.9,0.95,1],
                                yticks=[0.245, 0.249, 0.25])

bias_inset.pcolormesh(
        thetas_set_fine/np.pi,
        alphas_fine/np.pi,
        theta_bias_alphas_abs_fine/np.pi,
        cmap="jet",
        shading="auto",
        norm="log",
        rasterized=True,
        vmin=theta_bias_alphas_abs.min()/np.pi,
        vmax=theta_bias_alphas_abs.max()/np.pi,
    )
bias_inset.tick_params(axis='y',right=True,labelright=True,left=False,labelleft=False)

cbar_left = fig_S_alpha_bias_and_unct.colorbar(mesh_left, cax=ax_abs_bias_bar)

cbar_left.set_label(r"$|\theta_{\text{bias}}|/\pi$", labelpad=3)
cbar_left.ax.yaxis.set_label_position("left")

cbar_left.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())

cbar_left.ax.tick_params(
    axis='y',
    which='major',
    direction='in',
    length=3,
    width=0.8,
    labelsize=8
)

cbar_left.ax.tick_params(
    axis='y',
    which='minor',
    direction='in',
    length=2,
    width=0.6
)

alpha_edges = np.empty(len(alphas) + 1)
alpha_edges[1:-1] = 0.5 * (alphas[:-1] + alphas[1:])
alpha_edges[0] = alphas[0] - (alpha_edges[1] - alphas[0])
alpha_edges[-1] = alphas[-1] + (alphas[-1] - alpha_edges[-2])

# for a, th, y0, y1 in zip(alphas, thetas_change, alpha_edges[:-1], alpha_edges[1:]):
#     if np.isfinite(th):
#         ax_abs_bias.vlines(th / np.pi, y0/np.pi, y1/np.pi, color="black", linewidth=1)

ax_abs_bias.plot(hf.theta_change(0.824, 0.063,np.linspace(alphas.min(),alphas.max(),10000))/np.pi, 
                 np.linspace(alphas.min(),alphas.max(),10000)/np.pi, 
                 'k-', 
                 lw=1)

ax_abs_bias.text(
        0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
        'A',
        # fontsize=9,
        fontweight='bold',
        transform=ax_abs_bias.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='k'
    )
####################################################################
######################## END LEFT SUBPLOT #########################
####################################################################

####################################################################
####################### START RIGHT SUBPLOT #######################
####################################################################


mesh_right = ax_std_bias.pcolormesh(
        thetas_set/np.pi,
        alphas/np.pi,
        thetas_rec_alphas_std *1e3 / np.pi,
        cmap="jet",
        shading="auto",
        norm="log",
        rasterized=True,
        vmin=thetas_rec_alphas_std_fine.min()*1e3/np.pi,
        # vmax=vmax
    )

ax_std_bias.pcolormesh(
        thetas_set_fine/np.pi,
        alphas_fine/np.pi,
        thetas_rec_alphas_std_fine *1e3 / np.pi,
        cmap="jet",
        shading="auto",
        norm="log",
        rasterized=True,
        vmin=thetas_rec_alphas_std_fine.min()*1e3/np.pi,
        vmax=thetas_rec_alphas_std.max()*1e3/np.pi,
    )

ax_std_bias.text(
        0.025, 0.975,  # x=2.5% from left, y=97.5% from bottom
        'B',
        # fontsize=9,
        fontweight='bold',
        transform=ax_std_bias.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        color='k'
    )

##########################################
### start: inset ###

height = 0.45

std_inset = ax_std_bias.inset_axes([0.25,0.75-height,height*1.0,height],
                                xlim = (0.9,1),
                                xticklabels=[0.9,0.95,1], 
                                yticklabels=[0.245, 0.249, '0.250'],
                                xticks=[0.9,0.95,1],
                                yticks=[0.245, 0.249, 0.25])

std_inset.pcolormesh(
        thetas_set_fine/np.pi,
        alphas_fine/np.pi,
        thetas_rec_alphas_std_fine *1e3/ np.pi,
        cmap="jet",
        shading="auto",
        norm="log",
        rasterized=True,
        vmin=thetas_rec_alphas_std_fine.min()*1e3/np.pi,
        vmax=thetas_rec_alphas_std.max()*1e3/np.pi,
    )


# fig_S_alpha_bias_and_unct.colorbar(mesh_right, cax=ax_std_bias_bar)

cbar_right = fig_S_alpha_bias_and_unct.colorbar(mesh_right, cax=ax_std_bias_bar,
                                                ticks = [5,6,10,20],
                                                format = mticker.FixedFormatter(['5','','10','20']))

cbar_right.set_label(r"$\Delta \theta / \pi\times10^{-3}$", labelpad=3)
cbar_right.ax.yaxis.set_label_position("left")

# cbar_right.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
# cbar_right.ax.yaxis.set_minor_formatter(mticker.NullFormatter())

cbar_right.ax.tick_params(
    axis='y',
    which='major',
    direction='in',
    length=3,
    width=0.8,
    labelsize=8
)

cbar_right.ax.tick_params(
    axis='y',
    which='minor',
    direction='in',
    length=2,
    width=0.6
)

# for a, th, y0, y1 in zip(alphas, thetas_change, alpha_edges[:-1], alpha_edges[1:]):
#     if np.isfinite(th):
#         ax_std_bias.plot(th / np.pi, y0/np.pi, y1/np.pi, color="black", linewidth=1,marker='o',ms=1)

ax_std_bias.plot(hf.theta_change(0.824, 0.063,np.linspace(alphas.min(),alphas.max(),10000))/np.pi, 
                 np.linspace(alphas.min(),alphas.max(),10000)/np.pi, 
                 'k-', 
                 lw=1)

###################################################################
######################## END RIGHT SUBPLOT ########################
###################################################################

if save_fig:
    fig_S_alpha_bias_and_unct.savefig("figS5.pdf")

plt.show()
