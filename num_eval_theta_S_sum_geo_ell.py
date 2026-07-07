# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helper_functions as hf

plt.style.use('paper_mpl_style.mplstyle')  # Apply paper-style plotting template

colour_diff = 'C0'
colour_sum = 'C1'
colour_ell = 'C2'

##################################
### start: load numerical data ###
##################################
## geometric ellipse fits ##
data_theta_scan_geom_fits = np.load('num_data/num_fits_res_theta_S_sum_geo_ell.npz')

lambda_mean_theta_scan_geom_fits          = data_theta_scan_geom_fits['lambda_mean']
lambda_diff_theta_scan_geom_fits          = data_theta_scan_geom_fits['lambda_diff']
lambda_plus_theta_scan_geom_fits          = data_theta_scan_geom_fits['lambda_plus']
lambda_minus_theta_scan_geom_fits         = data_theta_scan_geom_fits['lambda_minus']

n_thetas_theta_scan_geom_fits             = data_theta_scan_geom_fits['n_thetas']
thetas_theta_scan_geom_fits               = data_theta_scan_geom_fits['thetas']
n_stoch_rep_theta_scan_geom_fits          = data_theta_scan_geom_fits['n_stoch_rep']
seed_offset_theta_scan_geom_fits          = data_theta_scan_geom_fits['seed_offset']
n_phis_theta_scan_geom_fits               = data_theta_scan_geom_fits['n_phis']
A0_theta_scan_geom_fits                   = data_theta_scan_geom_fits['A0']
sigma_theta_scan_geom_fits                = data_theta_scan_geom_fits['sigma']
results_ellipse_theta_scan_geom_fits      = data_theta_scan_geom_fits['results_ellipse']

theta_min = thetas_theta_scan_geom_fits[0]
theta_max = thetas_theta_scan_geom_fits[-1]

## PEAC fits ##
num_eval_theta_scan     = np.load('num_eval/num_eval_res_theta_S_sum_PEAC.npz')

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

################################
### end: load numerical data ###
################################

##################################
### start: raw theta reconstr. ###
##################################

### ellipse stuff ###
## ellipse axes ##
x0_ell_geom_fits    = results_ellipse_theta_scan_geom_fits[:,:,0]
y0_ell_geom_fits    = results_ellipse_theta_scan_geom_fits[:,:,1]
ap_ell_geom_fits    = results_ellipse_theta_scan_geom_fits[:,:,2]
bp_ell_geom_fits    = results_ellipse_theta_scan_geom_fits[:,:,3]
alpha_ell_geom_fits = results_ellipse_theta_scan_geom_fits[:,:,4]

major_axis_ell_mean_geom_fits = np.nanmean(ap_ell_geom_fits, axis=1)
major_axis_ell_std_geom_fits = np.nanstd(ap_ell_geom_fits, axis=1, ddof=1)

minor_axis_ell_mean_geom_fits = np.nanmean(bp_ell_geom_fits, axis=1)
minor_axis_ell_std_geom_fits = np.nanstd(bp_ell_geom_fits, axis=1, ddof=1)

minor_axis_real_geom_fits = hf.axis_sum(thetas_theta_scan_geom_fits, A0_theta_scan_geom_fits)
major_axis_real_geom_fits = hf.axis_diff(thetas_theta_scan_geom_fits, A0_theta_scan_geom_fits)

## theta ellipse ##
theta_ell_geom_fits = hf.geom_ell_to_theta(alpha_ell_geom_fits, ap_ell_geom_fits, bp_ell_geom_fits)

thetas_ell_mean_theta_scan_geom_fits_raw = np.nanmean(theta_ell_geom_fits, axis=1)
thetas_ell_std_theta_scan_geom_fits = np.nanstd(theta_ell_geom_fits, axis=1, ddof=1)

################################
### end: raw theta reconstr. ###
################################

###############################
### start: phase unwrapping ###
###############################
## create mask for branches ##
mask_branch1 = thetas_theta_scan_geom_fits <= np.pi
mask_branch2 = (np.pi < thetas_theta_scan_geom_fits) & (thetas_theta_scan_geom_fits <= 2*np.pi)
mask_branch3 = 2*np.pi < thetas_theta_scan_geom_fits

## phase unwrap per branch ##
branch_1_ell = thetas_ell_mean_theta_scan_geom_fits_raw
branch_2_ell = 2*np.pi - thetas_ell_mean_theta_scan_geom_fits_raw
branch_3_ell = 2*np.pi + thetas_ell_mean_theta_scan_geom_fits_raw

## combine branches for correct phase unwrapping ##
thetas_ell_mean_theta_scan_geom_fits = np.empty_like(thetas_theta_scan_geom_fits)

thetas_ell_mean_theta_scan_geom_fits[mask_branch1] = branch_1_ell[mask_branch1]
thetas_ell_mean_theta_scan_geom_fits[mask_branch2] = branch_2_ell[mask_branch2]
thetas_ell_mean_theta_scan_geom_fits[mask_branch3] = branch_3_ell[mask_branch3]


##################
### theta bias ###
##################
theta_bias_ell_geom_fits  = thetas_ell_mean_theta_scan_geom_fits - thetas_theta_scan_geom_fits

### theta bias plot ####
inch_to_cm = 2.54
phi_golden = (1 + np.sqrt(5)) / 2
width_inch = 15 / inch_to_cm
height_inch = width_inch / phi_golden
plt.rc('font', size=10)
fig_theta_bias, ax_theta_bias = plt.subplots(
    1, 1, figsize=(width_inch, height_inch)
)

# series values
x_ell_geom_fits = thetas_theta_scan_geom_fits/np.pi

## bias ellipse ##
ax_theta_bias.grid(True)
ax_theta_bias.minorticks_on()
ax_theta_bias.grid(which='minor', linestyle=':', linewidth=0.6)
ax_theta_bias.axhline(0, color="black", linewidth=1, ls="--")

theta_change = hf.theta_change(A0_theta_scan_geom_fits, sigma_theta_scan_geom_fits, np.pi/4)
ax_theta_bias.axvline(theta_change/np.pi, color="black", linewidth=1, ls="--")
ax_theta_bias.axvline(2-theta_change/np.pi, color="black", linewidth=1, ls="--")

## PEAC results for comparison ##
# hf.plot_line_with_wide_err(ax_theta_bias, thetas_set/np.pi, thetas_rec_sum_num-thetas_set, 0, thetas_rec_sum_std_num, colour_sum, r'$\theta_{\text{bias, ell}}$')
# hf.plot_line_with_wide_err(ax_theta_bias, x_ell_geom_fits, theta_bias_ell_geom_fits, 0, thetas_ell_std_theta_scan_geom_fits, colour_ell, r'$\theta_{\text{bias, ell}}$')
ax_theta_bias.plot(thetas_set/np.pi, thetas_rec_sum_num-thetas_set,
                   color=colour_sum, linewidth=1, label=r'$\theta_{\text{bias, sum}}$')
ax_theta_bias.fill_between(thetas_set/np.pi,
                           thetas_rec_sum_num-thetas_set - 1*thetas_rec_sum_std_num,
                           thetas_rec_sum_num-thetas_set + 1*thetas_rec_sum_std_num,
                           color=colour_sum, alpha=0.3)

ax_theta_bias.plot(x_ell_geom_fits, theta_bias_ell_geom_fits,
                   color=colour_ell, linewidth=1, label=r'$\theta_{\text{bias, ell}}$')
ax_theta_bias.fill_between(x_ell_geom_fits,
                           theta_bias_ell_geom_fits - 1*thetas_ell_std_theta_scan_geom_fits,
                           theta_bias_ell_geom_fits + 1*thetas_ell_std_theta_scan_geom_fits,
                           color=colour_ell, alpha=0.3)


ax_theta_bias.set_xlabel(r'$\theta/\pi$')
ax_theta_bias.set_xlim(x_ell_geom_fits[0], x_ell_geom_fits[-1])
ax_theta_bias.set_xlim(0.5, 1.5)
ax_theta_bias.set_ylim(-0.1, 0.1)
# ax_theta_bias.hlines([0.05], xmin=0.5, xmax=1.5, color='black', linewidth=1)
ax_theta_bias.legend(loc='lower right')

plt.tight_layout()

### theta uncertainty plot ###
fig_theta_unct, ax_theta_unct = plt.subplots(
    figsize=(width_inch, height_inch))
ax_theta_unct.plot(thetas_set/np.pi, thetas_rec_ell_std_num,
                    color="#929292", label=r'$\Delta \theta_\text{alg}$')
ax_theta_unct.plot(thetas_theta_scan_geom_fits/np.pi, thetas_ell_std_theta_scan_geom_fits,
                    color=colour_ell, label=r'$\Delta \theta_\text{geom}$')
ax_theta_unct.plot(thetas_set/np.pi, thetas_rec_sum_std_num,
                    color=colour_sum, label=r'$\Delta \theta_\text{sum}$')
ax_theta_unct.axvline(theta_change/np.pi, color="black", linewidth=1, ls="--")
ax_theta_unct.axvline(2-theta_change/np.pi, color="black", linewidth=1, ls="--")

ax_theta_unct.set_xlabel(r'$\theta/\pi$')
ax_theta_unct.set_xlim(0.5, 1.5)
ax_theta_unct.set_ylim(0.0, 0.08)
# ax_theta_unct.set_yscale("log")
ax_theta_unct.legend(loc='upper right')
ax_theta_unct.grid(True)
ax_theta_unct.minorticks_on()
ax_theta_unct.grid(which='minor', linestyle=':', linewidth=0.6)
plt.tight_layout()


### half-axes bias only for ellipse ###
fig_axes_bias, ax_axes_bias = plt.subplots(
    figsize=(width_inch, height_inch))
ax_axes_bias.grid(True)
ax_axes_bias.minorticks_on()
ax_axes_bias.grid(which='minor', linestyle=':', linewidth=0.6)

## bias major axis ##
ax_axes_bias.plot(thetas_theta_scan_geom_fits/np.pi, major_axis_ell_mean_geom_fits-major_axis_real_geom_fits, color="tab:red",
                    linewidth=1, label='bias major axis')
ax_axes_bias.fill_between(thetas_theta_scan_geom_fits/np.pi,
                            major_axis_ell_mean_geom_fits-major_axis_real_geom_fits - 1*major_axis_ell_std_geom_fits,
                            major_axis_ell_mean_geom_fits-major_axis_real_geom_fits + 1*major_axis_ell_std_geom_fits,
                            color='tab:red', alpha=0.3)
## bias minor axis ##
ax_axes_bias.plot(thetas_theta_scan_geom_fits/np.pi, minor_axis_ell_mean_geom_fits-minor_axis_real_geom_fits, color="tab:purple",
                    linewidth=1, label='bias minor axis')
ax_axes_bias.fill_between(thetas_theta_scan_geom_fits/np.pi,
                            minor_axis_ell_mean_geom_fits-minor_axis_real_geom_fits - 1*minor_axis_ell_std_geom_fits,
                            minor_axis_ell_mean_geom_fits-minor_axis_real_geom_fits + 1*minor_axis_ell_std_geom_fits,
                            color='tab:purple', alpha=0.3)

ax_axes_bias.set_xlabel(r'$\theta/\pi$')
ax_axes_bias.set_xlim(theta_min/np.pi, theta_max/np.pi)
ax_axes_bias.legend(loc='lower right')
plt.tight_layout()

### modulus of half-axes bias only for ellipse ###
fig_axes_bias_mod, ax_axes_bias_mod = plt.subplots(
    figsize=(width_inch, height_inch))
ax_axes_bias_mod.grid(True)
ax_axes_bias_mod.minorticks_on()
ax_axes_bias_mod.grid(which='minor', linestyle=':', linewidth=0.6)
ax_axes_bias_mod.set_yscale("log")

## bias major axis ##
ax_axes_bias_mod.plot(thetas_theta_scan_geom_fits/np.pi, np.abs(major_axis_ell_mean_geom_fits-major_axis_real_geom_fits), color="tab:red",
                    linewidth=1, label='modulus of bias major axis')

## bias minor axis ##
ax_axes_bias_mod.plot(thetas_theta_scan_geom_fits/np.pi, np.abs(minor_axis_ell_mean_geom_fits-minor_axis_real_geom_fits), color="tab:purple",
                    linewidth=1, label='modulus of bias minor axis')

ax_axes_bias_mod.set_xlabel(r'$\theta/\pi$')
ax_axes_bias_mod.set_xlim(theta_min/np.pi, theta_max/np.pi)
ax_axes_bias_mod.legend(loc='lower right')
plt.tight_layout()

### half-axes uncertainty only for ellipse ###
fig_axes_unct, ax_axes_unct = plt.subplots(
    figsize=(width_inch, height_inch))
ax_axes_unct.plot(thetas_theta_scan_geom_fits/np.pi, major_axis_ell_std_geom_fits,
                    color='tab:red', label='uncertainty major axis')
ax_axes_unct.plot(thetas_theta_scan_geom_fits/np.pi, minor_axis_ell_std_geom_fits,
                    color='tab:purple', label='uncertainty minor axis')

ax_axes_unct.set_xlabel(r'$\theta/\pi$')
ax_axes_unct.set_xlim(theta_min/np.pi, theta_max/np.pi)
ax_axes_unct.set_yscale("log")
ax_axes_unct.legend(loc='upper right')
ax_axes_unct.grid(True)
ax_axes_unct.minorticks_on()
ax_axes_unct.grid(which='minor', linestyle=':', linewidth=0.6)
plt.tight_layout()


np.savez_compressed('num_eval/num_eval_res_theta_S_sum_geo_ell.npz',
                    A0_set      = A0_theta_scan_geom_fits,
                    #
                    sigma_set   = sigma_theta_scan_geom_fits,
                    #
                    thetas_set  = thetas_theta_scan_geom_fits,
                    #
                    thetas_rec_ell  = thetas_ell_mean_theta_scan_geom_fits,
                    #
                    thetas_rec_ell_std   = thetas_ell_std_theta_scan_geom_fits
                    )