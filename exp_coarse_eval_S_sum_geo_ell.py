# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""

import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
import datetime

plt.style.use('paper_mpl_style.mplstyle')

colour_diff = 'C0'
colour_sum = 'C1'
colour_ell = 'C2'

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

to_import = "exp_coarse_fits_res_S_sum_geo_ell"

data_geom_fits = np.load(f'exp_eval/{to_import}.npz')

lambda_mean          = data_geom_fits['lambda_mean']
lambda_diff          = data_geom_fits['lambda_diff']
lambda_plus          = data_geom_fits['lambda_plus']
lambda_minus         = data_geom_fits['lambda_minus']

nTs                  = data_geom_fits['n_Ts']
Ts                   = data_geom_fits['Ts']
n_stoch_rep          = data_geom_fits['n_stoch_rep']
results_ellipse      = data_geom_fits['results_ellipse']




##########################################
########## start of parameters ###########
##########################################
k = 4*np.pi/780.226e-9

T_min = Ts[0]
T_max = Ts[-1]

phase_unwrapping = True
save_data = False
name_for_saving = "exp_coarse_eval_res_S_sum_geo_ell"
#########################################
########### end of parameters ###########
#########################################


### ellipse stuff ###
## ellipse axes ##
x0_ell_geom_fits    = results_ellipse[:,:,0]
y0_ell_geom_fits    = results_ellipse[:,:,1]
ap_ell_geom_fits    = results_ellipse[:,:,2]
bp_ell_geom_fits    = results_ellipse[:,:,3]
alpha_ell_geom_fits = results_ellipse[:,:,4]

major_axis_ell_mean_geom_fits = np.nanmean(ap_ell_geom_fits, axis=1)
major_axis_ell_std_geom_fits = np.nanstd(ap_ell_geom_fits, axis=1, ddof=1)

minor_axis_ell_mean_geom_fits = np.nanmean(bp_ell_geom_fits, axis=1)
minor_axis_ell_std_geom_fits = np.nanstd(bp_ell_geom_fits, axis=1, ddof=1)

## theta ellipse ##
thetas_ell_raw_geom_fits = hf.geom_ell_to_theta(alpha_ell_geom_fits, ap_ell_geom_fits, bp_ell_geom_fits)

thetas_ell_mean_raw = np.nanmean(thetas_ell_raw_geom_fits, axis=1)
thetas_ell_std = np.nanstd(thetas_ell_raw_geom_fits, axis=1, ddof=1)

################################
### end: raw theta reconstr. ###
################################

inch_to_cm = 2.54
phi_golden = (1 + np.sqrt(5)) / 2
width_inch = 15 / inch_to_cm
height_inch = width_inch / phi_golden
plt.rc('font', size=10)

###############################
### start: phase unwrapping ###
###############################
if phase_unwrapping:
    ### phase unwrapping ###
    # Arccos is implemented in numpy in such a way that only values between 0 and Pi are returned:
    # Because cos is an even function, every value returned by arccos also
    # has a negative counterpart. And given the Pi periodicity in our case, any integer multiples
    # of Pi can be added.
    # For a value x of arccos, +-x + l*Pi with l an integer is therefore also a possible solution.
    # By matching and considering the unmodified values of arrcos, its branches can be reconstructed
    # with correct phase unwrapping.

    ## create mask for branches ##
    mask_branch1 = Ts <= 1.70e-3
    mask_branch2 = (1.70e-3 < Ts) & (Ts < 2.5e-3)
    mask_branch3 = 2.5e-3 <= Ts

    ## phase unwrap per branch ##
    branch_1_ell = thetas_ell_raw_geom_fits
    branch_2_ell = 2*np.pi - thetas_ell_raw_geom_fits
    branch_3_ell = 2*np.pi + thetas_ell_raw_geom_fits


    ## combine branches for correct phase unwrapping ##
    thetas_ell_mean_raw_unwrapped = np.empty_like(thetas_ell_raw_geom_fits)

    thetas_ell_mean_raw_unwrapped[mask_branch1] = branch_1_ell[mask_branch1]
    thetas_ell_mean_raw_unwrapped[mask_branch2] = branch_2_ell[mask_branch2]
    thetas_ell_mean_raw_unwrapped[mask_branch3] = branch_3_ell[mask_branch3]

else:
    thetas_ell_mean_raw_mean = thetas_ell_mean_raw

if not phase_unwrapping:
    ### theta plot ####
    fig_theta_reconstructed, ax_theta_reconstructed = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_theta_reconstructed.grid(True)
    ax_theta_reconstructed.minorticks_on()
    ax_theta_reconstructed.grid(which='minor', linestyle=':', linewidth=0.6)

    ## phase ellipse ##
    ax_theta_reconstructed.plot(Ts/1e-3, thetas_ell_mean_raw_mean, color=colour_ell,
                        linewidth=0.5, marker="+", label=r'$\theta_{\text{ell}}$')
    
    ax_theta_reconstructed.axhline(0, color="black", linewidth=1, ls="--")
    ax_theta_reconstructed.axhline(np.pi, color="black", linewidth=1, ls="--")

    ax_theta_reconstructed.set_xlabel(r'$T$ (ms)')
    ax_theta_reconstructed.set_xlim(T_min/1e-3, T_max/1e-3)
    ax_theta_reconstructed.legend(loc='lower right')

if phase_unwrapping:
    ########################################
    ### fitting routine for acceleration ###
    ########################################

    # store for every reptition resulting acceleration,
    # such that afterwards mean and std can be calculated
    as_ell = np.empty_like(thetas_ell_raw_geom_fits[0,:])

    for i in range(len(thetas_ell_raw_geom_fits[0, :])):

        popt_ell, _, infodict_ell, _, _ = hf.curve_fit(
            hf.parabola_with_linear,
            Ts, thetas_ell_mean_raw_unwrapped[:,i], p0=[30e-3],
            full_output=True,
            maxfev=2000)

        as_ell[i] = popt_ell[0]

    a_ell_mean = np.mean(as_ell)

    delta_a_ell_mean = np.std(as_ell, ddof=1)


    Ts_fine = np.linspace(T_min, T_max, 100)
    thetas_ell_mean_raw_calib_fine = hf.parabola_with_linear(Ts_fine, a_ell_mean)


    # we choose S_sum as reference for all conversion purposes
    exp_coarse_eval  = np.load(f'exp_eval/exp_coarse_eval_res_S_sum_PEAC.npz')
    a_calib          = exp_coarse_eval['a_calib']
    a_calib_unct     = exp_coarse_eval['a_calib_unct']

    theta_all_calib_fine = hf.parabola_with_linear(Ts_fine, a_calib)

    # overwrite all to same calibration
    thetas_ell_mean_raw_calib_fine = theta_all_calib_fine


    thetas_ell_mean_raw_mean = np.mean(thetas_ell_mean_raw_unwrapped, axis=1)

    ### theta with fit plot ####
    fig_theta_acceleration, ax_theta_acceleration = plt.subplots(
        3, 1, figsize=(width_inch, height_inch * 3)
    )

    ## phase ellipse ##
    ax_theta_acceleration[0].grid(True)
    ax_theta_acceleration[0].minorticks_on()
    ax_theta_acceleration[0].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_acceleration[0].plot(Ts/1e-3, thetas_ell_mean_raw_mean, color=colour_ell, linewidth=0.5, marker="+", label=r'$\theta_{\text{ell}}$')
    ax_theta_acceleration[0].plot(Ts_fine/1e-3, thetas_ell_mean_raw_calib_fine, color=colour_ell, linewidth=1, label=rf'$\theta_{{\text{{ell, fit}}}}$: {a_ell_mean*1e3:.3f} +- {delta_a_ell_mean*1e3:.3f} mm/s^2')
    ax_theta_acceleration[0].set_xlabel(r'$T$ (ms)')
    ax_theta_acceleration[0].set_xlim(T_min/1e-3, T_max/1e-3)
    ax_theta_acceleration[0].legend(loc='lower right')

    ##################
    ### theta bias ###
    ##################
    thetas_calibrated_ell = hf.parabola_with_linear(Ts, a_ell_mean)

    # overwrite all to same calibration
    thetas_calibrated_all = hf.parabola_with_linear(Ts, a_calib)
    
    thetas_calibrated_ell = thetas_calibrated_all


    theta_bias_ell  = thetas_ell_mean_raw_mean - thetas_calibrated_ell

    if save_data:
        np.savez_compressed(f'exp_eval/{name_for_saving}.npz',
                            theta_calibrated_ell    = thetas_calibrated_ell,
                            theta_bias_ell          = theta_bias_ell,
                            theta_ell_std           = thetas_ell_std,
                            a_calib                 = a_calib,
                            a_calib_unct            = a_calib_unct
                            )

    ### theta bias plot ####
    fig_theta_bias, ax_theta_bias = plt.subplots(
        3, 1, figsize=(width_inch, height_inch * 3)
    )


    # series values
    x_ell = thetas_calibrated_ell/np.pi

    ## bias ellipse ##
    ax_theta_bias[0].grid(True)
    ax_theta_bias[0].minorticks_on()
    ax_theta_bias[0].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_bias[0].axhline(0, color="black", linewidth=1, ls="--")
    hf.plot_line_with_wide_err(ax_theta_bias[0], x_ell, theta_bias_ell, 0, thetas_ell_std, colour_ell, r'$\theta_{\text{bias, ell}}$')
    ax_theta_bias[0].set_xlabel(r'$\theta/\pi$')
    ax_theta_bias[0].set_xlim(x_ell[0], x_ell[-1])
    ax_theta_bias[0].legend(loc='lower right')

    plt.tight_layout()

### theta uncertainty plot ###
fig_theta_unct, ax_theta_unct = plt.subplots(
    figsize=(width_inch, height_inch))
ax_theta_unct.plot(Ts/1e-3, thetas_ell_std,
                    color=colour_ell, label=r'$\Delta \theta_\text{ell}$')

ax_theta_unct.set_xlabel(r'$T$ (ms)')
ax_theta_unct.set_xlim(T_min/1e-3, T_max/1e-3)
# ax_theta_unct.set_yscale("log")
ax_theta_unct.legend(loc='upper right')
ax_theta_unct.grid(True)
ax_theta_unct.minorticks_on()
ax_theta_unct.grid(which='minor', linestyle=':', linewidth=0.6)
plt.tight_layout()


### half-axes uncertainty only for ellipse ###
fig_axes_unct, ax_axes_unct = plt.subplots(
    figsize=(width_inch, height_inch))
ax_axes_unct.plot(Ts/1e-3, np.std(ap_ell_geom_fits, axis=1),
                    color='tab:red', label='uncertainty major axis')
ax_axes_unct.plot(Ts/1e-3, np.std(bp_ell_geom_fits, axis=1),
                    color='tab:purple', label='uncertainty minor axis')

ax_axes_unct.set_xlabel(r'$T$ (ms)')
ax_axes_unct.set_xlim(T_min/1e-3, T_max/1e-3)
ax_axes_unct.set_yscale("log")
ax_axes_unct.legend(loc='upper right')
ax_axes_unct.grid(True)
ax_axes_unct.minorticks_on()
ax_axes_unct.grid(which='minor', linestyle=':', linewidth=0.6)
plt.tight_layout()
