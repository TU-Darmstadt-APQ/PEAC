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

to_import = "exp_coarse_S_alpha_PEAC_and_ellipse_results"

data_full = np.load(f'exp_eval/{to_import}.npz')

lambda_mean         = data_full['lambda_mean']
lambda_diff         = data_full['lambda_diff']
lambda_plus         = data_full['lambda_plus']
lambda_minus        = data_full['lambda_minus']
n_Ts                = data_full['n_Ts']
Ts                  = data_full['Ts']
n_stoch_rep         = data_full['n_stoch_rep']
results_ellipse     = data_full['results_ellipse']
results_histogram   = data_full['results_histogram']


##########################################
########## start of parameters ###########
##########################################
k = 4*np.pi/780.226e-9

n_Ts = len(Ts)
T_min = Ts[0]
T_max = Ts[-1]

phase_unwrapping = True
save_data = False
name_for_saving = "exp_coarse_eval_results"
#########################################
########### end of parameters ###########
#########################################


### ellipse stuff ###
## ellipse axes ##
x0_ell, y0_ell, ap_ell, bp_ell, phi_ell = hf.parametric_to_polar_vectorised(
    results_ellipse)

major_axis_ell_mean = np.nanmean(ap_ell, axis=1)
major_axis_ell_std = np.nanstd(ap_ell, axis=1, ddof=1)

minor_axis_ell_mean = np.nanmean(bp_ell, axis=1)
minor_axis_ell_std = np.nanstd(bp_ell, axis=1, ddof=1)

## theta ellipse ##
theta_ell = hf.conic_section_to_theta(
    results_ellipse[:, :, 0], results_ellipse[:, :, 1], results_ellipse[:, :, 2])

theta_ell_mean_raw = np.nanmean(theta_ell, axis=1)
theta_ell_std = np.nanstd(theta_ell, axis=1, ddof=1)

### histogram stuff ###
## amplitudes ##
# first: mean of A_sum and A_diff as estimate for theta reconstructed
A0_hist_fits = (
    results_histogram[:, :, 0] + results_histogram[:, :, 3]) / 2
sigma_hist_fits = (
    results_histogram[:, :, 1] + results_histogram[:, :, 4]) / 2
print(f"A0 as mean of plus and minus: {np.mean(A0_hist_fits):.6f} +- {np.std(A0_hist_fits, ddof=1):.6f}")
print(f"sigma as mean of plus and minus: {np.mean(sigma_hist_fits):.6f} +- {np.std(sigma_hist_fits, ddof=1):.6f}")

A_sum_hist_fits = results_histogram[:, :, 6]
A_diff_hist_fits = results_histogram[:, :, 9]

A_sum_hist_mean = np.nanmean(A_sum_hist_fits, axis=1)
A_sum_hist_std = np.nanstd(A_sum_hist_fits, axis=1, ddof=1)

A_diff_hist_mean = np.nanmean(A_diff_hist_fits, axis=1)
A_diff_hist_std = np.nanstd(A_diff_hist_fits, axis=1, ddof=1)

## theta histogram ##
theta_hist_sum = hf.amplitude_to_theta(
    A_sum_hist_fits, A0_hist_fits, lambda_mean, lambda_diff)
theta_hist_diff = hf.amplitude_to_theta(
    A_diff_hist_fits, A0_hist_fits, lambda_mean, lambda_diff)

theta_hist_sum_mean_raw = np.nanmean(theta_hist_sum, axis=1)
theta_hist_sum_std = np.nanstd(theta_hist_sum, axis=1, ddof=1)

theta_hist_diff_mean_raw = np.nanmean(theta_hist_diff, axis=1)
theta_hist_diff_std = np.nanstd(theta_hist_diff, axis=1, ddof=1)

#############
### plots ###
#############
inch_to_cm = 2.54
phi_golden = (1 + np.sqrt(5)) / 2
width_inch = 15 / inch_to_cm
height_inch = width_inch / phi_golden
plt.rc('font', size=10)

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
    branch_1_ell = theta_ell
    branch_2_ell = 2*np.pi - theta_ell
    branch_3_ell = 2*np.pi + theta_ell

    branch_1_hist_sum = theta_hist_sum
    branch_2_hist_sum = 2*np.pi - theta_hist_sum
    branch_3_hist_sum = 2*np.pi + theta_hist_sum

    branch_1_hist_diff = np.pi - theta_hist_diff
    branch_2_hist_diff = np.pi + theta_hist_diff
    branch_3_hist_diff = 3*np.pi - theta_hist_diff

    ## combine branches for correct phase unwrapping ##
    theta_ell_unwrapped, theta_hist_sum_unwrapped, theta_hist_diff_unwrapped = np.empty_like(
        theta_ell), np.empty_like(theta_hist_sum), np.empty_like(theta_hist_diff)

    theta_ell_unwrapped[mask_branch1] = branch_1_ell[mask_branch1]
    theta_ell_unwrapped[mask_branch2] = branch_2_ell[mask_branch2]
    theta_ell_unwrapped[mask_branch3] = branch_3_ell[mask_branch3]

    theta_hist_sum_unwrapped[mask_branch1] = branch_1_hist_sum[mask_branch1]
    theta_hist_sum_unwrapped[mask_branch2] = branch_2_hist_sum[mask_branch2]
    theta_hist_sum_unwrapped[mask_branch3] = branch_3_hist_sum[mask_branch3]

    theta_hist_diff_unwrapped[mask_branch1] = branch_1_hist_diff[mask_branch1]
    theta_hist_diff_unwrapped[mask_branch2] = branch_2_hist_diff[mask_branch2]
    theta_hist_diff_unwrapped[mask_branch3] = branch_3_hist_diff[mask_branch3]

else:
    theta_ell_mean = theta_ell_mean_raw
    theta_hist_sum_mean = theta_hist_sum_mean_raw
    theta_hist_diff_mean =theta_hist_diff_mean_raw

if not phase_unwrapping:
    ### theta plot ####
    fig_theta_reconstructed, ax_theta_reconstructed = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_theta_reconstructed.grid(True)
    ax_theta_reconstructed.minorticks_on()
    ax_theta_reconstructed.grid(which='minor', linestyle=':', linewidth=0.6)

    ## phase ellipse ##
    ax_theta_reconstructed.plot(Ts/1e-3, theta_ell_mean, color=colour_ell,
                        linewidth=0.5, marker="+", label=r'$\theta_{\text{ell}}$')

    ## phase sum histogram ##
    ax_theta_reconstructed.plot(Ts/1e-3, theta_hist_sum_mean, color=colour_sum,
                        linewidth=0.5, marker="+", label=r'$\theta_{\text{sum}}$')

    ## phase difference histogram ##
    ax_theta_reconstructed.plot(Ts/1e-3, theta_hist_diff_mean, color=colour_diff,
                        linewidth=0.5, marker="+", label=r'$\theta_{\text{diff}}$')
    
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
    as_ell, as_hist_sum, as_hist_diff = np.empty_like(theta_ell[0,:]), np.empty_like(theta_hist_sum[0,:]), np.empty_like(theta_hist_diff[0,:])

    for i in range(len(theta_hist_sum[0, :])):

        popt_ell, _, infodict_ell, _, _ = hf.curve_fit(
            hf.parabola_with_linear,
            Ts, theta_ell_unwrapped[:,i], p0=[30e-3],
            full_output=True,
            maxfev=2000)

        popt_hist_sum, _, infodict_hist_sum, _, _ = hf.curve_fit(
            hf.parabola_with_linear,
            Ts, theta_hist_sum_unwrapped[:,i], p0=[30e-3],
            full_output=True,
            maxfev=2000)

        popt_hist_diff, _, infodict_hist_diff, _, _ = hf.curve_fit(
            hf.parabola_with_linear,
            Ts, theta_hist_diff_unwrapped[:,i], p0=[30e-3],
            full_output=True,
            maxfev=2000)
        
        as_ell[i] = popt_ell[0]
        as_hist_sum[i] = popt_hist_sum[0]
        as_hist_diff[i] = popt_hist_diff[0]

    a_ell_mean = np.mean(as_ell)
    a_hist_sum_mean = np.mean(as_hist_sum)
    a_hist_diff_mean = np.mean(as_hist_diff)

    delta_a_ell_mean = np.std(as_ell, ddof=1)
    delta_a_hist_sum_mean = np.std(as_hist_sum, ddof=1)
    delta_a_hist_diff_mean = np.std(as_hist_diff, ddof=1)

    Ts_fine = np.linspace(T_min, T_max, 100)
    theta_ell_calib_fine = hf.parabola_with_linear(Ts_fine, a_ell_mean)
    theta_hist_sum_calib_fine = hf.parabola_with_linear(Ts_fine, a_hist_sum_mean)
    theta_hist_diff_calib_fine = hf.parabola_with_linear(Ts_fine, a_hist_diff_mean)

    # we choose S_sum as reference for all conversion purposes
    a_calib = a_hist_sum_mean
    theta_all_calib_fine = hf.parabola_with_linear(Ts_fine, a_calib)

    # overwrite all to same calibration
    theta_ell_calib_fine = theta_all_calib_fine
    theta_hist_sum_calib_fine = theta_all_calib_fine
    theta_hist_diff_calib_fine = theta_all_calib_fine

    theta_ell_mean = np.mean(theta_ell_unwrapped, axis=1)
    theta_hist_sum_mean = np.mean(theta_hist_sum_unwrapped, axis=1)
    theta_hist_diff_mean = np.mean(theta_hist_diff_unwrapped, axis=1)

    ### theta with fit plot ####
    fig_theta_acceleration, ax_theta_acceleration = plt.subplots(
        3, 1, figsize=(width_inch, height_inch * 3)
    )

    ## phase ellipse ##
    ax_theta_acceleration[0].grid(True)
    ax_theta_acceleration[0].minorticks_on()
    ax_theta_acceleration[0].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_acceleration[0].plot(Ts/1e-3, theta_ell_mean, color=colour_ell, linewidth=0.5, marker="+", label=r'$\theta_{\text{ell}}$')
    ax_theta_acceleration[0].plot(Ts_fine/1e-3, theta_ell_calib_fine, color=colour_ell, linewidth=1, label=rf'$\theta_{{\text{{ell, fit}}}}$: {a_ell_mean*1e3:.3f} +- {delta_a_ell_mean*1e3:.3f} mm/s^2')
    ax_theta_acceleration[0].set_xlabel(r'$T$ (ms)')
    ax_theta_acceleration[0].set_xlim(T_min/1e-3, T_max/1e-3)
    ax_theta_acceleration[0].legend(loc='lower right')

    ## phase sum histogram ##
    ax_theta_acceleration[1].grid(True)
    ax_theta_acceleration[1].minorticks_on()
    ax_theta_acceleration[1].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_acceleration[1].plot(Ts/1e-3, theta_hist_sum_mean, color=colour_sum, linewidth=0.5, marker="+", label=r'$\theta_{\text{sum}}$')
    ax_theta_acceleration[1].plot(Ts_fine/1e-3, theta_hist_sum_calib_fine, color=colour_sum, linewidth=1, label=rf'$\theta_{{\text{{sum, fit}}}}$: {a_hist_sum_mean*1e3:.3f} +- {delta_a_hist_sum_mean*1e3:.3f} mm/s^2')
    ax_theta_acceleration[1].set_xlabel(r'$T$ (ms)')
    ax_theta_acceleration[1].set_xlim(T_min/1e-3, T_max/1e-3)
    ax_theta_acceleration[1].legend(loc='lower right')

    ## phase difference histogram ##
    ax_theta_acceleration[2].grid(True)
    ax_theta_acceleration[2].minorticks_on()
    ax_theta_acceleration[2].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_acceleration[2].plot(Ts/1e-3, theta_hist_diff_mean, color=colour_diff, linewidth=0.5, marker="+", label=r'$\theta_{\text{diff}}$')
    ax_theta_acceleration[2].plot(Ts_fine/1e-3, theta_hist_diff_calib_fine, color=colour_diff, linewidth=1, label=rf'$\theta_{{\text{{diff, fit}}}}$: {a_hist_diff_mean*1e3:.3f} +- {delta_a_hist_diff_mean*1e3:.3f} mm/s^2')
    ax_theta_acceleration[2].set_xlabel(r'$T$ (ms)')
    ax_theta_acceleration[2].set_xlim(T_min/1e-3, T_max/1e-3)
    ax_theta_acceleration[2].legend(loc='lower right')

    ##################
    ### theta bias ###
    ##################
    thetas_calibrated_ell = hf.parabola_with_linear(Ts, a_ell_mean)
    thetas_calibrated_hist_sum = hf.parabola_with_linear(Ts, a_hist_sum_mean)
    thetas_calibrated_hist_diff = hf.parabola_with_linear(Ts, a_hist_diff_mean)

    # overwrite all to same calibration
    thetas_calibrated_all = hf.parabola_with_linear(Ts, a_calib)
    
    thetas_calibrated_ell = thetas_calibrated_all
    thetas_calibrated_hist_sum = thetas_calibrated_all
    thetas_calibrated_hist_diff = thetas_calibrated_all


    theta_bias_ell  = theta_ell_mean - thetas_calibrated_ell
    theta_bias_hist_sum  = theta_hist_sum_mean - thetas_calibrated_hist_sum
    theta_bias_hist_diff = theta_hist_diff_mean - thetas_calibrated_hist_diff

    if save_data:
        np.savez_compressed(f'exp_eval/{name_for_saving}.npz',
                            thetas_calibrated       = thetas_calibrated_hist_sum,
                            theta_bias_ell          = theta_bias_ell,
                            theta_bias_hist_sum     = theta_bias_hist_sum,
                            theta_bias_hist_diff    = theta_bias_hist_diff,
                            theta_ell_std           = theta_ell_std,
                            theta_hist_sum_std      = theta_hist_sum_std,
                            theta_hist_diff_std     = theta_hist_diff_std,
                            A_sum_hist_mean         = A_sum_hist_mean,
                            A_sum_hist_std          = A_sum_hist_std,
                            A_diff_hist_mean        = A_diff_hist_mean,
                            A_diff_hist_std         = A_diff_hist_std,
                            a_calib                 = a_calib,
                            a_calib_unct            = delta_a_hist_sum_mean
                            )

    ### theta bias plot ####
    fig_theta_bias, ax_theta_bias = plt.subplots(
        3, 1, figsize=(width_inch, height_inch * 3)
    )


    # series values
    x_ell = thetas_calibrated_ell/np.pi
    x_hist_sum = thetas_calibrated_hist_sum/np.pi
    x_hist_diff = thetas_calibrated_hist_diff/np.pi

    ## bias ellipse ##
    ax_theta_bias[0].grid(True)
    ax_theta_bias[0].minorticks_on()
    ax_theta_bias[0].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_bias[0].axhline(0, color="black", linewidth=1, ls="--")
    hf.plot_line_with_wide_err(ax_theta_bias[0], x_ell, theta_bias_ell, 0, theta_ell_std, colour_ell, r'$\theta_{\text{bias, ell}}$')
    ax_theta_bias[0].set_xlabel(r'$\theta/\pi$')
    ax_theta_bias[0].set_xlim(x_ell[0], x_ell[-1])
    ax_theta_bias[0].legend(loc='lower right')

    ## bias sum histogram ##
    ax_theta_bias[1].grid(True)
    ax_theta_bias[1].minorticks_on()
    ax_theta_bias[1].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_bias[1].axhline(0, color="black", linewidth=1, ls="--")
    hf.plot_line_with_wide_err(ax_theta_bias[1], x_hist_sum, theta_bias_hist_sum, 0, theta_hist_sum_std, colour_sum, r'$\theta_{\text{bias, sum}}$')
    ax_theta_bias[1].set_xlabel(r'$\theta/\pi$')
    ax_theta_bias[1].set_xlim(x_hist_sum[0], x_hist_sum[-1])
    ax_theta_bias[1].legend(loc='lower right')

    ## bias sum histogram ##
    ax_theta_bias[2].grid(True)
    ax_theta_bias[2].minorticks_on()
    ax_theta_bias[2].grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_bias[2].axhline(0, color="black", linewidth=1, ls="--")
    hf.plot_line_with_wide_err(ax_theta_bias[2], x_hist_diff, theta_bias_hist_diff, 0, theta_hist_diff_std, colour_diff, r'$\theta_{\text{bias, diff}}$')
    ax_theta_bias[2].set_xlabel(r'$\theta/\pi$')
    ax_theta_bias[2].set_xlim(x_hist_diff[0], x_hist_diff[-1])
    ax_theta_bias[2].legend(loc='lower right')
    
    plt.tight_layout()

### theta uncertainty plot ###
fig_theta_unct, ax_theta_unct = plt.subplots(
    figsize=(width_inch, height_inch))
ax_theta_unct.plot(Ts/1e-3, theta_ell_std,
                    color=colour_ell, label=r'$\Delta \theta_\text{ell}$')
ax_theta_unct.plot(Ts/1e-3, theta_hist_sum_std,
                    color=colour_sum, label=r'$\Delta \theta_\text{sum}$')
ax_theta_unct.plot(Ts/1e-3, theta_hist_diff_std,
                    color=colour_diff, label=r'$\Delta \theta_\text{diff}$')

ax_theta_unct.set_xlabel(r'$T$ (ms)')
ax_theta_unct.set_xlim(T_min/1e-3, T_max/1e-3)
# ax_theta_unct.set_yscale("log")
ax_theta_unct.legend(loc='upper right')
ax_theta_unct.grid(True)
ax_theta_unct.minorticks_on()
ax_theta_unct.grid(which='minor', linestyle=':', linewidth=0.6)
plt.tight_layout()


### amplitude uncertainty only for histograms ###
fig_amp_unct, ax_amp_unct = plt.subplots(
    figsize=(width_inch, height_inch))
ax_amp_unct.plot(Ts/1e-3, A_sum_hist_std,
                    color=colour_sum, label=r'$\Delta A_\text{sum}$')
ax_amp_unct.plot(Ts/1e-3, A_diff_hist_std,
                    color=colour_diff, label=r'$\Delta A_\text{diff}$')

ax_amp_unct.set_xlabel(r'$T$ (ms)')
ax_amp_unct.set_xlim(T_min/1e-3, T_max/1e-3)
ax_amp_unct.set_yscale("log")
ax_amp_unct.legend(loc='upper right')
ax_amp_unct.grid(True)
ax_amp_unct.minorticks_on()
ax_amp_unct.grid(which='minor', linestyle=':', linewidth=0.6)


### half-axes uncertainty only for ellipse ###
fig_axes_unct, ax_axes_unct = plt.subplots(
    figsize=(width_inch, height_inch))
ax_axes_unct.plot(Ts/1e-3, major_axis_ell_std,
                    color='tab:red', label='uncertainty major axis')
ax_axes_unct.plot(Ts/1e-3, minor_axis_ell_std,
                    color='tab:purple', label='uncertainty minor axis')

ax_axes_unct.set_xlabel(r'$T$ (ms)')
ax_axes_unct.set_xlim(T_min/1e-3, T_max/1e-3)
ax_axes_unct.set_yscale("log")
ax_axes_unct.legend(loc='upper right')
ax_axes_unct.grid(True)
ax_axes_unct.minorticks_on()
ax_axes_unct.grid(which='minor', linestyle=':', linewidth=0.6)
plt.tight_layout()
