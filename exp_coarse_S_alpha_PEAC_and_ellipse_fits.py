# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import datetime
import logging
import os

plt.style.use('paper_mpl_style.mplstyle')

colour_diff = 'C0'
colour_sum = 'C1'
colour_ell = 'C2'

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
filename = os.path.splitext(os.path.basename(__file__))[0]

### configure logging once in the main file ###
logging.basicConfig(
    filename=f'{filename}-{timestamp}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Starting {filename}.py.")

import helper_functions as hf

def fit_ellipse_for_parallel(params):
    n_Ts, n_phis, _, _, folder, i, j, seed_offset = params
    seed = i + j * n_Ts + seed_offset

    S_plus = np.load(f'exp_data/{folder}/signal_p1.npy')[:, i,:].ravel()
    S_minus = np.load(f'exp_data/{folder}/signal_m1.npy')[:, i,:].ravel()

    ### bootstrapping in such a manner that the original data is included ###
    if j > 0:
        rng = np.random.default_rng(seed)
        S_plus, S_minus = rng.choice(np.vstack((S_plus, S_minus)), n_phis, axis=1)

    info_params = f"i = {i}, j = {j}, seed_offset = {seed_offset}, folder = {folder}"

    return hf.fit_ellipse(S_plus, S_minus, add_info="ell: "+info_params)


def fit_histogram_for_parallel(params):
    n_Ts, n_phis, lambda_plus, lambda_minus, folder, i, j, seed_offset = params
    seed = i + j * n_Ts + seed_offset

    S_plus = np.load(f'exp_data/{folder}/signal_p1.npy')[:, i,:].ravel()
    S_minus = np.load(f'exp_data/{folder}/signal_m1.npy')[:, i,:].ravel()

    ### bootstrapping in such a manner that the original data is included ###
    if j > 0:
        rng = np.random.default_rng(seed)
        S_plus, S_minus = rng.choice(np.vstack((S_plus, S_minus)), n_phis, axis=1)

    S_sum = lambda_plus * S_plus + lambda_minus * S_minus
    S_diff = lambda_plus * S_plus - lambda_minus * S_minus

    info_params = f"lambda_plus = {lambda_plus:.3f}, lambda_minus = {lambda_minus:.3f}, i = {i}, j = {j}, seed_offset = {seed_offset}, folder = {folder}"

    bins_plus, hist_vals_plus, fit_plus, peak_pos_plus, peak_frac_pos_plus, initial_guess_plus, ssr_plus = hf.fit_routine_hist(
        S_plus, add_info="plus: "+info_params)

    bins_minus, hist_vals_minus, fit_minus, peak_pos_minus, peak_frac_pos_minus, initial_guess_minus, ssr_minus = hf.fit_routine_hist(
        S_minus, add_info="minus: "+info_params)

    sigma_guess = hf.sigma_density(np.nanmean(
        [fit_plus[1], fit_minus[1]]), lambda_plus, lambda_minus)
    if np.isnan(sigma_guess):
        logger.warning(f'Fits of plus and minus failed for {info_params}')
        sigma_guess = hf.sigma_density(0.05, lambda_plus, lambda_minus)

    ## check for resolution of sigma ##
    if sigma_guess < (bins_plus[1]-bins_plus[0]) or sigma_guess < (bins_minus[1]-bins_minus[0]):
        logger.info(
            f'sigma_guess was less than resolution limit: {sigma_guess:.17g} versus {bins_plus[1]-bins_plus[0]:.17g} or {bins_minus[1]-bins_minus[0]:.17g}.')
        sigma_guess = None

    A0_mean = np.nanmean([fit_plus[0], fit_minus[0]])
    A_max = hf.amp_max_guess(A0_mean, lambda_plus, lambda_minus)

    bins_sum, hist_vals_sum, fit_sum, peak_pos_sum, peak_frac_pos_sum, initial_guess_sum, ssr_sum = hf.fit_routine_hist(
        S_sum, add_info="sum: "+info_params, sigma_guess=sigma_guess, A_max=A_max)

    bins_diff, hist_vals_diff, fit_diff, peak_pos_diff, peak_frac_pos_diff, initial_guess_diff, ssr_diff = hf.fit_routine_hist(
        S_diff, add_info="diff: "+info_params, sigma_guess=sigma_guess, A_max=A_max)

    return (*fit_plus, *fit_minus, *fit_sum, *fit_diff,
            *initial_guess_plus, *initial_guess_minus, *
            initial_guess_sum, *initial_guess_diff,
            ssr_plus, ssr_minus, ssr_sum, ssr_diff)


##################################################
###################### main ######################
##################################################
if __name__ == "__main__":
    ##########################################
    ########## start of parameters ###########
    ##########################################
    lambda_mean = 1/np.sqrt(2) # can be chosen freely, however, 1/sqrt(2) is the value if one really wants to describe a rotation
    lambda_diff = 0
    lambda_plus, lambda_minus = hf.rel_lambdas_to_plain(
        lambda_mean, lambda_diff)

    k = 4*np.pi/780.226e-9
    
    folder = "Coarse"

    Ts = np.linspace(1e-3, 3e-3, 21)

    n_phis = len(np.load(f'exp_data/{folder}/signal_p1.npy')[:, 0,:].ravel())
    
    n_Ts = len(Ts)
    T_min = Ts[0]
    T_max = Ts[-1]

    n_stoch_rep = 1000
    seed_offset = 0

    save_data = True

    max_kernels = 110
    #########################################
    ########### end of parameters ###########
    #########################################

    params_list = [(n_Ts, n_phis, lambda_plus, lambda_minus, folder, i, j, seed_offset) for i in range(n_Ts) for j in range(n_stoch_rep)]

    ### ellipse parallelisation ###
    futures_dict_ell = {}
    with ProcessPoolExecutor(max_workers=max_kernels) as executor:
        for idx, params in enumerate(params_list):
            future = executor.submit(fit_ellipse_for_parallel, params)
            futures_dict_ell[future] = idx
        results_ellipse_raw = [None] * len(params_list)
        for future in tqdm(as_completed(futures_dict_ell), total=len(futures_dict_ell)):
            idx = futures_dict_ell[future]
            results_ellipse_raw[idx] = future.result()
    ## ellipse parallelisation results ##
    results_ellipse = np.array(results_ellipse_raw)
    results_ellipse = results_ellipse.reshape(n_Ts, n_stoch_rep, -1)

    ### histogram parallelisation ###
    futures_dict_hist = {}
    with ProcessPoolExecutor(max_workers=max_kernels) as executor:
        for idx, params in enumerate(params_list):
            future = executor.submit(fit_histogram_for_parallel, params)
            futures_dict_hist[future] = idx
        results_histogram_raw = [None] * len(params_list)
        for future in tqdm(as_completed(futures_dict_hist), total=len(futures_dict_hist)):
            idx = futures_dict_hist[future]
            results_histogram_raw[idx] = future.result()
    ## histogram parallelisation results ##
    results_histogram = np.array(results_histogram_raw)
    results_histogram = results_histogram.reshape(n_Ts, n_stoch_rep, -1)

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
    # first: mean of A_plus and A_minus as estimate for theta reconstructed
    A0_hist_fits = (
        results_histogram[:, :, 0] + results_histogram[:, :, 3]) / 2
    sigma_hist_fits = (
        results_histogram[:, :, 1] + results_histogram[:, :, 4]) / 2
    logger.info(f"A0 as mean of plus and minus: {np.mean(A0_hist_fits):.6f} +- {np.std(A0_hist_fits, ddof=1):.6f}")
    logger.info(f"sigma as mean of plus and minus: {np.mean(sigma_hist_fits):.6f} +- {np.std(sigma_hist_fits, ddof=1):.6f}")
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

    ### theta plot ####
    fig_theta_reconstructed, ax_theta_reconstructed = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_theta_reconstructed.grid(True)
    ax_theta_reconstructed.minorticks_on()
    ax_theta_reconstructed.grid(which='minor', linestyle=':', linewidth=0.6)

    ## phase ellipse ##
    ax_theta_reconstructed.plot(Ts/1e-3, theta_ell_mean_raw, color=colour_ell,
                       linewidth=0.5, marker="+", label=r'$\theta_{\text{ell}}$')
   
    ## phase sum histogram ##
    ax_theta_reconstructed.plot(Ts/1e-3, theta_hist_sum_mean_raw, color=colour_sum,
                       linewidth=0.5, marker="+", label=r'$\theta_{\text{sum}}$')
    
    ## phase difference histogram ##
    ax_theta_reconstructed.plot(Ts/1e-3, theta_hist_diff_mean_raw, color=colour_diff,
                       linewidth=0.5, marker="+", label=r'$\theta_{\text{diff}}$')
    
    ax_theta_reconstructed.axhline(0, color="black", linewidth=1, ls="--")
    ax_theta_reconstructed.axhline(np.pi, color="black", linewidth=1, ls="--")


    ax_theta_reconstructed.set_xlabel(r'$T$ (ms)')
    ax_theta_reconstructed.set_xlim(T_min/1e-3, T_max/1e-3)
    ax_theta_reconstructed.set_ylim(0-5e-2, np.pi+5e-2)
    ax_theta_reconstructed.legend(loc='lower right')
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
    plt.tight_layout()


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

        
    ######################
    ### saving results ###
    ######################

    if save_data:
        np.savez_compressed('exp_eval/exp_coarse_S_alpha_PEAC_and_ellipse_results.npz',
                            lambda_mean         = np.array(lambda_mean),
                            lambda_diff         = np.array(lambda_diff),
                            lambda_plus         = np.array(lambda_plus),
                            lambda_minus        = np.array(lambda_minus),
                            n_Ts                = np.array(n_Ts),
                            Ts                  = Ts,
                            n_stoch_rep         = np.array(n_stoch_rep),
                            results_ellipse     = results_ellipse,
                            results_histogram   = results_histogram
                            )