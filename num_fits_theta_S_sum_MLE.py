# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
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

def fit_MLE_for_parallel(params):
    A0, sigma, theta, n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset, bounds = params
    seed = i + j * n_thetas + seed_offset

    S_plus, S_minus, S_sum, S_diff = hf.generate_signals(
        A0, sigma, theta, n_phis, lambda_plus, lambda_minus, seed)

    info_params = f"theta = {theta:.17g}, A0 = {A0:.17g}, sigma = {sigma:.17g}, seed = {seed}"
    
    _, _, MLE_plus, _, _, _,   = hf.fit_routine_hist(
        S_plus, add_info="plus: "+info_params+" MLE", method="MLE")
    
    _, _, MLE_minus, _, _, _,   = hf.fit_routine_hist(
        S_minus, add_info="minus: "+info_params+" MLE", method="MLE")
    
    sigma_guess = hf.sigma_density(np.nanmean(
        [MLE_plus[1], MLE_minus[1]]), lambda_plus, lambda_minus)
    if np.isnan(sigma_guess):
        logger.warning(f'Fits of plus and minus failed for {info_params}')
        sigma_guess = hf.sigma_density(0.05, lambda_plus, lambda_minus)

    A0_mean = np.nanmean([MLE_plus[0], MLE_minus[0]])
    A_max = hf.amp_max_guess(A0_mean, lambda_plus, lambda_minus)
    
    _, _, MLE_sum, _, _, _, = hf.fit_routine_hist(
        S_sum, add_info="sum: "+info_params+" MLE",sigma_guess=sigma_guess, A_max=A_max, method="MLE")
    
    _, _, MLE_diff, _, _, _,  = hf.fit_routine_hist(
        S_diff, add_info="diff: "+info_params+" MLE",sigma_guess=sigma_guess, A_max=A_max, method="MLE")

    return (*MLE_plus, *MLE_minus,*MLE_sum,*MLE_diff)


##################################################
###################### main ######################
##################################################
if __name__ == "__main__":
    ##########################################
    ########## start of parameters ###########
    ##########################################
    lambda_mean = 1/np.sqrt(2)
    lambda_diff = 0
    lambda_plus, lambda_minus = hf.rel_lambdas_to_plain(
        lambda_mean, lambda_diff)

    A0      = 0.824
    sigma   = 0.063
    n_phis  =   300

    theta_min = 0.0*np.pi
    theta_max = 3.0*np.pi
    n_thetas = 1000
    thetas = np.linspace(theta_min, theta_max, n_thetas)

    theta_min = thetas[0]
    theta_max = thetas[-1]

    seed_offset = 0
    n_stoch_rep = 1000
    bounds = True

    save_data = True

    folder_run = "num_data"
    saving_name = "num_fits_res_theta_S_sum_MLE"

    max_kernels = 120
    #########################################
    ########### end of parameters ###########
    #########################################

    params_list = [(A0, sigma, thetas[i], n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset, bounds)
                   for i in range(n_thetas) for j in range(n_stoch_rep)]

    ### MLE parallelisation ###
    futures_dict_hist = {}
    with ProcessPoolExecutor(max_workers=max_kernels) as executor:
        for idx, params in enumerate(params_list):
            future = executor.submit(fit_MLE_for_parallel, params)
            futures_dict_hist[future] = idx
        results_MLE_raw = [None] * len(params_list)
        for future in tqdm(as_completed(futures_dict_hist), total=len(futures_dict_hist)):
            idx = futures_dict_hist[future]
            results_MLE_raw[idx] = future.result()
    ## histogram parallelisation results ##
    results_MLE = np.array(results_MLE_raw)
    results_MLE = results_MLE.reshape(n_thetas, n_stoch_rep, -1)

    ### histogram stuff ###
    ## amplitudes ##
    # first: mean of A_plus and A_minus as estimate for theta reconstructed
    A0_hist_fits = (
        results_MLE[:, :, 0] + results_MLE[:, :, 3]) / 2
    sigma_hist_fits = (
        results_MLE[:, :, 1] + results_MLE[:, :, 4]) / 2
    logger.info(f"A0 as mean of plus and minus: {np.mean(A0_hist_fits):.6f} +- {np.std(A0_hist_fits, ddof=1):.6f}")
    logger.info(f"sigma as mean of plus and minus: {np.mean(sigma_hist_fits):.6f} +- {np.std(sigma_hist_fits, ddof=1):.6f}")
    print(f"A0 as mean of plus and minus: {np.mean(A0_hist_fits):.6f} +- {np.std(A0_hist_fits, ddof=1):.6f}")
    print(f"sigma as mean of plus and minus: {np.mean(sigma_hist_fits):.6f} +- {np.std(sigma_hist_fits, ddof=1):.6f}")

    A_sum_hist_fits = results_MLE[:, :, 6]
    A_diff_hist_fits = results_MLE[:, :, 9]

    A_sum_hist_mean = np.nanmean(A_sum_hist_fits, axis=1)
    A_sum_hist_std = np.nanstd(A_sum_hist_fits, axis=1, ddof=1)
    A_sum_real = hf.rel_lambdas_to_amplitude(
        thetas, A0, *hf.plain_lambdas_to_rel(lambda_plus, lambda_minus))

    A_diff_hist_mean = np.nanmean(A_diff_hist_fits, axis=1)
    A_diff_hist_std = np.nanstd(A_diff_hist_fits, axis=1, ddof=1)
    A_diff_real = hf.rel_lambdas_to_amplitude(
        thetas, A0, *hf.plain_lambdas_to_rel(lambda_plus, -lambda_minus))

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

    ## create mask for branches ##
    mask_branch1 = thetas <= np.pi
    mask_branch2 = (np.pi < thetas) & (thetas <= 2*np.pi)
    mask_branch3 = 2*np.pi < thetas

    branch_1_hist_sum = theta_hist_sum_mean_raw
    branch_2_hist_sum = 2*np.pi - theta_hist_sum_mean_raw
    branch_3_hist_sum = 2*np.pi + theta_hist_sum_mean_raw

    branch_1_hist_diff = np.pi - theta_hist_diff_mean_raw
    branch_2_hist_diff = np.pi + theta_hist_diff_mean_raw
    branch_3_hist_diff = 3*np.pi - theta_hist_diff_mean_raw

    ## combine branches for correct phase unwrapping ##
    theta_hist_sum_mean, theta_hist_diff_mean = np.empty_like(thetas), np.empty_like(thetas)

    theta_hist_sum_mean[mask_branch1] = branch_1_hist_sum[mask_branch1]
    theta_hist_sum_mean[mask_branch2] = branch_2_hist_sum[mask_branch2]
    theta_hist_sum_mean[mask_branch3] = branch_3_hist_sum[mask_branch3]

    theta_hist_diff_mean[mask_branch1] = branch_1_hist_diff[mask_branch1]
    theta_hist_diff_mean[mask_branch2] = branch_2_hist_diff[mask_branch2]
    theta_hist_diff_mean[mask_branch3] = branch_3_hist_diff[mask_branch3]

    ### theta bias plot ####
    fig_theta_bias, ax_theta_bias = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_theta_bias.grid(True)
    ax_theta_bias.minorticks_on()
    ax_theta_bias.grid(which='minor', linestyle=':', linewidth=0.6)


    ## bias sum histogram ##
    ax_theta_bias.plot(thetas/np.pi, theta_hist_sum_mean-thetas, color=colour_sum,
                       linewidth=1, label=r'$\theta_{\text{bias, sum}}$')
    ax_theta_bias.fill_between(thetas/np.pi,
                               theta_hist_sum_mean-thetas - 1*theta_hist_sum_std,
                               theta_hist_sum_mean-thetas + 1*theta_hist_sum_std,
                               color=colour_sum, alpha=0.3)
    ## bias difference histogram ##
    ax_theta_bias.plot(thetas/np.pi, theta_hist_diff_mean-thetas, color=colour_diff,
                       linewidth=1, label=r'$\theta_{\text{bias, diff}}$')
    ax_theta_bias.fill_between(thetas/np.pi,
                               theta_hist_diff_mean-thetas - 1*theta_hist_diff_std,
                               theta_hist_diff_mean-thetas + 1*theta_hist_diff_std,
                               color=colour_diff, alpha=0.3)

    ax_theta_bias.set_xlabel(r'$\theta/\pi$ (rad)')
    ax_theta_bias.set_xlim(theta_min/np.pi, theta_max/np.pi)
    ax_theta_bias.legend(loc='lower right')
    plt.tight_layout()
    

    ### theta uncertainty plot ###
    fig_theta_unct, ax_theta_unct = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_theta_unct.plot(thetas/np.pi, theta_hist_sum_std,
                       color=colour_sum, label=r'$\Delta \theta_\text{sum}$')
    ax_theta_unct.plot(thetas/np.pi, theta_hist_diff_std,
                       color=colour_diff, label=r'$\Delta \theta_\text{diff}$')

    ax_theta_unct.set_xlabel(r'$\theta/\pi$ (rad)')
    ax_theta_unct.set_xlim(theta_min/np.pi, theta_max/np.pi)
    # ax_theta_unct.set_yscale("log")
    ax_theta_unct.legend(loc='upper right')
    ax_theta_unct.grid(True)
    ax_theta_unct.minorticks_on()
    ax_theta_unct.grid(which='minor', linestyle=':', linewidth=0.6)
    plt.tight_layout()


    ### amplitude bias plot only for histograms ####
    fig_amp_bias, ax_amp_bias = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_amp_bias.grid(True)
    ax_amp_bias.minorticks_on()
    ax_amp_bias.grid(which='minor', linestyle=':', linewidth=0.6)

    ## bias sum histogram ##
    ax_amp_bias.plot(thetas/np.pi, A_sum_hist_mean-A_sum_real, color=colour_sum,
                     linewidth=1, label=r'$A_{\text{bias, sum}}$')
    ax_amp_bias.fill_between(thetas/np.pi,
                             A_sum_hist_mean-A_sum_real - 1*A_sum_hist_std,
                             A_sum_hist_mean-A_sum_real + 1*A_sum_hist_std,
                             color=colour_sum, alpha=0.3)
    ## bias diff histogram ##
    ax_amp_bias.plot(thetas/np.pi, A_diff_hist_mean-A_diff_real, color=colour_diff,
                     linewidth=1, label=r'$A_{\text{bias, diff}}$')
    ax_amp_bias.fill_between(thetas/np.pi,
                             A_diff_hist_mean-A_diff_real - 1*A_diff_hist_std,
                             A_diff_hist_mean-A_diff_real + 1*A_diff_hist_std,
                             color=colour_diff, alpha=0.3)

    ax_amp_bias.set_xlabel(r'$\theta/\pi$ (rad)')
    ax_amp_bias.set_xlim(theta_min/np.pi, theta_max/np.pi)
    ax_amp_bias.legend(loc='lower right')
    plt.tight_layout()

    ### amplitude uncertainty only for histograms ###
    fig_amp_unct, ax_amp_unct = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_amp_unct.plot(thetas/np.pi, A_sum_hist_std,
                     color=colour_sum, label=r'$\Delta A_\text{sum}$')
    ax_amp_unct.plot(thetas/np.pi, A_diff_hist_std,
                     color=colour_diff, label=r'$\Delta A_\text{diff}$')

    ax_amp_unct.set_xlabel(r'$\theta/\pi$ (rad)')
    ax_amp_unct.set_xlim(theta_min/np.pi, theta_max/np.pi)
    # ax_amp_unct.set_yscale("log")
    ax_amp_unct.legend(loc='upper right')
    ax_amp_unct.grid(True)
    ax_amp_unct.minorticks_on()
    ax_amp_unct.grid(which='minor', linestyle=':', linewidth=0.6)
    plt.tight_layout()


    ######################
    ### saving results ###
    ######################

    if save_data:
        np.savez_compressed(f'{folder_run}/{saving_name}.npz',
                            lambda_mean         = np.array(lambda_mean),
                            lambda_diff         = np.array(lambda_diff),
                            lambda_plus         = np.array(lambda_plus),
                            lambda_minus        = np.array(lambda_minus),
                            n_thetas            = np.array(n_thetas),
                            thetas              = thetas,
                            n_stoch_rep         = np.array(n_stoch_rep),
                            seed_offset         = np.array(seed_offset),
                            n_phis              = np.array(n_phis),
                            A0                  = np.array(A0),
                            sigma               = np.array(sigma),
                            results_MLE         = results_MLE
                            )