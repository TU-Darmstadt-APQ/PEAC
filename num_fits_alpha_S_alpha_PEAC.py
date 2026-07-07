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

colour_beta = 'C0'
colour_alpha = 'C1'
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


def fit_histogram_for_parallel(params):
    A0, sigma, theta, n_thetas, n_phis, lambda_plus_alpha, lambda_minus_alpha, i, j, seed_offset, bounds = params
    seed = i + j * n_thetas + seed_offset

    S_plus, S_minus, S_alpha, _ = hf.generate_signals_alpha(
    A0, sigma, theta, n_phis, lambda_plus_alpha, lambda_minus_alpha, seed)

    info_params = f"theta = {theta:.17g}, A0 = {A0:.17g}, sigma = {sigma:.17g}, seed = {seed}"

    bins_plus, hist_vals_plus, fit_plus, peak_pos_plus, peak_frac_pos_plus, initial_guess_plus, ssr_plus = hf.fit_routine_hist(
        S_plus, add_info="plus: "+info_params, bounds=bounds)

    bins_minus, hist_vals_minus, fit_minus, peak_pos_minus, peak_frac_pos_minus, initial_guess_minus, ssr_minus = hf.fit_routine_hist(
        S_minus, add_info="minus: "+info_params, bounds=bounds)

    sigma_guess = hf.sigma_density(np.nanmean(
        [fit_plus[1], fit_minus[1]]), lambda_plus_alpha, lambda_minus_alpha)
    if np.isnan(sigma_guess):
        logger.warning(f'Fits of plus and minus failed for {info_params}')
        sigma_guess = hf.sigma_density(sigma, lambda_plus_alpha, lambda_minus_alpha)

    ## check for resolution of sigma ##
    if sigma_guess < (bins_plus[1]-bins_plus[0]) or sigma_guess < (bins_minus[1]-bins_minus[0]):
        logger.info(
            f'sigma_guess was less than resolution limit: {sigma_guess:.17g} versus {bins_plus[1]-bins_plus[0]:.17g} or {bins_minus[1]-bins_minus[0]:.17g}.')
        sigma_guess = None

    A0_mean = np.nanmean([fit_plus[0], fit_minus[0]])
    A_max_alpha = hf.amp_max_guess(A0_mean, lambda_plus_alpha, lambda_minus_alpha)
    logger.info(f"A_max_alpha = {A_max_alpha:.17g}")

    _, _, fit_alpha, _, _, initial_guess_alpha, ssr_alpha = hf.fit_routine_hist(
        S_alpha, add_info="alpha: "+info_params, sigma_guess=sigma_guess, A_max=A_max_alpha, bounds=bounds)

    # _, _, fit_beta, _, _, initial_guess_beta, ssr_beta = hf.fit_routine_hist(
    #     S_beta, add_info="beta: "+info_params, sigma_guess=sigma_guess, A_max=A_max_alpha, bounds=bounds)

    # return (*fit_plus, *fit_minus, *fit_alpha, *fit_beta,
    #         *initial_guess_plus, *initial_guess_minus, *
    #         initial_guess_alpha, *initial_guess_beta,
    #         ssr_plus, ssr_minus, ssr_alpha, ssr_beta)
    return (*fit_plus,              *fit_minus,             *fit_alpha,
            *initial_guess_plus,    *initial_guess_minus,   *initial_guess_alpha,
            ssr_plus,               ssr_minus,              ssr_alpha)

##################################################
###################### main ######################
##################################################
if __name__ == "__main__":
    ##########################################
    ########## start of parameters ###########
    ##########################################

    # alpha = np.pi/4 is what PEAC uses
    # alpha   = np.pi/8

    # lambda_plus_alpha  = np.sin(alpha)
    # lambda_minus_alpha = np.cos(alpha)

    # lambda_mean_alpha, lambda_diff_alpha = hf.plain_lambdas_to_rel(lambda_plus_alpha, lambda_minus_alpha)

    alphas = np.linspace(0.249, 0.250, 1)*np.pi

    n_alphas = len(alphas)

    lambdas_plus_alpha  = np.sin(alphas)
    lambdas_minus_alpha = np.cos(alphas)

    lambdas_mean_alpha, lambdas_diff_alpha = hf.plain_lambdas_to_rel(lambdas_plus_alpha, lambdas_minus_alpha)

    A0      = 0.824
    sigma   = 0.063
    n_phis  =   300

    theta_min = 0.5*np.pi
    theta_max = 1.0*np.pi
    n_thetas = 501
    thetas = np.linspace(theta_min, theta_max, n_thetas)

    theta_min = thetas[0]
    theta_max = thetas[-1]

    seed_offset = 0
    n_stoch_rep = 1000
    bounds = True

    save_data = True

    folder_run = "num_data"
    saving_name = "num_fits_res_alpha_S_alpha_slice_PEAC"

    max_kernels = 100
    #########################################
    ########### end of parameters ###########
    #########################################

    params_list = [(A0, sigma, thetas[j], n_thetas, n_phis, lambdas_plus_alpha[i], lambdas_minus_alpha[i], j, k, seed_offset, bounds)
                   for i in range(n_alphas) for j in range(n_thetas) for k in range(n_stoch_rep)]

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
    results_histogram = results_histogram.reshape(n_alphas, n_thetas, n_stoch_rep, -1)

    #################
    ### save data ###
    #################
    if save_data:
        np.savez_compressed(f'{folder_run}/{saving_name}.npz',
                            alphas                    = alphas,
                            lambdas_mean_alpha         = lambdas_mean_alpha,
                            lambdas_diff_alpha         = lambdas_diff_alpha,
                            lambdas_plus_alpha         = lambdas_plus_alpha,
                            lambdas_minus_alpha        = lambdas_minus_alpha,
                            n_thetas            = np.array(n_thetas),
                            thetas              = thetas,
                            n_stoch_rep         = np.array(n_stoch_rep),
                            seed_offset         = np.array(seed_offset),
                            n_phis              = np.array(n_phis),
                            A0                  = np.array(A0),
                            sigma               = np.array(sigma),
                            results_histogram   = results_histogram
                            )