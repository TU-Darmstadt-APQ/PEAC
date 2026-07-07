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

def fit_ellipse_for_parallel(params):
    A0, sigma, theta, n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset, bounds = params
    seed = i + j * n_thetas + seed_offset

    S_plus, S_minus, _, _ = hf.generate_signals(
        A0, sigma, theta, n_phis, lambda_plus, lambda_minus, seed)

    info_params = f"theta = {theta:.17g}, A0 = {A0:.17g}, sigma = {sigma:.17g}, seed = {seed}"

    return hf.fit_ellipse_geometric(S_plus, S_minus, add_info="ell: "+info_params)


def fit_histogram_for_parallel(params):
    A0, sigma, theta, n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset, bounds = params
    seed = i + j * n_thetas + seed_offset

    S_plus, S_minus, S_sum, _ = hf.generate_signals(
        A0, sigma, theta, n_phis, lambda_plus, lambda_minus, seed)

    info_params = f"theta = {theta:.17g}, A0 = {A0:.17g}, sigma = {sigma:.17g}, seed = {seed}"

    bins_plus, _, fit_plus, _, _, initial_guess_plus, ssr_plus = hf.fit_routine_hist(
        S_plus, add_info="plus: "+info_params, bounds=bounds)

    bins_minus, _, fit_minus, _, _, initial_guess_minus, ssr_minus = hf.fit_routine_hist(
        S_minus, add_info="minus: "+info_params, bounds=bounds)

    sigma_guess = hf.sigma_density(np.nanmean(
        [fit_plus[1], fit_minus[1]]), lambda_plus, lambda_minus)
    if np.isnan(sigma_guess):
        logger.warning(f'Fits of plus and minus failed for {info_params}')
        sigma_guess = hf.sigma_density(sigma, lambda_plus, lambda_minus)

    ## check for resolution of sigma ##
    if sigma_guess < (bins_plus[1]-bins_plus[0]) or sigma_guess < (bins_minus[1]-bins_minus[0]):
        logger.info(
            f'sigma_guess was less than resolution limit: {sigma_guess:.17g} versus {bins_plus[1]-bins_plus[0]:.17g} or {bins_minus[1]-bins_minus[0]:.17g}.')
        sigma_guess = None

    A0_mean = np.nanmean([fit_plus[0], fit_minus[0]])
    A_max = hf.amp_max_guess(A0_mean, lambda_plus, lambda_minus)

    _, _, fit_sum, _, _, initial_guess_sum, ssr_sum = hf.fit_routine_hist(
        S_sum, add_info="sum: "+info_params, sigma_guess=sigma_guess, A_max=A_max, bounds=bounds)

    # _, _, fit_diff, _, _, initial_guess_diff, ssr_diff = hf.fit_routine_hist(
    #     S_diff, add_info="diff: "+info_params, sigma_guess=sigma_guess, A_max=A_max, bounds=bounds)

    return (*fit_plus,              *fit_minus,             *fit_sum,
            *initial_guess_plus,    *initial_guess_minus,   *initial_guess_sum,
            ssr_plus,               ssr_minus,              ssr_sum)


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


    # sigmas
    # 40 linearly spaced values from 0.15 to 0.005 (inclusive)
    linear_part = np.linspace(0.15, 0.005, 40)

    # Append the extra value
    sigmas = np.concatenate([linear_part, [1e-6]])

    n_sigmas = len(sigmas)

    n_phis  =   300

    # thetas
    # thetas = np.array([np.pi, np.pi+np.pi/32])
    # thetas = np.array([np.pi, np.pi + np.pi/32, np.pi + np.pi/2, np.pi])
    # thetas = np.array([np.pi - np.pi/32, np.pi - 1e-3,
    #                    np.pi, np.pi + 1e-3, np.pi + np.pi/32])
    # n_thetas = len(thetas)

    theta_min = 166/333*np.pi
    theta_max = 1.0*np.pi
    n_thetas = 168
    thetas = np.linspace(theta_min, theta_max, n_thetas)

    seed_offset = 0
    n_stoch_rep = 1000
    bounds = True

    save_data = True

    folder_run = "num_data"
    saving_name = "num_fits_res_sigma_S_sum_PEAC_and_geo_ell"

    max_kernels = 120
    #########################################
    ########### end of parameters ###########
    #########################################

    params_list = [(A0, sigmas[j], thetas[i], n_thetas, n_phis, lambda_plus, lambda_minus, i, k, seed_offset, bounds)
                   for i in range(n_thetas) for j in range(n_sigmas) for k in range(n_stoch_rep)]

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
    results_ellipse = results_ellipse.reshape(n_thetas, n_sigmas, n_stoch_rep, -1)

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
    results_histogram = results_histogram.reshape(n_thetas, n_sigmas, n_stoch_rep, -1)

    #################
    ### save data ###
    #################
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
                            sigmas              = sigmas,
                            results_ellipse     = results_ellipse,
                            results_histogram   = results_histogram
                            )