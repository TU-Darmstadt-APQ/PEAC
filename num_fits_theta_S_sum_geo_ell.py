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
    A0, sigma, theta, n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset = params
    seed = i + j * n_thetas + seed_offset

    S_plus, S_minus, _, _ = hf.generate_signals(
        A0, sigma, theta, n_phis, lambda_plus, lambda_minus, seed)

    info_params = f"theta = {theta:.17g}, A0 = {A0:.17g}, sigma = {sigma:.17g}, seed = {seed}"

    return hf.fit_ellipse_geometric(S_plus, S_minus, add_info="ell: "+info_params)


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

    save_data = True

    folder_run = "num_data"
    saving_name = "num_fits_res_theta_S_sum_geo_ell"

    max_kernels = 120
    #########################################
    ########### end of parameters ###########
    #########################################

    params_list = [(A0, sigma, thetas[i], n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset)
                   for i in range(n_thetas) for j in range(n_stoch_rep)]

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
    results_ellipse = results_ellipse.reshape(n_thetas, n_stoch_rep, -1)

    ### ellipse stuff ###
    ## ellipse axes ##
    x0_ell    = results_ellipse[:,:,0]
    y0_ell    = results_ellipse[:,:,1]
    ap_ell    = results_ellipse[:,:,2]
    bp_ell    = results_ellipse[:,:,3]
    alpha_ell = results_ellipse[:,:,4]

    major_axis_ell_mean = np.nanmean(ap_ell, axis=1)
    major_axis_ell_std = np.nanstd(ap_ell, axis=1, ddof=1)

    minor_axis_ell_mean = np.nanmean(bp_ell, axis=1)
    minor_axis_ell_std = np.nanstd(bp_ell, axis=1, ddof=1)

    minor_axis_real = hf.axis_sum(thetas, A0)
    major_axis_real = hf.axis_diff(thetas, A0)

    ## theta ellipse ##
    theta_ell = hf.geom_ell_to_theta(alpha_ell, ap_ell, bp_ell)

    theta_ell_mean_raw = np.nanmean(theta_ell, axis=1)
    theta_ell_std = np.nanstd(theta_ell, axis=1, ddof=1)


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
                            sigma               = np.array(sigma),
                            results_ellipse     = results_ellipse
                            )

    #############
    ### plots ###
    #############
    inch_to_cm = 2.54
    phi_golden = (1 + np.sqrt(5)) / 2
    width_inch = 15 / inch_to_cm
    height_inch = width_inch / phi_golden
    plt.rc('font', size=10)

    ### phase unwrapping ###
    ## create mask for branches ##
    mask_branch1 = thetas <= np.pi
    mask_branch2 = (np.pi < thetas) & (thetas <= 2*np.pi)
    mask_branch3 = 2*np.pi < thetas

    ## phase unwrap per branch ##
    branch_1_ell = theta_ell_mean_raw
    branch_2_ell = 2*np.pi - theta_ell_mean_raw
    branch_3_ell = 2*np.pi + theta_ell_mean_raw

    ## combine branches for correct phase unwrapping ##
    theta_ell_mean = np.empty_like(thetas)

    theta_ell_mean[mask_branch1] = branch_1_ell[mask_branch1]
    theta_ell_mean[mask_branch2] = branch_2_ell[mask_branch2]
    theta_ell_mean[mask_branch3] = branch_3_ell[mask_branch3]


    ##################
    ### theta bias ###
    ##################
    theta_bias_ell  = theta_ell_mean - thetas

    ### theta bias plot ####
    fig_theta_bias, ax_theta_bias = plt.subplots(
        1, 1, figsize=(width_inch, height_inch)
    )

    # series values
    x_ell = thetas/np.pi

    ## bias ellipse ##
    ax_theta_bias.grid(True)
    ax_theta_bias.minorticks_on()
    ax_theta_bias.grid(which='minor', linestyle=':', linewidth=0.6)
    ax_theta_bias.axhline(0, color="black", linewidth=1, ls="--")
    hf.plot_line_with_wide_err(ax_theta_bias, x_ell, theta_bias_ell, 0, theta_ell_std, colour_ell, r'$\theta_{\text{bias, ell}}$')
    ax_theta_bias.set_xlabel(r'$\theta/\pi$')
    ax_theta_bias.set_xlim(x_ell[0], x_ell[-1])
    ax_theta_bias.legend(loc='lower right')

    plt.tight_layout()

    ### theta uncertainty plot ###
    fig_theta_unct, ax_theta_unct = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_theta_unct.plot(thetas/np.pi, theta_ell_std,
                       color=colour_ell, label=r'$\Delta \theta_\text{ell}$')

    ax_theta_unct.set_xlabel(r'$\theta/\pi$ (rad)')
    ax_theta_unct.set_xlim(theta_min/np.pi, theta_max/np.pi)
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
    ax_axes_bias.plot(thetas/np.pi, major_axis_ell_mean-major_axis_real, color="tab:red",
                      linewidth=1, label='bias major axis')
    ax_axes_bias.fill_between(thetas/np.pi,
                              major_axis_ell_mean-major_axis_real - 1*major_axis_ell_std,
                              major_axis_ell_mean-major_axis_real + 1*major_axis_ell_std,
                              color='tab:red', alpha=0.3)
    ## bias minor axis ##
    ax_axes_bias.plot(thetas/np.pi, minor_axis_ell_mean-minor_axis_real, color="tab:purple",
                      linewidth=1, label='bias minor axis')
    ax_axes_bias.fill_between(thetas/np.pi,
                              minor_axis_ell_mean-minor_axis_real - 1*minor_axis_ell_std,
                              minor_axis_ell_mean-minor_axis_real + 1*minor_axis_ell_std,
                              color='tab:purple', alpha=0.3)

    ax_axes_bias.set_xlabel(r'$\theta/\pi$ (rad)')
    ax_axes_bias.set_xlim(theta_min/np.pi, theta_max/np.pi)
    ax_axes_bias.legend(loc='lower right')
    plt.tight_layout()

    ### half-axes uncertainty only for ellipse ###
    fig_axes_unct, ax_axes_unct = plt.subplots(
        figsize=(width_inch, height_inch))
    ax_axes_unct.plot(thetas/np.pi, major_axis_ell_std,
                      color='tab:red', label='uncertainty major axis')
    ax_axes_unct.plot(thetas/np.pi, minor_axis_ell_std,
                      color='tab:purple', label='uncertainty minor axis')

    ax_axes_unct.set_xlabel(r'$\theta/\pi$ (rad)')
    ax_axes_unct.set_xlim(theta_min/np.pi, theta_max/np.pi)
    ax_axes_unct.set_yscale("log")
    ax_axes_unct.legend(loc='upper right')
    ax_axes_unct.grid(True)
    ax_axes_unct.minorticks_on()
    ax_axes_unct.grid(which='minor', linestyle=':', linewidth=0.6)
    plt.tight_layout()
