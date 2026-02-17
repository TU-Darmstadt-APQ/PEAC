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

    return hf.fit_ellipse(S_plus, S_minus, add_info="ell: "+info_params)


def fit_histogram_for_parallel(params):
    A0, sigma, theta, n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset, bounds = params
    seed = i + j * n_thetas + seed_offset

    S_plus, S_minus, S_sum, S_diff = hf.generate_signals(
        A0, sigma, theta, n_phis, lambda_plus, lambda_minus, seed)

    info_params = f"theta = {theta:.17g}, A0 = {A0:.17g}, sigma = {sigma:.17g}, seed = {seed}"

    bins_plus, hist_vals_plus, fit_plus, peak_pos_plus, peak_frac_pos_plus, initial_guess_plus, ssr_plus = hf.fit_routine_hist(
        S_plus, add_info="plus: "+info_params, bounds=bounds)

    bins_minus, hist_vals_minus, fit_minus, peak_pos_minus, peak_frac_pos_minus, initial_guess_minus, ssr_minus = hf.fit_routine_hist(
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

    bins_sum, hist_vals_sum, fit_sum, peak_pos_sum, peak_frac_pos_sum, initial_guess_sum, ssr_sum = hf.fit_routine_hist(
        S_sum, add_info="sum: "+info_params, sigma_guess=sigma_guess, A_max=A_max, bounds=bounds)

    bins_diff, hist_vals_diff, fit_diff, peak_pos_diff, peak_frac_pos_diff, initial_guess_diff, ssr_diff = hf.fit_routine_hist(
        S_diff, add_info="diff: "+info_params, sigma_guess=sigma_guess, A_max=A_max, bounds=bounds)

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
    saving_name = "theta_scan"

    max_kernels = 110
    #########################################
    ########### end of parameters ###########
    #########################################

    params_list = [(A0, sigma, thetas[i], n_thetas, n_phis, lambda_plus, lambda_minus, i, j, seed_offset, bounds)
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
    results_histogram = results_histogram.reshape(n_thetas, n_stoch_rep, -1)

    ### ellipse stuff ###
    ## ellipse axes ##
    x0_ell, y0_ell, ap_ell, bp_ell, phi_ell = hf.parametric_to_polar_vectorised(
        results_ellipse)

    major_axis_ell_mean = np.nanmean(ap_ell, axis=1)
    major_axis_ell_std = np.nanstd(ap_ell, axis=1, ddof=1)

    minor_axis_ell_mean = np.nanmean(bp_ell, axis=1)
    minor_axis_ell_std = np.nanstd(bp_ell, axis=1, ddof=1)

    minor_axis_real = hf.axis_sum(thetas, A0)
    major_axis_real = hf.axis_diff(thetas, A0)

    ## theta ellipse ##
    theta_ell = hf.conic_section_to_theta(
        results_ellipse[:, :, 0], results_ellipse[:, :, 1], results_ellipse[:, :, 2])

    theta_ell_mean_raw = np.nanmean(theta_ell, axis=1)
    theta_ell_std = np.nanstd(theta_ell, axis=1, ddof=1)

    ### histogram stuff ###
    ## amplitudes ##
    # first: mean of A_plus and A_minus as estimate for theta recon
    A0_hist_fits = (
        results_histogram[:, :, 0] + results_histogram[:, :, 3]) / 2
    A_sum_hist_fits = results_histogram[:, :, 6]
    A_diff_hist_fits = results_histogram[:, :, 9]

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
                            results_ellipse     = results_ellipse,
                            results_histogram   = results_histogram
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
    # Arccos is implemented in numpy in such a way that only values between 0 and Pi are returned:
    # in our case, 0 to Pi/2. Because cos is an even function, every value returned by arccos also
    # has a negative counterpart. And given the Pi/2 periodicity in our case, any integer multiples
    # of Pi/2 can be added.
    # For a value x of arccos, +-x + l*Pi/2 with l an integer is therefore also a possible solution.
    # By matching and considering the unmodified values of arrcos, its branches can be reconstructed
    # with correct phase unwrapping.

    ## create mask for branches ##
    mask_branch1 = thetas <= np.pi
    mask_branch2 = (np.pi < thetas) & (thetas <= 2*np.pi)
    mask_branch3 = 2*np.pi < thetas

    ## phase unwrap per branch ##
    branch_1_ell = theta_ell_mean_raw
    branch_2_ell = 2*np.pi - theta_ell_mean_raw
    branch_3_ell = 2*np.pi + theta_ell_mean_raw

    branch_1_hist_sum = theta_hist_sum_mean_raw
    branch_2_hist_sum = 2*np.pi - theta_hist_sum_mean_raw
    branch_3_hist_sum = 2*np.pi + theta_hist_sum_mean_raw

    branch_1_hist_diff = np.pi - theta_hist_diff_mean_raw
    branch_2_hist_diff = np.pi + theta_hist_diff_mean_raw
    branch_3_hist_diff = 3*np.pi - theta_hist_diff_mean_raw

    ## combine branches for correct phase unwrapping ##
    theta_ell_mean, theta_hist_sum_mean, theta_hist_diff_mean = np.empty_like(
        thetas), np.empty_like(thetas), np.empty_like(thetas)

    theta_ell_mean[mask_branch1] = branch_1_ell[mask_branch1]
    theta_ell_mean[mask_branch2] = branch_2_ell[mask_branch2]
    theta_ell_mean[mask_branch3] = branch_3_ell[mask_branch3]

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

    ## bias ellipse ##
    ax_theta_bias.plot(thetas/np.pi, theta_ell_mean-thetas, color=colour_ell,
                       linewidth=1, label=r'$\theta_{\text{bias, ell}}$')
    ax_theta_bias.fill_between(thetas/np.pi,
                               theta_ell_mean-thetas - 1*theta_ell_std,
                               theta_ell_mean-thetas + 1*theta_ell_std,
                               color=colour_ell, alpha=0.3)
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
    
    ##################
    ### theta bias ###
    ##################
    theta_bias_ell  = theta_ell_mean - thetas
    theta_bias_hist_sum  = theta_hist_sum_mean - thetas
    theta_bias_hist_diff = theta_hist_diff_mean - thetas

    ### theta bias plot ####
    fig_theta_bias, ax_theta_bias = plt.subplots(
        3, 1, figsize=(width_inch, height_inch * 3)
    )

    # series values
    x_ell = thetas/np.pi
    x_hist_sum = thetas/np.pi
    x_hist_diff = thetas/np.pi

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
    ax_theta_unct.plot(thetas/np.pi, theta_ell_std,
                       color=colour_ell, label=r'$\Delta \theta_\text{ell}$')
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
