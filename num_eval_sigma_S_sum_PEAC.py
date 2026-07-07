# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import helper_functions as hf

plt.style.use('paper_mpl_style.mplstyle')

colour_diff = 'C0'
colour_sum = 'C1'
colour_ell = 'C2'

# sigma scan
data_sigma_scan = np.load('num_data/num_fits_res_sigma_S_sum_PEAC_and_geo_ell.npz')
save_data = True
name_for_saving = "num_eval_res_sigma_S_sum_PEAC_and_geo_ell"

lambda_mean_sigma_scan          = data_sigma_scan['lambda_mean']
lambda_diff_sigma_scan          = data_sigma_scan['lambda_diff']
lambda_plus_sigma_scan          = data_sigma_scan['lambda_plus']
lambda_minus_sigma_scan         = data_sigma_scan['lambda_minus']

n_thetas_sigma_scan             = data_sigma_scan['n_thetas']
thetas_sigma_scan               = data_sigma_scan['thetas']
n_stoch_rep_sigma_scan          = data_sigma_scan['n_stoch_rep']
seed_offset_sigma_scan          = data_sigma_scan['seed_offset']
n_phis_sigma_scan               = data_sigma_scan['n_phis']
A0_sigma_scan                   = data_sigma_scan['A0']
sigmas_sigma_scan               = data_sigma_scan['sigmas']
results_ellipse_sigma_scan      = data_sigma_scan['results_ellipse']    # shape (n_thetas, n_sigmas, n_stoch_rep, XXX)
results_histogram_sigma_scan    = data_sigma_scan['results_histogram']  # shape (n_thetas, n_sigmas, n_stoch_rep, XXX)

# swap axes
results_ellipse_sigma_scan = results_ellipse_sigma_scan.swapaxes(0, 1)
results_histogram_sigma_scan = results_histogram_sigma_scan.swapaxes(0, 1)

n_sigmas = len(sigmas_sigma_scan)

A0s = (results_histogram_sigma_scan[:, :, :, 0] + results_histogram_sigma_scan[:, :, :, 3]) / 2
A_sum_sigmas = results_histogram_sigma_scan[:, :, :, 6]
A_diff_sigmas = results_histogram_sigma_scan[:, :, :, 9]

A_sum_sigmas_real = np.empty((n_sigmas, n_thetas_sigma_scan))
A_diff_sigmas_real = np.empty((n_sigmas, n_thetas_sigma_scan))

theta_sum_sigmas = np.empty((n_sigmas, n_thetas_sigma_scan))
theta_diff_sigmas = np.empty((n_sigmas, n_thetas_sigma_scan))
theta_ell_sigmas = np.empty((n_sigmas, n_thetas_sigma_scan))

theta_unct_sum_sigmas = np.empty((n_sigmas, n_thetas_sigma_scan))
theta_unct_diff_sigmas = np.empty((n_sigmas, n_thetas_sigma_scan))
theta_unct_ell_sigmas = np.empty((n_sigmas, n_thetas_sigma_scan))


for i in range(n_sigmas):
    ### histogram stuff ###
    ## amplitudes ##
    # first: mean of A_plus and A_minus as estimate for theta recon
    A0_hist_fits = A0s[i]
    A_sum_hist_fits = A_sum_sigmas[i]
    A_diff_hist_fits = A_diff_sigmas[i]

    A_sum_hist_mean = np.nanmean(A_sum_hist_fits, axis=1)
    A_sum_hist_std = np.nanstd(A_sum_hist_fits, axis=1, ddof=1)
    A_sum_sigmas_real[i] = hf.rel_lambdas_to_amplitude(
        thetas_sigma_scan, A0_sigma_scan, lambda_mean_sigma_scan, lambda_diff_sigma_scan)

    A_diff_hist_mean = np.nanmean(A_diff_hist_fits, axis=1)
    A_diff_hist_std = np.nanstd(A_diff_hist_fits, axis=1, ddof=1)
    A_diff_sigmas_real[i] = hf.rel_lambdas_to_amplitude(
        thetas_sigma_scan, A0_sigma_scan, *hf.plain_lambdas_to_rel(lambda_plus_sigma_scan, -lambda_minus_sigma_scan))

    ## theta histogram ##
    theta_hist_sum = hf.amplitude_to_theta(
        A_sum_hist_fits, A0_hist_fits, lambda_mean_sigma_scan, lambda_diff_sigma_scan)
    
    theta_hist_diff = hf.amplitude_to_theta(
        A_diff_hist_fits, A0_hist_fits, *hf.plain_lambdas_to_rel(lambda_plus_sigma_scan, -lambda_minus_sigma_scan))

    theta_hist_sum_mean_raw = np.nanmean(theta_hist_sum, axis=1)
    theta_unct_sum_sigmas[i] = np.nanstd(theta_hist_sum, axis=1, ddof=1)

    theta_hist_diff_mean_raw = np.nanmean(theta_hist_diff, axis=1)
    theta_unct_diff_sigmas[i] = np.nanstd(theta_hist_diff, axis=1, ddof=1)

    ### ellipse stuff ###
    ## ellipse axes ##
    x0_ell    = results_ellipse_sigma_scan[i, :, :, 0]
    y0_ell    = results_ellipse_sigma_scan[i, :, :, 1]
    ap_ell    = results_ellipse_sigma_scan[i, :, :, 2]
    bp_ell    = results_ellipse_sigma_scan[i, :, :, 3]
    alpha_ell = results_ellipse_sigma_scan[i, :, :, 4]

    ## theta ellipse ##
    theta_ell = hf.geom_ell_to_theta(alpha_ell, ap_ell, bp_ell)

    theta_ell_mean_raw = np.nanmean(theta_ell, axis=1)
    theta_unct_ell_sigmas[i] = np.nanstd(theta_ell, axis=1, ddof=1)

    ### phase unwrapping ###
    ## create mask for branches ##
    mask_branch1 = thetas_sigma_scan <= np.pi
    mask_branch2 = (np.pi < thetas_sigma_scan) & (thetas_sigma_scan <= 2*np.pi)
    mask_branch3 = 2*np.pi < thetas_sigma_scan

    ## phase unwrap per branch ##
    branch_1_ell = theta_ell_mean_raw
    branch_2_ell = 2*np.pi - theta_ell_mean_raw
    branch_3_ell = 2*np.pi + theta_ell_mean_raw

    branch_1_hist_sum = theta_hist_sum_mean_raw
    branch_2_hist_sum = 2*np.pi - theta_hist_sum_mean_raw
    branch_3_hist_sum = 2*np.pi + theta_hist_sum_mean_raw

    branch_1_hist_diff = theta_hist_diff_mean_raw
    branch_2_hist_diff = 2*np.pi - theta_hist_diff_mean_raw
    branch_3_hist_diff = 2*np.pi + theta_hist_diff_mean_raw

    ## combine branches for correct phase unwrapping ##
    theta_hist_sum_mean, theta_hist_diff_mean, theta_ell_mean = np.empty_like(
        thetas_sigma_scan), np.empty_like(thetas_sigma_scan), np.empty_like(thetas_sigma_scan)

    theta_hist_sum_mean[mask_branch1] = branch_1_hist_sum[mask_branch1]
    theta_hist_sum_mean[mask_branch2] = branch_2_hist_sum[mask_branch2]
    theta_hist_sum_mean[mask_branch3] = branch_3_hist_sum[mask_branch3]

    theta_hist_diff_mean[mask_branch1] = branch_1_hist_diff[mask_branch1]
    theta_hist_diff_mean[mask_branch2] = branch_2_hist_diff[mask_branch2]
    theta_hist_diff_mean[mask_branch3] = branch_3_hist_diff[mask_branch3]

    theta_ell_mean[mask_branch1] = branch_1_ell[mask_branch1]
    theta_ell_mean[mask_branch2] = branch_2_ell[mask_branch2]
    theta_ell_mean[mask_branch3] = branch_3_ell[mask_branch3]

    theta_sum_sigmas[i] = theta_hist_sum_mean
    theta_diff_sigmas[i] = theta_hist_diff_mean
    theta_ell_sigmas[i] = theta_ell_mean

if save_data:
    np.savez_compressed(f'num_eval/{name_for_saving}.npz',
                        A0_set                  =A0_sigma_scan,
                        A_sum_sigmas_set            =A_sum_sigmas_real,
                        A_sum_sigmas_rec            =A_sum_sigmas,
                        A_diff_sigmas_set             =A_diff_sigmas_real,
                        A_diff_sigmas_rec             =A_diff_sigmas,
                        #
                        sigmas_set               =sigmas_sigma_scan,
                        #
                        thetas_set              =thetas_sigma_scan,
                        #
                        thetas_rec_sum_sigmas       =theta_sum_sigmas,
                        thetas_rec_diff_sigmas        =theta_diff_sigmas,
                        thetas_rec_ell_sigmas        =theta_ell_sigmas,
                        #
                        thetas_rec_sum_sigmas_std   =theta_unct_sum_sigmas,
                        thetas_rec_diff_sigmas_std    =theta_unct_diff_sigmas,
                        thetas_rec_ell_sigmas_std    =theta_unct_ell_sigmas
                        )