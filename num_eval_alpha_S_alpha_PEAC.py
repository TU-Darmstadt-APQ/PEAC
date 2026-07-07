# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import helper_functions as hf

plt.style.use('paper_mpl_style.mplstyle')

colour_beta = 'C0'
colour_alpha = 'C1'
colour_ell = 'C2'

# alpha scan
num_eval_theta_scan     = np.load('num_data/num_fits_res_alpha_S_alpha_slice_PEAC.npz')
save_data = True
name_for_saving = "num_eval_res_alpha_S_alpha_slice_PEAC"

alphas              = num_eval_theta_scan['alphas']
lambdas_mean_alpha  = num_eval_theta_scan['lambdas_mean_alpha']
lambdas_diff_alpha  = num_eval_theta_scan['lambdas_diff_alpha']
lambdas_plus_alpha  = num_eval_theta_scan['lambdas_plus_alpha']
lambdas_minus_alpha = num_eval_theta_scan['lambdas_minus_alpha']
n_thetas            = num_eval_theta_scan['n_thetas']
thetas              = num_eval_theta_scan['thetas']
n_stoch_rep         = num_eval_theta_scan['n_stoch_rep']
seed_offset         = num_eval_theta_scan['seed_offset']
n_phis              = num_eval_theta_scan['n_phis']
A0                  = num_eval_theta_scan['A0']
sigma               = num_eval_theta_scan['sigma']
results_histogram   = num_eval_theta_scan['results_histogram']

n_alphas = len(alphas)

A0s = (results_histogram[:, :, :, 0] + results_histogram[:, :, :, 3]) / 2
A_alphas = results_histogram[:, :, :, 6]
A_betas = results_histogram[:, :, :, 9]

A_alphas_real = np.empty((n_alphas, n_thetas))
A_betas_real = np.empty((n_alphas, n_thetas))

theta_alphas = np.empty((n_alphas, n_thetas))
theta_betas = np.empty((n_alphas, n_thetas))

theta_unct_alphas = np.empty((n_alphas, n_thetas))
theta_unct_betas = np.empty((n_alphas, n_thetas))


for i in range(n_alphas):
    lambda_plus_alpha = lambdas_plus_alpha[i]
    lambda_minus_alpha = lambdas_minus_alpha[i]
    lambda_mean_alpha = lambdas_mean_alpha[i]
    lambda_diff_alpha = lambdas_diff_alpha[i]
    # lambdas in beta direction
    lambda_mean_beta = - lambda_diff_alpha/2
    lambda_diff_beta = 2 * lambda_mean_alpha

    ### histogram stuff ###
    ## amplitudes ##
    # first: mean of A_plus and A_minus as estimate for theta recon
    A0_hist_fits = A0s[i]
    A_alpha_hist_fits = A_alphas[i]
    A_beta_hist_fits = A_betas[i]

    A_alpha_hist_mean = np.nanmean(A_alpha_hist_fits, axis=1)
    A_alpha_hist_std = np.nanstd(A_alpha_hist_fits, axis=1, ddof=1)
    A_alphas_real[i] = hf.rel_lambdas_to_amplitude(
        thetas, A0, lambda_mean_alpha, lambda_diff_alpha)

    A_beta_hist_mean = np.nanmean(A_beta_hist_fits, axis=1)
    A_beta_hist_std = np.nanstd(A_beta_hist_fits, axis=1, ddof=1)
    A_betas_real[i] = hf.rel_lambdas_to_amplitude(
        thetas, A0, lambda_mean_beta, lambda_diff_beta)

    ## theta histogram ##
    theta_hist_alpha = hf.amplitude_to_theta(
        A_alpha_hist_fits, A0_hist_fits, lambda_mean_alpha, lambda_diff_alpha)
    
    theta_hist_beta = hf.amplitude_to_theta(
        A_beta_hist_fits, A0_hist_fits, lambda_mean_beta, lambda_diff_beta)

    theta_hist_alpha_mean_raw = np.nanmean(theta_hist_alpha, axis=1)
    theta_unct_alphas[i] = np.nanstd(theta_hist_alpha, axis=1, ddof=1)

    theta_hist_beta_mean_raw = np.nanmean(theta_hist_beta, axis=1)
    theta_unct_betas[i] = np.nanstd(theta_hist_beta, axis=1, ddof=1)

    ### phase unwrapping ###
    ## create mask for branches ##
    mask_branch1 = thetas <= np.pi
    mask_branch2 = (np.pi < thetas) & (thetas <= 2*np.pi)
    mask_branch3 = 2*np.pi < thetas

    ## phase unwrap per branch ##
    branch_1_hist_alpha = theta_hist_alpha_mean_raw
    branch_2_hist_alpha = 2*np.pi - theta_hist_alpha_mean_raw
    branch_3_hist_alpha = 2*np.pi + theta_hist_alpha_mean_raw

    branch_1_hist_beta = theta_hist_beta_mean_raw
    branch_2_hist_beta = 2*np.pi - theta_hist_beta_mean_raw
    branch_3_hist_beta = 2*np.pi + theta_hist_beta_mean_raw

    ## combine branches for correct phase unwrapping ##
    theta_hist_alpha_mean, theta_hist_beta_mean = np.empty_like(
        thetas), np.empty_like(thetas)

    theta_hist_alpha_mean[mask_branch1] = branch_1_hist_alpha[mask_branch1]
    theta_hist_alpha_mean[mask_branch2] = branch_2_hist_alpha[mask_branch2]
    theta_hist_alpha_mean[mask_branch3] = branch_3_hist_alpha[mask_branch3]

    theta_hist_beta_mean[mask_branch1] = branch_1_hist_beta[mask_branch1]
    theta_hist_beta_mean[mask_branch2] = branch_2_hist_beta[mask_branch2]
    theta_hist_beta_mean[mask_branch3] = branch_3_hist_beta[mask_branch3]

    theta_alphas[i] = theta_hist_alpha_mean
    theta_betas[i] = theta_hist_beta_mean

if save_data:
    np.savez_compressed(f'num_eval/{name_for_saving}.npz',
                        alphas                  =alphas,
                        A0_set                  =A0,
                        A_alphas_set            =A_alphas_real,
                        A_alphas_rec            =A_alphas,
                        A_betas_set             =A_betas_real,
                        A_betas_rec             =A_betas,
                        #
                        sigma_set               =sigma,
                        #
                        thetas_set              =thetas,
                        #
                        thetas_rec_alphas       =theta_alphas,
                        thetas_rec_betas        =theta_betas,
                        #
                        thetas_rec_alphas_std   =theta_unct_alphas,
                        thetas_rec_betas_std    =theta_unct_betas
                        )