import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helper_functions as hf

plt.style.use('paper_mpl_style.mplstyle')  # Apply paper-style plotting template

##################################
### start: load numerical data ###
##################################
data_theta_scan = np.load('num_data/theta_scan.npz')

lambda_mean_theta_scan          = data_theta_scan['lambda_mean']
lambda_diff_theta_scan          = data_theta_scan['lambda_diff']
lambda_plus_theta_scan          = data_theta_scan['lambda_plus']
lambda_minus_theta_scan         = data_theta_scan['lambda_minus']

n_thetas_theta_scan             = data_theta_scan['n_thetas']
thetas_theta_scan               = data_theta_scan['thetas']
n_stoch_rep_theta_scan          = data_theta_scan['n_stoch_rep']
seed_offset_theta_scan          = data_theta_scan['seed_offset']
n_phis_theta_scan               = data_theta_scan['n_phis']
A0_theta_scan                   = data_theta_scan['A0']
sigma_theta_scan                = data_theta_scan['sigma']
results_ellipse_theta_scan      = data_theta_scan['results_ellipse']
results_histogram_theta_scan    = data_theta_scan['results_histogram']

################################
### end: load numerical data ###
################################

##################################
### start: raw theta reconstr. ###
##################################
# mean of A_plus and A_minus
A0_hist_fits_theta_scan = (
    results_histogram_theta_scan[:, :, 0] + results_histogram_theta_scan[:, :, 3]) / 2

# sum signal fit results
A_sum_hist_fits_theta_scan      = results_histogram_theta_scan[:, :, 6]
sigma_sum_hist_fits_theta_scan  = results_histogram_theta_scan[:, :, 7]
mu_sum_hist_fits_theta_scan     = results_histogram_theta_scan[:, :, 8]

# set sum signal amplitde
A_sum_real_theta_scan = hf.rel_lambdas_to_amplitude(
    thetas_theta_scan, A0_theta_scan, *hf.plain_lambdas_to_rel(lambda_plus_theta_scan, lambda_minus_theta_scan))

# difference signal fit results
A_diff_hist_fits_theta_scan     = results_histogram_theta_scan[:, :, 9]
sigma_diff_hist_fits_theta_scan = results_histogram_theta_scan[:, :, 10]
mu_diff_hist_fits_theta_scan    = results_histogram_theta_scan[:, :, 11]

# set difference signal amplitde
A_diff_real_theta_scan = hf.rel_lambdas_to_amplitude(
    thetas_theta_scan, A0_theta_scan, *hf.plain_lambdas_to_rel(lambda_plus_theta_scan, -lambda_minus_theta_scan))

# theta via sum and difference signal
thetas_sum_theta_scan = hf.amplitude_to_theta(
    A_sum_hist_fits_theta_scan, A0_hist_fits_theta_scan, lambda_mean_theta_scan, lambda_diff_theta_scan)
thetas_diff_theta_scan = hf.amplitude_to_theta(
    A_diff_hist_fits_theta_scan, A0_hist_fits_theta_scan, lambda_mean_theta_scan, lambda_diff_theta_scan)

# theta via ellipse fits
thetas_ell_theta_scan = hf.conic_section_to_theta(
    results_ellipse_theta_scan[:, :, 0], results_ellipse_theta_scan[:, :, 1], results_ellipse_theta_scan[:, :, 2])

# all thetas have shape (n_thetas_theta_scan, n_stoch_rep_theta_scan)
# calculate wrt to each theta setting mean and empirical standard deviation
thetas_ell_mean_raw_theta_scan = np.mean(thetas_ell_theta_scan, axis=1)
thetas_ell_std_theta_scan = np.std(thetas_ell_theta_scan, axis=1, ddof=1)

thetas_sum_mean_raw_theta_scan = np.mean(thetas_sum_theta_scan, axis=1)
thetas_sum_std_theta_scan = np.std(thetas_sum_theta_scan, axis=1, ddof=1)

thetas_diff_mean_raw_theta_scan = np.mean(thetas_diff_theta_scan, axis=1)
thetas_diff_std_theta_scan = np.std(thetas_diff_theta_scan, axis=1, ddof=1)

################################
### end: raw theta reconstr. ###
################################

###############################
### start: phase unwrapping ###
###############################
## create mask for branches ##
mask_branch1_theta_scan = thetas_theta_scan <= np.pi
mask_branch2_theta_scan = (thetas_theta_scan > np.pi) & (thetas_theta_scan <= 2*np.pi)
mask_branch3_theta_scan = thetas_theta_scan > 2*np.pi

## phase unwrap per branch ##
branch_1_ell_theta_scan = thetas_ell_mean_raw_theta_scan
branch_2_ell_theta_scan = 2*np.pi - thetas_ell_mean_raw_theta_scan
branch_3_ell_theta_scan = 2*np.pi + thetas_ell_mean_raw_theta_scan

branch_1_hist_sum_theta_scan = thetas_sum_mean_raw_theta_scan
branch_2_hist_sum_theta_scan = 2*np.pi - thetas_sum_mean_raw_theta_scan
branch_3_hist_sum_theta_scan = 2*np.pi + thetas_sum_mean_raw_theta_scan

branch_1_hist_diff_theta_scan = np.pi - thetas_diff_mean_raw_theta_scan
branch_2_hist_diff_theta_scan = np.pi + thetas_diff_mean_raw_theta_scan
branch_3_hist_diff_theta_scan = 3*np.pi - thetas_diff_mean_raw_theta_scan

## combine branches for correct phase unwrapping ##
thetas_ell_mean_theta_scan, thetas_sum_mean_theta_scan, thetas_diff_mean_theta_scan = np.empty_like(
    thetas_theta_scan), np.empty_like(thetas_theta_scan), np.empty_like(thetas_theta_scan)


thetas_ell_mean_theta_scan[mask_branch1_theta_scan] = branch_1_ell_theta_scan[mask_branch1_theta_scan]
thetas_ell_mean_theta_scan[mask_branch2_theta_scan] = branch_2_ell_theta_scan[mask_branch2_theta_scan]
thetas_ell_mean_theta_scan[mask_branch3_theta_scan] = branch_3_ell_theta_scan[mask_branch3_theta_scan]


thetas_sum_mean_theta_scan[mask_branch1_theta_scan] = branch_1_hist_sum_theta_scan[mask_branch1_theta_scan]
thetas_sum_mean_theta_scan[mask_branch2_theta_scan] = branch_2_hist_sum_theta_scan[mask_branch2_theta_scan]
thetas_sum_mean_theta_scan[mask_branch3_theta_scan] = branch_3_hist_sum_theta_scan[mask_branch3_theta_scan]


thetas_diff_mean_theta_scan[mask_branch1_theta_scan] = branch_1_hist_diff_theta_scan[mask_branch1_theta_scan]
thetas_diff_mean_theta_scan[mask_branch2_theta_scan] = branch_2_hist_diff_theta_scan[mask_branch2_theta_scan]
thetas_diff_mean_theta_scan[mask_branch3_theta_scan] = branch_3_hist_diff_theta_scan[mask_branch3_theta_scan]
#############################
### end: phase unwrapping ###
#############################

np.savez_compressed('num_eval/num_eval_results.npz',
                    A0_set      = A0_theta_scan,
                    A_sum_set   = A_sum_real_theta_scan,
                    A_sum_rec   = A_sum_hist_fits_theta_scan,
                    A_diff_set  = A_diff_real_theta_scan,
                    A_diff_rec  = A_diff_hist_fits_theta_scan,
                    #
                    sigma_set   = sigma_theta_scan,
                    #
                    thetas_set  = thetas_theta_scan,
                    #
                    thetas_rec_ell  = thetas_ell_mean_theta_scan,
                    thetas_rec_sum  = thetas_sum_mean_theta_scan,
                    thetas_rec_diff = thetas_diff_mean_theta_scan,
                    #
                    thetas_rec_ell_std   = thetas_ell_std_theta_scan,
                    thetas_rec_sum_std   = thetas_sum_std_theta_scan,
                    thetas_rec_diff_std  = thetas_diff_std_theta_scan
                    )