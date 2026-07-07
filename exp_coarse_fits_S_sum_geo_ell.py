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

    return hf.fit_ellipse_geometric(S_plus, S_minus, add_info="ell: "+info_params)


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

    max_kernels = 120
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

    ######################
    ### saving results ###
    ######################

    if save_data:
        np.savez_compressed('exp_eval/exp_coarse_fits_res_S_sum_geo_ell.npz',
                            lambda_mean         = np.array(lambda_mean),
                            lambda_diff         = np.array(lambda_diff),
                            lambda_plus         = np.array(lambda_plus),
                            lambda_minus        = np.array(lambda_minus),
                            n_Ts                = np.array(n_Ts),
                            Ts                  = Ts,
                            n_stoch_rep         = np.array(n_stoch_rep),
                            results_ellipse     = results_ellipse
                            )