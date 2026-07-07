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
    n_Ts, n_phis, folder, i, j, seed_offset = params
    seed = i + j * n_Ts + seed_offset
    S_null = np.load(f'exp_data/Contrast_Envelope/signal_0.npy')[:, i,:].ravel()

    ### bootstrapping in such a manner that the original data is included ###
    if j > 0:
        rng = np.random.default_rng(seed)
        S_null = rng.choice(S_null, n_phis)

    info_params = f"i = {i}, j = {j}, seed_offset = {seed_offset}, folder = {folder}"
    
    bins_null, hist_vals_null, fit_null, peak_pos_null, peak_frac_pos_null, initial_guess_null, ssr_null = hf.fit_routine_hist(
        S_null, add_info="null: "+info_params)

    return (*fit_null,
            *initial_guess_null,
            ssr_null)


##################################################
###################### main ######################
##################################################
if __name__ == "__main__":
    ##########################################
    ########## start of parameters ###########
    ##########################################

    k = 4*np.pi/780.226e-9
    
    folder = "Contrast_Envelope"

    Ts = np.linspace(-150,150,31)*1e-6

    n_phis = len(np.load(f'exp_data/{folder}/signal_0.npy')[:,0,:].ravel())
    
    n_Ts = len(Ts)
    T_min = Ts[0]
    T_max = Ts[-1]    

    n_stoch_rep = 1000
    seed_offset = 0

    save_plots = False
    save_data = True

    max_kernels = 12
    #########################################
    ########### end of parameters ###########
    #########################################

    params_list = [(n_Ts, n_phis, folder, i, j, seed_offset) for i in range(n_Ts) for j in range(n_stoch_rep)]

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


    ### histogram stuff ###
    ## visibilities ##
    # first: mean of V_plus and V_minus as estimate for theta calc
    V0_hist_fits = results_histogram[:, :, 0]
    sigma_hist_fits = results_histogram[:, :, 1]
    logger.info(f"V0 as mean of plus, minus and zero: {np.mean(V0_hist_fits):.6f} +- {np.std(V0_hist_fits, ddof=1):.6f}")
    logger.info(f"sigma as mean of plus, minus and zero: {np.mean(sigma_hist_fits):.6f} +- {np.std(sigma_hist_fits, ddof=1):.6f}")
    print(f"V0 as mean of plus and minus: {np.mean(V0_hist_fits):.6f} +- {np.std(V0_hist_fits, ddof=1):.6f}")
    print(f"sigma as mean of plus and minus: {np.mean(sigma_hist_fits):.6f} +- {np.std(sigma_hist_fits, ddof=1):.6f}")

        
    ######################
    ### saving results ###
    ######################

    if save_data:
        np.savez_compressed('exp_eval/exp_contrast_envelope_fits_res_S0.npz',
                            n_Ts=np.array(n_Ts),
                            Ts=Ts,
                            n_stoch_rep=np.array(n_stoch_rep),
                            results_histogram=results_histogram
                            )