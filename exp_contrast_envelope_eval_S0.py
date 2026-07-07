# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""

import numpy as np 
from scipy.optimize import curve_fit as c
import scipy.constants as sc


"""
    Import PEAC results for contrast envelope fitting
"""

hist_res = np.load('exp_eval/exp_contrast_envelope_fits_res_S0.npz')['results_histogram']

amplitudes = hist_res[:,:,0]

delta_T = np.linspace(-150,150,31)

"""
    Define functions for evaluation
"""
m87 = 86.909 * sc.u

def temp(sigma):
    k = 2 * 2*np.pi / 780.226e-9
    # return sc.hbar**2/(sc.k * (5.8845e-3)**2 * m87) * 1/sigma**2 * 1e9
    return m87 / (k**2 * sc.k * sigma**2) * 1e9

def deltap(sigma):
    k = 4*np.pi/780.226e-9
    return m87 / (k**2 * sc.hbar * sigma)

def gauss(x,amp,x0,sig,off):
    return amp * np.exp(-(x-x0)**2/(2*sig**2)) + off


"""
    Fit gaussian normal distribution to amplitudes
"""

try:
    fit_peac = np.load('exp_eval/exp_contrast_envelope_gauss_fits.npy')
except FileNotFoundError:
    fit_peac = []
    
    for i in range(amplitudes.shape[1]):
        res, _ = c(gauss, 
                   delta_T,amplitudes[:,i], 
                   p0 = [0.75,0,70,0],
                   bounds = ([0,-100,0,0],np.inf))
        fit_peac.append(res)
    
    np.save('exp_eval/exp_contrast_envelope_gauss_fits.npy', fit_peac)
    fit_peac = np.load('exp_eval/exp_contrast_envelope_gauss_fits.npy')