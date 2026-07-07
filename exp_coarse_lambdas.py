# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""

import numpy as np

# =============================================================================
# LOAD EXPERIMENTAL DATA
# =============================================================================

print("Loading atom count data from Coarse experiment...")
atoms_in_m1 = np.load('exp_data/Coarse/all_atoms_in_m1.npy')  # mF = -1 state
atoms_in_0  = np.load('exp_data/Coarse/all_atoms_in_0.npy')   # mF = 0 state  
atoms_in_p1 = np.load('exp_data/Coarse/all_atoms_in_p1.npy')  # mF = +1 state

# Total atom number per shot (sum across all substates)
all_atoms = (atoms_in_m1 + atoms_in_0 + atoms_in_p1)

# =============================================================================
# COMPUTE RELATIVE POPULATIONS (λ)
# =============================================================================
# λ_i = N_i / N_total for each state i ∈ {m1, 0, p1}

lam_m1 = atoms_in_m1 / all_atoms   # Fraction in mF = -1 state
lam_0  = atoms_in_0  / all_atoms   # Fraction in mF = 0 state
lam_p1 = atoms_in_p1 / all_atoms   # Fraction in mF = +1 state

# Differential population imbalance (used for phase reconstruction)
delta_lam = lam_p1 - lam_m1

# =============================================================================
# STATISTICS AND RESULTS
# =============================================================================
# Compute sample mean and standard deviation (ddof=1 for sample std)

stats = {
    'λ_m1': (lam_m1.mean(), lam_m1.std(ddof=1)),
    'λ_0':  (lam_0.mean(),  lam_0.std(ddof=1)), 
    'λ_p1': (lam_p1.mean(), lam_p1.std(ddof=1)),
    'Δλ':   (delta_lam.mean(), delta_lam.std(ddof=1))
}

# Print results with 2 decimal places
print("\nPopulation fractions (λ_i = N_i/N_total):")
print(f"λ_m1 = {stats['λ_m1'][0]:.2f} ± {stats['λ_m1'][1]:.2f}")
print(f"λ_0  = {stats['λ_0'][0]:.2f} ± {stats['λ_0'][1]:.2f}")
print(f"λ_p1 = {stats['λ_p1'][0]:.2f} ± {stats['λ_p1'][1]:.2f}")
print(f"Δλ   = {stats['Δλ'][0]:.2f} ± {stats['Δλ'][1]:.2f}")

