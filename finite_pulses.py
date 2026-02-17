# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit as sc

import os
import sys

plt.style.use('paper_mpl_style.mplstyle')  # Use custom Matplotlib style (e.g. for paper plots)

# Experimental / physical parameters
tau = 100e-6            # Pulse duration (s)
k = 4*np.pi/780.226e-9  # Effective two-photon wavevector (1/m) for 780 nm light (Bragg/Raman)
g = 31e-3               # Effective acceleration (m/s^2), here 31 mm/s^2 (probably residual g)

# According to Phys. Rev. A 99, 033619 (2019)
def phase(T):
    """
    Compute the interferometer phase for a Mach--Zehnder-like 3-pulse sequence
    with finite-duration shaped pulses, following PRA 99, 033619.

    Parameters
    ----------
    T : float
        Interrogation time between pulses (s).

    Returns
    -------
    phi : float
        Total interferometer phase (rad) due to acceleration g.
    """

    # Dimensionless integrated pulse area F(t, tau) from our experiment (Blackman):
    # It encodes the time-dependent Rabi frequency envelope.
    def F(t, tau):
        return (0.42 * t
                - 0.5 * tau/(2*np.pi) * np.sin(2*np.pi*t/tau)
                + 0.08 * tau/(4*np.pi) * np.sin(4*np.pi*t/tau))

    # Effective Rabi frequencies for π/2 – π – π/2 pulses
    Omega_0 = np.pi/2 * 1/(0.42 * tau)  # π/2 pulse
    Omega_1 = np.pi   * 1/(0.42 * tau)  # π   pulse
    Omega_2 = np.pi/2 * 1/(0.42 * tau)  # π/2 pulse

    # detuning
    def detuning(t):
        return -k * g * t

    # First pulse: include sinusoidal coupling during the π/2 pulse window [-τ/2, τ/2]
    def integrand_0(t):
        return detuning(t) * np.sin(Omega_0 * F(t - (0*T - tau/2), tau))

    # Second pulse (π pulse) centered at time T:
    # pulse window [T - τ/2, T + τ/2], with appropriate time shift and phase offset π/2
    def integrand_1(t):
        return detuning(t)* np.sin(Omega_1 * F(t - (T - tau/2), tau) + np.pi/2)

    # Third pulse (π/2 pulse) centered at time 2T:
    # pulse window [2T - τ/2, 2T+- τ/2], with shift and phase offset 3π/2
    def integrand_2(t):
        return detuning(t) * np.sin(Omega_2 * F(t - (2*T - tau/2), tau) + 3*np.pi/2)

    # Integrate the contributions piecewise over the full interferometer sequence:
    # 1) First pulse (π/2))
    theta_0 = integrate.quad(integrand_0, -tau/2, tau/2)[0]

    # 2) Free evolution between first and second pulse
    #    Note the plus sign because of sin(1/2 pi) = +1
    theta_1 = +integrate.quad(detuning, tau/2, T - tau/2)[0]

    # 3) Second pulse (π)
    theta_2 = integrate.quad(integrand_1, T - tau/2, T + tau/2)[0]

    # 4) Free evolution between second and third pulse
    #    Note the minus sign because of sin(3/2 pi) = -1
    theta_3 = -integrate.quad(detuning, T + tau/2, 2*T - tau/2)[0]

    # 5) Third pulse (π/2)
    theta_4 = integrate.quad(integrand_2, 2*T - tau/2, 2*T + tau/2)[0]

    # Total interferometer phase = sum of all time segments
    phi = (theta_0 +
           theta_1 +
           theta_2 +
           theta_3 +
           theta_4)

    return phi


# Define interrogation times T between 1 ms and 3 ms (in ms for plotting)
times = np.linspace(1,3, 10000)  # times in ms (for x-axis)
phis = []

# Compute phase for each T (converted to seconds)
for t in times:
    phis.append(phase(t * 1e-3))


# Simple analytic estimate including leading finite-pulse correction:
# φ_est(t) = k g t^2 (1 + γ τ / t),
# where γ is fitted to the numerical result.
def estimated_phase(t, gamma):
    """
    Estimated interferometer phase for finite-duration pulses:
    φ ≈ k g t^2 (1 + γ τ / t).

    Parameters
    ----------
    t : float or array
        Interrogation time (s).
    gamma : float
        Dimensionless correction parameter to be fitted.

    Returns
    -------
    float or array
        Estimated phase (rad).
    """
    return k * g * t**2 * (1 + gamma * tau / t)


# Fit γ to the numerically calculated phases
gamma, _ = sc(estimated_phase, times * 1e-3, phis, p0=[0.1])

# Plot numerical phase vs interrogation time
plt.plot(times, phis, label='Numerical (finite pulses)')

# Plot fitted analytical approximation
plt.plot(times,
         estimated_phase(times * 1e-3, *gamma),
         ls=':',
         label=f'Estimated (fit γ = {gamma[0]:.3g})')

# Plot ideal t^2 scaling without finite-pulse correction
plt.plot(times,
         k * g * (times * 1e-3)**2,
         label='only $kgT^2$')

plt.xlabel(r'Interrogation time $T$ (ms)')
plt.ylabel(r'Phase $\phi$ (rad)')
plt.legend()
plt.tight_layout()
plt.show()

print(f"γ = {gamma[0]:.17g}")