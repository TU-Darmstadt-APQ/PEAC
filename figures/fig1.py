# -*- coding: utf-8 -*-
"""
Created on Thu Okc 09 15:35:14 2025

@author: Dominik Pfeiffer
"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit as c 
from matplotlib.patches import Ellipse
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch as FAP
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from curlyBrace import curlyBrace

import os
import sys

sys.path.append(os.path.abspath('..'))
from helper_functions import Amp_all

plt.style.use('../paper_mpl_style.mplstyle')

save_fig = False

"""
Data import, processing, and visualization for interferometer analysis.
This code loads experimental datasets, processes them, applies masks, 
and visualizes the interferometer sequence with custom color maps 
and annotated insets representing atomic states.
"""

# =============================================================================
# Global Style and Setup
# =============================================================================

# Define interferometer times (ms) and phases (rad)
times = np.linspace(1, 3, 21)
times_fine = np.linspace(0.95, 3.05, 201)
phase = np.linspace(0, 1, 21, endpoint=True) * 2 * np.pi

# colours
colour_hist_plus = 'C4'
colour_hist_minus = 'C5'
colour_hist_zero = 'C6'

# =============================================================================
# Data Import
# =============================================================================
# Interferometer signals (bootstrapped results)
delta_N_0 = np.load('../exp_data/Coarse/signal_0.npy').swapaxes(0,1)
delta_N_m1 = np.load('../exp_data/Coarse/signal_m1.npy').swapaxes(0,1)
delta_N_p1 = np.load('../exp_data/Coarse/signal_p1.npy').swapaxes(0,1)

S_all = np.load('../exp_data/Coarse/signal_all.npy').swapaxes(0,1)

# Fringe scan and no-Stern–Gerlach datasets
fringe_data = np.load('../exp_data/Coarse/Experimental_schematics/250410_190540_.npy')
combined_data = np.load('../exp_data/Coarse/Experimental_schematics/250117_172609_.npy')


# Histogram results
amplitude_results = np.load('../exp_eval/amplitude_all_results.npy')
# =============================================================================
# Data Masking
# =============================================================================
# Preprocessing for upper and lower SG signal channels
upper_SG_data = fringe_data.mean(axis=0)[0, 5, 0]
lower_SG_data = fringe_data.mean(axis=0)[0, 5, 0]

# SG-free version of datasets
upper_wo_SG = combined_data.mean(axis=(0, 3))[0, 0, :90, 20:80]
lower_wo_SG = combined_data.mean(axis=(0, 3))[0, 0, :90, 20:80]

# Apply spatial masks to isolate relevant regions
for i in range(len(upper_SG_data)):
    upper_SG_data[199 - i:, i] = 0
    lower_SG_data[:199 - i, i] = 0

upper_wo_SG[46:] = 0
lower_wo_SG[:46] = 0

# =============================================================================
# Colormap Definitions
# =============================================================================
# Define custom semi-transparent purple and green maps for overlay visualizations
purple_rgba = to_rgba('tab:purple', alpha=1.0)
transparent_purple = to_rgba('tab:purple', alpha=0.0)
green_rgba = to_rgba('tab:green', alpha=1.0)
transparent_green = to_rgba('tab:green', alpha=0.0)

custom_purple = LinearSegmentedColormap.from_list('transparent2purple', [transparent_purple, purple_rgba])
custom_green = LinearSegmentedColormap.from_list('transparent2green', [transparent_green, green_rgba])

# =============================================================================
# Helper Function
# =============================================================================
def blackman_pulse(t, tau, omega):
    """
    Generate a Blackman pulse envelope.
    t: time array
    tau: pulse duration
    omega: angular frequency (unused parameter in this version)
    """
    return 0.42 - 0.5 * np.cos(2 * np.pi * t / tau) + 0.08 * np.cos(4 * np.pi * t / tau)

# =============================================================================
# Plot Setup
# =============================================================================
time = np.linspace(0, 1, 1000)
fig, axs = plt.subplots(3,6, 
                        figsize=(3.54, 5.31), 
                        width_ratios=(0.04, 0.09, 0.6, 0.09, 0.09, 0.09),
                        sharey='row')

gs = axs[0, 0].get_gridspec()
for a in axs[0, :]:
    a.remove()
ax_1 = fig.add_axes([0.05, 0.75, 0.95, 0.24])

# Bragg pulse sequence and Stern–Gerlach visualization
ax_1.fill_between(time + 3, blackman_pulse(time, 1, 100) * 0.5, color='tab:blue', alpha=0.5)
ax_1.fill_between(time + 7, blackman_pulse(time, 1, 100) * 0.75, color='tab:blue', alpha=0.5)
ax_1.fill_between(time + 11, blackman_pulse(time, 1, 100) * 0.5, color='tab:blue', alpha=0.5, label=r'Bragg pulses')
ax_1.fill_between([14, 15, 17], [00.75, 0.75, 0.75], color='tab:orange', label=r'Stern–Gerlach current')

# Annotated phase and timing markers
p1 = FAP((3.4, 0.1), (7.6, 0.1), arrowstyle='<|-|>', color='black', mutation_scale=10, zorder=3)
p2 = FAP((11.6, 0.1), (7.4, 0.1), arrowstyle='<|-|>', color='black', mutation_scale=10, zorder=3)
ax_1.add_patch(p1)
ax_1.add_patch(p2)

ax_1.text(5.6, 0.12, r'$T$', ha='center')
ax_1.text(9.6, 0.12, r'$T$', ha='center')
ax_1.text(11.5, 0.03, r'$\phi_\mathrm{las}$', ha='center',rotation=90)

# Styling and line parameters
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.solid_capstyle'] = 'round'

# Atom trajectories showing different momentum components
ax_1.plot([0, 3.5], [0.2, 0.2], color='purple')
ax_1.plot([3.5, 7.5], [0.2, 0.2], color='purple')
ax_1.plot([3.5, 7.5], [0.2, 0.4], color='green')
ax_1.plot([7.5, 11.5], [0.2, 0.4], color='green', label=r'$n\hbar k_\mathrm{eff}$')
ax_1.plot([7.5, 11.5], [0.4, 0.4], color='purple', label=r'$0\hbar k_\mathrm{eff}$')

ax_1.plot([11.5,14.25],[0.4,0.4],ls=':',color='purple')
ax_1.plot([14.25,17],[0.4,0.4],ls=':',color='purple')

ax_1.plot(np.linspace(14.25,17,10),
         np.linspace(0,2.75,10)**2*0.01+0.4, 
         ls=':', 
         color='purple')

ax_1.plot(np.linspace(14.25,17,10),
         -np.linspace(0,2.75,10)**2*0.01+0.4, 
         ls=':', 
         color='purple')

ax_1.plot([11.5,14.25],[0.4,0.5375],ls=':',color='green')
ax_1.plot([14.25,17],[0.5375,0.675],ls=':',color='green')

ax_1.plot(np.linspace(14.25,17,10),
         (np.linspace(0,2.75,10)**2*0.01+
          np.linspace(0,2.75,10)*0.05 + 
          0.5375), 
         ls=':',color='green')

ax_1.plot(np.linspace(14.25,17,10), 
         (-np.linspace(0,2.75,10)**2*0.01+
          np.linspace(0,2.75,10)*0.05 + 
          0.5375), 
         ls=':',color='green')

# =============================================================================
# Inset Plots
# =============================================================================
# ax2 = axs.twinx()
# ax2.set_yticks([])
padding = 55
# Insets with and without Stern–Gerlach effect
inset = ax_1.inset_axes([0.625, 0.24, 0.55,0.8], zorder=1)
inset.vlines([47, 94], 0, 199 + padding , color='r', ls='--', lw=0.5)
inset.imshow(np.pad(upper_SG_data[:, 26:167],((0,padding),(0,0))), cmap=custom_green, vmin=5e-3)
inset.imshow(np.pad(lower_SG_data[:, 26:167],((padding,0),(0,0))), cmap=custom_purple, vmin=5e-3)
inset.set_xticks([])
inset.set_yticks([])
inset.spines[:].set_visible(False)

inset2 = ax_1.inset_axes([0.524, 0.425, 0.2, 0.2])
inset2.imshow(upper_wo_SG, cmap=custom_green, vmin=5e-3)
inset2.imshow(lower_wo_SG, cmap=custom_purple, vmin=5e-3)
inset2.set_xticks([])
inset2.set_yticks([])
inset2.spines[:].set_visible(False)

# Annotate momentum branches with curly braces
for color, coords, label in zip([colour_hist_minus, colour_hist_zero, colour_hist_plus], [(0, 199 + padding), (47, 199 + padding), (93, 199 + padding)], [r'$-1$', r'$0$', r'$+1$']):
    theta, summit, *_ = curlyBrace(fig=fig, ax=inset, p1=coords, p2=(coords[0] + 46, 199 + padding),
                                   k_r=0.15, bool_auto=False, str_text='', color=color, lw=0.5)
    inset.text(summit[0], summit[1], label, ha='center', va='top', fontsize=6, color=color)


# =============================================================================
# Modify ticks, spines and size of plot
# =============================================================================

# plt.subplots_adjust(left=0, right=1, top=0.98, bottom=0.12)

ax_1.set_xticks([3.5, 7.5, 11.5, 15.5, 19], [r'$\pi\,/\,2$',r'$\pi$',r'$\pi\,/\,2$',r'SG & TOF ','Detection'])
ax_1.tick_params(axis='x', which='both', tick1On=False, tick2On=False)
ax_1.set_yticks([])
ax_1.set_xlim(-0.2, 21)
ax_1.set_ylim(0, 0.85)

inset.set_ylim(top=0, bottom=199+padding)
ax_1.spines[:].set_visible(False)
# ax2.spines[:].set_visible(False)


"""
Plot interferometer signals for different m_F states and visualize their distributions.
This script plots the mean interferometer phase responses for each magnetic sublevel (m_F),
along with histograms showing their distributions and a combined pseudo-color mesh.
"""

# =============================================================================
# Configuration and Parameters
# =============================================================================
plot_index = 6  # Index selecting which dataset slice to visualize
phase = np.linspace(0, 2 * np.pi, 21, endpoint=True)  # Phase array for plotting (0 to 2π)

# =============================================================================
# Figure and Style Setup
# =============================================================================

# Customize general plot aesthetics

mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['lines.linewidth'] = 0.25

# =============================================================================
# Plot Mean Interferometer Signals
# =============================================================================
# m_F = -1 state
axs[1,2].plot(phase[:20] / (np.pi), delta_N_m1.mean(axis=1)[plot_index],
            color=colour_hist_minus, ls='--', marker='X', label=r'$S_{-1}$')

# m_F = 0 state
axs[1,2].plot(phase[:20] / (np.pi), delta_N_0.mean(axis=1)[plot_index],
            color=colour_hist_zero, ls='--', marker='o', label=r'$S_{0}$')

# m_F = +1 state
axs[1,2].plot(phase[:20] / (np.pi), delta_N_p1.mean(axis=1)[plot_index],
            color=colour_hist_plus, ls='--', marker='P', label=r'$S_{+1}$')

# Combined ±1 states (coherent sum)
axs[1,2].plot(phase[:20] / (np.pi),
            S_all.mean(axis=1)[plot_index],
            color='k', ls='--', marker='s', label=r'$S_\text{all}$',zorder=-1)


axs[1,2].legend(loc='center', 
              ncols=4, 
              borderpad=0.15, 
              handletextpad = 0.1, 
              handlelength = 0.75, 
              bbox_to_anchor = (0.5,0.93), 
              columnspacing = 1
              )

axs[1,2].hlines(0,0,2*0.95,lw=0.75,ls='dashed',color='k',zorder=-1)

# =============================================================================
# Histograms for Each Magnetic Sublevel
# =============================================================================
# Create histograms to display signal distributions for each population
axs[1,3].hist(delta_N_m1[plot_index].ravel(),
            color=colour_hist_minus,
            bins=int(1 / abs(delta_N_m1[plot_index].ravel()).max() * np.sqrt(len(delta_N_m1[plot_index].ravel()))),
            range=(-1, 1),
            orientation='horizontal',
            density=True)

axs[1,4].hist(delta_N_0[plot_index].ravel(),
            color=colour_hist_zero,
            bins=int(1 / abs(delta_N_0[plot_index].ravel()).max() * np.sqrt(len(delta_N_0[plot_index].ravel()))),
            range=(-1, 1),
            orientation='horizontal',
            density=True)

axs[1,5].hist(delta_N_p1[plot_index].ravel(),
            color=colour_hist_plus,
            bins=int(1 / abs(delta_N_p1[plot_index].ravel()).max() * np.sqrt(len(delta_N_p1[plot_index].ravel()))),
            range=(-1, 1),
            orientation='horizontal',
            density=True)
    
# =============================================================================
# Combined Histogram of Superposed ±1 States
# =============================================================================
# Plot histogram of the normalized combination signal
N, bins, patches = axs[1,1].hist(S_all[plot_index].ravel(),
                               color='k',
                               bins=int(1 / abs(S_all[plot_index].ravel()).max()
                                        * np.sqrt(len(S_all[plot_index].ravel()))),
                               range=(-1, 1),
                               orientation='horizontal',
                               density=True)

# # Normalize frequencies to color-code histogram bins
# fracs = N / N.max()
# norm = colors.Normalize(fracs.min(), fracs.max())

# # Apply a colormap gradient (Viridis) to bins based on frequency
# for frac, patch in zip(fracs, patches):
#     patch.set_facecolor(plt.cm.viridis(norm(frac)))

# =============================================================================
# Pseudo-Color Mesh for Distribution Overview
# =============================================================================
# 2D representation of combined state distribution
hist_sum, x_sum = np.histogram(S_all[plot_index].ravel(),
                               bins=int(1 / abs(S_all[plot_index].ravel()).max()
                                        * np.sqrt(len(S_all[plot_index].ravel()))),
                               range=(-1, 1),
                               density=True)

X = np.array([0, 1])
Y = x_sum
Z = hist_sum[:, np.newaxis]

# Rasterized color mesh (improves performance for dense data)
axs[1,0].pcolormesh(X, Y, Z, rasterized=True)

# =============================================================================
# Axis Customization
# =============================================================================
# Set consistent vertical and horizontal limits
axs[1,2].set_ylim(-1, 1)
# axs[1,0].set_xlim(right=1)
axs[1,2].set_xticks([0,2],['$0$',r'$2$'])
axs[1,2].xaxis.set_minor_locator(MultipleLocator(phase[1] / (np.pi)))

# Color-coded tick marks to visualize phase cycles
xticks = axs[1,2].get_xticks(minor=True)
# axs[1,0].tick_params(axis='x', which='minor', width=6.25, length=6, direction='in')

# norm = plt.Normalize(phase.min() / (2 * np.pi), phase.max() / (2 * np.pi))
# cmap = plt.cm.jet

# for tick, loc in zip(axs[1,0].xaxis.get_minor_ticks(), xticks):
    # col = cmap(norm(loc))
    # tick.tick1line.set_markeredgecolor(col)
    # tick.tick2line.set_markeredgecolor(col)

axs[1,2].xaxis.remove_overlapping_locs = False

for a in axs[1,:2]:
    a.set_xticks([])
    
for a in axs[1,3:]:
    a.set_xticks([])
    
axs[1,0].set_ylabel(r'$S_{m_F}$, $S_\text{all}$',labelpad=-5)
axs[1,2].set_xlabel(r'$\phi_\text{las}/\pi$',labelpad=-8)
# =============================================================================
# Layout and Output
# =============================================================================

# Annotate momentum branches with curly braces
for axis, color, label in zip([axs[1,3],axs[1,4],axs[1,5],axs[1,1]], 
                              [colour_hist_minus, colour_hist_zero, colour_hist_plus, 'k'], 
                              [r'$S_{-1}$', r'$S_{0\quad}$', r'$S_{+1}$',r'$S_\text{all}$']
                              ):
    theta, summit, *_ = curlyBrace(fig=fig, ax=axis, p1=(axis.get_xlim()[1],axis.get_ylim()[0]), p2=(axis.get_xlim()[0],axis.get_ylim()[0]),
                                   k_r=0.1, bool_auto=True, str_text='', color=color, lw=0.5)
    axis.text(summit[0], summit[1], label, ha='center', va='top', fontsize=6, color=color)


# =============================================================================
# Pseudocolor Population Distribution Map
# =============================================================================
gs = axs[2, 0].get_gridspec()
for a in axs[2, :]:
    a.remove()
ax_3 = fig.add_subplot(gs[2, :])

# Build a 2D time-resolved histogram of the population signal.
# Each iteration adds a vertical "stripe" representing the population distribution at time t.
for i, t in enumerate(times):
    y_all = S_all[i].ravel()                         # Flatten 2D data array for histogram computation
    v_max_all = y_all.max()                          # Determine maximum value for scaling reference
    bins_all = int(np.sqrt(len(y_all)) * (1 / abs(y_all).max()))  # Dynamically set bin count based on data size

    # Compute normalized histogram of the population distribution within [-1, 1]
    hist_all, x_all = np.histogram(y_all, bins=bins_all, range=(-1, 1), density=True)

    # Construct rectangular grid coordinates for the pseudocolor mesh
    X = np.array([t - np.diff(times)[0] / 2, t + np.diff(times)[0] / 2])  # Horizontal (time) bounds per slice
    Y = x_all                                                             # Vertical data range (signal axis)
    Z = hist_all[:, np.newaxis]                                           # Convert histogram to column vector

    # Render the 2D color mesh (rasterized for efficient export)
    ax_3.pcolormesh(X, Y, Z, rasterized=True)

# =============================================================================
# Overlay Analytical Visibility Fit Curves
# =============================================================================
# Display the theoretical amplitude envelope from the analytic model (upper and lower bounds)



ax_3.plot(times_fine,
        Amp_all(times_fine * 1e-3,
                     *amplitude_results.mean(axis=0)),
         color='k', lw=0.75)

ax_3.plot(times_fine,
           -Amp_all(times_fine * 1e-3,
                        *amplitude_results.mean(axis=0)),
         color='k', lw=0.75)

# =============================================================================
# Add Vertical Markers for Example Time Points
# =============================================================================
# Mark specific experimental time index with a vertical dashed line for emphasis
ax_3.vlines(times[plot_index], -1, 1, ls='--', lw=0.75, color='w')

# =============================================================================
# Axis Labels and Formatting
# =============================================================================
# Label axes and configure minor tick spacing
ax_3.set_xlabel(r'$T$ (ms)')
ax_3.set_ylabel(r'$S_\text{all}$', labelpad=-3)
ax_3.set_xticks([1,1.5,2,2.5,3])
ax_3.xaxis.set_minor_locator(MultipleLocator(0.1))

# Optimize figure layout and spacing
plt.subplots_adjust(left=0.13,
                    right=0.99,
                    bottom=0.075,
                    top=1.025,
                    wspace=0, 
                    hspace=0.2)

for ax in axs[2, :]:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - 0.025, box.width, box.height])

fig.text(0,0.98,'A',fontweight='bold',)
fig.text(0,0.6815,'B',fontweight='bold',)
fig.text(0,0.3465,'C',fontweight='bold',)
# =============================================================================
# Export and Display Figure
# =============================================================================
# Save the figure for publication and display it on screen
if save_fig:
    plt.savefig('fig1.pdf')

plt.show()
