# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 15:35:14 2025

Author: Dominik Pfeiffer

Description:
    This script imports interferometer experimental data, visualizes the dependence
    of interferometric signals on laser phase for selected interferometer times,
    plots state correlations (S- vs. S+), and shows histogram-based amplitude
    evolution over time.
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit as c
from matplotlib.patches import Ellipse, FancyArrowPatch as FAP, ConnectionPatch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib as mpl

import os
import sys

sys.path.append(os.path.abspath('..'))

plt.style.use('../paper_mpl_style.mplstyle')# Apply paper-style plotting template

save_fig = False

# Local helper functions
from helper_functions import  Amp_sum

# =============================================================================
# Global Style and Setup
# =============================================================================


# Define interferometer times (ms) and phases (rad)
times = np.linspace(1, 3, 21)
times_fine = np.linspace(0.95, 3.05, 201)
phase = np.linspace(0, 1, 20, endpoint=False) * 2 * np.pi

# Auxiliary parameters
size = 100
bins = int(np.ceil(np.sqrt(300)))

# Construct laser phase array replicated for all experimental shots
phases = (np.ones((15, len(phase))) * phase).ravel()

# Colours
colour_hist_sum = 'C1'
colour_hist_plus = 'C4'
colour_hist_minus = 'C5'

# =============================================================================
# Data Import
# =============================================================================
# Load data arrays for spin states and sums from bootstrapping results
signal_m1 = np.load('../exp_data/Coarse/signal_m1.npy').swapaxes(0,1)
signal_p1 = np.load('../exp_data/Coarse/signal_p1.npy').swapaxes(0,1)

signal_sum = signal_m1 * np.cos(np.pi/4) + signal_p1 * np.sin(np.pi/4)

amplitude_results = np.load('../exp_eval/amplitude_sum_results.npy')


# =============================================================================
# Figure and Axes Layout
# =============================================================================
fig, axs = plt.subplots(
    3, 4, sharey=False,
    figsize=(3.54, 3.54),
    height_ratios=(0.25, 0.3, 0.4)
)

size = 0.75      # Marker size for plotted data points
line_width = 0.75  # Line width for overlaid curves

# =============================================================================
# Top Row: Phase-dependent Interferometer Signals
# =============================================================================
# Display fringe scans at selected interferometer times to show phase evolution
for col, idx in enumerate([0, 4, 7, 15]):
    axs[0, col].plot(phase, signal_m1.mean(axis=1)[idx],
                     marker='o', color=colour_hist_minus, ls='None', markersize=size)
    axs[0, col].plot(phase, signal_p1.mean(axis=1)[idx],
                     marker='o', color=colour_hist_plus, ls='None', markersize=size)
    axs[0, col].plot(phase, signal_sum[idx].mean(axis=0),
                     marker='o', color=colour_hist_sum, ls='None', markersize=size)
    axs[0, col].set_xticks([])
    axs[0, col].xaxis.set_minor_locator(MultipleLocator(phase[1]))
    axs[0, col].set_aspect(phase.max()/(2*abs(signal_sum[15]).max()))
    axs[0, col].set_ylim(-abs(signal_sum[15]).max(),abs(signal_sum[15]).max())

# Apply gradient-colored minor ticks to all four subplots in the first row
norm = plt.Normalize(phase.min(), phase.max())
cmap = plt.cm.jet
labels = [r'$i$',r'$ii$',r'$iii$',r'$iv$']
for col in range(4):
    axs[0, col].tick_params(axis='x', which='both', width=3, length=4,direction='in')
    xticks = axs[0, col].get_xticks(minor=True)
    axs[0,col].text(0,0.95,labels[col])

    for tick, loc in zip(axs[0, col].xaxis.get_minor_ticks(), xticks):
        colr = cmap(norm(loc))
        tick.tick1line.set_markeredgecolor(colr)
        tick.tick2line.set_markeredgecolor(colr)
    if col>0:
        axs[0,col].set_yticks([])
    axs[0,col].text(0,-abs(signal_sum[15]).max()-0.45,'0',ha='center')
    axs[0,col].text(np.pi,-abs(signal_sum[15]).max()-0.45,'1',ha='center')
axs[0,col].text(2*np.pi,-abs(signal_sum[15]).max()-0.45,'2',ha='center')
axs[0, 0].set_ylabel(r'$S_{\pm1}$, $S_\text{sum}$',labelpad=-3)

# =============================================================================
# Middle Row: amplitude Temporal Evolution
# =============================================================================
# Merge bottom row into a single subplot representing time-varying signal distributions
gs = axs[1, 0].get_gridspec()
for a in axs[1, :]:
    a.remove()
ax_1 = fig.add_subplot(gs[1, :])


# Create 2D histogram mesh (pseudocolor map) showing population vs. time
for i, t in enumerate(times):
    y_sum = signal_sum[i]
    v_max_sum = y_sum.max()
    bins_sum = max(np.sqrt(2), int(bins * np.sqrt(2) / v_max_sum))  # Ensure valid count
    hist_sum, x_sum = np.histogram(y_sum, bins=bins_sum, range=(-np.sqrt(2), np.sqrt(2)), density=True)

    X = np.array([t - np.diff(times)[0] / 2, t + np.diff(times)[0] / 2])
    Y = x_sum
    Z = hist_sum[:, np.newaxis]

    ax_1.pcolormesh(X, Y, Z, rasterized=True)

# Overlay analytical amplitude fits (upper and lower mirror)
for sign in [1, -1]:
    ax_1.plot(
        times_fine,
        sign * Amp_sum(times_fine * 1e-3, *amplitude_results.mean(axis=0)),
        color=colour_hist_sum, lw=line_width
    )

# Add vertical markers for example time points
ax_1.vlines([1, 1.4, 1.7, 2.5], -np.sqrt(2), np.sqrt(2), ls='--', lw=0.75, color='w')

# Axis labels and formatting
ax_1.set_xlabel(r'$T$ (ms)')
ax_1.set_ylabel(r'$S_\text{sum}$',labelpad=-3)
ax_1.set_xticks([1,1.5,2,2.5,3])
ax_1.xaxis.set_minor_locator(MultipleLocator(0.1))


# =============================================================================
# Bottom Row: Scatter Correlations of S- vs. S+
# =============================================================================
# Each subplot: Scatter plot of S- vs. S+ at different interferometer times
limit = 1.2
indices = [0, 4, 7, 15]
locs = [-1*limit/1.1,1*limit/1.1,1*limit/1.1,-1*limit/1.1]
aligns = ['left', 'right', 'right', 'left']
for i, ind in enumerate(indices):
    im = axs[2, i].scatter(
        signal_m1[ind], signal_p1[ind],
        c=phases / np.pi, cmap='jet', s=size
    )
    axs[2, i].set_facecolor('grey')
    axs[2, i].set_xlim(-limit, limit)
    axs[2, i].set_ylim(-limit, limit)
    axs[2, i].set_aspect(1)
    axs[2, i].set_yticks([-1, 0, 1], [-1, 0, 1])
    axs[2, i].set_xticks([-1, 0, 1], [-1, r'$S_{-1}$', 1])
    axs[2,i].text(locs[i],1*limit/1.1,labels[i],c='w',ha=aligns[i],va='top')
    if i>0:
        axs[2,i].set_yticks([])
    

# Label axes and add colorbar to indicate relative laser phase Φ_L
axs[2, 0].set_ylabel(r'$S_{+1}$',labelpad=-3)
# cbar_ax = fig.add_axes([0.89, 0.59, 0.02, 0.25])
# cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label=r'$\Phi_\mathrm{L}$')

# =============================================================================
# Layout and Output
# =============================================================================
fig.align_ylabels([axs[0,0],axs[2,0],ax_1])

plt.subplots_adjust(bottom=0.15, 
                    top=0.99, 
                    left=0.1, 
                    right=0.985, 
                    wspace=0.0, 
                    hspace=0.15)

pos1 = axs[0, 0].get_position()
pos3 = ax_1.get_position()
ax_1.set_position([pos3.x0, pos3.y0 - 0.1, pos3.width,1.25*pos3.height])

#Shift second row downward to increase spacing
for ax in axs[2, :]:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - 0.125, box.width, box.height])
    

fig.text(0.55,0.74,r'$\phi_\text{las}/\pi$',va='center',ha='center')
fig.text(0.0,0.99,r'A',va='top',ha='left',fontweight='bold')
fig.text(0.0,0.715,r'B',va='top',ha='left',fontweight='bold')
fig.text(0.0,0.285,r'C',va='center',ha='left',fontweight='bold')

fig.text(0.13,0.705,r'$i$',va='top',ha='left',c='w')
fig.text(0.3,0.705,r'$ii$',va='top',ha='left',c='w')
fig.text(0.42,0.705,r'$iii$',va='top',ha='left',c='w')
fig.text(0.76,0.705,r'$iv$',va='top',ha='left',c='w')

if save_fig:
    plt.savefig('fig2.pdf')

plt.show()
