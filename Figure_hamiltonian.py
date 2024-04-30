#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This file generates the figures of Hamiltonian

"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt

# %%
data_h = np.load('results/results_stochastic_harmonic.npz')
data_p = np.load('results/results_stochastic_pendulum.npz')
data_d = np.load('results/results_stochastic_Duffing.npz')
data_3 = np.load('results/results_stochastic_3dof.npz')
data_w = np.load('results/results_stochastic_wave.npz')
data_b = np.load('results/results_stochastic_Beam.npz')

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 24
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

# %%
""" Hamonic """
H_a_h = data_h['H_a']
H_i_h = data_h['H_i'][0]
label_h = np.linspace(0, 1, len(H_a_h))

""" Pendulum """
H_a_p = data_p['H_a']*-1
H_i_p = data_p['H_i'][0]*-1
label_p = np.linspace(0, 5, len(H_a_p))

""" Duffing """
H_a_d = data_d['H_a']
H_i_d = data_d['H_i'][0]
label_d = np.linspace(0, 1, len(H_a_d))

""" 3DOF """
H_a_3 = data_3['H_a']
H_i_3 = data_3['H_i']
label_3 = np.linspace(0, 1, len(H_a_3))

""" String """
H_a_w = data_w['H_a']
H_i_w = data_w['H_i']
label_w = np.linspace(0, 1, len(H_a_w))

""" Beam """
H_a_b = data_b['H_a']
H_i_b = data_b['H_i']
label_b = np.linspace(0, 2, len(H_a_b))

# %%
fig1 = plt.figure(figsize=(12,8))
plt.plot(label_h, H_a_h, linewidth=2, color='tab:blue', label='Harmonic-Truth')
plt.plot(label_h, H_i_h, linewidth=2, linestyle='--', color='blue', label='Harmonic-Discovered')

plt.plot(label_p, H_a_p, linewidth=2, color='tab:red', label='Pendulum-Truth')
plt.plot(label_p, H_i_p, linewidth=2, linestyle='--', color='red', label='Pendulum-Discovered')

plt.plot(label_d, H_a_d, linewidth=2, color='tab:green', label='Duffing-Truth')
plt.plot(label_d, H_i_d, linewidth=2, linestyle='--', color='green', label='Duffing-Discovered')

plt.plot(label_3, H_a_3, linewidth=2, color='tab:orange', label='3DOF SS-Truth')
plt.plot(label_3, H_i_3, linewidth=2, linestyle='--', color='orange', label='3DOF SS-Discovered')

plt.plot(label_w, H_a_w, linewidth=2, color='tab:pink', label='Wave-Truth')
plt.plot(label_w, H_i_w, linewidth=2, linestyle='--', color='pink', label='Wave-Discovered')

plt.plot(label_b, H_a_b, linewidth=2, color='tab:cyan', label='Beam-Truth')
plt.plot(label_b, H_i_b, linewidth=2, linestyle='--', color='cyan', label='Beam-Discovered')

plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian, $\mathcal{H}$(t)')
plt.xlim([0,1]); 
plt.ylim([1e0, 1e4])
plt.yscale('log')
plt.grid('True', alpha=0.30)
plt.legend(loc=4, bbox_to_anchor=(1.45,0.25), ncol=1, columnspacing=0.5, 
            borderpad=0.25, labelspacing=0.15, handlelength=1.5,
            handletextpad=0.25)
# plt.margins(0)

fig1.savefig('figures/Hamiltonian.pdf', format='pdf', dpi=600, bbox_inches='tight')

