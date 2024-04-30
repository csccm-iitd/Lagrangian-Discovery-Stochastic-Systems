#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This code is for "discovering Lagrangian of Stochastic Harmonic Oscillator" 
-- The data are generated using Stochastic Taylor strong order 1.5 scheme

"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym 
import seaborn as sns
import pandas as pd

np.random.seed(0)

# %%
""" Generating system response """

# The time and system parameters:
x0 = np.array([0.5, 0])
m, k, b = 1, 1000, 1
dt, T, Nsamp = 0.0001, 1, 200
t_param = [dt, T, Nsamp]
sys_param = [m, k, b]

# System responses:
xt, t_eval = utils_data.harmonic(x0, t_param, sys_param)
xt_mean = np.mean(xt, 0)

# %%
# Plot the responses:
fig1 = plt.figure(figsize=(10,6), dpi=300)

plt.subplot(2,1,1); 
plt.plot(t_eval, xt_mean[0, :], label='Displacement'); plt.ylabel('$x$(t)');
plt.grid(True); plt.legend(loc=1); plt.margins(0)
plt.subplot(2,1,2); 
plt.plot(t_eval, xt_mean[1, :], label='Velocity'); plt.ylabel('$\dot{x}$(t)');
plt.legend(loc=1); plt.xlabel('Time (s)'); plt.grid(True); plt.margins(0)
plt.suptitle('Stochastic Harmonic Oscillator', y=0.95)

# %%
""" Generating the design matrix """

# Define the symbolic library:
x, y = sym.symbols('x, xd')
xvar = [x, y]
D, nd = utils.library_sdof(xvar,polyn=5,harmonic=1)

# Obtain the numerical library:
# For demonstration, in this example, the Euler-Lagrange library is derived manually,
# For automation use,
# Rl, dxdt = utils.euler_lagrange_mdof(D,xvar,xt[sample, ...],dt)
Dxdx, Dydt = [], []
for sample in range(Nsamp):
    Dxdx_temp, Dydt_temp = utils.euler_lagrange(D, xvar, xt[sample, ...], dt)
    Dxdx.append(Dxdx_temp[0]), Dydt.append(Dydt_temp[0]) 

Dxdx, Dydt = np.array(Dxdx), np.array(Dydt)
Dxdx, Dydt = np.mean(Dxdx, axis=0), np.mean(Dydt, axis=0) 

# Find the Euler-Lagrange operator: 
Rl = Dydt - Dxdx

# Find the target vector from Euler-Lagrange equation:
dxdt = Rl[:, np.where(D == y**2)]
dxdt = dxdt.reshape(1, len(dxdt))

# Remove the target vector from Euler-Lagrange library:
Rl = np.delete(-1*Rl, np.where(D == y**2), axis=1)

# %%
""" Sparse regression for the System/Drift identification """

# compute Sparse regression: sequential least squares
lam = 400      # lam is our sparsification knob.
Xi = utils.sparsifyDynamics(Rl,dxdt,lam)

# Now insert the target vector back to the Euler-Lagrange library:
Xi = np.insert(Xi, np.where(D == y**2)[0], 1, axis=0)
Theta = pd.DataFrame( {'Basis': D, 'Theta':Xi[:,0]} )
print('The identified parameter vector...\n', Theta)

# %%
""" Print the Lagrangian and Hamiltonian """

""" Lagrangian """
L = sym.Array(0.5*np.dot(D,Xi))

""" Hamiltonian """
H = (sym.diff(L, y)*y - L)      # The Legendre transformation,
print('Lagrangian: %s,\nHamiltonian: %s' % (L, H))

# Create functions for Identified system:
Lfun = sym.lambdify([x,y], L, 'numpy') 
Hfun = sym.lambdify([x,y], H, 'numpy')

# %%
""" Sparse regression for the identification of volatility constant"""
# Numerically evaluate the identified Lagrangian, 
L_drift = []
for i in range(Nsamp):
    L_drift.append( Lfun(xt[i,0,:], xt[i,1,:])**2 )
L_drift = np.mean( np.array(L_drift), axis=0 )

# compute Sparse regression: for diffusion
D_stoch = []
for sample in range(Nsamp):
    D_stoch.append( utils.library_diffusion( xt[sample, ...] ) )
D_stoch = np.mean( np.array(D_stoch), axis=0 )

beta = utils.sparsifyDynamics(library=D_stoch, target=(dt*L_drift), lam=0.5)
beta_exact = 0.5*np.sqrt(beta)

X = sym.Symbol('X', commutative=False)
Xd = sym.Symbol('Xd', commutative=False)
xvar = [X, Xd]
D_diffusion = utils.library_visualize(xvar)
Beta = pd.DataFrame( {'Basis':D_diffusion, 'Beta':beta_exact[:,0]} )
print('The identified Stochastic parameter vector...\n', Beta)

# %%
""" Hamiltonian time series """
H_a = 0.5*xt_mean[1,:]**2 + 0.5*(k/m)*xt_mean[0,:]**2 
H_i = Hfun(xt_mean[0], xt_mean[1]) 

""" Lagrangian time series """
L_a = 0.5*xt_mean[1,:]**2 - 0.5*(k/m)*xt_mean[0,:]**2 
L_i = Lfun(xt_mean[0,:], xt_mean[1,:]) 

print('Percentage Hamiltonian error: %0.4f' % (100*np.linalg.norm(H_a-H_i)/np.linalg.norm(H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.linalg.norm(L_a-L_i)/np.linalg.norm(L_a)) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(10,6))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i[0], 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.title('Stochastic Harmonic Oscillator')
plt.ylim([120,130])
plt.grid(True)
plt.legend()
plt.margins(0)

# %%
""" Plotting the theta """
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 16

Xi_plot_drift = np.array(Xi.T)
Xi_plot_drift[Xi_plot_drift != 0] = -1
Xi_plot_diffusion = np.array(beta_exact.T)
Xi_plot_diffusion[Xi_plot_diffusion != 0] = -1

fig3, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,3))
plt.subplots_adjust(hspace=6) 
plt.suptitle('Stochastic Harmonic Oscillator', y=1.1)

ax1 = sns.heatmap( Xi_plot_drift, linewidth=0.5, cmap='Set1', ax=ax[0], cbar=False, yticklabels=[0])
ax1.set_xlabel('Library functions')
ax1.set_ylabel('Drift')
ax1.set_title('(a) Lagrangian for Drift', y=1)
ax1.set_xticklabels(D, rotation=45)

ax2 = sns.heatmap( Xi_plot_diffusion, linewidth=0.5, cmap='Set1', ax=ax[1], cbar=False, yticklabels=[0])
ax2.set_xlabel('Library functions')
ax2.set_ylabel('Diffusion')
ax2.set_title('(b) Lagrangian for Diffusion', y=1)
ax2.set_xticklabels(D_diffusion, rotation=45)

# %% Save results/figures, if required:
# np.savez('results/review_results_stochastic_harmonic_n'+str(Nsamp), Xi=Xi, 
#           beta_exact=beta_exact, t_eval=t_eval, H_a=H_a, H_i=H_i, L_a=L_a, L_i=L_i)
