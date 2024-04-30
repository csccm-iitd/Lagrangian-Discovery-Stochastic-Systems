#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- This code is for "discovering Lagrangian of Stochastic 3DOF Oscillator" 
-- The data are generated using Stochastic Taylor strong order 1.5 scheme

"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym
import pandas as pd
import seaborn as sns

np.random.seed(0)

# %%
""" Generating system response """

# The time and system parameters:
x0 = np.array([0.25, 0, 0.5, 0, 0, 0])
m1, m2, m3, k1, k2, k3, b1, b2, b3 = 10, 10, 10, 10000, 10000, 10000, 1, 1, 1 
dt, T, Nsamp = 0.0001, 1, 200
t_param = [dt, T, Nsamp]
sys_param = [m1, m2, m3, k1, k2, k3, b1, b2, b3]

# System responses:
xt, t_eval = utils_data.mdof_system_stoschastic(x0, t_param, sys_param)
xt_mean = np.mean(xt, 0)

# %%
fig1 = plt.figure(figsize=(10,10))
fig1.subplots_adjust(hspace=0.5)
plt.subplot(6,1,1); plt.plot(t_eval, xt_mean[0,:], 'r'); plt.ylabel('$x_1$(t)');
plt.grid(True); plt.margins(0)
plt.subplot(6,1,2); plt.plot(t_eval, xt_mean[1,:]); plt.ylabel('$\dot{x}_1$(t)');
plt.grid(True); plt.margins(0)
plt.subplot(6,1,3); plt.plot(t_eval, xt_mean[2,:], 'r'); plt.ylabel('$x_2$(t)');
plt.grid(True); plt.margins(0)
plt.subplot(6,1,4); plt.plot(t_eval, xt_mean[3,:]); plt.ylabel('$\dot{x}_2$(t)');
plt.grid(True); plt.margins(0)
plt.subplot(6,1,5); plt.plot(t_eval, xt_mean[4,:], 'r'); plt.ylabel('$x_3$(t)');
plt.grid(True); plt.margins(0)
plt.subplot(6,1,6); plt.plot(t_eval, xt_mean[5,:]); plt.ylabel('$\dot{x}_3$(t)');
plt.xlabel('Time (t)'); plt.grid(True); plt.margins(0)

# %%
""" Generating the design matrix """

# Define the symbolic library:
xvar = [sym.symbols('x'+str(i)) for i in range(1, 6+1)]
D, nd = utils.library_mdof(xvar, polyn=3, funofvelocity=1, harmonic=1)

# Obtain the numerical library:
Rl, dxdt = [], []
for sample in range(Nsamp):
    Rl_temp, dxdt_temp = utils.euler_lagrange_library(D, xvar, xt[sample, ...], dt)
    Rl.append(Rl_temp), dxdt.append(dxdt_temp) 

Rl, dxdt = np.array(Rl), np.array(dxdt)
Rl, dxdt = np.mean(Rl, axis=0), np.mean(dxdt, axis=0) 

# %%
""" Sparse regression: sequential least squares """
Xi = []
lam = 500    # lam is the sparsification constant
for i in range(3):
    Xi.append(utils.sparsifyDynamics(Rl[i], dxdt[i:i+1], lam))
Xi = (np.array(Xi).squeeze(-1)).T

xvar_vel = xvar[1::2]
Xi_final = np.zeros([nd, 3])
for i in range(3):
    Xi_final[:,i] = np.insert(Xi[:,i], np.where(D == xvar_vel[i]**2)[0], 1)

keys=['Theta-'+str(i) for i in range(3)]
keys.insert(0, 'Basis')
Theta = pd.DataFrame( dict(zip(keys, np.append(D[:,None], Xi_final, axis=1).T)) )
print('The identified parameter vector...\n', Theta)

# %%
""" Lagrangian """
xdot = xvar[1::2]
Xi_reduced = utils.nonequal_sum(np.array(Xi_final))
L = np.sum(sym.Array(0.5*np.dot(D,Xi_reduced)))

""" Hamiltonian """
H = 0
for i in range(len(xdot)):
    H += (sym.diff(L, xdot[i])*xdot[i])
H = H - L
print('Lagrangian: %s, Hamiltonian: %s' % (L, H))

Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

# %%
""" Sparse regression for the identification of volatility constant"""
# Numerically evaluate the identified Lagrangian,
L_drift = []
for i in range(Nsamp):
    L_drift.append( Lfun(xt[i,:,:])**2 )
L_drift = np.mean( np.array(L_drift), axis=0, keepdims=True )

# compute Sparse regression: for diffusion
D_stoch = []
for sample in range(Nsamp):
    D_stoch.append( utils.library_diffusion( xt[sample, ...], polyn=3, tpp=False ) )
D_stoch = np.mean( np.array(D_stoch), axis=0 )

# %%
beta = utils.sparsifyDynamics(library=D_stoch, target=(dt*L_drift), lam=2.5)
beta_exact = 0.5*np.sqrt(beta)

Beta = pd.DataFrame(beta_exact, columns=['Beta'])
print('The identified Stochastic parameter vector...\n', Beta)

# %%
""" Hamiltonian """
H_a = (0.5*xt_mean[1,:]**2 + 0.5*xt_mean[3,:]**2 + 0.5*xt_mean[5,:]**2) + \
      (0.5*(k1/m1)*xt_mean[0,:]**2 + 0.5*(k2/m2)*(xt_mean[2,:]-xt_mean[0,:])**2 + \
       0.5*(k3/m3)*(xt_mean[4,:]-xt_mean[2,:])**2)
H_i = Hfun(xt_mean)
        
L_a = (0.5*xt_mean[1,:]**2 + 0.5*xt_mean[3,:]**2 + 0.5*xt_mean[5,:]**2) - \
      (0.5*(k1/m1)*xt_mean[0,:]**2 + 0.5*(k2/m2)*(xt_mean[2,:]-xt_mean[0,:])**2 + \
       0.5*(k3/m3)*(xt_mean[4,:]-xt_mean[2,:])**2)
L_i = Lfun(xt_mean)
    
print('Percentage Hamiltonian error: %0.4f' % (100*np.linalg.norm(H_a-H_i)/np.linalg.norm(H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.linalg.norm(L_a-L_i)/np.linalg.norm(L_a)) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([0,400])
plt.grid(True)
plt.legend()
plt.margins(0)

# %%
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 14

Xi_plot_drift = np.array(Xi_final.T)
Xi_plot_drift[Xi_plot_drift != 0 ] = -1
Xi_plot_diffusion = np.array(beta_exact.T)
Xi_plot_diffusion[Xi_plot_diffusion != 0] = -1

fig3, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,3))
plt.subplots_adjust(hspace=4) 
plt.suptitle('Stochastic 3DOF Oscillator', y=1.4)

ax1 = sns.heatmap( Xi_plot_drift, linewidth=0.5, cmap='Set1', square=False, ax=ax[0], cbar=False, yticklabels=[0,1,2])
ax1.set_xlabel('Library functions')
ax1.set_ylabel('Drift')
ax1.set_title('(a) Lagrangian for Drift', y=3)

ax2 = sns.heatmap( Xi_plot_diffusion, linewidth=0.5, cmap='Set1', square=False, ax=ax[1], cbar=False, yticklabels=[0])
ax2.set_xlabel('Library functions')
ax2.set_ylabel('Diffusion')
ax2.set_title('(b) Lagrangian for Diffusion', y=2)

ax1.text(7,-0.25, str(r'$x_1^2$'), color='b', fontsize=10)
ax1.text(8,-0.25, str(r'$\dot{x}_1^2$'), color='b', fontsize=10)
ax1.text(10,-0.25, str(r'$\dot{x}_2^2$'), color='b', fontsize=10)
ax1.text(12,-0.25, str(r'$\dot{x}_3^2$'), color='b', fontsize=10)
ax1.text(20,-0.25, str(r'$(x_2 - x_1)^2$'), color='b', fontsize=10, ha='right')
ax1.text(21,-0.25, str(r'$(x_3 - x_2)^2$'), color='b', fontsize=10)

ax2.text(0,-0.25, str(r'$x_1$'), color='k', rotation=90)
ax2.text(1,-0.25, str(r'$\dot{x}_1$'), color='k', rotation=90)
ax2.text(2,-0.25, str(r'$x_2$'), color='k', rotation=90)
ax2.text(3,-0.25, str(r'$\dot{x}_2$'), color='k', rotation=90)
ax2.text(4,-0.25, str(r'$x_3$'), color='k', rotation=90)
ax2.text(5,-0.25, str(r'$\dot{x}_3$'), color='k', rotation=90)
ax2.text(6,-0.25, str(r'$x_1^2$'), color='b', rotation=90)
ax2.text(7,-0.25, str(r'$\dot{x}_1^2$'), color='k', rotation=90)
ax2.text(8,-0.25, str(r'$x_2^2$'), color='b', rotation=90)
ax2.text(9,-0.25, str(r'$\dot{x}_2^2$'), color='k', rotation=90)
ax2.text(10,-0.25, str(r'$x_3^2$'), color='b', rotation=90)
ax2.text(11,-0.25, str(r'$\dot{x}_3^2$'), color='k', rotation=90)

# %% Save results/figures, if required:
# np.savez('results/results_stochastic_3dof', Xi_final=Xi_final, 
#           beta_exact=beta_exact, t_eval=t_eval, H_a=H_a, H_i=H_i, L_a=L_a, L_i=L_i)
