#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This code is for "discovering Lagrangian of Stochastic Pendulum" 
-- The Harmonic bases are removed from the library to simulate an
    ill-conditioned case of library
-- The data are generated using Stochastic Taylor strong order 1.5 scheme

"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils_data
import utils
import sympy as sym 
import seaborn as sns
import pandas as pd

np.random.seed(0)

# %%
""" Generating system response """

# The time and system parameters:
x0 = np.array([0.9, 0])
m, g, l, b = 1, 9.81, 1, 0.10
dt, T, Nsamp = 0.0005, 10, 200
t_param = [dt, T, Nsamp]
sys_param = [m, g, l, b]

# System responses:
xt_10s, t_eval_10s = utils_data.SimplePendulum(x0, t_param, sys_param)

xt, t_eval = xt_10s[:, :, :10000], t_eval_10s[:10000] 
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
plt.suptitle('Stochastic Pendulum', y=0.95)

# %%
""" Generating the design matrix """

# Define the symbolic library:
x, y = sym.symbols('x, y')
xvar = [x, y]
D, nd = utils.library_sdof(xvar,polyn=6,harmonic=0) # Dont use Harmonic function in library

# Obtain the numerical library:
Rl, dxdt = [], []
for sample in range(Nsamp):
    Rl_temp, dxdt_temp = utils.euler_lagrange_library(D,xvar,xt[sample, ...],dt)
    Rl.append(Rl_temp), dxdt.append(dxdt_temp) 

Rl, dxdt = np.array(Rl), np.array(dxdt)
Rl, dxdt = np.mean(Rl, axis=0).squeeze(0), np.mean(dxdt, axis=0) 

# %%
""" Sparse regression for the System/Drift identification """

# compute Sparse regression: sequential least squares
lam = 2      # lam is our sparsification knob.
Xi = utils.sparsifyDynamics(Rl, dxdt, lam)

# Now insert the target vector back to the Euler-Lagrange library:
Xi = np.insert(Xi, np.where(D == y**2)[0], 1, axis=0)
Theta = pd.DataFrame( {'Basis': D, 'Theta':Xi[:,0]} )
print('The identified parameter vector...\n', Theta)

# %%
""" Print the Lagrangian and Hamiltonian """

""" Lagrangian """
L = -1*sym.Array(0.5*np.dot(D ,Xi))

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

beta = utils.sparsifyDynamics(library=D_stoch, target=(dt*L_drift), lam=0.003) 
beta_exact = 0.5*np.sqrt(beta)

X = sym.Symbol('X', commutative=False)
Xd = sym.Symbol('Xd', commutative=False)
xvar = [X, Xd]
D_diffusion = utils.library_visualize(xvar)
Beta = pd.DataFrame( {'Basis':D_diffusion, 'Beta':beta_exact[:,0]} )
print('The identified Stochastic parameter vector...\n', Beta)

# %%
""" Hamiltonian time series """
xt_mean_10s = np.mean(xt_10s, axis=0)

H_a = 0.5*xt_mean_10s[1,:]**2 - (g/l)*np.cos(xt_mean_10s[0,:]) 
H_i = Hfun(xt_mean_10s[0], xt_mean_10s[1]) 

""" Lagrangian time series """
L_a = 0.5*xt_mean_10s[1,:]**2 + (g/l)*np.cos(xt_mean_10s[0,:]) 
L_i = Lfun(xt_mean_10s[0,:], xt_mean_10s[1,:]) 

print('Percentage Hamiltonian error: %0.4f' % (100*np.linalg.norm(H_a-H_i)/np.linalg.norm(H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.linalg.norm(L_a-L_i)/np.linalg.norm(L_a)) )

# %%
""" Generating system response for different initial condition """

# The time and system parameters:
x0 = np.array([0.7, 0])
m, g, l, b = 1, 9.81, 1, 0.10
dt, T, Nsamp = 0.0005, 10, 200
t_param = [dt, T, Nsamp]
sys_param = [m, g, l, b]

# System responses:
xt_10d, t_eval_10d = utils_data.SimplePendulum(x0, t_param, sys_param)

""" Hamiltonian time series """
xt_mean_10d = np.mean(xt_10d, axis=0)

H_a10 = 0.5*xt_mean_10d[1,:]**2 - (g/l)*np.cos(xt_mean_10d[0,:]) 
H_i10 = Hfun(xt_mean_10d[0], xt_mean_10d[1]) 

""" Lagrangian time series """
L_a10 = 0.5*xt_mean_10d[1,:]**2 + (g/l)*np.cos(xt_mean_10d[0,:]) 
L_i10 = Lfun(xt_mean_10d[0,:], xt_mean_10d[1,:]) 

print('Percentage Hamiltonian error: %0.4f' % (100*np.linalg.norm(H_a10-H_i10)/np.linalg.norm(H_a10)) )
print('Percentage Lagrange error: %0.4f' % (100*np.linalg.norm(L_a10-L_i10)/np.linalg.norm(L_a10)) )

# %%
def harmonic_ident(xinit, t_param, sys_param):
    """
    Parameters
    ----------
    x0 : array, initial condition.
    t_param : list, temporal parameters.
    sys_param : list, system parameters.

    Returns
    -------
    y : array, ensemble of system states.
    """
    # parameters of Duffing oscillator in Equation
    m, k, b = sys_param
    dt, T, Nsamp = t_param
    delmat = np.row_stack(([np.sqrt(dt), 0],[(dt**1.5)/2, (dt**1.5)/(2*np.sqrt(3))]))
    
    # np.random.seed(0)
    t = np.linspace(0, T, int(T/dt))
    
    y = []
    # Simulation Starts Here ::
    for ensemble in range(Nsamp):
        if ensemble % 20 == 0:
            print('Data generation, ensemble count-{}'.format(ensemble))
        x0 = np.array(xinit)
        x = np.vstack(xinit)  # Initial condition
        for n in range(len(t)-1):
            delgen = np.dot(delmat, np.random.normal(0,1,2))
            dW, dZ = delgen
            
            a1 = x0[1]
            a2 = -(k/m)*x0[0]
            b2 = b/m
            L0a1 = a2
            L0a2 = a1*-(k/m)
            L0b2 = 0
            L1a1 = b2
            L1a2 = 0
            
            # Taylor 1.5 Mapping:
            sol1 = x0[0] + a1*dt + 0.5*L0a1*dt**2 + L1a1*dZ
            sol2 = x0[1] + a2*dt + b2*dW + 0.5*L0a2*(dt**2) + L1a2*dZ + L0b2*(dW*dt-dZ)
            
            x0 = np.array([sol1, sol2])
            x = np.column_stack((x, x0))
        y.append(x)
        
    y = np.array(y)
    return y, t

# Identified system responses:
xt_i, _ = harmonic_ident([0.9, 0], [0.0005, 10, 200], [1, -Xi[3], b])
xt_i_mean = np.mean(xt_i, 0)
xt_i_std = np.std(xt_i, 0)
xt_i_u = xt_i_mean + 2 * xt_i_std
xt_i_l = xt_i_mean - 2 * xt_i_std

# Identified system responses:
xt_id, _ = harmonic_ident([0.7, 0], [0.0005, 10, 200], [1, -Xi[3], b])
xt_id_mean = np.mean(xt_id, 0)
xt_id_std = np.std(xt_id, 0)
xt_id_u = xt_id_mean + 2 * xt_id_std
xt_id_l = xt_id_mean - 2 * xt_id_std

# %%
""" Plotting the theta """
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 22

Xi_plot_drift = np.array(Xi.T)
Xi_plot_drift[Xi_plot_drift != 0] = -1
Xi_plot_diffusion = np.array(beta_exact.T)
Xi_plot_diffusion[Xi_plot_diffusion != 0] = -1

fig3 = plt.figure(figsize=(20,12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1,1], height_ratios=[1,0.7,0.7])
plt.subplots_adjust(hspace=0.55, wspace=0.20) 

label_drift = np.arange(0,len(D),1)
ax0 = fig3.add_subplot(gs[0, 0])
ax0.stem( label_drift, np.abs(Xi[:,0]), linefmt='blue', markerfmt ='bo', basefmt="w")
ax0.set_ylabel('Parameters')
ax0.set_xlabel('Library functions')
ax0.set_title('(a) Lagrangian basis for Drift')
ax0.grid(True, alpha=0.4)
ax0.set_xticks([0,5,10,15,20,25,30])
ax0.set_ylim([0,10])
ax0.text(3.5, 8.5, ''r'$\theta^2$=8.8403', color='b')
ax0.text(5.5, 1, ''r'$\dot{\theta}^2$=1', color='b')

label_diffusion = np.arange(0,len(D_diffusion),1)
ax1 = fig3.add_subplot(gs[0, 1])
ax1.stem( label_diffusion, np.sqrt(beta[:,0]), linefmt='red', markerfmt ='ro', basefmt="w")
ax1.set_xlabel('Library functions')
ax1.set_title('(b) Basis for Diffusion')
ax1.grid(True, alpha=0.4)
ax1.set_ylim([0,0.1])
ax1.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax1.text(2, 0.08, ''r'$\theta^2$=0.0739', color='r')

ax2 = fig3.add_subplot(gs[1, 0])
ax2.plot(t_eval_10s, H_a, 'b', label='Truth')
ax2.plot(t_eval_10s, H_i[0], 'r--', linewidth=2, label='Discovered')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Hamiltonian')
ax2.set_title('(c) Hamiltonian evolution for seen IC')
ax2.set_ylim([-10,10])
ax2.grid(True,alpha=0.35)
ax2.legend(labelspacing=0.1,borderpad=0.25,handletextpad=0.25)
ax2.axvline(x=5, color='k', linestyle='--')
ax2.text(1,-9, 'Training Regime', color='brown')
ax2.text(6,-9, 'Prediction Regime', color='brown')
ax2.margins(0)

ax3 = fig3.add_subplot(gs[1, 1])
ax3.plot(t_eval_10s, xt_mean_10s[0], color='b', label='Truth')
ax3.plot(t_eval_10s, xt_i_mean[0], '--', color='r', linewidth=2, label='Discovered')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('$x(t)$')
ax3.set_title('(d) Displacement evolution for seen IC')
ax3.grid(True,alpha=0.35)
ax3.axvline(x=5, color='k', linestyle='--')
ax3.text(1,-0.8, 'Training Regime', color='brown')
ax3.text(6,-0.8, 'Prediction Regime', color='brown')
ax3.legend(loc=1, labelspacing=0.1,borderpad=0.25,handletextpad=0.25)
ax3.margins(0)

ax4 = fig3.add_subplot(gs[2, 0])
ax4.plot(t_eval_10d, H_a10, 'b', label='Truth')
ax4.plot(t_eval_10d, H_i10[0], 'r--', linewidth=2, label='Discovered')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Hamiltonian')
ax4.set_title('(e) Hamiltonian evolution for unseen IC')
ax4.set_ylim([-10,10])
ax4.grid(True,alpha=0.35)
ax4.legend(labelspacing=0.1,borderpad=0.25,handletextpad=0.25)
ax4.margins(0)

ax5 = fig3.add_subplot(gs[2, 1])
ax5.plot(t_eval_10s, xt_mean_10d[0], color='b', label='Truth')
ax5.plot(t_eval_10s, xt_id_mean[0], '--', color='r', linewidth=2, label='Discovered')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('$x(t)$')
ax5.set_title('(f) Displacement evolution for unseen IC')
ax5.grid(True,alpha=0.35)
ax5.legend(loc=1, labelspacing=0.1,borderpad=0.25,handletextpad=0.25)
ax5.margins(0)

fig3.savefig('figures/missing_hamiltonian.pdf', format='pdf', dpi=600, bbox_inches='tight')

# %% Save results/figures, if required:
np.savez('results/results_stochastic_pendulum_missing'+str(Nsamp), Xi=Xi, 
          beta_exact=beta_exact, t_eval=t_eval, H_a=H_a, H_i=H_i, L_a=L_a, L_i=L_i,
          H_a10=H_a10, H_i10=H_i10, L_a10=L_a10, L_i10=L_i10)
