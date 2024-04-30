#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This code is for "discovering Lagrangian of the Stochastic Wave equation"
-- This code is implemented particle-wise, i.e., the Lagrangian is discovered 
   for each particle/spatial location. 
   
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

# The time parameters:
dx = 0.01       # Spacing of points on string
dt = 0.0001      # Size of time step
c = 2           # Speed of wave propagation
L = 0.5         # Length of string
T = 1           # Time to run the simulation
sigma = c**2*2 
Nsamp = 30
t = np.arange(0, T+dt, dt)      # time vector 

Dis_stoch, Vel_stoch, Acc_stoch, a, b = utils_data.wave_stoch(c=c,b=sigma,Nsamp=Nsamp,
                                                         L=L,T=T,dx=dx,dt=dt)
Dis_mean = np.mean( Dis_stoch, axis=0 )
vel_mean = np.mean( Vel_stoch, axis=0 )
acc_mean = np.mean( Acc_stoch, axis=0 )

nx = 2*int(L/dx) + 1
nt = int(T/dt) + 1

# %%
fig1 = plt.figure(figsize=(14,8))
plt.subplots_adjust(hspace=0.4)

plt.subplot(3,1,1); plt.imshow(Dis_mean, cmap='seismic', aspect='auto'); 
plt.title('Displacement-$u(t,x)$'); plt.ylabel('Space $x(t)$') 
plt.colorbar(aspect=5, pad=0.01) 
plt.subplot(3,1,2); plt.imshow(vel_mean, cmap='seismic', aspect='auto'); 
plt.title('Velocity-$\partial_{t} u(t,x)$'); plt.ylabel('Space $x(t)$') 
plt.colorbar(aspect=5, pad=0.01)
plt.subplot(3,1,3); plt.imshow(acc_mean, cmap='seismic', aspect='auto'); 
plt.title('Acceleration-$\partial_{tt} u(t,x)$'); plt.ylabel('Space $x(t)$') 
plt.xlabel('Time $t \in T$') 
plt.colorbar(aspect=5, pad=0.01)
plt.suptitle('1-D Wave Equation: $\partial^2_t u(x,t) = \partial_x^2 u(x,t)$')

del Acc_stoch

# %%
""" Generating the design matrix """
xvar = [sym.symbols('x'+str(i)) for i in range(1, nx)]
D, nd = utils.library_pde(xvar, Type='order1', dx=dx, polyn=4)

Rl, dxdt = [], []
for sample in range(Nsamp):
    if sample % 5 == 0:
        print('Library generation, ensemble count-{}'.format(sample))
        
    xt = np.zeros((nx + 1, nt))
    xt[::2] = Dis_stoch[sample, ...]  
    xt[1::2] = Vel_stoch[sample, ...] 
    
    Rl_temp, dxdt_temp = utils.euler_lagrange_library(D,xvar,xt,dt)
    Rl.append(Rl_temp), dxdt.append(dxdt_temp) 
    
    del Rl_temp, dxdt_temp
    
Rl, dxdt = np.array(Rl), np.array(dxdt)
Rl, dxdt = np.mean(Rl, axis=0), np.mean(dxdt, axis=0) 

# %%
""" Sparse regression: sequential least squares """

Xi = []
lam = 0.5     # lam is the sparsification constant
data = dxdt
for kk in range(int(nx/2)):
    print('Element- ', kk)
    if len(np.where( data[kk] != 0)[0]) < 5: # to check, if the vector is full of zeros
        Xi.append(np.zeros([Rl[0].shape[1],1]))
    else:
        Xi.append(utils.sparsifyDynamics(Rl[kk], data[kk:kk+1], lam))

Xi = np.column_stack( (np.array(Xi).squeeze(-1)) )

xvar_vel = xvar[1::2]
Xi_final = np.zeros([nd, int(nx/2)])
for kk in range(len(xvar_vel)):
    if len(np.where( data[kk] != 0)[0]) < 5:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 0)
    else:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 1)

# %%
predict = np.sqrt( np.abs(np.mean(Xi_final[Xi_final < 0])) )
rel_error = 100*np.abs(predict-c)/c
print("Actual: %d, Predicted: %0.4f, Relative error: %0.4f percent." % (c,predict,rel_error))

# %%
""" Lagrangian """
xvar_vel = xvar[1::2]
Xi_reduced = utils.nonequal_sum(np.array(Xi_final))
L = np.sum(sym.Array(0.5*np.dot(D,Xi_reduced)))

""" Hamiltonian """
H = 0
for i in range(len(xvar_vel)):
    H += (sym.diff(L, xvar_vel[i])*xvar_vel[i])
H = H - L

# print('Lagrangian: %s, Hamiltonian: %s' % (L, H))
Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

# %%
""" Sparse regression for the identification of volatility constant"""
# Numerically evaluate the identified Lagrangian,
L_drift = []
for sample in range(Nsamp):
    xt = np.zeros((nx + 1, nt))
    xt[::2] = Dis_stoch[sample, ...]  
    xt[1::2] = Vel_stoch[sample, ...] 
    
    L_drift.append( Lfun(xt[:-2, :])**2 )
L_drift = np.mean( np.array(L_drift), axis=0, keepdims=True )

# %%
D_stoch = []
for sample in range(Nsamp):
    xt = np.zeros((nx + 1, nt))
    xt[::2] = Dis_stoch[sample, ...]  
    xt[1::2] = Vel_stoch[sample, ...] 
    
    D_stoch.append( utils.library_diffusion_pde( xt, polyn=2, tpp=False ) ) 
D_stoch = np.array(D_stoch).mean(axis = 0) 

beta = utils.sparsifyDynamics(library=D_stoch, target=(dt*L_drift), lam=10)

beta_exact = 0.5*np.sqrt(beta[:-4:2].mean())/c**4 
rel_error_b = 100*np.abs(beta_exact-(sigma/c**2))/(sigma/c**2)

print('The identified Stochastic parameter vector...\n', beta_exact)
print("Actual: %d, Relative error: %0.4f percent." % (sigma/c**2,rel_error_b))

# %%
""" Hamiltonian """
Te = 0.5*np.sum(vel_mean[:]**2, axis=0)
Ve = 0.5*c**2/dx**2*(np.sum(np.diff(Dis_mean[:],axis=0)**2, axis=0))

xt = np.zeros((nx + 1, nt))
xt[::2] = Dis_mean 
xt[1::2] = vel_mean 

H_a = Te + Ve
L_a = Te - Ve
H_i = Hfun(xt[:-2, :])
L_i = Lfun(xt[:-2, :])

print('Percentage Hamiltonian error: %0.4f' % (100*np.linalg.norm(H_a-H_i)/np.linalg.norm(H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.linalg.norm(L_a-L_i)/np.linalg.norm(L_a)) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,6))
plt.plot(t, H_a, 'b', label='Actual')
plt.plot(t, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([0,5000])
plt.grid(True)
plt.legend()
plt.margins(0)

# %%
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 20

Xi_plot = np.array(Xi.T)
Xi_plot[Xi_plot != 0 ] = -1

fig3, ax = plt.subplots(figsize=(20,6))
ax = sns.heatmap( Xi_plot, linewidth=0.5, cmap='Set1', cbar=False, )
ax.set_xlabel('Library functions', fontsize=24)
ax.set_ylabel('Segments', fontsize=24)
ax.set_title('(a) Elastic Transversal Waves in a Solid', fontweight='bold', pad=30)

ax.text(5, -0.5, r'$\{u(x,t)^2_{(i)}\}$', color='b', fontsize=20)
ax.text(100, -0.5, r'$\{(u(x,t)_{(i-1)} - u(x,t)_{(i+1)})^2\}$', color='b', fontsize=20)
plt.margins(0)

# %% Save results/figures, if required:
# np.savez('results/results_stochastic_wave', Xi_final=Xi_final, beta=beta, 
#           beta_exact=beta_exact, t=t, H_a=H_a, H_i=H_i, L_a=L_a, L_i=L_i)
