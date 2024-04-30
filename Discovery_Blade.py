#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This code is for "discovering Lagrangian of Stochastic Euler-Bernoulli beam"
-- This code is implemented particle-wise, i.e., the Lagrangian is discovered 
   for each particle/spatial location. 
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym
import seaborn as sns

np.random.seed(0)

# %%
""" Generating system response """

# System/Blade parameters:
rho = 8050              # Density of material
b, d = 0.02, 0.001      # Dimensions of blade
A = b*d                 # C/Sectional area of blade 
E = 2e10                # Elasticity 
I = (b*d**3)/12         # Moment-of-inertia
Length = 1                   # Length of string
c1, c2 = 0, 0           # Rayleigh-Damping coefficients

params = [rho, b, d, A, Length, E, I, c1, c2]

c = (E*I)/(2*rho*A)
print('Wave coefficient-{}'.format(c))

# Simulation parameters:
Ne = 100
dt = 0.0001     # Size of time step
T_end = 2      # Time to run the simulation
sigma = (rho*A)*15
Nsamp = 20
print('r-{}'.format(c*dt**2/(1/Ne)**4))

tt  = np.arange(0, T_end+dt, dt)
xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)

Ds, Vs, As = utils_data.cantilever(params,sigma,Length,T_end,dt,Ne,Nsamp)

dim = int(Ne/2) 
nx = 2*dim
nt = int(T_end/dt) + 1

Ds = Ds[:, :dim, :]
Vs = Vs[:, :dim, :]
As = As[:, :dim, :]

Dis = np.mean(Ds, axis=0)
Vel = np.mean(Vs, axis=0)
Acc = np.mean(As, axis=0)

# %%
fig1 = plt.figure(figsize=(14,8))
plt.subplots_adjust(hspace=0.4)

plt.subplot(3,1,1)
plt.imshow(Dis, aspect='auto', cmap='seismic')
plt.title('Displacement-$u(t,x)$'); plt.ylabel('Space $x(t)$') 
plt.subplot(3,1,2)
plt.imshow(Vel, aspect='auto', cmap='seismic')
plt.title('Velocity-$\partial_{t} u(t,x)$'); plt.ylabel('Space $x(t)$') 
plt.subplot(3,1,3)
plt.imshow(Acc, aspect='auto', cmap='seismic')
plt.title('Acceleration-$\partial_{tt} u(t,x)$'); plt.ylabel('Space $x(t)$') 

plt.suptitle('1-D Euler-Bernoulli Equation: $\partial^2_t u(x,t) = \partial_x^4 u(x,t)$')

# %%
""" Generating the design matrix """
xvar = [sym.symbols('x'+str(i)) for i in range(1, nx+1)]
D, nd = utils.library_pde(xvar, Type='order2', dx=Length/Ne, polyn=2)

Rl, dxdt = [], []
for sample in range(Nsamp):
    print('Library generation, ensemble count-{}'.format(sample))

    xt = np.zeros([nx, nt])
    xt[::2] = Ds[sample, ...]  
    xt[1::2] = Vs[sample, ...] 
    
    Rl_temp, dxdt_temp = utils.euler_lagrange_library(D,xvar,xt,dt)
    Rl.append(Rl_temp), dxdt.append(dxdt_temp) 
    del Rl_temp, dxdt_temp

Rl, dxdt = np.array(Rl), np.array(dxdt)
Rl, dxdt = np.mean(Rl, axis=0), np.mean(dxdt, axis=0) 

# %%
""" Sparse regression: sequential least squares """
Xi = []
lam = 0.05      # lam is the sparsification constant
for kk in range(int(len(xvar)/2)):
    print('Element-', kk)
    data = dxdt[kk:kk+1] 
    Xi.append(utils.sparsifyDynamics(Rl[kk], data, lam))

Xi = np.column_stack( (np.array(Xi).squeeze(-1)) )

predict = np.abs(np.mean(Xi[np.nonzero(Xi)]))
rel_error = 100*np.abs(predict-c)/c
print("Actual: %0.4f, Predicted: %0.4f, Relative error: %0.4f percent." % (c,predict,rel_error))

xvar_vel = xvar[1::2]
Xi_final = np.zeros([nd, int(len(xvar)/2)])
for kk in range(len(xvar_vel)):
    if len(np.where( dxdt[kk] != 0)[0]) < 5:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 0)
    else:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 1)

# %%
""" Lagrangian """
xvar_vel = xvar[1::2]
Xi_reduced = utils.nonequal_sum(np.array(Xi_final))
L = np.sum(0.5*sym.Array(np.dot(D,Xi_reduced)))
H = 0
for i in range(len(xvar_vel)):
    H += (sym.diff(L, xvar_vel[i])*xvar_vel[i])
H = H - L

Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

# %%
""" Sparse regression for the System/Diffusion identification """
# Numerically evaluate the identified Lagrangian,
L_drift = []
for sample in range(Nsamp):
    xt = np.zeros((nx, nt))
    xt[::2] = Ds[sample, ...]  
    xt[1::2] = Vs[sample, ...] 
    
    L_drift.append( Lfun(xt)**2 )
L_drift = np.mean( np.array(L_drift), axis=0, keepdims=True )

# %%
D_stoch = []
for sample in range(Nsamp):
    xt = np.zeros((nx, nt))
    xt[::2] = Ds[sample, ...]  
    xt[1::2] = Vs[sample, ...] 
    
    D_stoch.append( utils.library_diffusion_pde( xt, polyn=2, tpp=False ) ) 
D_stoch = np.array(D_stoch).mean(axis = 0) 

beta = utils.sparsifyDynamics(library=D_stoch, target=(dt*L_drift), lam=100)

# %%
beta_exact = 0.5*np.sqrt(beta[4::2].mean())*(rho*A)**2
rel_error_b = 100*np.abs(beta_exact-sigma)/(sigma)

print('The identified Stochastic parameter vector...\n', beta_exact)
print("Actual: %0.4f, Relative error: %0.4f percent." % (sigma,rel_error_b))

# %%
""" Hamiltonian """
T = 0.5*np.sum(Vel[2:-2,:]**2, axis=0)
V = (c/(1/Ne)**4)*np.sum((-np.diff(Dis[:-1,:], axis=0) + np.diff(Dis[1:,:], axis=0))**2, axis=0)
    
T_i = 0.5*np.sum(Vel[2:-2,:]**2, axis=0)
V_i = (predict/(1/Ne)**4)*np.sum((-np.diff(Dis[:-1,:], axis=0) + np.diff(Dis[1:,:], axis=0))**2, axis=0)

H_a = 0.5*T + 0.5*V
L_a = 0.5*T - 0.5*V
H_i = 0.5*T_i + 0.5*V_i
L_i = 0.5*T_i - 0.5*V_i

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22
t_eval = np.arange(0,T_end+dt,dt)

fig2 = plt.figure(figsize=(8,6))
plt.plot(t_eval, L_a, 'b', label='Actual')
plt.plot(t_eval, L_i, 'r:', linewidth=1, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
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
ax.set_title('(a) Flexion Vibration of a Blade', fontweight='bold', pad=30)

ax.text(5, -0.5, r'$\{u(x,t)^2_{(i)}\}$', color='b', fontsize=20)
plt.text(32, -0.5 ,r'$\{u(x,t)_{(i-1)} -2u(x,t)_{(i)} +u(x,t)_{(i+1)}\}^2$',
          color='b', fontsize=22)
plt.margins(0)

# %% Save results/figures, if required:
# np.savez('results/results_stochastic_Beam', Xi_final=Xi_final, beta=beta, 
#           beta_exact=beta_exact, t_eval=t_eval, H_a=H_a, H_i=H_i, L_a=L_a, L_i=L_i)
