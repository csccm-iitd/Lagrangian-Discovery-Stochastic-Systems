#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This file generates the figures of predicted responses

"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import utils_data

# %%
""" Stochastic Harmonic Oscillator """
np.random.seed(0)

# The time and system parameters:
x0 = np.array([0.5, 0])
m, k, k_i, b, b_i = 1, 1000, 1000.21, 1, 1.0308 
dt, T, Nsamp = 0.0001, 3, 200
t_param = [dt, T, Nsamp]
sys_param = [m, k, b]
sys_param_i = [m, k_i, b_i]

# System responses:
xt_a_h, t_eval_h = utils_data.harmonic(x0, t_param, sys_param)
xt_i_h, t_eval_h = utils_data.harmonic(x0, t_param, sys_param_i)

xt_mean_a_h = np.mean(xt_a_h, 0)
xt_mean_i_h = np.mean(xt_i_h, 0)
xt_mean_u_h = np.mean(xt_i_h, 0) + 2*np.std(xt_i_h, 0)
xt_mean_l_h = np.mean(xt_i_h, 0) - 2*np.std(xt_i_h, 0)


# %%
""" Stochastic Pendulum """
np.random.seed(0)

# The time and system parameters:
x0 = np.array([0.9, 0])
m, g, g_i, l, b, b_i = 1, 9.81, 0.5*19.6175, 1, 0.10, 0.1172
dt, T, Nsamp = 0.0005, 10, 200
t_param = [dt, T, Nsamp]
sys_param = [m, g, l, b]
sys_param_i = [m, g_i, l, b_i]

# System responses:
xt_a_p, t_eval_p = utils_data.SimplePendulum(x0, t_param, sys_param)
xt_i_p, t_eval_p = utils_data.SimplePendulum(x0, t_param, sys_param_i)

xt_mean_a_p = np.mean(xt_a_p, 0)
xt_mean_i_p = np.mean(xt_i_p, 0)
xt_mean_u_p = np.mean(xt_i_p, 0) + 2*np.std(xt_i_p, 0)
xt_mean_l_p = np.mean(xt_i_p, 0) - 2*np.std(xt_i_p, 0)


# %%
""" Stochastic Duffing Oscillator """
np.random.seed(0)

# The time and system parameters:
x0 = np.array([0.4, 0])
m, k, k_i, alpha, alpha_i, b, b_i = 1, 1000, 999.357, 5000, 2*2502.31, 1, 1.0308
dt, T, Nsamp = 0.0001, 3, 200
t_param = [dt, T, Nsamp]
sys_param = [m, k, alpha, b]
sys_param_i = [m, k_i, alpha_i, b_i]

# System responses:
xt_a_d, t_eval_d = utils_data.Duffing(x0, t_param, sys_param)
xt_i_d, t_eval_d = utils_data.Duffing(x0, t_param, sys_param_i)

xt_mean_a_d = np.mean(xt_a_d, 0)
xt_mean_i_d = np.mean(xt_i_d, 0)
xt_mean_u_d = np.mean(xt_i_d, 0) + 2*np.std(xt_i_d, 0)
xt_mean_l_d = np.mean(xt_i_d, 0) - 2*np.std(xt_i_d, 0)


# %%
""" Stochastic 3DOF Oscillator """
np.random.seed(0)

# The time and system parameters:
x0 = np.array([0.25, 0, 0.5, 0, 0, 0])
m1, m2, m3, k1, k2, k3, b1, b2, b3 = 10, 10, 10, 10000, 10000, 10000, 1, 1, 1 
m1, m2, m3, k1_i, k2_i, k3_i, b1_i, b2_i, b3_i = 10, 10, 10, 10*999.99, 10*1000.04, 10*999.965, 1.0308, 1.0308, 1.0308 
dt, T, Nsamp = 0.0001, 3, 200
t_param = [dt, T, Nsamp]
sys_param = [m1, m2, m3, k1, k2, k3, b1, b2, b3]
sys_param_i = [m1, m2, m3, k1_i, k2_i, k3_i, b1_i, b2_i, b3_i]

# System responses:
xt_a_3, t_eval_3 = utils_data.mdof_system_stoschastic(x0, t_param, sys_param)
xt_i_3, t_eval_3 = utils_data.mdof_system_stoschastic(x0, t_param, sys_param_i)

xt_mean_a_3 = np.mean(xt_a_3, 0)
xt_mean_i_3 = np.mean(xt_i_3, 0)
xt_mean_u_3 = np.mean(xt_i_3, 0) + 2*np.std(xt_i_3, 0)
xt_mean_l_3 = np.mean(xt_i_3, 0) - 2*np.std(xt_i_3, 0)


# %%
""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

r = 4 

# Plot the responses:
fig1, ax = plt.subplots(nrows=6, ncols=2, figsize=(16,18), dpi=300)
plt.subplots_adjust(hspace=0.5)

plt.subplot(6,2,1); 
plt.plot(t_eval_h[::r], xt_mean_a_h[0, ::r], color='r', label='True');
plt.plot(t_eval_h[::r], xt_mean_i_h[0, ::r], linestyle='--', color='blue', label='Discovered'); 
plt.fill_between(t_eval_h[::r], xt_mean_u_h[0, ::r], xt_mean_l_h[0, ::r], color='tab:green', alpha=0.5) 
plt.axvline(x=1, linewidth='4', color='k')
plt.ylabel('$x$(t)'); plt.grid(True, alpha=0.25); 
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.margins(0)

plt.subplot(6,2,2); 
plt.plot(t_eval_h[::r], xt_mean_a_h[1, ::r], color='r', label='True');
plt.plot(t_eval_h[::r], xt_mean_i_h[1, ::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$\dot{x}$(t)');
plt.fill_between(t_eval_h[::r], xt_mean_u_h[1, ::r], xt_mean_l_h[1, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.grid(True, alpha=0.25); plt.margins(0)
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.title('(a) Stochastic Harmonic Oscillator', x=-0.15)

plt.subplot(6,2,3); 
plt.plot(t_eval_p[::r], xt_mean_a_p[0, ::r], color='r', label='True'); 
plt.plot(t_eval_p[::r], xt_mean_i_p[0, ::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel(''r'$\theta$(t)');
plt.fill_between(t_eval_p[::r], xt_mean_u_p[0, ::r], xt_mean_l_p[0, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=5, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)

plt.subplot(6,2,4); 
plt.plot(t_eval_p[::r], xt_mean_a_p[1, ::r], color='r', label='True'); 
plt.plot(t_eval_p[::r], xt_mean_i_p[1, ::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel(''r'$\dot{\theta}$(t)');
plt.fill_between(t_eval_p[::r], xt_mean_u_p[1, ::r], xt_mean_l_p[1, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=5, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)
plt.title('(b) Stochastic Pendulum', x=-0.10)

plt.subplot(6,2,5); 
plt.plot(t_eval_d[::r], xt_mean_a_d[0, ::r], color='r', label='True'); 
plt.plot(t_eval_d[::r], xt_mean_i_d[0, ::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$x$(t)');
plt.fill_between(t_eval_d[::r], xt_mean_u_d[0, ::r], xt_mean_l_d[0, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)

plt.subplot(6,2,6); 
plt.plot(t_eval_d[::r], xt_mean_a_d[1, ::r], color='r', label='True'); 
plt.plot(t_eval_d[::r], xt_mean_i_d[1, ::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$\dot{x}$(t)');
plt.fill_between(t_eval_d[::r], xt_mean_u_d[1, ::r], xt_mean_l_d[1, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)
plt.title('(c) Stochastic Duffing Oscillator', x=-0.10)

plt.subplot(6,2,7);
plt.plot(t_eval_3[::r], xt_mean_a_3[0,::r], color='r', label='True');
plt.plot(t_eval_3[::r], xt_mean_i_3[0,::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$x_1$(t)');
plt.fill_between(t_eval_3[::r], xt_mean_u_3[0, ::r], xt_mean_l_3[0, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)

plt.subplot(6,2,8); 
plt.plot(t_eval_3[::r], xt_mean_a_3[1,::r], color='r', label='True'); 
plt.plot(t_eval_3[::r], xt_mean_i_3[1,::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$\dot{x}_1$(t)');
plt.fill_between(t_eval_3[::r], xt_mean_u_3[1, ::r], xt_mean_l_3[1, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)
plt.title('(d) Stochastic 3DOF Structural system', x=-0.15)

plt.subplot(6,2,9); 
plt.plot(t_eval_3[::r], xt_mean_a_3[2,::r], color='r', label='True'); 
plt.plot(t_eval_3[::r], xt_mean_i_3[2,::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$x_2$(t)');
plt.fill_between(t_eval_3[::r], xt_mean_u_3[2, ::r], xt_mean_l_3[2, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)

plt.subplot(6,2,10); 
plt.plot(t_eval_3[::r], xt_mean_a_3[3,::r], color='r', label='True'); 
plt.plot(t_eval_3[::r], xt_mean_i_3[3,::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$\dot{x}_2$(t)');
plt.fill_between(t_eval_3[::r], xt_mean_u_3[3, ::r], xt_mean_l_3[3, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.grid(True, alpha=0.25); plt.margins(0)

plt.subplot(6,2,11); 
plt.plot(t_eval_3[::r], xt_mean_a_3[4,::r], color='r', label='True'); 
plt.plot(t_eval_3[::r], xt_mean_i_3[4,::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$x_3$(t)');
plt.fill_between(t_eval_3[::r], xt_mean_u_3[4, ::r], xt_mean_l_3[4, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.xlabel('Time (t)'); 
plt.grid(True, alpha=0.25); plt.margins(0)

plt.subplot(6,2,12); 
plt.plot(t_eval_3[::r], xt_mean_a_3[5,::r], color='r', label='True'); 
plt.plot(t_eval_3[::r], xt_mean_i_3[5,::r], linestyle='--', color='blue', label='Discovered'); 
plt.ylabel('$\dot{x}_3$(t)');
plt.fill_between(t_eval_3[::r], xt_mean_u_3[5, ::r], xt_mean_l_3[5, ::r], color='tab:green', alpha=0.5)
plt.axvline(x=1, linewidth='4', color='k')
plt.legend(loc=3, ncol=2, handletextpad=0, handlelength=1, columnspacing=0.5, borderpad=0.2)
plt.xlabel('Time (t)'); 
plt.grid(True, alpha=0.25); plt.margins(0)

fig1.savefig('results/Response_sde.pdf', format='pdf', dpi=100, bbox_inches='tight')

# %%
""" Stochastic Wave equation """
np.random.seed(0)

# The time parameters:
dx = 0.01       # Spacing of points on string
dt = 0.001     # Size of time step
c = 2           # Speed of wave propagation
c_i = np.sqrt(4.00126)           # Speed of wave propagation
L = 0.5         # Length of string
T = 4           # Time to run the simulation
sigma = c**2*2 
sigma_i = c**2*2.1249
Nsamp = 30
t = np.arange(0, T+dt, dt)      # time vector 

Dis_stoch, _, _, _, _ = utils_data.wave_stoch(c=c,b=sigma,Nsamp=Nsamp,
                                                         L=L,T=T,dx=dx,dt=dt)

Dis_stoch_i, _, _, _, _ = utils_data.wave_stoch(c=c_i,b=sigma_i,Nsamp=Nsamp,
                                                         L=L,T=T,dx=dx,dt=dt)

Dis_mean = np.mean( Dis_stoch, axis=0 )[:, ::2]
Dis_mean_i = np.mean( Dis_stoch_i, axis=0 )[:, ::2]


# %%
""" Stochastic Euler-Bernoulli Beam """

# System/Blade parameters:
rho = 8050              # Density of material
rho_i = 8150
b, d = 0.02, 0.001      # Dimensions of blade
A = b*d                 # C/Sectional area of blade 
E = 2e10                # Elasticity 
I = (b*d**3)/12         # Moment-of-inertia
Length = 1                   # Length of string
c1, c2 = 0, 0           # Rayleigh-Damping coefficients

params = [rho, b, d, A, Length, E, I, c1, c2]
params_i = [rho_i, b, d, A, Length, E, I, c1, c2]

c = (E*I)/(2*rho*A)
print('Wave coefficient-{}'.format(c))

# Simulation parameters:
Ne = 100
dt = 0.001     # Size of time step
T_end = 4      # Time to run the simulation
sigma = (rho*A)*20
sigma_i = 3.2222
Nsamp = 20
print('r-{}'.format(c*dt**2/(1/Ne)**4))

tt  = np.arange(0, T_end+dt, dt)
xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)

Ds, _, _ = utils_data.cantilever(params,sigma,Length,T_end,dt,Ne,Nsamp)
Ds_i, _, _ = utils_data.cantilever(params_i,sigma_i,Length,T_end,dt,Ne,Nsamp)

Dis = np.mean(Ds, axis=0)
Dis_i = np.mean(Ds_i, axis=0)

# %%
plt.rcParams['font.size'] = 18

fig2 = plt.figure(figsize=(20,10), dpi=100)
plt.subplots_adjust(hspace=0.6, wspace=0.1)

plt.subplot(4,2,1); 
plt.imshow(Dis_mean, cmap='nipy_spectral', aspect='auto', extent=[0,4,0,0.5]); 
plt.axvline(x=1, linewidth=4, color='white')
plt.title('(a) Wave Equation: $\partial^2_t u(x,t) = \partial_x^2 u(x,t) + \partial_tW(x,t)$\nGround Truth, $u(t,x)$'); 
plt.ylabel('$x$') 
plt.colorbar(aspect=5, pad=0.01) 
plt.subplot(4,2,3); 
plt.imshow(Dis_mean_i, cmap='nipy_spectral', aspect='auto', extent=[0,4,0,0.5]); 
plt.axvline(x=1, linewidth=4, color='white')
plt.title('Prediction, $u(x,t)$'); plt.ylabel('$x$') 
plt.colorbar(aspect=5, pad=0.01)
plt.subplot(4,2,5); 
plt.imshow(np.abs(Dis_mean-Dis_mean_i), cmap='nipy_spectral', aspect='auto',
           vmin=0, vmax=0.5, extent=[0,4,0,0.5]); 
plt.axvline(x=1, linewidth=4, color='white')
plt.title('Predictive error in $u(x,t)$'); plt.ylabel('$x$') 
plt.colorbar(aspect=5, pad=0.01)
plt.subplot(4,2,7); 
plt.imshow(np.std(Dis_stoch_i, axis=0), cmap='nipy_spectral', aspect='auto',
           vmin=0, vmax=0.5, extent=[0,4,0,0.5]); 
plt.axvline(x=1, linewidth=4, color='white')
plt.title('Standard deviation in $u(x,t)$'); plt.ylabel('$x$') 
plt.xlabel('Time $t \in T$') 
plt.colorbar(aspect=5, pad=0.01)

plt.subplot(4,2,2)
plt.imshow(Dis, aspect='auto', cmap='nipy_spectral', extent=[0,4,0,0.5])
plt.axvline(x=1, linewidth=4, color='white')
plt.title('(b) Euler-Bernoulli Beam: $\partial^2_t u(x,t) = \partial_x^4 u(x,t) + \partial_tW(x,t)$\nGround Truth, $u(t,x)$'); 
plt.ylabel('$x$') 
plt.colorbar(aspect=5, pad=0.01)
plt.subplot(4,2,4)
plt.imshow(Dis_i, aspect='auto', cmap='nipy_spectral', extent=[0,4,0,0.5])
plt.axvline(x=1, linewidth=4, color='white')
plt.title('Prediction, $u(x,t)$'); plt.ylabel('$x$') 
plt.colorbar(aspect=5, pad=0.01)
plt.subplot(4,2,6)
plt.imshow(np.abs(Dis-Dis_i), aspect='auto', cmap='nipy_spectral',
           vmin=0, vmax=1, extent=[0,4,0,0.5])
plt.axvline(x=1, linewidth=4, color='white')
plt.title('Predictive error in $u(x,t)$'); plt.ylabel('$x$') 
plt.colorbar(aspect=5, pad=0.01)
plt.subplot(4,2,8)
plt.imshow(np.std(Ds_i, axis=0), aspect='auto', cmap='nipy_spectral',
           vmin=0, vmax=1, extent=[0,4,0,0.5])
plt.axvline(x=1, linewidth=4, color='white')
plt.title('Standard deviation in $u(x,t)$'); plt.ylabel('$x$') 
plt.colorbar(aspect=5, pad=0.01)

fig2.savefig('results/Response_pde.pdf', format='pdf', dpi=200, bbox_inches='tight')
