#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This file generates the stem plots of basis functions

"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import utils
import numpy as np
import matplotlib.pyplot as plt

# %%
data_h = np.load('results/results_stochastic_harmonic.npz')
data_p = np.load('results/results_stochastic_pendulum.npz')
data_d = np.load('results/results_stochastic_Duffing.npz')
data_3 = np.load('results/results_stochastic_3dof.npz')
data_w = np.load('results/results_stochastic_wave.npz')
data_b = np.load('results/results_stochastic_Beam.npz')

# %%
basis_h = np.where(data_h['Xi'] >= 0, data_h['Xi'], data_h['Xi']*-1)[:, 0]
label_h = np.arange(0, len(basis_h), 1)  
beta_h = data_h['beta_exact'][:, 0]
label_bh = np.arange(0, len(beta_h), 1)  

basis_p = np.where(data_p['Xi'] >= 0, data_p['Xi'], data_p['Xi']*-1)[:, 0]
label_p = np.arange(0, len(basis_p), 1)  
beta_p = data_p['beta_exact'][:, 0]
label_bp = np.arange(0, len(beta_p), 1)  

basis_d = np.where(data_d['Xi'] >= 0, data_d['Xi'], data_d['Xi']*-1)[:, 0]
label_d = np.arange(0, len(basis_d), 1)  
beta_d = data_h['beta_exact'][:, 0]
label_pd = np.arange(0, len(beta_d), 1) 

basis_3 = np.where(data_3['Xi_final'] >= 0, data_3['Xi_final'], data_3['Xi_final']*-1) 
basis_3 = utils.nonequal_sum( basis_3 ).sum(axis=1) 
label_3 = np.arange(0, len(basis_3), 1)  
beta_3 = data_3['beta_exact'][:, 0]
label_b3 = np.arange(0, len(beta_3), 1) 

basis_w = utils.nonequal_sum( data_w['Xi_final'] ).sum(axis=1) 
basis_w = np.where( basis_w >= 0, basis_w, basis_w*-1 )  
label_w = np.arange(0, len(basis_w), 1)  
beta_w = data_w['beta'][:, 0]
beta_w[ np.where( beta_w != 0 ) ] = data_w['beta_exact']
label_bw = np.arange(0, len(beta_w), 1) 

basis_b = utils.nonequal_sum( data_b['Xi_final'] ).sum(axis=1) 
basis_b = np.concatenate( (basis_b, np.zeros(len(basis_b))) ) 
basis_b = np.where( basis_b >= 0, basis_b, basis_b*-1 )  
label_b = np.arange(0, len(basis_b), 1)  
beta_b = data_b['beta'][:, 0] 
beta_b[ np.where( beta_b != 0 ) ] = data_b['beta_exact']
label_bb = np.arange(0, len(beta_b), 1) 

# %%
basis = [basis_h, basis_p, basis_d, basis_3, basis_w, basis_b]
beta = [beta_h, beta_p, beta_d, beta_3, beta_w, beta_b]
label_basis = [label_h, label_p, label_d, label_3, label_w, label_b]
label_beta = [label_bh, label_bp, label_pd, label_b3, label_bw, label_bb]

# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22

barWidth = 1
ylim_beta = [1.1, 0.15, 1.1, 1.1, 4, 6]

fig1, ax = plt.subplots(nrows=6, ncols=2, figsize=(18,16), dpi=100,
                        gridspec_kw={'width_ratios': [1,0.8]})
plt.subplots_adjust(hspace=0.3)

for i in range(6):
    ax[i,0].stem( label_basis[i], basis[i], linefmt='blue', markerfmt ='bo', basefmt="w" )
    ax[i,1].stem( label_beta[i], beta[i], linefmt='r', markerfmt ='ro', basefmt="w" )
        
    ax[i,1].set_ylim([0, ylim_beta[i]])
    if i < 3:
        ax[i,1].set_xticks( [0,2,4,6,8,10,12] )
    elif i == 3:
        ax[i,1].set_xticks( [0,2,4,6,8,10,12,14,16] )
        
    if i == 0:
        ax[i,1].text(2.2, 0.9, '$x^2$', color='r')
        ax[i,1].legend(['Harmonic'], handletextpad=0, borderpad=0.2, loc=1)
    elif i == 1:
        ax[i,1].text(2.2, 0.1, ''r'$\theta^2$', color='r')
        ax[i,1].legend(['Pendulum'], handletextpad=0, borderpad=0.2, loc=4)
    elif i == 2:
        ax[i,1].text(2.2, 0.9, '$x^2$', color='r')
        ax[i,1].legend(['Duffing'], handletextpad=0, borderpad=0.2, loc=1)
    elif i == 3:
        ax[i,1].text(5, 0.9, '$x_1^2$', color='r', ha='center')
        ax[i,1].text(7, 0.9, '$x_2^2$', color='r', ha='center')
        ax[i,1].text(10.2, 0.9, '$x_3^2$', color='r')
        ax[i,1].legend(['3DOF SS'], handletextpad=0, borderpad=0.2, loc=4)
    elif i == 4:
        ax[i,1].text(50, 3, '$u_i^2$', color='r')
        ax[i,1].legend(['Wave Eqn.'], handletextpad=0, borderpad=0.2, loc=4)
    else:
        ax[i,1].text(50, 4, '$u_i^2$', color='r')
        ax[i,1].legend(['Euler-Bernoulli-Beam'], handletextpad=0, borderpad=0.2, loc=4)
    
    ax[i,0].grid(True, alpha=0.35)
    ax[i,1].grid(True, alpha=0.35)
    
    if i == 0:
        ax[i,0].set_yscale('log')
        ax[i,0].set_xlim( [0,25] )
        ax[i,0].set_ylim( [1e-2, 1e4])
        ax[i,0].text(3.5, 1e3, '$x^2$', color='b')
        ax[i,0].text(5.5, 0.9, '$\dot{x}^2$', color='b')
        ax[i,0].legend(['Harmonic'], handletextpad=0, borderpad=0.2, loc=1)
    elif i == 1:
        ax[i,0].set_yscale('log')
        ax[i,0].set_xlim( [0,25] )
        ax[i,0].set_ylim( [1e-2, 1e2])
        ax[i,0].text(3.5, 0.9, ''r'$\dot{\theta}^2$', color='b')
        ax[i,0].text(22, 20, ''r'$cos{\theta}$', color='b')
        ax[i,0].legend(['Pendulum'], handletextpad=0, borderpad=0.2, loc='center')
    elif i == 2:
        ax[i,0].set_yscale('log')
        ax[i,0].set_xlim( [0,15] )
        ax[i,0].set_ylim( [1e-2, 1e4])
        ax[i,0].text(5.5, 0.9, '$\dot{x}^2$', color='b')
        ax[i,0].text(3.5, 1e3, '$x^2$', color='b')
        ax[i,0].text(10.5, 1e3, '$x^4$', color='b')
        ax[i,0].legend(['Duffing'], handletextpad=0, borderpad=0.2, loc=4)
    elif i == 3:
        ax[i,0].set_yscale('log')
        ax[i,0].set_xlim( [0,50] )
        ax[i,0].set_ylim( [1e-2, 1e4])
        ax[i,0].text(3.5, 1e3, '$x_1^2$', color='b')
        ax[i,0].text(25, 1e3, '$(x_2-x_1)^2$', color='b', ha='right')
        ax[i,0].text(27, 1e3, '$(x_3-x_2)^2$', color='b')
        ax[i,0].text(7, 5, '$\dot{x}_1^2$', color='b')
        ax[i,0].text(10, 3, '$\dot{x}_2^2$', color='b')
        ax[i,0].text(13, 0.5, '$\dot{x}_3^2$', color='b') 
        ax[i,0].legend(['3DOF SS'], handletextpad=0, borderpad=0.2, loc=4)
    elif i == 4:
        ax[i,0].set_xlim( [0,260] )
        ax[i,0].set_ylim( [0, 5])
        ax[i,0].text(25, 2.0, "$\partial_t {u}_i^2$", color='b', ha='right')
        ax[i,0].text(150, 4, ''r'${(\partial u_{i}/\partial x)^2}$', color='b')
        ax[i,0].legend(['Wave Eqn.'], handletextpad=0, borderpad=0.2, loc=4)
    else:
        ax[i,0].set_xlim( [0,420] )
        ax[i,0].set_ylim( [0, 2])
        ax[i,0].text(50, 1.5, "$\partial_t {u}_i^2$", color='b', ha='right')
        ax[i,0].text(180, 0.50, ''r'${(\partial^2 u_{i}/\partial x^2)^2}$', color='b') 
        ax[i,0].legend(['Euler-Bernoulli-Beam'], handletextpad=0, borderpad=0.2, loc=1)
 
    if i == 2:
        ax[i,0].set_ylabel('Parameter values', fontweight='bold', fontsize=26)
        
    if i == 0:
        ax[i,0].set_title('(a) Lagrangian basis for Drift', fontweight ='bold')
        ax[i,1].set_title('(b) Basis for Diffusion', fontweight ='bold' ) 
        
    if i == 5:
        ax[i,0].set_xlabel('Basis functions', fontweight ='bold' ) 
        ax[i,1].set_xlabel('Basis functions', fontweight ='bold' ) 

fig1.savefig('figures/Basis.pdf', format='pdf', dpi=600, bbox_inches='tight')
