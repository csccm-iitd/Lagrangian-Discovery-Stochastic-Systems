#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This code contains the useful source codes for data generation. 

"""

import numpy as np
import utils
import beam3fun

"""
Stochastic Euler-Bernoulli Beam with Brownian excitation
"""
def cantilever(params,sigma=1,L=1,T=1,dt=0.001,Ne=100,Nsamp=100):
    """
    Solves the Stochastic Euler-Bernoulli Beam equation using Order 0.5 Strong Euler-Maruyama scheme 
    
    Parameters
    ----------
    params : list, the system parameters.
    sigma : scalar, optional
            Diffusion coefficient of Brownian motion. The default is 1.
    L : scalar, optional
        Length of string. The default is 1.
    T : scalar, optional
        Time of simulation. The default is 1.
    dt : scalar, optional
         Time step size. The default is 0.001.
    Ne : integer, optional
         Number of finite element. The default is 100.
    Nsamp : scalar, optional
            Ensemble size. The default is 1.

    Returns
    -------
    Dis : matrix, displacement.
    Vel : matrix, velocity.
    Acc : matrix, acceleration.
    """
    
    # System/Blade parameters:
    rho, b, d, A, L, E, I, c1, c2 = params
    
    tt  = np.arange(0, T+dt, dt)
    xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)
    
    [Ma, Ka, _, _] = beam3fun.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')
    Ca = (c1*Ma + c2*Ka)

    # % ------------------------------------------------
    Lambda = 1.875104069/L
    # Lambda = 4.694091133/L
    # Lambda = 7.854757438/L
    # Lambda = 10.99554073/L
    # Lambda = 14.13716839/L

    h1 = np.cosh(Lambda*xx) -np.cos(Lambda*xx) -(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.sinh(Lambda*xx)-np.sin(Lambda*xx))
    h2 = Lambda*(np.sinh(Lambda*xx)+np.sin(Lambda*xx))-(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.cosh(Lambda*xx)-np.cos(Lambda*xx))*Lambda
    
    D0 = np.zeros(2*Ne)
    D0[0::2] = h1
    D0[1::2] = h2
    V0 = np.zeros(2*Ne)

    D, V, A = beam3fun.Newmark_stoch(Ma, Ca, Ka, sigma, D0, V0, dt, T, Nsamp)

    # % -------------------------------------------------
    Dis = D[:, 0::2, :]
    # Vel = V[:, 0::2, :]
    Acc = A[:, 0::2, :]
    
    Vel = np.zeros_like(Dis)
    for sample in range(Nsamp): 
        Vel[sample, ...] = utils.FiniteDiffVec(Dis[sample, ...],dx=dt,d=1)
    
    return Dis, Vel, Acc

"""
Free vibration of Euler-Bernoulli Beam 
"""
def cantilever_determ(params,T,dt,Ne=100):
    """
    Solves the Beam equation with fixed-free boundary using Newmark scheme 
    
    Parameters
    ----------
    params : list, the system parameters.
    T : scalar, terminal time.
    dt : float, time step.
    Ne : integer, number of finite element. The default is 100.

    Returns
    -------
    Dis : matrix, displacement.
    Vel : matrix, velocity.
    Acc : matrix, acceleration.
    """
    rho, b, d, A, L, E, I = params
    c1 = 0
    c2 = 0
    xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)
    
    [Ma, Ka, _, _] = beam3fun.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')

    Ca = (c1*Ma + c2*Ka)
    F = lambda t : 0    # for forced vibration, e.g., F = lambda t: np.sin(2*t)
    
    # % ------------------------------------------------
    Lambda = 1.875104069/L
    # Lambda = 4.694091133/L
    # Lambda = 7.854757438/L
    # Lambda = 10.99554073/L
    # Lambda = 14.13716839/L

    h1 = np.cosh(Lambda*xx) -np.cos(Lambda*xx) -(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.sinh(Lambda*xx)-np.sin(Lambda*xx))
    h2 = Lambda*(np.sinh(Lambda*xx)+np.sin(Lambda*xx))-(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.cosh(Lambda*xx)-np.cos(Lambda*xx))*Lambda
    
    D0 = np.zeros(2*Ne)
    D0[0::2] = h1
    D0[1::2] = h2
    V0 = np.zeros(2*Ne)
    D, V, A = beam3fun.Newmark(Ma,Ca,Ka,F,D0,V0,dt,T)
    
    # % -------------------------------------------------
    Dis = D[0::2]
    # Vel = V[0::2]
    Acc = A[0::2]
    
    Vel = utils.FiniteDiffVec(Dis,dx=dt,d=1)
    
    xt = np.zeros([2*Dis.shape[0],Dis.shape[1]])
    xt[::2] = Dis
    xt[1::2] = Vel
    
    return Dis, Vel, Acc


"""
Stochastic Wave equation using Euler-Maruyama
"""
def wave_stoch(c=5,b=1,Nsamp=1,L=1,T=1,dx=0.01,dt=0.001):
    """
    Solves the Stochastic Wave equation using Order 0.5 Strong Euler-Maruyama scheme 

    Parameters
    ----------
    c : scalar, optional
        Spped of wave. The default is 5.
    b : scalar, optional
        Diffusion coefficient of Brownian motion. The default is 1.
    Nsamp : scalar, optional
        Ensemble size. The default is 1.
    L : scalar, optional
        Length of string. The default is 1.
    T : scalar, optional
        Time of simulation. The default is 1.
    dx : scalar, optional
        Spatial grid spacing, assuming uniform spacing. The default is 0.01.
    dt : scalar, optional
        Time step size. The default is 0.001.

    Returns
    -------
    usol : tensor, ensemble of displacement.
    Vel : tensor, ensemble of velocity.
    Acc : tensor, ensemble of acceleration.
    a, b : matrix, matrix, meshgrid of the solution space.

    """
    # Parameters
    s = b*dt        # Coefficient - 0 
    r = c*dt/dx     # Coefficient - 1
    
    n = int(L/dx) + 1               # no of space grid 
    t = np.arange(0, T+dt, dt)      # time vector 
    mesh = np.arange(0, L+dx, dx)   # mesh grid 
    
    usol = [] 
    for ensemble in range(Nsamp):
        if ensemble % 20 == 0:
            print('Data generation, ensemble count-{}'.format(ensemble))
                
        # Set current and past to the graph of a plucked string
        current = 0.5 - 0.5*np.cos( 2*np.pi/L*mesh ) 
        past = current
        
        sol = np.zeros([len(mesh), len(t)])
        sol[:, 0] = current
        
        for i in range(len(t)):
            dW = np.random.randn(n-3)*np.sqrt(dt) 
            future = np.zeros(n)
        
            # Calculate the future position of the string
            future[0] = 0 
            future[1:n-2] = r**2*( current[0:n-3] + current[2:n-1] ) + \
                            2*(1-r**2)*current[1:n-2] - past[1:n-2] + s*dW
            future[n-1] = 0 
            sol[:, i] = current
            
            # Settings up for the next time step
            past = current 
            current = future
        usol.append(sol)

    usol = np.array(usol)
    Vel = np.zeros_like(usol)
    Acc = np.zeros_like(usol)
    for ensemble in range(Nsamp):
        Vel[ensemble, ...] = utils.FiniteDiffVec(usol[ensemble, ...], dx=dt, d=1) 
        Acc[ensemble, ...] = utils.FiniteDiffVec(usol[ensemble, ...], dx=dt, d=2) 
    a, b = np.meshgrid(t, mesh)

    return usol, Vel, Acc, a, b


"""
Stochastic Harmonic oscillator 
----------------------------------------------------------------------
"""
def harmonic(xinit, t_param, sys_param):
    """
    Solves the Stochastic Harmonic oscillator using Order 1.5 Strong Ito-Taylor scheme 
    
    Parameters
    ----------
    xinit : vector, initial condition.
    t_param : list, contains [dt, T, Nsamp].
    sys_param : list, contains system parameters.

    Returns
    -------
    y : tensor, ensemble of the state responses at time vector t.
    t : vector, the time vector.
    """ 
    # parameters of Harmonic oscillator in Equation
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


"""
Stochastic Pendulum 
----------------------------------------------------------------------
"""
def SimplePendulum(xinit, t_param, sys_param):
    """
    Solves the Stochastic Pendulum using Order 1.5 Strong Ito-Taylor scheme 
    
    Parameters
    ----------
    xinit : vector, initial condition.
    t_param : list, contains [dt, T, Nsamp].
    sys_param : list, contains system parameters.

    Returns
    -------
    y : tensor, ensemble of the state responses at time vector t.
    t : vector, the time vector.
    """ 
    # parameters of Pendulum 
    m, g, l, b = sys_param
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
            a2 = -(g/l)*np.sin(x0[0])
            b2 = b/(l**2)
            L0a1 = a2
            L0a2 = a1*-(g/l)*np.cos(x0[0])
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


"""
Stochastic duffing oscillator
----------------------------------------------------------------------
"""
def Duffing(xinit, t_param, sys_param):
    """
    Solves the stochastic Duffing oscillator using Order 1.5 Strong Ito-Taylor scheme 
    
    Parameters
    ----------
    xinit : vector, initial condition.
    t_param : list, contains [dt, T, Nsamp].
    sys_param : list, contains system parameters.

    Returns
    -------
    y : tensor, ensemble of the state responses at time vector t.
    t : vector, the time vector.
    """ 
    # parameters of Duffing oscillator 
    m, k, alpha, b = sys_param
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
            a2 = -k*x0[0] - alpha*x0[0]**3
            b2 = b
            L0a1 = a2
            L0a2 = a1*(-k -3*alpha*x0[0]**2) 
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


"""
Stochastic 3-DOF system:
"""
def mdof_system_stoschastic(xinit, t_param, sys_param):
    """
    Solves the 3DOF-SDE using Order 1.5 Strong Ito-Taylor scheme 
    
    Parameters
    ----------
    xinit : vector, initial condition.
    t_param : list, contains [dt, T, Nsamp].
    sys_param : list, contains system parameters.

    Returns
    -------
    y : tensor, ensemble of the state responses at time vector t.
    t : vector, the time vector.
    """    
    # parameters of 3DOF oscillator in Equation
    m1, m2, m3, k1, k2, k3, b1, b2, b3 = sys_param
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
            a2 = (- (k1 + k2)*x0[0] + k2*x0[2])/m1
            a3 = x0[3] 
            a4 = (k2*x0[0] - (k2 + k3)*x0[2] + k3*x0[4])/m2 
            a5 = x0[5]
            a6 = (k3*x0[2] - k3*x0[4])/m3
            b21 = b1/m1 
            b22 = b2/m2 
            b23 = b3/m3 
            
            L0a1 = a2
            L0a2 = - a1*(k1 + k2)/m1 + a3*k2/m1 
            L0a3 = a4
            L0a4 = a1*k2/m2 - a3*(k2 + k3)/m2 + a5*k3/m2 
            L0a5 = a6
            L0a6 = a3*k3/m3 - a5*k3/m3
            
            L1a1 = b21
            L1a2 = 0
            L1a3 = b22
            L1a4 = 0
            L1a5 = b23
            L1a6 = 0
            
            L0b21 = 0
            L0b22 = 0
            L0b23 = 0
            
            # Taylor 1.5 Mapping:
            sol1 = x0[0] + a1*dt + 0.5*L0a1*dt**2 + L1a1*dZ
            sol2 = x0[1] + a2*dt + b21*dW + 0.5*L0a2*(dt**2) + L1a2*dZ + L0b21*(dW*dt-dZ)
            sol3 = x0[2] + a3*dt + 0.5*L0a3*dt**2 + L1a3*dZ
            sol4 = x0[3] + a4*dt + b22*dW + 0.5*L0a4*(dt**2) + L1a4*dZ + L0b22*(dW*dt-dZ)
            sol5 = x0[4] + a5*dt + 0.5*L0a5*dt**2 + L1a5*dZ
            sol6 = x0[5] + a6*dt + b23*dW + 0.5*L0a6*(dt**2) + L1a6*dZ + L0b23*(dW*dt-dZ)
    
            x0 = np.array([sol1, sol2, sol3, sol4, sol5, sol6])
            x = np.column_stack((x, x0))
        y.append(x)
        
    y = np.array(y)
    return y, t
