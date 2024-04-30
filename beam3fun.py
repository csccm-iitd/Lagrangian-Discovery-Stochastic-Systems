#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-- This file contains the functions 
    - to generate mass and striffness matrices of beam
    - to solve beam equation

"""

import numpy as np

def Beam3(rho,A,E,I,Le,n,bc,k_t1=0,k_t2=0,k_r1=0,k_r2=0):
    """
    Euler-Bernoulli Beam 
    - spatial discretization by finite element method

    Parameters
    ----------
    rho : scalar, density of beam material.
    A : scalar, Cross-sectional area of beam.
    E : scalar, Elastic modulus of beam.
    I : scalar, MOI of beam.
    Le : scalar, length of beam element.
    n : scalar, number of nodes.
    bc : string, boundary condition.
    k_t1 : scalar, optional
        Defines BC, if not given. The default is 0.
    k_t2 : scalar, optional
        Defines BC, if not given. The default is 0.
    k_r1 : scalar, optional
        Defines BC, if not given. The default is 0.
    k_r2 : scalar, optional
        Defines BC, if not given. The default is 0.

    Returns
    -------
    Ma : matrix, global mass matrix.
    Ka : matrix, global mass matrix.
    omega : vector, natural frequencies.
    lambdaL : vector, modeshape parameters of beam.

    """

    # element mass and stiffness matrix
    Me = rho*A*Le/420*np.array([[156,    22*Le,   54,     -13*Le],
                                [22*Le,  4*Le**2,  13*Le,  -3*Le**2],
                                [54,     13*Le,   156,    -22*Le],
                                [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])
                       
    Ke = E*I/(Le**3)*np.array([[12,    6*Le,    -12,    6*Le],
                            [6*Le,  4*Le**2,  -6*Le,  2*Le**2],
                            [-12,   -6*Le,   12,     -6*Le],
                            [6*Le,  2*Le**2,  -6*Le,  4*Le**2]])
    
    # global stiffness matrix
    Ma = np.zeros([2*n,2*n])
    Ka = np.zeros([2*n,2*n])
    for i in range(0, 2*n-3, 2):
        Ma[i:i+4,i:i+4] = Ma[i:i+4,i:i+4] + Me
        Ka[i:i+4,i:i+4] = Ka[i:i+4,i:i+4] + Ke

    # boundary conditions !
    # bcs = 'general';
    # bcs = 'simply-supported';
    if bc == 'cantilever':
        # the left end is clamped !
        Ma = np.delete(Ma, [0,1], 1) # column delete
        Ma = np.delete(Ma, [0,1], 0) # row delete
        Ka = np.delete(Ka, [0,1], 1)
        Ka = np.delete(Ka, [0,1], 0)
        
    elif bc == 'simply-supported':
        # simply supported at two ends
        Ma = np.delete(Ma, [0,-2], 1) # first and second last column
        Ma = np.delete(Ma, [0,-2], 0) # first and second last row
        Ka = np.delete(Ka, [0,-2], 1)
        Ka = np.delete(Ka, [0,-2], 0)
        
    elif bc == 'general':
          # linear translational and rotational springs at both ends
          # E I y''' = - k_t2 * y   --- right end
          # E I y''' =   k_t1 * y   --- left end
          # E I y''  =   k_r2 * y'  --- right end
          # E I y''  = - k_r1 * y'  --- left end 
        Ka[0,0] = Ka(1,1) + k_t1;
        Ka[1,1] = Ka(2,2) + k_r1;
        Ka[-2,-2] = Ka(-2,-2) + k_t2; 
        Ka[-1,-1] = Ka[-1,-1] + k_r2;

    # natural frequency
    w2, _ = np.linalg.eig( np.matmul( np.linalg.inv(Ma),Ka ) )
    w2 = np.sort(w2)
    omega = np.sqrt(w2)
    lambdaL = np.sqrt(omega)*(rho*A/E/I)**(1/4)
    return Ma, Ka, omega, lambdaL


def Newmark(M,C,K,F,D0,V0,dt,T,Beta=1/4,Gamma=1/2):
    """
    Newmark method for deterministic linear time invariant system 
    Governing equation: M*Y''(t)+C*Y'(t)+K*Y(t)=S*Bd(t) 
    
    M  : matrix, global mass matrix
    C  : matrix, global damping matrix
    K  : matrix, global striffness matrix
    F  : scalar/vector, force at the beam
    D0 : vector, initial displacement
    V0 : vector, initial velocity
    dt : scalar, time increment
    T  : scalar, final time 
    Beta, Gamma : scalar, scalar, parameters of Newmark-beta
    
    Reference
    Dynamics of Structures. Chopra A K
    Dynamics of Structures. Clough R W, Penzien J
    """
    
    # integration constant
    c1 = 1/Beta/dt**2
    c2 = Gamma/Beta/dt
    c3 = 1/Beta/dt
    c4 = 1/2/Beta-1
    c5 = Gamma/Beta-1
    c6 = (Gamma/2/Beta-1)*dt
    c7 = (1-Gamma)*dt
    c8 = Gamma*dt

    A0 = np.dot( np.linalg.inv(M), (F(0)- np.dot(K,D0)- np.dot(C,V0)) )      # initial acceleration
    n = int(T/dt+1)
    m = len(D0)
    D = np.zeros([m,n])
    V = np.zeros([m,n])
    A = np.zeros([m,n])
    D[:, 0] = D0
    V[:, 0] = V0
    A[:, 0] = A0

    Kbar = c1*M + c2*C + K          # linear time-invariant system
    for i in range(n-1):
        Da = D[:, i]
        Va = V[:, i]
        Aa = A[:, i]
        Fbar = F(i*dt) + np.dot(M, (c1*Da+c3*Va+c4*Aa)) + np.dot(C, (c2*Da+c5*Va+c6*Aa))
        D[:,i+1] = np.matmul(np.linalg.inv(Kbar), Fbar)
        A[:,i+1] = c1*(D[:,i+1]-Da) -c3*Va -c4*Aa
        V[:,i+1] = Va +c7*Aa +c8*A[:,i+1]
    return D, V, A


def Newmark_stoch(M,C,K,b,D0,V0,dt,T,Nsamp,Beta=1/4,Gamma=1/2):
    """
    Newmark method for linear time invariant system with Euler-Maruyama for
    time-forwarding the stochastic Brownian motion 
    Governing equation: M*Y''(t)+C*Y'(t)+K*Y(t)=S*Bd(t) 
    
    M  : matrix, global mass matrix
    C  : matrix, global damping matrix
    K  : matrix, global striffness matrix
    b  : scalar, diffusion coefficient of Brownian motion
    D0 : vector, initial displacement
    V0 : vector, initial velocity
    dt : scalar, time increment
    T  : scalar, final time 
    Nsamp : scalar, ensemble size
    Beta, Gamma : scalar, scalar, parameters of Newmark-beta
    """
    
    # integration constant
    c1 = 1/Beta/dt**2
    c2 = Gamma/Beta/dt
    c3 = 1/Beta/dt
    c4 = 1/2/Beta-1
    c5 = Gamma/Beta-1
    c6 = (Gamma/2/Beta-1)*dt
    c7 = (1-Gamma)*dt
    c8 = Gamma*dt

    A0 = np.dot( np.linalg.inv(M), (0 - np.dot(K,D0)- np.dot(C,V0)) )  # initial acceleration
    n = int(T/dt+1)
    m = len(D0)

    Kbar = c1*M + c2*C + K          # linear time-invariant system
    D, V, A = [], [], [] 
    for ensemble in range(Nsamp):
        if ensemble % 20 == 0:
            print('Data generation, ensemble count-{}'.format(ensemble))
            
        Ds = np.zeros([m,n])
        Vs = np.zeros([m,n])
        As = np.zeros([m,n])
        Ds[:, 0] = D0
        Vs[:, 0] = V0
        As[:, 0] = A0
        
        for i in range(n-1):
            Da = Ds[:, i]
            Va = Vs[:, i]
            Aa = As[:, i]
            
            dW = np.random.randn(m)*np.sqrt(dt)
            Fbar = b*dW + np.dot(M, (c1*Da+c3*Va+c4*Aa)) + np.dot(C, (c2*Da+c5*Va+c6*Aa))
            
            Ds[:,i+1] = np.matmul(np.linalg.inv(Kbar), Fbar)
            As[:,i+1] = c1*(Ds[:,i+1]-Da) -c3*Va -c4*Aa
            Vs[:,i+1] = Va +c7*Aa +c8*As[:,i+1]
        D.append(Ds)
        V.append(Vs)
        A.append(As)
        
    D = np.array(D)
    V = np.array(V)
    A = np.array(A)
    return D, V, A


def E_Maruyama(M,C,K,b,D0,V0,dt,T):
    """
    Euler-Maruyama method for linear time invariant system 
    (Requires very fine time step to avoid overflow)
    Governing equation: M*Y''(t)+C*Y'(t)+K*Y(t)=S*Bd(t) 
    
    M  : matrix, global mass matrix
    C  : matrix, global damping matrix
    K  : matrix, global striffness matrix
    b  : scalar, diffusion coefficient of Brownian motion
    D0 : vector, initial displacement
    V0 : vector, initial velocity
    dt : scalar, time increment
    T  : scalar, final time 
    """
    
    n = int(T/dt+1)
    m = len(D0)
    xt = np.zeros([2*m,n])
    xt[:m, 0] = D0 
    xt[m:, 0] = V0 

    sys = np.row_stack((np.column_stack((np.zeros((m,m)), np.eye(m,m))),
                        np.column_stack((-np.matmul(np.linalg.inv(M),K), -np.matmul(np.linalg.inv(M),C))) ))
    
    for i in range(n-1):
        dW = np.random.randn(m)*np.sqrt(dt) 
        force = np.concatenate(( np.zeros(m), np.dot(np.linalg.inv(M), dW) ))
        
        xt_i = np.array(xt[:, i] )
        xt[:, i+1] = xt[:, i] + np.matmul(sys, xt_i)*dt + b*force  
        
    
    D = np.array(xt[:m, :])
    V = np.array(xt[m:, :])
    return D, V 
