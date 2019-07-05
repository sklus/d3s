#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.polynomial
import scipy as sp
import matplotlib.pyplot as plt

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables
import d3s.systems as systems

from d3s.tools import printVector, printMatrix

plt.ion()

#%% Ornstein-Uhlenbeck process --------------------------------------------------------------------

# define domain
bounds = sp.array([[-2, 2]])
boxes = sp.array([1000])
Omega = domain.discretization(bounds, boxes)

# define system
alpha = 1
beta = 4

def b(x):
    return -alpha*x

def sigma(x):
    return np.sqrt(2/beta)*np.ones((1, 1, X.shape[1]))

# Euler-Maruyama integrator for Ornstein-Uhlenbeck (for comparison with EDMD; make sure the same parameters are used)
h = 0.001
tau = 0.5
f = systems.OrnsteinUhlenbeck(h, int(tau/h))

# define observables
psi = observables.monomials(10)

evs = 5 # number of eigenvalues/eigenfunctions to be computed
X = Omega.rand(10000) # generate test points

# apply generator EDMD
Y1 = b(X)
Z1 = sigma(X)

K1, d1, V1 = algorithms.gedmd(X, Y1, Z1, psi, evs=evs, operator='K')
printMatrix(K1, 'K_gEDMD')
printVector(np.real(d1), 'd_gEDMD')

# apply standard EDMD
Y2 = f(X)

K2, d2, V2 = algorithms.edmd(X, Y2, psi, evs=evs, operator='K')
printVector(1/tau*np.log(np.real(d2)), 'd_EDMD')

# collect results
c = Omega.midpointGrid()
R1 = V1.T @ psi(c) # gEDMD eigenfunctions
R2 = V2.T @ psi(c) #  EDMD eigenfunctions
R3 = np.zeros((evs, c.shape[1]))
for i in range(evs):
    q = np.zeros((evs,))
    q[i] = 1
    he = np.polynomial.hermite_e.HermiteE(q)
    R3[i, :] = he(np.sqrt(alpha*beta)*c) # rescaled probabilists' Hermite polynomials
    
    # normalize
    R1[i, :] = R1[i, :]/np.amax(abs(R1[i, :]))
    R2[i, :] = R2[i, :]/np.amax(abs(R2[i, :]))
    R3[i, :] = R3[i, :]/np.amax(abs(R3[i, :]))

# plot results
plt.figure(1)
plt.clf()
plt.plot(c.T, R1.T)
plt.title('gEDMD')
plt.legend([ 'phi_%i'% (i+1) for i in range(evs) ])

plt.figure(2)
plt.clf()
plt.plot(c.T, R2.T)
plt.title('EDMD')
plt.legend([ 'phi_%i'% (i+1) for i in range(evs) ])

plt.figure(3)
plt.clf()
plt.plot(c.T, R3.T)
plt.title('True solution')
plt.legend([ 'phi_%i'% (i+1) for i in range(evs) ])

#%% Simple dobule-well process --------------------------------------------------------------------

# define domain
bounds = sp.array([[-2, 2], [-1.5, 1.5]])
boxes = sp.array([20, 15])
Omega = domain.discretization(bounds, boxes)

# define system
def b(x):
     return np.vstack((-4*x[0, :]**3 + 4*x[0, :], -2*x[1, :]))
 
def sigma(x):
    n = x.shape[1]
    y = np.zeros((2, 2, n))
    y[0, 0, :] = 0.7
    y[0, 1, :] = x[0, :]
    y[1, 1, :] = 0.5
    return y

# define observables
order = 10
psi = observables.monomials(order)
# psi = observables.gaussians(Omega, 0.5)

X = Omega.randPerBox(200)
Y = b(X)
Z = sigma(X)

# apply generator EDMD
evs = 3
K, d, V = algorithms.gedmd(X, Y, Z, psi, evs=evs, operator='K')

# compute eigenfunctions
printVector(np.real(d), 'd')

c = Omega.midpointGrid()
Psi_c = psi(c)
for i in range(evs):
    plt.figure(i+1);
    plt.clf()
    Omega.plot(np.real( V[:, i].T @ Psi_c ), mode='3D')

#%% system identification
order = 4 # reduce order of monomials
p = observables.allMonomialPowers(2, order)
n = p.shape[1] # number of monomials up to order 4

printMatrix(p, 'p')
printMatrix(K[:n, :n], 'K')

# compute entries of a evaluated in c
b_c = K[:, 1:3].T @Psi_c

a_11 = K[:, 3].T @ Psi_c - 2*b_c[0, :]*c[0, :]
a_12 = K[:, 4].T @ Psi_c - b_c[0, :]*c[1, :] - b_c[1, :]*c[0, :]
a_22 = K[:, 5].T @ Psi_c - 2*b_c[1, :]*c[1, :]

plt.figure(evs+1)
Omega.plot(a_11, mode='3D')
plt.figure(evs+2)
Omega.plot(a_12, mode='3D')
plt.figure(evs+3)
Omega.plot(a_22, mode='3D')
