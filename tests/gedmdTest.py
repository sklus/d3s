#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import d3s.algorithms as algorithms
import d3s.observables as observables
import d3s.domain as domain

from d3s.tools import printVector, printMatrix

plt.ion()

#%% Ornstein-Uhlenbeck

# define system
alpha = 1.5
beta = 1

def b(x):
    return -alpha*x

def sigma(x):
    return np.sqrt(2/beta)*np.ones((1, 1, X.shape[1]))

# define observables
order = 10
psi = observables.monomials(order)

X = domain.randb(5000, (-2, 2))
Y = b(X)
Z = sigma(X)

# apply generator EDMD
K = algorithms.gedmd(X, Y, Z, psi)

printMatrix(K)

#%% simple dobule-well process

# define domain
bounds = sp.array([[-2, 2], [-1, 1]])
boxes = sp.array([40, 20])
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
order = 4
psi = observables.monomials(order)

X = Omega.randPerBox(10)
Y = b(X)
Z = sigma(X)

# apply generator EDMD
K = algorithms.gedmd(X, Y, Z, psi)

c = observables.allMonomialPowers(2, order)
printMatrix(c, 'c')

printMatrix(K, 'K')

# compute entries of a evaluated in c
c = Omega.midpointGrid()
Psi_c = psi(c)
b_c = K[:, 1:3].T @Psi_c

a_11 = K[:, 3].T @ Psi_c - 2*b_c[0, :]*c[0, :]
a_12 = K[:, 4].T @ Psi_c - b_c[0, :]*c[1, :] - b_c[1, :]*c[0, :]
a_22 = K[:, 5].T @ Psi_c - 2*b_c[1, :]*c[1, :]
