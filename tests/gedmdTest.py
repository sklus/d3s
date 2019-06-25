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
bounds = sp.array([[-2, 2], [-2, 2]])
boxes = sp.array([30, 30])
Omega = domain.discretization(bounds, boxes)

# define system
def b(x):
     return np.vstack((-4*x[0, :]**3 + 4*x[0, :], -2*x[1, :]))
 
def sigma(x):
    n = x.shape[1]
    y = np.zeros((2, 2, n))
    y[0, 0, :] = 0.7
    y[1, 1, :] = 0.7
    return y

# define observables
order = 3
psi = observables.monomials(order)

X = Omega.randPerBox(10)
Y = b(X)
Z = sigma(X)

# apply generator EDMD
K = algorithms.gedmd(X, Y, Z, psi)

printMatrix(K)