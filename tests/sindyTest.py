#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.io
import d3s.algorithms as algorithms
import d3s.observables as observables

from d3s.tools import printVector, printMatrix

#%% load variables from mat file into main scope
data = sp.io.loadmat('data/lorenz.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

#%% apply SINDy
d = X.shape[0]
p = 2 # maximum order of monomials

psi = observables.monomials(p)
Xi1 = algorithms.sindy(X, Y, psi, iterations=1)

c = observables.allMonomialPowers(d, p)
n = c.shape[1] # number of functions

#%% output results
printMatrix(c)
printMatrix(Xi1)

#%% apply gEDMD
K, _, _ = algorithms.gedmd(X, Y, None, psi)

# construct projection onto full-state observable
B = np.zeros((10, d))
for i in range(3):
    B[i+1, i] = 1
Xi2 = (K@B).T
printMatrix(Xi2)