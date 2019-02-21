#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy
import scipy.io
import matplotlib.pyplot
import d3s.algorithms as algorithms
import d3s.observables as observables

from d3s.tools import printVector, printMatrix

# load variables from mat file into main scope
data = scipy.io.loadmat('data/lorenz.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

d = X.shape[0]
p = 2 # maximum order of monomials

psi = observables.monomials(p)
Xi = algorithms.sindy(X, Y, psi, iterations=1)

c = observables.allMonomialPowers(d, p)

printMatrix(c)
printMatrix(Xi)
