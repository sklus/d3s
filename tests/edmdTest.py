#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy
import scipy.io
import matplotlib
import d3s.domain as domain
import d3s.observables as observables
import d3s.algorithms as algorithms
import d3s.systems as systems

# define domain
bounds = scipy.array([[-2, 2], [-2, 2]])
boxes = scipy.array([30, 30])
Omega = domain.discretization(bounds, boxes)

# generate training data
f = systems.DoubleWell2D(1e-3, 10000)
X = Omega.rand(50000)
Y = f(X) # double-well in two dimensions

# choose observables
psi = observables.monomials(10)
#psi = observables.indicators(Omega)

evs = 4 # number of eigenvalues/eigenfunctions to be computed
PsiC = psi(Omega.midpointGrid()) # observables evaluated at midpoints of the grid

#%% EDMD for Perron-Frobenius operator
d, V = algorithms.edmd(X, Y, psi, operator='P', evs=evs)
for i in range(evs):
    matplotlib.pyplot.figure()
    r = scipy.real(V[:,i].transpose() @ PsiC)
    Omega.plot(r, '3D')
    matplotlib.pyplot.title('EDMD P, eigenfunction  %d' % i)

#%% EDMD for Koopman operator    
d, V = algorithms.edmd(X, Y, psi, operator='K', evs=evs)
for i in range(evs):
    matplotlib.pyplot.figure()
    r = scipy.real(V[:,i].transpose() @ PsiC)
    Omega.plot(r, '3D')
    matplotlib.pyplot.title('EDMD K, eigenfunction %d' % i)

#%% Ulam's method for Perron-Frobenius operator
d, V = algorithms.ulam(X, Y, Omega, operator='P', evs=evs)
for i in range(evs):
    matplotlib.pyplot.figure()
    r = scipy.real(V[:,i])
    Omega.plot(r, '3D')
    matplotlib.pyplot.title('Ulam P, eigenfunction %d' % i)

#%% Ulam's method for Koopman operator
d, V = algorithms.ulam(X, Y, Omega, operator='K', evs=evs)
for i in range(evs):
    matplotlib.pyplot.figure()
    r = scipy.real(V[:,i])
    Omega.plot(r, '3D')
    matplotlib.pyplot.title('Ulam K, eigenfunction %d' % i)
