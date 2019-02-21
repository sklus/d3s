#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import scipy
import scipy.io
import matplotlib

import d3s.domain as domain
import d3s.kernels as kernels
import d3s.algorithms as algorithms
import d3s.systems as systems

#%% Ornstein-Uhlenbeck process
Omega = domain.discretization(scipy.array([[-2, 2]]), scipy.array([50]))

f = systems.OrnsteinUhlenbeck(0.001, 500)
X = Omega.randPerBox(100)
Y = f(X)

sigma = scipy.sqrt(0.3)
epsilon = 0.1
k = kernels.gaussianKernel(sigma)

evs = 4 #d number of eigenfunctions to be computed

# Perron-Frobenius
d, V = algorithms.kernelEdmd(X, Y, k, epsilon=epsilon, evs=evs, operator='P')
for i in range(evs):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(X, V[:, i])

# Koopman
d, V = algorithms.kernelEdmd(X, Y, k, epsilon=epsilon, evs=evs, operator='K')
for i in range(evs):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(X, V[:, i]/scipy.amax(abs(V[:, i])))

#%% quadruple-well problem
data = scipy.io.loadmat('data/quadrupleWell_uniform.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s)) # load variables from mat file into main scope

Omega = domain.discretization(bounds, boxes)

evs = 4 #d number of eigenfunctions to be computed

# define kernel and regularization parameter
sigma = scipy.sqrt(2)
epsilon = 0
k = kernels.gaussianKernel(sigma)

# Perron-Frobenius
d, V = algorithms.kernelEdmd(X, Y, k, epsilon=epsilon, evs=evs, operator='P')
for i in range(evs):
    matplotlib.pyplot.figure()
    Omega.plot(scipy.real(V[:, i]))

# change bandwidth of kernel and regularization parameter
k.sigma = 0.5
epsilon = 0.1

# Koopman
d, V = algorithms.kernelEdmd(X, Y, k, epsilon=epsilon, evs=evs, operator='K')
for i in range(evs):
    matplotlib.pyplot.figure()
    Omega.plot(scipy.real(V[:, i]))

#%% string kernel example
words = ('computer', 'browser', 'tablet', 'internet', 'e-mail',
         'hurricane', 'storm', 'rain', 'damage', 'weather',
         'president', 'state', 'department', 'election', 'midterm',
         'science', 'stem', 'therapy', 'cell', 'disease')
data = [line.rstrip('\n') for line in open('data/news.txt')]

X = data[0:-1]
Y = data[1:]

k_s = kernels.stringKernel()
k = lambda x, y : scipy.exp(-k_s(x, y)**2/0.4)

d, V = algorithms.kernelEdmd(X, Y, k, epsilon=0.1, evs=4, operator='P')

# normalize eigenfunctions
for i in range(4):
    V[:, i] /= numpy.max(abs(V[:, i]))

# plot second vs. third dominant eigenfunction
matplotlib.pyplot.axis([-1, 1, -1, 1])
for i in range(len(words)):
    indices = [j for j, x in enumerate(data[:-1]) if x == words[i]]
    matplotlib.pyplot.plot(numpy.real(V[indices, 1]), numpy.real(V[indices, 2]), '.')
matplotlib.pyplot.show()
