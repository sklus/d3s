#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.io
import scipy.cluster
import matplotlib.pyplot as plt

import d3s.kernels as kernels
import d3s.algorithms as algorithms

plt.ion()

#%% Bickley jet -----------------------------------------------------------------------------------

#%% load variables from mat file into main scope
data = sp.io.loadmat('data/bickley.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

#%% apply kernel CCA to detect coherent sets
sigma = 1
k = kernels.gaussianKernel(sigma)

evs = 9 # number of eigenfunctions to be computed
d, V = algorithms.kcca(X, Y, k, evs, epsilon=1e-3)

#%% plot eigenvectors
for i in range(evs):
    plt.figure(figsize=(3, 2))
    plt.scatter(X[0, :], X[1, :], c=V[:, i])
plt.show()

#%% k-means of eigenvectors
c, l = sp.cluster.vq.kmeans2(np.real(V[:, :5]), 5)
plt.figure(figsize=(3, 2))
plt.scatter(X[0, :], X[1, :], c=l)
plt.show()

#%% seba
S = algorithms.seba(np.real(V))
plt.figure(figsize=(3, 2))
plt.scatter(X[0, :], X[1, :], c=S[:, 0:7].sum(axis=1))
plt.show()

#%% time-dependent 5-well potential ---------------------------------------------------------------

#%% load variables from mat file into main scope
data = sp.io.loadmat('data/moving5well.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

#%% apply kernel CCA to detect coherent sets
evs = 10 # number of eigenfunctions to be computed
d, V = algorithms.kcca(X, Y, k, evs, epsilon=1e-2)

#%% plot eigenvalues
plt.figure()
plt.plot(d, 'o')
plt.show()

#%% k-means of eigenvectors
c, l = sp.cluster.vq.kmeans2(np.real(V[:, 0:5]), 5)
fig = plt.figure()
plt.scatter(X[0, :], X[1, :], c=l)
plt.show()

#%% apply diffusion maps algorithm

# because the algorithm works with any number of timesteps, we nee to adjust
# the data format: X.shape = (n_dim x n_points); Y same but one timestep later

X_coords = _np.vstack((X[0], Y[0])).T  # For x coordinates across timesteps
Y_coords = _np.vstack((X[1], Y[1])).T  # For y coordinates across timesteps

E = algorithms.diffMaps(X_coords, Y_coords, eps='bh', r=-1)

#%% plot eigenvalues
plt.figure()
plt.plot(E[0], 'o')
plt.show()

#%% k-means of eigenvectors

c, l = sp.cluster.vq.kmeans2(np.real(E[1][:, 0:6]), 5)
fig = plt.figure()
plt.scatter(X[0, :], X[1, :], c=l)
plt.show()


