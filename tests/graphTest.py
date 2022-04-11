#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import d3s.networks as networks

import d3s.algorithms as algorithms
from d3s.tools import printVector, printMatrix

#%% Simple guiding example ------------------------------------------------------------------------

# construct adjacency matrix
n = 12
A = np.zeros((n, n));
C = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]);
A[0:4, 0:4] = C
A[4:8, 4:8] = C
A[8:12, 8:12] = C
A[3, 4] = 0.01
A[7, 8] = 0.01
A[11, 0] = 0.01

# apply spectral clustering
G = networks.graph(A)
d, V, c = networks.spectralClustering(G, 3, 'fb')
x = G.randomWalk(1, 10000)
printVector(c)

plt.figure(1)
plt.clf()
G.draw(c)

plt.figure(2)
plt.clf()
plt.plot(np.real(V))

plt.figure(3)
plt.clf()
plt.plot(x)

#%% Randomly generated graph ----------------------------------------------------------------------

# load variables from mat file into main scope
data = sp.io.loadmat('data/randomGraph.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

# apply spectral clustering
G = networks.graph(A)
d, V, c = networks.spectralClustering(G, 10, 'fb')

plt.figure(4)
plt.clf()
G.draw(c)

plt.figure(5)
plt.clf()
plt.plot(c)

#%% Double-well graph -----------------------------------------------------------------------------

# load variables from mat file into main scope
data = sp.io.loadmat('data/doubleWellGraph.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

[n_v, n_t] = A.shape[1:]
pos = {i: (p[0, i], p[1, i]) for i in range(n_v)} # positions of the vertices

# generate random walks in time-evolving graph
m = 5000 # number of random walkers
Z = np.zeros((m, 10*n_t+1), dtype=np.uint64)
Z[:, 0] = np.random.randint(0, n_v, size=m)

plt.figure(6)
for i in range(n_t):
    plt.clf()
    G = networks.graph(A[:, :, i])
    
    for j in range(m):
        z = G.randomWalk(Z[j, 10*i], 10)
        Z[j, 10*i+1:10*i+11] = z
    
    G.draw(pos=pos)
    
    z = Z[:, 10*i+10]
    r0 = p[0, z] + 0.1*np.random.randn(m,)
    r1 = p[1, z] + 0.1*np.random.randn(m,)
    plt.plot(r0, r1, 'r.')
    
    plt.pause(0.5)

# compute (cross-)covariance matrices
X = np.zeros((n_v, m))
Y = np.zeros((n_v, m))

for i in range(m):
    p = Z[i, 0]
    X[p, i] = 1
    
    q = Z[i, -1]
    Y[q, i] = 1
    
C_xx = 1/m*(X@X.T)
C_xy = 1/m*(X@Y.T)
C_yy = 1/m*(Y@Y.T)

eps = 1e-8
F = algorithms.dinv(C_xx + eps*np.eye(n_v)) @ C_xy @ algorithms.dinv(C_yy + eps*np.eye(n_v)) @ C_xy.T

d, V = algorithms.sortEig(F, 5)

# plot results
plt.figure(7)
plt.clf()
plt.plot(d, '.')

plt.figure(8)
plt.clf()
plt.plot(V[:, :2], '.-')

s = algorithms.seba(V[:, :2])
c = np.argmax(s, axis=1) + 1
ind, = np.where(np.sum(s, axis=1) < 0.1)
c[ind] = 0

c[c == 0] = 123 # yellow (transition region)
c[c == 1] = 50  # green (coherent set 1)
c[c == 2] = 99  # red (coherent set 2)

plt.figure(9)
G = networks.graph(A[:, :, 0])
G.draw(c=c, pos=pos)