#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import d3s.networks as networks
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

[n, m] = A.shape[1:]
pos = {i: (p[0, i], p[1, i]) for i in range(n)} # positions of the vertices

plt.figure(5)
for i in range(m):
    print(i)
    plt.clf()
    G = networks.graph(A[:, :, i])
    G.draw(pos=pos)
    plt.pause(0.5)

