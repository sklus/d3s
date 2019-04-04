#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt

import d3s.algorithms as algorithms

plt.ion()

# load variables from mat file into main scope
data = scipy.io.loadmat('data/vonKarman.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

nx, ny, m = x.shape
x = x.reshape([nx*ny, m])
X = x[:, :-1]
Y = x[:, 1:]

d, V = algorithms.dmd(X, Y)

for i in range(10):
    plt.figure()
    v = sp.real(V[:, i].reshape(nx, ny))
    plt.imshow(v)
