import numpy as np
import scipy as sp

import d3s.algorithms as algorithms

import matplotlib.pyplot as plt

c = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

def plot(x, l):
    k = l.max()+1
    plt.figure()
    for i in range(k):
        plt.plot(x[0, l==i], x[1, l==i], c[np.mod(i, len(c))]+'.', markersize=10)

k = 5
m = 1000
X = np.random.rand(2, m)
l = algorithms.kmeans(X, k, 1000)

plot(X, l)
