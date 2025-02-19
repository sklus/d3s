import numpy as np
import matplotlib.pyplot as plt

import d3s.kernels as kernels
import d3s.algorithms as algorithms

#%% create data set
n = 100
theta = np.linspace(0, 2*np.pi, n)
noise = 0.0
X = np.vstack((np.cos(theta), np.sin(theta))) + noise*np.random.randn(2, n)
y = theta < np.pi
X[:, y] = X[:, y] - 1/np.sqrt(2);

plt.figure(1)
plt.clf()
plt.scatter(X[0, y==0], X[1, y==0], color='red', alpha=0.5)
plt.scatter(X[0, y==1], X[1, y==1], color='blue', alpha=0.5)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

#%% apply kernel PCA
sigma = 1/np.sqrt(200)
k = kernels.gaussianKernel(sigma)

d, V = algorithms.kpca(X, k, 2)

plt.figure(2)
plt.clf()
plt.scatter(V[y==0, 0], V[y==0, 1], color='red', alpha=0.5)
plt.scatter(V[y==1, 0], V[y==1, 1], color='blue', alpha=0.5)
plt.xlabel('pc_1')
plt.ylabel('pc_2')
plt.show()
