import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import d3s.domain as domain
import d3s.algorithms as algorithms
import d3s.observables as observables
import d3s.kernels as kernels
import d3s.networks as networks

#%% define graphon
def w(x, y):
    '''
    Symmetric graphon with three peaks.
    '''
    return 0.2*np.exp(-((x - 0.2)**2 + (y - 0.2)**2 ) / 0.02) \
         + 0.1*np.exp(-((x - 0.5)**2 + (y - 0.5)**2 ) / 0.02) \
         + 0.2*np.exp(-((x - 0.8)**4 + (y - 0.8)**4 ) / 0.0005)

g = networks.graphon(w)

#%% plot graphon
n = 101
x = np.linspace(0, 1, n) # equidistant grid points
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for i in range(n):
    for j in range(n):
        Z[i, j] = g.w(X[i, j], Y[i, j])

fig = plt.figure(1)
plt.clf()
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('graphon w')

#%% plot degree function
degree = np.zeros_like(x)
for i in range(n):
    degree[i] = g.d(x[i])

plt.figure(2)
plt.clf()
plt.plot(x, degree)
plt.title('degree function d')

#%% plot transition density function
for i in range(n):
    for j in range(n):
        Z[i, j] = g.p(X[i, j], Y[i, j])

fig = plt.figure(3)
plt.clf()
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('transition density function p')

#%% compute one long random walk
m = 20000
r = g.randomWalk(0.5, m) # 0.5 is the initial position of the random walker

plt.figure(4)
plt.clf()
plt.plot(r)
plt.title('random walk')

Xd = r[:-1].reshape([1, m]) #  data matrices ...
Yd = r[1:].reshape([1, m])  #  ... for EDMD

#%% estimate invariant density
k = kernels.gaussianKernel(0.03)
pi = kernels.kde(Xd, k)
pi_x = pi(x[np.newaxis, :]).squeeze()

plt.figure(5)
plt.clf()
plt.plot(x, pi_x)
plt.title('estimated invariant density')

#%% apply EDMD and plot eigenfunctions
Omega = domain.discretization(np.array([[-0.1, 1.1]]), np.array([20]))
psi = observables.gaussians(Omega, sigma=0.05)

evs = 4 # number of eigenvalues to compute
_, d, V = algorithms.edmd(Xd, Yd, psi, evs=evs, operator='K')

phi_x = np.real(V.T @ psi(x[np.newaxis, :])) # Koopman eigenfunctions
phi_hat_x = np.real(V.T @ psi(x[np.newaxis, :])) # PF eigenfunctions, still need to be multiplied by pi

for i in range(evs):
    phi_i = lambda x: (V[:, i].T @ psi(np.array([[x]])))[0]
    f = lambda x: phi_i(x)**2 * pi(x)
    I = np.sqrt(sp.integrate.quad(f, 0, 1)[0])
    phi_x[i, :] = phi_x[i, :] / I # normalize w.r.t. pi-weighted norm
    
    phi_hat_x[i, :] = phi_hat_x[i, :] * pi_x
    phi_hat_i = lambda x: (V[:, i].T @ psi(np.array([[x]])))[0] * pi(x)
    f = lambda x: phi_hat_i(x)**2 / pi(x)
    I = np.sqrt(sp.integrate.quad(f, 0, 1)[0])
    phi_hat_x[i, :] = phi_hat_x[i, :] / I # normalize w.r.t. 1/pi-weighted norm

_, l = sp.cluster.vq.kmeans2(np.real(phi_x[:3, :]).T, 3) # apply k-means clustering

plt.figure(6)
plt.clf()
plt.plot(np.real(d), '.')
plt.title('eigenvalues')

plt.figure(7)
plt.clf()
plt.plot(x, phi_x[:3, :].T)
plt.title('Koopman eigenfunctions')

plt.figure(8)
plt.clf()
plt.plot(x, phi_hat_x[:3, :].T)
plt.title('Perron-Frobenius eigenfunctions')

plt.figure(9)
plt.clf()
plt.plot(x, l, '.')
plt.title('clustering')

#%% approximate p and w using dominant eigenfunctions
Z1 = np.zeros_like(X)
Z2 = np.zeros_like(X)
for i in range(n):
    for j in range(n):
        for k in range(3): # use first three eigenfunctions for the reconstruction
            Z1[i, j] = Z1[i, j] + d[k] * phi_hat_x[k, i] * phi_x[k, j] # p
            Z2[i, j] = Z2[i, j] + d[k] * phi_hat_x[k, i] * phi_hat_x[k, j] # w

fig = plt.figure(10)
plt.clf()
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z1, cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('reconstructed p')

fig = plt.figure(11)
plt.clf()
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z2, cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('reconstructed w')

#%% sample graph from graphon
n_v = 200
x_A, A = g.sampleGraph(n_v, makeSymmetric=False)

plt.figure(12)
plt.clf()
plt.imshow(A)
