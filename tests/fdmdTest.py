#--------------------------------------------------------------------------------------------------
# Functional DMD examples, see https://arxiv.org/pdf/2609.29159 for details.
#--------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.kernels as kernels
import d3s.systems as systems
import d3s.networks as networks

#%% Graphon example -------------------------------------------------------------------------------

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
n = 101 # number of subintervals
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

#%% degree function
degree = np.zeros_like(x)
for i in range(n):
    degree[i] = g.d(x[i])
    
#%% plot transition density function
for i in range(n):
    for j in range(n):
        Z[i, j] = g.p(X[i, j], Y[i, j])

fig = plt.figure(2)
plt.clf()
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('transition density function p')

#%% simulate continuous-time random walk process
def rhs(u, t, g, x):
    print(f't = {t}')
    
    n = x.shape[0]
    h = 1/(n-1)
    du = -u
    for i in range(n):
        for j in range(n-1):
            du[i] += h/2 * (g.p(x[j], x[i])*u[j] + g.p(x[j+1], x[i])*u[j+1]) # trapezoidal rule
            
    return du
    
sigma = 0.2
u0 = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-0.5)**2/(2*sigma**2)) # initial density
t = np.linspace(0.0, 5, 51) # time points
u = odeint(rhs, u0, t, args=(g, x)) # solve ODE

#%% plot solution
plt.figure(3)
gamma = np.sum(u0) / np.sum(degree) # multiplicative factor
for i in range(t.shape[0]):
    plt.clf()
    plt.plot(x, u[i, :])
    plt.plot(x, gamma*degree) # invariant density for comparison
    plt.title(f't = {t[i]:.3}')
    
    plt.ylim([0, 4.5])
    plt.pause(0.1)
    
#%% construct data matrices
U = u[:-1, :]
V = u[1:, :]

m = U.shape[0]

C_uu = np.zeros((m, m))
C_uv = np.zeros((m, m))

def inner_product(u, v, x):
    n = x.shape[0]
    h = 1/(n-1)
    s = 0
    for i in range(n-1):
        s += h/2 * (u[i]*v[i] + u[i+1]*v[i+1]) # trapezoidal rule
    return s

for i in range(m):
    for j in range(m):
        C_uu[i, j] = inner_product(U[i, :], U[j, :], x)
        C_uv[i, j] = inner_product(U[i, :], V[j, :], x)

#%% compute eigenvalues and eigenfunctions
evs = 4 # number of eigenvalues
A = sp.linalg.pinv(C_uu) @ C_uv
d, Xi = algorithms.sortEig(A, evs)

tau = t[1]-t[0] # lag time
mu = np.log(d)/tau + 1 # eigenvalues of Perron-Frobenius operator for discrete-time random walk

plt.figure(3)
plt.clf()
plt.plot(np.real(mu), '.')

phi = U.T @ Xi
for i in range(evs):
    phi[:, i] = phi[:, i] / np.max(abs(phi[:, i]))

plt.figure(4)
plt.clf()
plt.plot(x, phi)
plt.legend([f'$\phi_{i}$' for i in range(evs)])

#%% reconstruct graphon
w_r = np.zeros_like(X)

pi = phi[:, 0]
for i in range(3):
    phi_s = phi[:, i]
    s = np.sum(phi[:, i]**2 / pi) # normalize w.r.t. 1/pi
    phi_s = phi_s / s
    
    w_r = w_r + mu[i] * np.outer(phi_s, phi_s)
    
fig = plt.figure(5)
plt.clf()
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, 0.2 * w_r / np.max(abs(w_r)), cmap=plt.cm.coolwarm)
plt.xlabel('x')
plt.ylabel('y')
plt.title('reconstructed graphon w')

#%% clustering
c, l = sp.cluster.vq.kmeans2(np.real(phi[:, :3]), 3)

plt.figure(4)
plt.plot(x, l)
plt.show()

#%% Himmelblau system -----------------------------------------------------------------------------

#%% generate training data
Omega = domain.discretization(np.array([[-6, 6], [-6, 6]]), np.array([50, 50]))
f = systems.Himmelblau(0.001, 100)

n = 5000 # number of samples for kernel density estimation
m = 15 # overall number of snapshots

U = np.zeros((m, 2, n))
V = np.zeros((m, 2, n))

s = 3 # number of iterations per initial condition
for i in range(m//s):
    X = 0.5*np.random.randn(2, n) # normally distributed samples
    X[0, :] = X[0, :] + 6*(np.random.rand() - 0.5) # shift in x-direction
    X[1, :] = X[1, :] + 6*(np.random.rand() - 0.5) # shift in y-direction
    
    for j in range(s):
        Y = f(X)
        U[s*i+j, :, :] = X
        V[s*i+j, :, :] = Y
        X = Y

#%% plot training data
c = Omega.midpointGrid()
k = kernels.gaussianKernel(0.5)

for i in range(m):
    plt.figure(1)
    plt.clf()
    kde = kernels.kde(U[i, :, :], k)
    f_c = kde(c)

    Omega.plot(f_c, '3D')
    plt.title(f'i = {i}')
    
    plt.pause(0.5)  
    
#%% compute Gram matrices
C_uu = np.zeros((m, m))
C_uv = np.zeros((m, m))

for i in range(m):
    print(f'Row {i+1}/{m} ...')
    for j in range(m):
        C_uu[i, j] = np.sum(kernels.gramian2(U[i, :, :], U[j, :, :], k)) # could use symmetry here ...
        C_uv[i, j] = np.sum(kernels.gramian2(U[i, :, :], V[j, :, :], k))

#%% compute eigenvalues
evs = 5 # number of eigenvalues
A = sp.linalg.pinv(C_uu) @ C_uv
d, Xi = algorithms.sortEig(A, evs)

plt.figure(2)
plt.clf()
plt.plot(np.real(d), '.')

for l in range(3):
    phi_c = np.zeros(c.shape[1])
    for i in range(m):
        kde = kernels.kde(U[i, :, :], k) # function u_i
        f_i = kde(c) # evaluate u_i in c
        phi_c += np.real(Xi[i, l]) * f_i.squeeze()

    plt.figure(3+l)
    plt.clf()
    Omega.plot(phi_c, '3D')
    plt.title(f'$\lambda = {{{np.real(d[l])}}}$')
