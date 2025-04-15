import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import d3s.algorithms as algorithms
from d3s.tools import printMatrix

#%% auxiliary functions
def qendyEval(z, A, B, C):
    '''
    Evaluate quadratic system given by the matrices A, B, and C in z.
    '''
    z_ = z[:, np.newaxis]
    dz = A @ np.kron(z_, z_) +  B @ z_ + C
    return dz[:, 0]

def sindyEval(x, psi, Xi):
    '''
    Evaluate SINDy approximation in x.
    '''
    x_ = x[:, np.newaxis]
    dx = Xi @ psi(x_)
    return dx[:, 0]

def sysSimulation(x0, b, t):
    '''
    Simulate original system.
    '''
    sol = sp.integrate.solve_ivp(lambda t, x: b(x[:, np.newaxis])[:, 0], [t[0], t[-1]], x0[:, 0], t_eval=t, rtol=1e-12, atol=1e-12)
    return (sol.t, sol.y)

def qendySimulation(z0, A, B, C, t):
    '''
    Simulate system identified by QENDy.
    '''
    sol = sp.integrate.solve_ivp(lambda t, z: qendyEval(z, A, B, C), [t[0], t[-1]], z0[:, 0], t_eval=t, rtol=1e-12, atol=1e-12)
    return (sol.t, sol.y)

def sindySimulation(x0, psi, Xi, t):
    '''
    Simulate system identified by SINDy.
    '''
    sol = sp.integrate.solve_ivp(lambda t, x: sindyEval(x, psi, Xi),  [t[0], t[-1]], x0[:, 0], t_eval=t, rtol=1e-12, atol=1e-12)
    return (sol.t, sol.y)

#%% simple benchmark problem

# define rational system (see Goyal & Benner, 2024)
def b(x):
    return -x / (1 + x)

# define observables
class rationalBasis(object):
    def __call__(self, x):
        return np.vstack( (x, 1/(1+x), x/(1+x)**2) )
    
    def diff(self, x):
        m = x.shape[1]
        y = np.zeros((3, 1, m))
        y[0, 0, :] = np.ones_like(x)
        y[1, 0, :] = -1/(1+x)**2
        y[2, 0, :] = 1/(1+x)**2 - 2*x/(1+x)**3
        return y

# generate training data
X = np.linspace(0, 1, 1001).reshape((1, 1001))
dX = b(X)

# choose obvervables
psi = rationalBasis()

# apply QENDy
A, B, C = algorithms.qendy(X, dX, psi)
printMatrix(A, 'A')
printMatrix(B, 'B')
printMatrix(C, 'C')

# apply SINDy
Xi = algorithms.sindy(X, dX, psi)
printMatrix(Xi, 'Xi')

#%% compare identified models
t0 = 0
t1 = 10
ts = np.linspace(t0, t1, 50)

#% simulate original system
x0 = np.array([[2]])
t1, x1 = sysSimulation(x0, b, ts)

#% simulate QENDy system
z0 = psi(x0)
t2, x2 = qendySimulation(z0, A, B, C, ts)

#% simulate SINDy system
t3, x3 = sindySimulation(x0, psi, Xi, ts)

#% plot solutions
plt.clf()
plt.plot(t1, x1[0, :], label='original system')
plt.plot(t2, x2[0, :], '.', label='QENDy') # use 1 here since FSO is the second state, the first one is the constant function
plt.plot(t3, x3[0, :], '.', label='SINDy')
plt.legend()

#%% damped pendulum

# define system
def b(x):
    c = 0.1 # damping coefficient
    return np.vstack((x[1, :], -np.sin(x[0,:]) - c*x[1, :]))

# define observables
class trigonometricBasis(object):
    def __call__(self, x):
        return np.vstack( (x, np.sin(x[0, :]), np.cos(x[0, :])) )
    
    def diff(self, x):
        m = x.shape[1]
        y = np.zeros((4, 2, m))
        y[0, 0, :] = np.ones(m)
        y[1, 1, :] = np.ones(m)
        y[2, 0, :] = np.cos(x[0, :])
        y[3, 0, :] = -np.sin(x[0, :])
        return y

# generate training data
X = 2*(np.random.rand(2, 100) - 0.5)
dX = b(X)

# choose obvervables
psi = trigonometricBasis()

# apply QENDy
A, B, C = algorithms.qendy(X, dX, psi)
printMatrix(A, 'A')
printMatrix(B, 'B')
printMatrix(C, 'C')

# apply SINDy
Xi = algorithms.sindy(X, dX, psi)
printMatrix(Xi, 'Xi')

#%% compare identified models
t0 = 0
t1 = 20
ts = np.linspace(t0, t1, 200)

#% simulate original system
x0 = np.array([[1], [0]])
t1, x1 = sysSimulation(x0, b, ts)

#% simulate QENDy system
z0 = psi(x0)
t2, x2 = qendySimulation(z0, A, B, C, ts)

#% simulate SINDy system
t3, x3 = sindySimulation(x0, psi, Xi, ts)

#% plot solutions
plt.clf()
plt.plot(t1, x1.T, label='original system')
plt.plot(t2, x2[:2, :].T, '.', label='QENDy')
plt.plot(t3, x3.T, '.', label='SINDy')
plt.legend()

#%% plot solution in phase space
plt.clf()
plt.plot(x1[0, :], x1[1, :], label='original system')
plt.plot(x2[0, :], x2[1, :], '.', label='QENDy')
plt.legend()

#%% Thomas' cyclically symmetric attractor

epsilon = 0 # case 1
#epsilon = 1 # case 2

# define system
def b(x):
    alpha = 0.2; beta = 0 # case 1
    #alpha = 0.25; beta = 0.15 # case 2
    return np.vstack(( np.sin(x[1, :]) - alpha*x[0, :] - beta*x[1, :]*np.cos(x[0, :]),
                       np.sin(x[2, :]) - alpha*x[1, :] - beta*x[2, :]*np.cos(x[1, :]),
                       np.sin(x[0, :]) - alpha*x[2, :] - beta*x[0, :]*np.cos(x[2, :]) ))

# define observables
class thomasBasis(object):
    def __call__(self, x):
        return np.vstack( (x, np.sin(x), np.cos(x)) )
    
    def diff(self, x):
        m = x.shape[1]
        y = np.zeros((9, 3, m))
        y[0, 0, :] = np.ones(m)
        y[1, 1, :] = np.ones(m)
        y[2, 2, :] = np.ones(m)
        y[3, 0, :] = np.cos(x[0, :])
        y[4, 1, :] = np.cos(x[1, :])
        y[5, 2, :] = np.cos(x[2, :])
        y[6, 0, :] = -np.sin(x[0, :])
        y[7, 1, :] = -np.sin(x[1, :])
        y[8, 2, :] = -np.sin(x[2, :])
        return y

# generate training data
x0 = np.array([[1.0], [-1.0], [0.0]]) # case 1
#x0 = np.array([[0], [1.0], [1.0]]) # case 2
_, X = sysSimulation(x0, b, np.linspace(0, 100, 1000))
dX = b(X)

# visualize training data
plt.clf()
ax = plt.figure(1).add_subplot(projection='3d')
ax.plot(X[0, :], X[1, :], X[2, :], label='training data')
plt.legend()

# choose obvervables
psi = thomasBasis()

# apply QENDy
A, B, C = algorithms.qendy(X, dX, psi, epsilon=epsilon)

printMatrix(A, 'A')
printMatrix(B, 'B')
printMatrix(C, 'C')

# apply SINDy
Xi = algorithms.sindy(X, dX, psi)
printMatrix(Xi, 'Xi')

#%% compare identified models
t0 = 0
t1 = 100
ts = np.linspace(t0, t1, 1000)

#% simulate original system
x0 = np.array([[0.0], [1.0], [1.0]]) # case 1
# x0 = np.array([[1.0], [-1.0], [0.0]]) # case 2
t1, x1 = sysSimulation(x0, b, ts)

#% simulate QENDy system
z0 = psi(x0)
t2, x2 = qendySimulation(z0, A, B, C, ts)

#% simulate SINDy system
t3, x3 = sindySimulation(x0, psi, Xi, ts)

#% plot solutions
plt.clf()
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(t1, x1[i, :].T, label='original system')
    plt.ylim([-5, 5])
    plt.plot(t2, x2[i, :].T, label='QENDy')
    plt.ylim([-5, 5])
    plt.plot(t3, x3[i, :].T, label='SINDy')
    plt.ylim([-5, 5])
    
plt.legend()

#%% plot solution in phase space
plt.clf()

ax = plt.figure(1).add_subplot(projection='3d')
ax.plot(x1[0, :], x1[1, :], x1[2, :], label='original system')
ax.plot(x2[0, :], x2[1, :], x2[2, :], label='QENDy')
ax.plot(x3[0, :], x3[1, :], x3[2, :], label='SINDy')

plt.legend()
