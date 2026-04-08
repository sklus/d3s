import numpy as np
import jax.numpy as jnp
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt

import d3s.algorithms as algorithms
import d3s.domain as domain
import d3s.observables as observables

from d3s.tools import printVector, printMatrix

#%% auxiliary functions and classes
def uniformInCircle(r):
    '''
    Generates uniformly distributed data points within the circle of radius r.
    '''
    phi = 2*np.pi*np.random.rand()
    rho = r*np.sqrt(np.random.rand())
    return np.array([rho*np.cos(phi), rho*np.sin(phi)])

def uniformInEllipse(a, b, c, r):
    '''
    Generates uniformly distributed data points within the ellipse defined
    by the equation a x_1^2 + b x_1 x_2 + c x_2^2 < r^2.
        
    The semi-major and semi-minor axes are given by 1/sqrt(lambda_i) and the
    eigenvectors v_i define the new coordinate system.
    '''
    d, V = sp.linalg.eigh(np.array([[a, 0.5*b], [0.5*b, c]]))
    p = uniformInCircle(r)
    return V @ (p/np.sqrt(d))

class myBasis(observables.jaxBasis):
    def __init__(self, a, b, c, k):
        '''
        Generates basis functions of the form
            (1 - a*x_1^2 - b*x_1*x_2 - c*x_2^2) * x_1^{i-j} * x_2^j.
        
        a, b, c: coefficients of the quadratic equation describing the ellipse
        k: order of monomials
        '''
        self.a = a
        self.b = b
        self.c = c
        self.k = k # maximum degree of basis functions
        self.n = (k+1)*(k+2)//2 # overall number of basis functions

    def __call__(self, x):
        s = x.shape
        if len(s) == 1:
            y = jnp.zeros(self.n)
        else:
            y = jnp.zeros((self.n, s[1]))
        
        r = 0
        for i in range(self.k+1):
            for j in range(i+1):
                y = y.at[r].set((1 - self.a*x[0]**2 - self.b*x[0]*x[1] - self.c*x[1]**2) * x[0]**(i-j) * x[1]**j)
                r += 1
        return y

def gedmd_KvN(X, Y, psi):
    '''
    Modified version of gEDMD that computes the Koopman generator L,
    the Perron-Frobenius generator L_adj, and the Koopman-von Neumann
    generator Q.
    '''
    PsiX = psi(X)
    dPsiX = psi.diff(X)
    dPsiY = np.einsum('ijk,jk->ik', dPsiX, Y)
    m = X.shape[1]
    
    C_0      = 1/m * PsiX @ PsiX.T
    C_0_pinv = sp.linalg.pinv(C_0)
    C_1      = 1/m * PsiX @ dPsiY.T

    L     = C_0_pinv @ C_1 # Koopman generator
    L_adj = C_0_pinv @ C_1.T # Perron-Frobenius generator
    Q     = C_0_pinv @ (-0.5*C_1 + 0.5*C_1.T) # Koopman-von Neumann generator

    return (L, L_adj, Q)

#%% damped or undamped oscillator
damped = False

# define domain
bounds = np.array([[-2, 2], [-2, 2]])
boxes = np.array([30, 30])
Omega = domain.discretization(bounds, boxes)

# define system
if not damped:
    A = np.array([[0.0, 1.0], [-2.0, 0.0]]) # undamped oscillator
    a, b, c, r = 1.0, 0.0, 0.5, 1.0         # parameters defining the domain
else:
    A = np.array([[0.0, 1.0], [-2.0, -2.0]]) # undamped oscillator
    a, b, c, r = 1.0, 1.0, 0.5, 1.0          # parameters defining the domain

def f(x):
    return A @ x

# generate training data
m = 2000 # number of data points
X = np.zeros((2, m))
for i in range(m):
    X[:, i] = uniformInEllipse(a, b, c, r)
Y = f(X)

# choose observables
k = 2 # maximum degree of basis functions
psi_ = myBasis(a, b, c, k)
n = psi_.n

psi = psi_ # without whitening transformation
# psi = observables.whitening(psi_, X) # with whitening transformation w.r.t. X

# apply gEDMD
L, L_adj, Q = gedmd_KvN(X, Y, psi)

printMatrix(L, 'L')
printMatrix(L_adj, 'L_adj')
printMatrix(Q, 'Q')

Q = np.array(Q) # jnp -> np
d, V = algorithms.sortEig(Q, evs=m)
printVector(d, 'lambda')

plt.figure(1)
plt.clf()
plt.plot(np.real(d), np.imag(d), '.')
plt.title('eigenvalues')
plt.axis('equal');

#%% plot eigenfunctions
Phi = np.zeros((n, m), dtype=complex)
for i in range(n):
    Phi[i, :] = V[:,i].T @ psi(X) # i-th eigenfunction
    
    s = np.real(Phi[i, :] / max(abs(Phi[i, :]))) # normalize and consider only real part
    
    fig = plt.figure(2+i)
    plt.clf()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[0, :], X[1, :], s, c=s)
    ax.set_aspect('equalxy')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.title(f'$\lambda_{i} = {d[i]:.2f}$')
    
#%% plot time evolution
alpha_0 = np.zeros(n, dtype=complex)
alpha_0[4] = 1

n_t = 401 # number of time steps
t = np.linspace(0, 2*np.pi/np.sqrt(2), n_t) # 2*np.pi/np.sqrt(2) is the period of the system

prop = sp.linalg.expm(np.diag(d)*t[1]) # propagator with lag time t[1]

alpha = np.zeros((n, n_t), dtype=complex)
alpha[:, 0] = alpha_0
for i in range(n_t-1):
    alpha[:, i+1] = prop @ alpha[:, i] # evolution of the coefficients

plt.figure(8)
plt.clf()
plt.plot(t, alpha.T)
plt.legend([f'$\\alpha_{i}$' for i in range(m)])

#%% visualize evolution of the corresponding function
fig = plt.figure(9)
plt.clf()
for i in range(0, n_t, 10):
    plt.clf()
    ax = fig.add_subplot(projection='3d')
    r = np.real(alpha[:, i].T @ Phi)
    ax.scatter(X[0, :], X[1, :], r, c=r)
    plt.title(f't = {t[i]:.2f}')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlim([-0.4, 0.4])
    ax.set_aspect('equalxy')
    
    plt.pause(0.05)
