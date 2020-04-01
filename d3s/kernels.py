#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as _np
from scipy.spatial import distance


class gaussianKernel(object):
    '''Gaussian kernel with bandwidth sigma.'''
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x, y):
        return _np.exp(-_np.linalg.norm(x-y)**2/(2*self.sigma**2))
    def diff(self, x, y):
        return -1/self.sigma**2*(x - y) * self(x, y)
    def ddiff(self, x, y):
        return (1/self.sigma**4*_np.outer(x-y, x-y) - 1/self.sigma**2 *_np.eye(x.shape[0])) * self(x, y) 
    def __repr__(self):
        return 'Gaussian kernel with bandwidth sigma = %f.' % self.sigma


class laplacianKernel(object):
    '''Laplacian kernel with bandwidth sigma.'''
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x, y):
        return _np.exp(-_np.linalg.norm(x-y)/self.sigma)
    def __repr__(self):
        return 'Laplacian kernel with bandwidth sigma = %f.' % self.sigma


class polynomialKernel(object):
    '''Polynomial kernel with degree p and inhomogeneity c.'''
    def __init__(self, p, c=1):
        self.p = p
        self.c = c
    def __call__(self, x, y):
        return (self.c + x.T @ y)**self.p
    def diff(self, x, y):
        return self.p*(self.c + x.T @ y)**(self.p-1)*y;
    def ddiff(self, x, y):
        return self.p*(self.p-1)*(self.c + x.T @ y)**(self.p-2) * _np.outer(y, y);
    def __repr__(self):
        return 'Polynomial kernel with degree p = %f and inhomogeneity c = %f.' % (self.p, self.c)


class stringKernel(object):
    '''
    String kernel implementation based on Marianna Madry's C++ code, see
    https://github.com/mmadry/string_kernel.
    '''
    def __init__(self, kn = 2, l = 0.9):
        self._kn = kn # level of subsequence matching
        self._l  = l  # decay factor

    def __call__(self, x, y):
        return self.evaluate(x, y) / _np.sqrt(self.evaluate(x, x)*self.evaluate(y, y))

    def __repr__(self):
        return 'String kernel.'

    def evaluate(self, x, y):
        '''Unnormalized string kernel evaluation.'''
        lx = len(x)
        ly = len(y)
        Kd = _np.zeros([2, lx+1, ly+1])

        # dynamic programming
        for i in range(2):
            Kd[i, :, :] = (i + 1) % 2

        # calculate Kd and Kdd
        for i in range(1, self._kn):
            # set the Kd to zero for those lengths of s and t where s (or t) has exactly length i-1 and t (or s)
            # has length >= i-1. L-shaped upside down matrix
            for j in range(i - 1,  lx):
                Kd[i % 2, j, i - 1] = 0
            for j in range(i - 1, ly):
                Kd[i % 2, i - 1, j] = 0
            for j in range(i, lx):
                Kdd = 0
                for m in range(i, ly):
                    if x[j - 1] != y[m - 1]:
                        Kdd = self._l * Kdd
                    else:
                        Kdd = self._l * (Kdd + self._l * Kd[(i + 1) % 2, j - 1, m - 1])
                    Kd[i % 2, j, m] = self._l * Kd[i % 2, j - 1, m] + Kdd

        # calculate value of kernel function evaluation
        s = 0
        for i in range(self._kn, len(x) + 1):
            for j in range(self._kn, len(y)+1):
                if x[i - 1] == y[j - 1]:
                    s += self._l**2 * Kd[(self._kn - 1) % 2, i - 1, j - 1]

        return s


def gramian(X, k):
    '''Compute Gram matrix for training data X with kernel k.'''
    name = k.__class__.__name__
    if name == 'gaussianKernel':
        return _np.exp(-distance.squareform(distance.pdist(X.T, 'sqeuclidean'))/(2*k.sigma**2))
    elif name == 'laplacianKernel':
        return _np.exp(-distance.squareform(distance.pdist(X.T, 'euclidean'))/k.sigma)
    elif name == 'polynomialKernel':
        return (k.c + X.T @ X)**k.p
    elif name == 'stringKernel':
        n = len(X)
        # compute weights for normalization
        d = _np.zeros(n)
        for i in range(n):
            d[i] = k.evaluate(X[i], X[i])
        # compute Gram matrix
        G = _np.ones([n, n]) # diagonal automatically set to 1
        for i in range(n):
            for j in range(i):
                G[i, j] = k.evaluate(X[i], X[j]) / _np.sqrt(d[i]*d[j])
                G[j, i] = G[i, j]
        return G
    else:
        #print('User-defined kernel.')
        if isinstance(X, list): # e.g., for strings
            n = len(X)
            G = _np.zeros([n, n])
            for i in range(n):
                for j in range(i+1):
                    G[i, j] = k(X[i], X[j])
                    G[j, i] = G[i, j]
        else:
            n = X.shape[1]
            G = _np.zeros([n, n])
            for i in range(n):
                for j in range(i+1):
                    G[i, j] = k(X[:, i], X[:, j])
                    G[j, i] = G[i, j]
        return G


def gramian2(X, Y, k):
    '''Compute Gram matrix for training data X and Y with kernel k.'''
    name = k.__class__.__name__
    if name == 'gaussianKernel':
        #print('Gaussian kernel with sigma = %f.' % k.sigma)
        return _np.exp(-distance.cdist(X.T, Y.T, 'sqeuclidean')/(2*k.sigma**2))
    elif name == 'laplacianKernel':
        #print('Laplacian kernel with sigma = %f.' % k.sigma)
        return _np.exp(-distance.cdist(X.T, Y.T, 'euclidean')/k.sigma)
    elif name == 'polynomialKernel':
        #print('Polynomial kernel with degree = %f and c = %f.' % (k.p, k.c))
        return (k.c + X.T@Y)**k.p
    elif name == 'stringKernel':
        n = len(X)
        d = _np.zeros([n, 2])
        for i in range(n):
            d[i, 0] = k.evaluate(X[i], X[i])
            d[i, 1] = k.evaluate(Y[i], Y[i])
        # compute Gram matrix
        G = _np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                G[i, j] = k.evaluate(X[i], Y[j]) / _np.sqrt(d[i, 0]*d[j, 1])
        return G
    else:
        #print('User-defined kernel.')
        if isinstance(X, list): # e.g., for strings
            n = len(X)
            G = _np.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    G[i, j] = k(X[i], Y[j])
        else:
            n = X.shape[1]
            G = _np.zeros([n, n])
            for i in range(n):
                for j in range(n):
                    G[i, j] = k(X[:, i], Y[:, j])
        return G


class densityEstimate(object):
    '''Kernel density estimation using the Gaussian kernel.'''
    def __init__(self, X, k, beta=1):
        if k.__class__.__name__ != 'gaussianKernel':
            print('Error: Only implemented for Gaussian kernel.')
            return
        self.X = X                                     # points for density estimation
        self.k = k                                     # kernel
        self.d, self.n = X.shape                       # dimension and number of data points
        self.c = 1/_np.sqrt(2*_np.pi*k.sigma**2)**self.d # normalization constant
        self.beta = beta                               # inverse temperature, for MD applications
      
    def rho(self, x):
        G2 = gramian2(x, self.X, self.k)
        return self.c/self.n * G2.sum(axis=1, keepdims=True).T
    
    def V(self, x):
        return -_np.log(self.rho(x))/self.beta
    
    def gradV(self, x):
        G2 = gramian2(x, self.X, self.k)
        m = x.shape[1]
        y = _np.zeros_like(x)
        for i in range(m):
            for j in range(self.n):
                y[:, i] = y[:, i] + (x[:, i] - self.X[:, j])*G2[i, j]
            y[:, i] =  1/(self.beta*self.rho(x[:, i, None])) * self.c/(self.n * self.k.sigma**2)*y[:, i]
        return y
    
    # def rho(self, x):
    #     y = 0
    #     for i in range(self.n):
    #         y = y + self.k(x, self.X[:, i])
    #     return self.c/self.n * y
    
    # def V(self, x):
    #     return -1/self.beta * _np.log(self.rho(x))
    
    # def gradV(self, x):
    #     y = _np.zeros((self.d,))
    #     for i in range(self.n):
    #         y = y + self.k.diff(x, self.X[:, i])
    #     return -1/(self.beta*self.rho(x)) * self.c/self.n * y
