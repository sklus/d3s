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
        return -1/self.sigma**2*(x-y) * self(x, y)
    def ddiff(self, x, y):
        d = 1 if x.ndim == 0 else x.shape[0]
        return (1/self.sigma**4*_np.outer(x-y, x-y) - 1/self.sigma**2 *_np.eye(d)) * self(x, y)
    def laplace(self, x, y):
        return (1/self.sigma**4*_np.linalg.norm(x-y)**2 - len(x)/self.sigma**2) * self(x, y)
    def __repr__(self):
        return 'Gaussian kernel with bandwidth sigma = %f.' % self.sigma


class gaussianKernelGeneralized(object):
    '''Generalized Gaussian kernel with bandwidths sigma = (sigma_1, ..., sigma_d).'''
    def __init__(self, sigma):
        self.sigma = sigma
        self.D = _np.diag(1/(2*sigma**2))
    def __call__(self, x, y):
        xy = _np.squeeze(x-y) # (d, 1) vs. (d, )
        return _np.exp(-xy.T @ self.D @ xy )
    def diff(self, x, y):
        return -2*self.D @ (x-y) * self(x, y)
    def ddiff(self, x, y):
        return (_np.outer(2*self.D@(x-y), 2*self.D@(x-y)) - 2*self.D) * self(x, y)
    def laplace(self, x, y):
        return (_np.linalg.norm(2*self.D@(x-y))**2 - 2*_np.trace(self.D)) * self(x, y)
    def __repr__(self):
        return 'Generalized Gaussian kernel with bandwidths '+_np.array_str(self.sigma)+'.'


class laplacianKernel(object):
    '''Laplacian kernel with bandwidth sigma.'''
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x, y):
        return _np.exp(-_np.linalg.norm(x-y)/self.sigma)
    def diff(self, x, y):
        return -1/self.sigma*(x - y) / _np.linalg.norm(x-y) * self(x, y)
    def ddiff(self, x, y):
        # TODO: check x \ne y
        n_xy = _np.linalg.norm(x-y)
        return ( (1/(self.sigma**2*n_xy**2) + 1/(self.sigma*n_xy**3)) * _np.outer(x-y, x-y) - 1/(self.sigma*n_xy)*_np.eye(x.shape[0]) ) * self(x, y)
    def laplace(self, x, y):
        # TODO: check x \ne y
        n_xy = _np.linalg.norm(x-y)
        return ( 1/self.sigma**2 + (1-len(x))/(self.sigma*n_xy)) * self(x, y)
    def __repr__(self):
        return 'Laplacian kernel with bandwidth sigma = %f.' % self.sigma


class polynomialKernel(object):
    '''Polynomial kernel with degree p and inhomogeneity c.'''
    def __init__(self, p, c=1):
        self.p = p
        self.c = c
    def __call__(self, x, y):
        if x.ndim == 0:
            return (self.c + x * y)**self.p
        return (self.c + x.T @ y)**self.p
    def diff(self, x, y):
        if x.ndim == 0:
            return self.p*(self.c + x * y)**(self.p-1)*y;
        return self.p*(self.c + x.T @ y)**(self.p-1)*y;
    def ddiff(self, x, y):
        if x.ndim == 0:
            return self.p*(self.p-1)*(self.c + x.T * y)**(self.p-2) * _np.outer(y, y)
        return self.p*(self.p-1)*(self.c + x.T @ y)**(self.p-2) * _np.outer(y, y)
    def laplace(self, x, y):
        if x.ndim == 0:
            self.p*(self.p-1)*(self.c + x.T * y)**(self.p-2) * _np.linalg.norm(y)**2
        return self.p*(self.p-1)*(self.c + x.T @ y)**(self.p-2) * _np.linalg.norm(y)**2
    def __repr__(self):
        return 'Polynomial kernel with degree p = %f and inhomogeneity c = %f.' % (self.p, self.c)


class periodicKernel1D(object):
    '''One-dimensional periodic kernel with frequency p and bandwidth sigma.'''
    def __init__(self, p, sigma):
        self.p = p
        self.sigma = sigma
        
    def __call__(self, x, y):
        return _np.exp(-2*_np.sin((x-y)/self.p)**2/self.sigma**2)
    
    def diff(self, x, y):
        s = _np.zeros((1,))
        s[0] = -4*_np.sin((x-y)/self.p)*_np.cos((x-y)/self.p)/(self.sigma**2*self.p) * self(x, y)
        return s
    
    def ddiff(self, x, y):
        s = _np.zeros((1, 1))
        s[0, 0] = -(4*(4*_np.cos((x-y)/self.p)**4 + 2*_np.cos((x-y)/self.p)**2*self.sigma**2 - 4*_np.cos((x-y)/self.p)**2 - self.sigma**2))/(self.sigma**4*self.p**2) * self(x, y)
        return s
    
    def __repr__(self):
        return 'One-dimensional periodic kernel with frequency p = %f and bandwidth sigma = %f.' % (self.p, self.sigma)


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


class productKernel(object):
    '''Product of one-dimensional kernels, i.e., k(x) = k(x_1) ... k(x_d).'''
    def __init__(self, k):
        self.k = k
        self.d = len(k)
    
    def __call__(self, x, y):
        s = 1
        for i in range(self.d):
            s *= self.k[i](x[i], y[i])
        return s
    
    def diff(self, x, y):
        ds = self(x, y) * _np.ones((self.d, 1))
        for i in range(self.d):
            ds[i] *= self.k[i].diff(x[i], y[i]) / self.k[i](x[i], y[i])
        return ds
    
    def ddiff(self, x, y):
        dds = self(x, y) * _np.ones((self.d, self.d))
        for i in range(self.d):
            for j in range(i+1):
                if i == j:
                    dds[i, j] *= self.k[i].ddiff(x[i], y[i]) / self.k[i](x[i], y[i])
                else:
                    dds[i, j] *= self.k[i].diff(x[i], y[i]) / self.k[i](x[i], y[i]) * self.k[j].diff(x[j], y[j]) / self.k[j](x[j], y[j])
                    dds[j, i] = dds[i, j]
        return dds
            
    def laplace(self, x, y):
        s = self(x, y)
        ls = 0
        for i in range(self.d):
            ls += s * self.k[i].ddiff(x[i], y[i])[0, 0] / self.k[i](x[i], y[i])
        return ls
    
    def __repr__(self):
        return 'Product kernel with ' + str(self.k) + '.'
    

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
        m = len(X)
        n = len(Y)
        dx = _np.zeros((m,))
        dy = _np.zeros((n,))
        for i in range(m):
            dx[i] = k.evaluate(X[i], X[i])
        for j in range(n):
            dy[j] = k.evaluate(Y[j], Y[j])
        
        G = _np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                G[i, j] = k.evaluate(X[i], Y[j]) / _np.sqrt(dx[i]*dy[j])
        return G
    else:
        # print('User-defined kernel.')
        if isinstance(X, list): # e.g., for strings
            m = len(X)
            n = len(Y)
            G = _np.zeros([m, n])
            for i in range(m):
                for j in range(n):
                    G[i, j] = k(X[i], Y[j])
        else:
            m = X.shape[1]
            n = Y.shape[1]
            G = _np.zeros([m, n])
            for i in range(m):
                for j in range(n):
                    G[i, j] = k(X[:, i], Y[:, j])
        return G


class densityEstimate(object):
    '''Kernel density estimation using the Gaussian kernel.'''
    def __init__(self, X, k, beta=1):
        if k.__class__.__name__ != 'gaussianKernel':
            print('Error: Only implemented for Gaussian kernel.')
            return
        self.X = X                                       # points for density estimation
        self.k = k                                       # kernel
        self.d, self.n = X.shape                         # dimension and number of data points
        self.c = 1/_np.sqrt(2*_np.pi*k.sigma**2)**self.d # normalization constant
        self.beta = beta                                 # inverse temperature, for MD applications
      
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
