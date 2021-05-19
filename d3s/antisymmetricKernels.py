#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as _np
import itertools

class antisymmetrizedKernel(object):
    '''
    Antisymmetrized version k_a of the kernel k, where k must be permutation invariant.
    '''
    def __init__(self, k, d):
        self.k = k           # kernel
        self.d = d           # dimension
        self.P = allperms(d) # list of all permutations, precomputed for the sake of efficiency

    def __call__(self, x, y):
        n_perms = len(self.P)
        k_a = 0
        for i in range(n_perms):
            p = list(self.P[i])
            k_a += sgn(p)*self.k(x, y[p])
        return k_a / n_perms
    
    def diff(self, x, y):
        n_perms = len(self.P)
        D_k_a = _np.zeros(self.d)
        for i in range(n_perms):
            p = list(self.P[i])
            D_k_a += sgn(p)*self.k.diff(x, y[p])
        return D_k_a / n_perms
    
    def ddiff(self, x, y):
        n_perms = len(self.P)
        D_k_a = _np.zeros((self.d, self.d))
        for i in range(n_perms):
            p = list(self.P[i])
            D_k_a += sgn(p)*self.k.ddiff(x, y[p])
        return D_k_a / n_perms
    
    def laplace(self, x, y):
        n_perms = len(self.P)
        Delta_k_a = 0
        for i in range(n_perms):
            p = list(self.P[i])
            Delta_k_a += sgn(p)*self.k.laplace(x, y[p])
        return Delta_k_a / n_perms
    

class gaussianSlaterKernel(object):
    '''
    Antisymmetric Gaussian kernel computed via determinants.
    '''
    def __init__(self, sigma):
        self.sigma = sigma # bandwidth

    def __call__(self, x, y):
        d = x.shape[0]
        A = _np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                A[i, j] = _np.exp(-(x[i]-y[j])**2/(2*self.sigma**2))
        return _np.linalg.det(A) / _np.math.factorial(d)
    
    def diff(self, x, y):
        d = x.shape[0]
        D_k = _np.zeros(d)
        A = _np.zeros((d, d))
        for i in range(d):
            for mu in range(d):
                for nu in range(d):
                    A[mu, nu] = _np.exp(-(x[mu]-y[nu])**2/(2*self.sigma**2))
            for nu in range(d):
                A[i, nu] *= -1/self.sigma**2 * (x[i] - y[nu])
            D_k[i] = _np.linalg.det(A)
        return D_k / _np.math.factorial(d)
    
    def ddiff(self, x, y):
        d = x.shape[0]
        D_k = _np.zeros((d, d))
        A = _np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                for mu in range(d):
                    for nu in range(d):
                        A[mu, nu] = _np.exp(-(x[mu]-y[nu])**2/(2*self.sigma**2))
                if i == j:
                    for nu in range(d):
                        A[i, nu] *= (1/self.sigma**4 * (x[i] - y[nu])**2 - 1/self.sigma**2)
                else:
                    for nu in range(d):
                        A[i, nu] *= (1/self.sigma**2 * (x[i] - y[nu]))
                    for nu in range(d):
                        A[j, nu] *= (1/self.sigma**2 * (x[j] - y[nu]))
                            
                D_k[i, j] = _np.linalg.det(A)
        return D_k / _np.math.factorial(d)
    

class symmetrizedKernel(object):
    '''
    Symmetrized version k_s of the kernel k, where k must be permutation invariant.
    '''
    def __init__(self, k, d):
        self.k = k           # kernel
        self.d = d           # dimension
        self.P = allperms(d) # list of all permutations, precomputed for the sake of efficiency

    def __call__(self, x, y):
        n_perms = len(self.P)
        k_s = 0
        for i in range(n_perms):
            p = list(self.P[i])
            k_s += self.k(x, y[p])
        return k_s / n_perms


#%% auxiliary functions
def allperms(d):
    '''
    Generate all permutations of the numbers 0, ..., d-1.
    '''
    P = list(itertools.permutations(range(d)))
    return P


def sgn(p):
    '''
    Compute the sign of permutation p.
    '''
    n = len(p)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]: 
                s += 1
    return 1 if s % 2 == 0 else -1


def permute(X, p):
    '''
    Permute rows of a data matrix X w.r.t. p.
    '''
    return X[list(p), :]
