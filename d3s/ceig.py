#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as _np
import scipy as _sp
import d3s.algorithms as algorithms

'''Implementation of constrained eigenvalue solvers.'''

def hceig(A, N):
    '''
    Solve standard eigenvalue problem with homogeneous constraints:
      min x^T A x
      s.t. N^T x = 0
           ||x|| = 1
   '''
    n = A.shape[0]
    P = _np.eye(n) - N @ _sp.linalg.pinv(N)
    d, X = algorithms.sortEig(P @ A, evs=n)
    return d, X


def hcgeig(A, B, N):
    '''
    Solve generalized eigenvalue problem with homogeneous constraints:
      min x^T A x
      s.t. N^T x = 0
         ||x||_B = (x^T B x)^1/2 = 1, B s.p.d.
    '''
    L = _sp.linalg.cholesky(B, lower=True)
    L_inv = trinv(L, lower=True)
    A_ = L_inv @ A @ L_inv.T
    N_ = L_inv @ N
    d, Y = hceig(A_, N_)
    X = L_inv.T @ Y
    return d, X


def iceig(A, N, t):
    '''
    Solve standard eigenvalue problem with inhomogeneous constraints:
      min x^T A x
      s.t. N^T x = t
           ||x|| = 1
   '''
    n, nc = N.shape
    
    if _np.linalg.norm(_sp.linalg.pinv(N.T) @ t) >= 1:
        print('Warning: Might not be solvable.');

    [Q, R] = _sp.linalg.qr(N)

    A_ = Q.T @ A @ Q

    # B = A_[:nc, :nc];
    Gamma = A_[nc:, :nc]
    C = A_[nc:, nc:]

    y = _sp.linalg.solve(R[:nc, :nc].T, t)
    b = -Gamma @ y
    s = _np.sqrt(1 - y.T @ y)

    [d, Z] = ieig(C, b, s)

    X = Q @ _np.vstack(( _np.tile(y[:, _np.newaxis], [1, Z.shape[1]]), Z ))
    return d, X


def icgeig(A, B, N, t):
    '''
    Solve generalized eigenvalue problem with inhomogeneous constraints:
      min x^T A x
      s.t. N^T x = t
         ||x||_B = (x^T B x)^1/2 = 1, B s.p.d.
    '''
    L = _sp.linalg.cholesky(B, lower=True)
    L_inv = trinv(L, lower=True)
    A_ = L_inv @ A @ L_inv.T
    N_ = L_inv @ N
    d, Y = iceig(A_, N_, t)
    X = L_inv.T @ Y
    return d, X


def ieig(A, b, s):
    '''
    Solve inhomogeneous eigenvalue problem, symmetric case:
      A x = lambda x + b
      s.t. ||x|| = s
  '''
    m, n = A.shape
    if m != n:
        print('Not a square matrix.')
        return
    
    M = _np.vstack(( _np.hstack(( A, -_np.eye(n) )),
                    _np.hstack(( -1/s**2 * _np.outer(b, b), A )) ))
    
    d, V = algorithms.sortEig(M, evs=2*n)

    e = _np.zeros((2*n))
    X = _np.zeros((n, 2*n), dtype=complex)
    for ev in range(2*n):
        gamma = V[:n, ev]
        f = gamma.conj().T @ b / s**2
        gamma = gamma/f # normalize

        z = (A - d[ev]*_np.eye(n)) @ gamma
        
        e[ev] = abs(z.conj().T @ z - s**2) # check if constraint is satisfied
     
        X[:, ev] = z
    
    ind, = _np.where(e < 1e-6) # select only valid solutions
    X = X[:, ind]
    d = d[ind]
    return d, X


def trinv(L, lower=False):
    '''
    Invert triangular matrix L.
    '''
    n = L.shape[0]
    I = _np.eye(n)
    return _sp.linalg.solve_triangular(L, I, lower=lower)
