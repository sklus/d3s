# -*- coding: utf-8 -*-
import scipy as _scipy
import scipy.sparse.linalg

import d3s.observables as _observables
import d3s.kernels as _kernels

# The algorithms are implemented based on the following publications:
# S. Brunton, J. Proctor, J. Kutz:
#    Discovering governing equations from data by sparse identification of nonlinear dynamical systems.
# M. Williams, I. Kevrekidis, C. Rowley:
#    A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition.
# M. Williams, C. Rowley, I. Kevrekidis:
#    A kernel-based method for data-driven Koopman spectral analysis.
# S. Klus, P. Koltai, C. Schütte:
#    On the numerical approximation of the Perron-Frobenius and Koopman operator.
# S. Klus, F. Nüske, P. Koltai, H. Wu, I. Kevrekidis, C. Schütte, F. Noé:
#    Data-driven model reduction and transfer operator approximation.
# S. Klus, I. Schuster, K. Muandet:
#    Eigendecompositions of transfer operators in reproducing kernel Hilbert spaces.


def dmd(X, Y, mode='exact'):
    '''
    Exact and standard DMD of the data matrices X and Y.

    mode: 'exact' for exact DMD or 'standard' for standard DMD
    '''
    U, s, Vt = _scipy.linalg.svd(X, full_matrices=False)
    S_inv = _scipy.diag(1/s)
    A = U.transpose() @ Y @ Vt.transpose() @ S_inv
    d, W = sortEig(A, A.shape[0])

    if mode == 'exact':
        Phi = Y @ Vt.transpose() @ S_inv @ W @ _scipy.diag(1/d)
    elif mode == 'standard':
        Phi = U @ W
    else:
        raise ValueError('Only exact and standard DMD available.')

    return d, Phi


def amuse(X, Y, evs=5):
    '''
    AMUSE implementation of TICA, see TICA documentation.
    '''
    U, s, _ = _scipy.linalg.svd(X, full_matrices=False)
    S_inv = _scipy.diag(1/s)
    Xp = S_inv @ U.transpose() @ X
    Yp = S_inv @ U.transpose() @ Y
    K = Xp @ Yp.transpose()
    d, W = sortEig(K, evs)
    Phi = U @ S_inv @ W

    # normalize eigenvectors
    for i in range(Phi.shape[1]):
        Phi[:, i] /= _scipy.linalg.norm(Phi[:, i])
    return d, Phi


def tica(X, Y, evs=5):
    '''
    Time-lagged independent component analysis of the data matrices X and Y.

    evs: number of eigenvalues/eigenvectors
    '''
    return edmd(X, Y, _observables.identity, evs=evs)


def ulam(X, Y, Omega, evs=5, operator='K'):
    '''
    Ulam's method for the Koopman or Perron-Frobenius operator. The matrices X and Y contain
    the input data.

    Omega:    box discretization of type topy.domain.discretization
    evs:      number of eigenvalues/eigenvectors
    operator: 'K' for Koopman or 'P' for Perron-Frobenius

    TODO: Switch to sparse matrices.
    '''
    m = X.shape[1] # number of test points
    n = Omega.numBoxes() # number of boxes
    A = _scipy.zeros([n, n])
    # compute transitions
    for i in range(m):
        ix = Omega.index(X[:, i])
        iy = Omega.index(Y[:, i])
        A[ix, iy] += 1
    # normalize
    for i in range(n):
        s = A[i, :].sum()
        if s != 0:
            A[i, :] /= s
    if operator == 'P': A = A.transpose()
    d, V = sortEig(A, evs)
    return (d, V)


def edmd(X, Y, psi, evs=5, operator='K'):
    '''
    Conventional EDMD for the Koopman or Perron-Frobenius operator. The matrices X and Y
    contain the input data.

    psi:      set of basis functions, see d3s.observables
    evs:      number of eigenvalues/eigenvectors
    operator: 'K' for Koopman or 'P' for Perron-Frobenius
    '''
    PsiX = psi(X)
    PsiY = psi(Y)
    C_0 = PsiX @ PsiX.transpose()
    C_1 = PsiX @ PsiY.transpose()
    if operator == 'P': C_1 = C_1.transpose()

    A = _scipy.linalg.pinv(C_0) @ C_1
    d, V = sortEig(A, evs)
    return (d, V)


def kernelEdmd(X, Y, k, epsilon=0, evs=5, operator='P'):
    '''
    Kernel EDMD for the Koopman or Perron-Frobenius operator. The matrices X and Y
    contain the input data.

    k:        kernel, see d3s.kernels
    epsilon:  regularization parameter
    evs:      number of eigenvalues/eigenvectors
    operator: 'K' for Koopman or 'P' for Perron-Frobenius (note that the default is P here)
    '''
    if isinstance(X, list): # e.g., for strings
        n = len(X)
    else:
        n = X.shape[1]

    G_0 = _kernels.gramian(X, k)
    G_1 = _kernels.gramian2(X, Y, k)
    if operator == 'K': G_1 = G_1.transpose()

    A = _scipy.linalg.pinv(G_0 + epsilon*_scipy.eye(n), rcond=1e-15) @ G_1
    d, V = sortEig(A, evs)
    if operator == 'K': V = G_0 @ V
    return (d, V)


def sindy(X, Y, psi, eps=0.001, iterations=10):
    '''
    Sparse indentification of nonlinear dynamics for the data given by X and Y.

    psi:        set of basis functions, see topy.observables
    eps:        cutoff threshold
    iterations: number of sparsification steps
    '''
    PsiX = psi(X)
    Xi = Y @ _scipy.linalg.pinv(PsiX) # least-squares initial guess

    for k in range(iterations):
        s = abs(Xi) < eps # find coefficients less than eps ...
        Xi[s] = 0         # ... and set them to zero
        for ind in range(X.shape[0]):
            b = ~s[ind, :] # consider only functions corresponding to coefficients greater than eps
            Xi[ind, b] = Y[ind, :] @ _scipy.linalg.pinv(PsiX[b, :])
    return Xi


# auxiliary functions
def sortEig(A, evs=5):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    evs: number of eigenvalues/eigenvectors
    '''
    n = A.shape[0]
    if evs < n:
        d, V = _scipy.sparse.linalg.eigs(A, evs)
    else:
        d, V = _scipy.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])
