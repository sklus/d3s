# -*- coding: utf-8 -*-
import numpy as _numpy
import scipy as _scipy
import scipy.sparse.linalg

import d3s.observables as _observables
import d3s.kernels as _kernels

'''
The implementations of the methods 
 
    - DMD, TICA, AMUSE
    - Ulam's method
    - EDMD, kernel EDMD
    - SINDy
    - kernel PCA, kernel CCA
    - CMD
    - SEBA
 
are based on the publications listed here:
 
    https://github.com/sklus/d3s

'''


def dmd(X, Y, mode='exact'):
    '''
    Exact and standard DMD of the data matrices X and Y.

    :param mode: 'exact' for exact DMD or 'standard' for standard DMD
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


def dmdc(X, Y, U, svThresh=1e-10):
    '''
    DMD + control where control matrix B is unknown, https://arxiv.org/abs/1409.6358
    :param X: State matrix in Reals NxM-1, where N is dim of state vector, M is number of samples
    :param Y: One step time-laged state matrix in Reals NxM-1
    :param U: Control input matrix, in Reals QxM-1, where Q is dim of control vector
    :param svThresh: Threshold below which to discard singular values
    :return: A_approx, B_approx, Phi  (where Phi are dynamic modes of A)
    '''
    n = X.shape[0] # size of state vector
    q = U.shape[0] # size of control vector

    # Y = G * Gamma
    Omega = scipy.vstack((X, U))
    U, svs, V = scipy.linalg.svd(Omega)
    V = V.T
    svs_to_keep = svs[scipy.where(svs > svThresh)] # todo: ensure exist svs that are greater than thresh
    n_svs = len(svs_to_keep)
    Sigma_truncated = scipy.diag(svs_to_keep)
    U_truncated = U[:, :n_svs]
    V_truncated = V[:, :n_svs]

    U2, svs2, V2 = scipy.linalg.svd(Y, full_matrices=False)
    V2 = V2.T
    svs_to_keep2 = svs2[scipy.where(svs2 > svThresh)]
    n_svs2 = len(svs_to_keep2)
    Sigma2_truncated = scipy.diag(svs_to_keep2)
    U2_truncated = U2[:, :n_svs2]
    V2_truncated = V2[:, :n_svs2]

    # separate into POD modes for A, B matrices
    UA = U_truncated[:n, :]
    UB = U_truncated[n:, :]

    A_approx = U2_truncated.T @ Y @ V_truncated @ scipy.linalg.inv(Sigma_truncated) @ UA.T @ U2_truncated
    B_approx = U2_truncated.T @ Y @ V_truncated @ scipy.linalg.inv(Sigma_truncated) @ UB.T

    # Eigen decomposition of A_approx
    w, _ = scipy.linalg.eig(A_approx)
    W = scipy.diag(w)

    # Compute dynamic modes of A
    Phi = Y @ V_truncated @ scipy.linalg.inv(Sigma_truncated) @ UA.T @ U2_truncated @ W

    return A_approx, B_approx, Phi


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

    :param evs: number of eigenvalues/eigenvectors
    '''
    return edmd(X, Y, _observables.identity, evs=evs)


def ulam(X, Y, Omega, evs=5, operator='K'):
    '''
    Ulam's method for the Koopman or Perron-Frobenius operator. The matrices X and Y contain
    the input data.

    :param Omega:    box discretization of type topy.domain.discretization
    :param evs:      number of eigenvalues/eigenvectors
    :param operator: 'K' for Koopman or 'P' for Perron-Frobenius

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

    :param psi:      set of basis functions, see d3s.observables
    :param evs:      number of eigenvalues/eigenvectors
    :param operator: 'K' for Koopman or 'P' for Perron-Frobenius
    '''
    PsiX = psi(X)
    PsiY = psi(Y)
    C_0 = PsiX @ PsiX.transpose()
    C_1 = PsiX @ PsiY.transpose()
    if operator == 'P': C_1 = C_1.transpose()

    A = _scipy.linalg.pinv(C_0) @ C_1
    d, V = sortEig(A, evs)
    return (d, V)


def kedmd(X, Y, k, epsilon=0, evs=5, operator='P'):
    '''
    Kernel EDMD for the Koopman or Perron-Frobenius operator. The matrices X and Y
    contain the input data.

    :param k:        kernel, see d3s.kernels
    :param epsilon:  regularization parameter
    :param evs:      number of eigenvalues/eigenvectors
    :param operator: 'K' for Koopman or 'P' for Perron-Frobenius (note that the default is P here)
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

    :param psi:        set of basis functions, see topy.observables
    :param eps:        cutoff threshold
    :param iterations: number of sparsification steps
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


def kpca(X, k, evs=5):
    '''
    Kernel PCA. Returns data projected onto principal components.
    
    :param X:    data matrix, each column represents a data point
    :param k:    kernel
    :param evs:  number of eigenvalues/eigenvectors
    '''
    G = _kernels.gramian(X, k) # Gram matrix
    
    # center Gram matrix
    n = X.shape[1]
    N = _scipy.eye(n) - 1/n*_scipy.ones((n, n))
    G = N @ G @ N    
    d, V = sortEig(G, evs)
    return (d, V)


def kcca(X, Y, k, evs=5, epsilon=1e-6):
    '''
    Kernel CCA. Returns nonlinear transformation of the data X.
    
    :param X:    data matrix, each column represents a data point
    :param Y:    lime-lagged data, each column y_i is x_i mapped forward by the dynamical system
    :param k:    kernel
    :param evs:  number of eigenvalues/eigenvectors
    :epsilon:    regularization parameter
    '''
    G_0 = _kernels.gramian(X, k)
    G_1 = _kernels.gramian(Y, k)
    
    # center Gram matrices
    n = X.shape[1]
    I = _scipy.eye(n)
    N = I - 1/n*_scipy.ones((n, n))
    G_0 = N @ G_0 @ N
    G_1 = N @ G_1 @ N
    
    A = _scipy.linalg.solve(G_0 + epsilon*I, G_0, assume_a='sym') \
      @ _scipy.linalg.solve(G_1 + epsilon*I, G_1, assume_a='sym')
    
    d, V = sortEig(A, evs)
    return (d, V)


def cmd(X, Y, evs=5, epsilon=1e-6):
    '''
    Coherent mode decomposition. Returns modes xi and eta.
    
    :param X:    data matrix, each column represents a data point
    :param Y:    lime-lagged data, each column y_i is x_i mapped forward by the dynamical system
    :param evs:  number of eigenvalues/eigenvectors
    :epsilon:    regularization parameter
    '''
    G_0 = X.T @ X
    G_1 = Y.T @ Y
    
    # center Gram matrices
    n = X.shape[1]
    I = _scipy.eye(n)
    N = I - 1/n*_scipy.ones((n, n))
    G_0 = N @ G_0 @ N
    G_1 = N @ G_1 @ N
    
    A = _scipy.linalg.solve(G_0 + epsilon*I, _scipy.linalg.solve(G_1 + epsilon*I, G_1, assume_a='sym')) @ G_0
    
    d, V = sortEig(A, evs)
    rho = _scipy.sqrt(d);
    W = _scipy.linalg.solve(G_1 + epsilon*I, G_0) @ V @ _scipy.diag(rho)
    
    Xi = X @ V
    Eta = Y @ W
    
    return (rho, Xi, Eta)


def seba(V, R0=None, maxIter=5000):
    '''
    Sparse eigenbasis approximation as described in 
    
    "Sparse eigenbasis approximation: Multiple feature extraction across spatiotemporal scales with
    application to coherent set identification" by G. Froyland, C. Rock, and K. Sakellariou.
    
    Based on the original Matlab implementation, see https://github.com/gfroyland/SEBA.
    
    :param V:        eigenvectors
    :param R0:       optional initial rotation
    :param maxIter:  maximum number of iterations
    :return:         sparse basis output
    
    TODO: perturb near-constant vectors?
    '''
    n, r = V.shape
    
    V, _ = _scipy.linalg.qr(V, mode='economic')
    mu = 0.99/_scipy.sqrt(n)
    
    if R0 == None:
        R0 = _scipy.eye(r)
    else:
        R0, _ = _scipy.linalg.polar(R0)
    
    S = _scipy.zeros((n, r))
    
    for i in range(maxIter):
        Z = V @ R0.T
        
        # threshold
        for j in range(r):
            S[:, j] = _scipy.sign(Z[:, j]) * _scipy.maximum(abs(Z[:, j]) - mu, 0)
            S[:, j] = S[:, j]/_scipy.linalg.norm(S[:, j])
        
        # polar decomposition
        R1, _ = _scipy.linalg.polar(S.T @ V)
        
        # check whether converged
        if _scipy.linalg.norm(R1 - R0) < 1e-14:
            break
        
        # overwrite initial matrix with new matrix
        R0 = R1.copy()
    
    # choose correct parity and normalize
    for j in range(r):
        S[:, j] = S[:, j] * _scipy.sign(S[:, j].sum())
        S[:, j] = S[:, j] / _scipy.amax(S[:, j])
    
    # sort vectors
    ind = _scipy.argsort(-_numpy.min(S, axis=0))
    S = S[:, ind]
        
    return S


# auxiliary functions
def sortEig(A, evs=5):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    '''
    n = A.shape[0]
    if evs < n:
        d, V = _scipy.sparse.linalg.eigs(A, evs)
    else:
        d, V = _scipy.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])
