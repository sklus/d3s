# -*- coding: utf-8 -*-
import numpy as _np
import scipy as _sp
import scipy.sparse.linalg

import d3s.observables as _observables
import d3s.kernels as _kernels

'''
The implementations of the methods 
 
    - DMD, TICA, AMUSE
    - Ulam's method
    - EDMD, kernel EDMD, generator EDMD
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
    :return:     eigenvalues d and modes Phi
    '''
    U, s, Vt = _sp.linalg.svd(X, full_matrices=False)
    S_inv = _sp.diag(1/s)
    A = U.T @ Y @ Vt.T @ S_inv
    d, W = sortEig(A, A.shape[0])

    if mode == 'exact':
        Phi = Y @ Vt.T @ S_inv @ W @ _sp.diag(1/d)
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

    # eigendecomposition of A_approx
    w, _ = scipy.linalg.eig(A_approx)
    W = scipy.diag(w)

    # compute dynamic modes of A
    Phi = Y @ V_truncated @ scipy.linalg.inv(Sigma_truncated) @ UA.T @ U2_truncated @ W

    return A_approx, B_approx, Phi


def amuse(X, Y, evs=5):
    '''
    AMUSE implementation of TICA, see TICA documentation.
    
    :return:    eigenvalues d and corresponding eigenvectors Phi containing the coefficients for the eigenfunctions
    '''
    U, s, _ = _sp.linalg.svd(X, full_matrices=False)
    S_inv = _sp.diag(1/s)
    Xp = S_inv @ U.T @ X
    Yp = S_inv @ U.T @ Y
    K = Xp @ Yp.T
    d, W = sortEig(K, evs)
    Phi = U @ S_inv @ W

    # normalize eigenvectors
    for i in range(Phi.shape[1]):
        Phi[:, i] /= _sp.linalg.norm(Phi[:, i])
    return d, Phi


def tica(X, Y, evs=5):
    '''
    Time-lagged independent component analysis of the data matrices X and Y.

    :param evs: number of eigenvalues/eigenvectors
    :return:    eigenvalues d and corresponding eigenvectors V containing the coefficients for the eigenfunctions
    '''
    _, d, V = edmd(X, Y, _observables.identity, evs=evs)
    return (d, V)


def ulam(X, Y, Omega, evs=5, operator='K'):
    '''
    Ulam's method for the Koopman or Perron-Frobenius operator. The matrices X and Y contain
    the input data.

    :param Omega:    box discretization of type topy.domain.discretization
    :param evs:      number of eigenvalues/eigenvectors
    :param operator: 'K' for Koopman or 'P' for Perron-Frobenius
    :return:         eigenvalues d and corresponding eigenvectors V containing the coefficients for the eigenfunctions

    TODO: Switch to sparse matrices.
    '''
    m = X.shape[1] # number of test points
    n = Omega.numBoxes() # number of boxes
    A = _sp.zeros([n, n])
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
    if operator == 'P': A = A.T
    d, V = sortEig(A, evs)
    return (d, V)


def edmd(X, Y, psi, evs=5, operator='K'):
    '''
    Conventional EDMD for the Koopman or Perron-Frobenius operator. The matrices X and Y
    contain the input data.

    :param psi:      set of basis functions, see d3s.observables
    :param evs:      number of eigenvalues/eigenvectors
    :param operator: 'K' for Koopman or 'P' for Perron-Frobenius
    :return:         eigenvalues d and corresponding eigenvectors V containing the coefficients for the eigenfunctions
    '''
    PsiX = psi(X)
    PsiY = psi(Y)
    C_0 = PsiX @ PsiX.T
    C_1 = PsiX @ PsiY.T
    if operator == 'P': C_1 = C_1.T

    A = _sp.linalg.pinv(C_0) @ C_1
    d, V = sortEig(A, evs)
    return (A, d, V)


def gedmd(X, Y, Z, psi, evs=5, operator='K'):
    '''
    Generator EDMD for the Koopman operator. The matrices X and Y
    contain the input data. For stochastic systems, Z contains the
    diffusion term evaluated in all data points X. If the system is
    deterministic, set Z = None.
    '''
    PsiX = psi(X)
    dPsiY = _np.einsum('ijk,jk->ik', psi.diff(X), Y)
    if not (Z is None): # stochastic dynamical system
        n = PsiX.shape[0] # number of basis functions
        ddPsiX = psi.ddiff(X) # second-order derivatives
        S = _np.einsum('ijk,ljk->ilk', Z, Z) # sigma \cdot sigma^T
        for i in range(n):
            dPsiY[i, :] += 0.5*_np.sum( ddPsiX[i, :, :, :] * S, axis=(0,1) )
    
    C_0 = PsiX @ PsiX.T
    C_1 = PsiX @ dPsiY.T
    if operator == 'P': C_1 = C_1.T

    A = _sp.linalg.pinv(C_0) @ C_1
    
    d, V = sortEig(A, evs, which='SM')
    
    return (A, d, V)


def kedmd(X, Y, k, epsilon=0, evs=5, operator='P'):
    '''
    Kernel EDMD for the Koopman or Perron-Frobenius operator. The matrices X and Y
    contain the input data.

    :param k:        kernel, see d3s.kernels
    :param epsilon:  regularization parameter
    :param evs:      number of eigenvalues/eigenvectors
    :param operator: 'K' for Koopman or 'P' for Perron-Frobenius (note that the default is P here)
    :return:         eigenvalues d and eigenfunctions evaluated in X
    '''
    if isinstance(X, list): # e.g., for strings
        n = len(X)
    else:
        n = X.shape[1]

    G_0 = _kernels.gramian(X, k)
    G_1 = _kernels.gramian2(X, Y, k)
    if operator == 'K': G_1 = G_1.T

    A = _sp.linalg.pinv(G_0 + epsilon*_np.eye(n), rcond=1e-15) @ G_1
    d, V = sortEig(A, evs)
    if operator == 'K': V = G_0 @ V
    return (d, V)


def sindy(X, Y, psi, eps=0.001, iterations=10):
    '''
    Sparse indentification of nonlinear dynamics for the data given by X and Y.

    :param psi:        set of basis functions, see topy.observables
    :param eps:        cutoff threshold
    :param iterations: number of sparsification steps
    :return:           coefficient matrix Xi
    '''
    PsiX = psi(X)
    Xi = Y @ _sp.linalg.pinv(PsiX) # least-squares initial guess

    for k in range(iterations):
        s = abs(Xi) < eps # find coefficients less than eps ...
        Xi[s] = 0         # ... and set them to zero
        for ind in range(X.shape[0]):
            b = ~s[ind, :] # consider only functions corresponding to coefficients greater than eps
            Xi[ind, b] = Y[ind, :] @ _sp.linalg.pinv(PsiX[b, :])
    return Xi


def kpca(X, k, evs=5):
    '''
    Kernel PCA.
    
    :param X:    data matrix, each column represents a data point
    :param k:    kernel
    :param evs:  number of eigenvalues/eigenvectors
    :return:     data X projected onto principal components
    '''
    G = _kernels.gramian(X, k) # Gram matrix
    
    # center Gram matrix
    n = X.shape[1]
    N = _np.eye(n) - 1/n*_sp.ones((n, n))
    G = N @ G @ N    
    d, V = sortEig(G, evs)
    return (d, V)


def kcca(X, Y, k, evs=5, epsilon=1e-6):
    '''
    Kernel CCA.
    
    :param X:    data matrix, each column represents a data point
    :param Y:    lime-lagged data, each column y_i is x_i mapped forward by the dynamical system
    :param k:    kernel
    :param evs:  number of eigenvalues/eigenvectors
    :epsilon:    regularization parameter
    :return:     nonlinear transformation of the data X
    '''
    G_0 = _kernels.gramian(X, k)
    G_1 = _kernels.gramian(Y, k)
    
    # center Gram matrices
    n = X.shape[1]
    I = _np.eye(n)
    N = I - 1/n*_sp.ones((n, n))
    G_0 = N @ G_0 @ N
    G_1 = N @ G_1 @ N
    
    A = _sp.linalg.solve(G_0 + epsilon*I, G_0, assume_a='sym') \
      @ _sp.linalg.solve(G_1 + epsilon*I, G_1, assume_a='sym')
    
    d, V = sortEig(A, evs)
    return (d, V)


def cmd(X, Y, evs=5, epsilon=1e-6):
    '''
    Coherent mode decomposition.
    
    :param X:    data matrix, each column represents a data point
    :param Y:    lime-lagged data, each column y_i is x_i mapped forward by the dynamical system
    :param evs:  number of eigenvalues/eigenvectors
    :epsilon:    regularization parameter
    :return:     eigenvalues and modes xi and eta
    '''
    G_0 = X.T @ X
    G_1 = Y.T @ Y
    
    # center Gram matrices
    n = X.shape[1]
    I = _np.eye(n)
    N = I - 1/n*_sp.ones((n, n))
    G_0 = N @ G_0 @ N
    G_1 = N @ G_1 @ N
    
    A = _sp.linalg.solve(G_0 + epsilon*I, _sp.linalg.solve(G_1 + epsilon*I, G_1, assume_a='sym')) @ G_0
    
    d, V = sortEig(A, evs)
    rho = _sp.sqrt(d);
    W = _sp.linalg.solve(G_1 + epsilon*I, G_0) @ V @ _sp.diag(rho)
    
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
    
    V, _ = _sp.linalg.qr(V, mode='economic')
    mu = 0.99/_sp.sqrt(n)
    
    if R0 == None:
        R0 = _np.eye(r)
    else:
        R0, _ = _sp.linalg.polar(R0)
    
    S = _sp.zeros((n, r))
    
    for i in range(maxIter):
        Z = V @ R0.T
        
        # threshold
        for j in range(r):
            S[:, j] = _sp.sign(Z[:, j]) * _sp.maximum(abs(Z[:, j]) - mu, 0)
            S[:, j] = S[:, j]/_sp.linalg.norm(S[:, j])
        
        # polar decomposition
        R1, _ = _sp.linalg.polar(S.T @ V)
        
        # check whether converged
        if _sp.linalg.norm(R1 - R0) < 1e-14:
            break
        
        # overwrite initial matrix with new matrix
        R0 = R1.copy()
    
    # choose correct parity and normalize
    for j in range(r):
        S[:, j] = S[:, j] * _sp.sign(S[:, j].sum())
        S[:, j] = S[:, j] / _sp.amax(S[:, j])
    
    # sort vectors
    ind = _sp.argsort(_np.min(S, axis=0))[::-1]
    S = S[:, ind]
        
    return S


def kmeans(x, k, maxIter=100):
    '''
    Simple k-means implementation.
    
    :param x: data matrix, each column is a data point
    :param k: number of clusters
    :param maxIter: maximum number of iterations
    :return: cluster numbers for all data points
    '''
    def update(l0):
        c = _np.zeros((d, k))
        for i in range(k):
            c[:, i] = _np.mean(x[:, l0==i], axis=1)
        D = _sp.spatial.distance.cdist(x.T, c.T)
        l1 = D.argmin(axis=1)
        return l1
    
    d, m = x.shape # number of dimensions and data points
    l0 = _np.random.randint(0, k, size=m) # initial cluster assignments
    
    it = 0
    while it < maxIter:
        l1 = update(l0)
        it += 1
      
        if (l1 == l0).all():
            print('k-means converged after %d iterations.' % it)
            return l1
        l0 = l1
    print('k-means did not converge.')
    return l1


# auxiliary functions
def sortEig(A, evs=5, which='LM'):
    '''
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    '''
    n = A.shape[0]
    if evs < n:
        d, V = _sp.sparse.linalg.eigs(A, evs, which=which)
    else:
        d, V = _sp.linalg.eig(A)
    ind = d.argsort()[::-1] # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])
