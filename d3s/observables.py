# -*- coding: utf-8 -*-

import math
import numpy as _numpy
from scipy.spatial import distance


def identity(x):
    '''
    Identity function.
    '''
    return x


class monomials(object):
    '''
    Computation of monomials in d dimensions.
    '''

    def __init__(self, p):
        '''
        The parameter p defines the maximum order of the monomials.
        '''
        self.p = p

    def __call__(self, x):
        '''
        Evaluate all monomials of order up to p for all data points in x.
        '''
        [d, m] = x.shape # d = dimension of state space, m = number of test points
        c = allMonomialPowers(d, self.p) # matrix containing all powers for the monomials
        n = c.shape[1] # number of monomials
        y = _numpy.ones([n, m])
        for i in range(n):
            for j in range(d):
                y[i, :] = _numpy.multiply(y[i, :], _numpy.power(x[j, :], c[j, i]))
        return y

    def __repr__(self):
        return 'Monomials of order up to = %d.' % self.p


class indicators(object):
    '''
    Indicator functions for box discretization Omega.
    '''
    def __init__(self, Omega):
        self.Omega = Omega

    def __call__(self, x):
        [d, m] = x.shape # d = dimension of state space, m = number of test points
        n = self.Omega.numBoxes()
        y = _numpy.zeros([n, m])
        for i in range(m):
            ind = self.Omega.index(x[:, i])
            pass
            if ind == -1:
                continue
            y[ind, i] = 1
        return y

    def __repr__(self):
        return 'Indicator functions for box discretization.'


class gaussians(object):
    '''
    Gaussians whose centers are the centers of the box discretization Omega.

    sigma: width of Gaussians
    '''
    def __init__(self, Omega, sigma=1):
        self.Omega = Omega
        self.sigma = sigma

    def __call__(self, x):
        c = self.Omega.midpointGrid()
        D = distance.cdist(c.transpose(), x.transpose(), 'sqeuclidean')
        y = _numpy.exp(-1/(self.sigma**2)*D)
        return y

    def __repr__(self):
        return 'Gaussian functions for box discretization with bandwidth %f.' % self.sigma


# auxiliary functions
def nchoosek(n, k):
    '''
    Computes binomial coefficients.
    '''
    return math.factorial(n)//math.factorial(k)//math.factorial(n-k) # integer division operator


def nextMonomialPowers(x):
    '''
    Returns powers for the next monomial. Implementation based on John Burkardt's MONOMIAL toolbox, see
    http://people.sc.fsu.edu/~jburkardt/m_src/monomial/monomial.html.
    '''
    m = len(x)
    j = 0
    for i in range(1, m): # find the first index j > 1 s.t. x[j] > 0
        if x[i] > 0:
            j = i
            break
    if j == 0:
        t = x[0]
        x[0] = 0
        x[m - 1] = t + 1
    elif j < m - 1:
        x[j] = x[j] - 1
        t = x[0] + 1
        x[0] = 0
        x[j-1] = x[j-1] + t
    elif j == m - 1:
        t = x[0]
        x[0] = 0
        x[j - 1] = t + 1
        x[j] = x[j] - 1
    return x


def allMonomialPowers(d, p):
    '''
    All monomials in d dimensions of order up to p.
    '''
    # Example: For d = 3 and p = 2, we obtain
    #[[ 0  1  0  0  2  1  1  0  0  0]
    # [ 0  0  1  0  0  1  0  2  1  0]
    # [ 0  0  0  1  0  0  1  0  1  2]]
    n = nchoosek(p + d, p) # number of monomials
    x = _numpy.zeros(d) # vector containing powers for the monomials, initially zero
    c = _numpy.zeros([d, n]) # matrix containing all powers for the monomials
    for i in range(1, n):
        c[:, i] = nextMonomialPowers(x)
    c = _numpy.flipud(c) # flip array in the up/down direction
    return c
