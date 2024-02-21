# -*- coding: utf-8 -*-
import numpy as _np
import scipy as _sp

import matplotlib
import matplotlib.cm
import matplotlib.pyplot

from mpl_toolkits.mplot3d import axes3d, Axes3D

from d3s.tools import indexS2M, indexM2S


class discretization(object):
    '''
    Box disretization of d-dimensional hypercubes.
    '''

    def __init__(self, bounds, boxes):
        '''
        Initializes box discretization object.
        '''
        assert bounds.shape[0] == boxes.size
        self._bounds = bounds # lower and upper bounds for intervals
        self._boxes = boxes # number of boxes per dimension
        self._h = _np.divide(bounds[:, 1] - bounds[:, 0], boxes) # length of interval in each direction
        self._d = boxes.size # number of dimensions

    def __repr__(self):
        return 'Disretization of %s into %s boxes.' \
            % ('x'.join(['[%.2f, %.2f]' % (self._bounds[i, 0], self._bounds[i, 1]) for i in range(self._d)]),
               'x'.join(['%d' % (self._boxes[i]) for i in range(self._d)]))
    
    def dimension(self):
        '''
        Returns dimension of the domain.
        '''
        return self._d

    def numBoxes(self):
        '''
        Returns number of boxes.
        '''
        return self._boxes.prod()
    
    def numVertices(self):
        '''
        Returns number of vertices of the grid.
        '''
        return (self._boxes+1).prod()

    def rand(self, n):
        '''
        Generates n random test points in Omega.
        '''
        x = _np.zeros([self._d, n])
        for i in range(self._d):
            x[i, :] = randb(n, self._bounds[i, :])
        return x

    def randPerBox(self, n):
        '''
        Generates n random test points per box.
        '''
        d = self._d # dimension of state space
        nBoxes = self.numBoxes() # number of boxes
        nTestPoints = n*nBoxes # number of all test points

        x = _np.zeros([d, nTestPoints]) # for the test points
        for i in range(nBoxes):
             index = indexS2M(i, self._boxes) # corresponding multi-index
             lb = self._bounds[:, 0] + _np.multiply(index,    self._h) # lower bounds for box
             ub = self._bounds[:, 0] + _np.multiply(index +1, self._h) # upper bounds for box
             for mu in range(d):
                 x[mu, n*i:n*(i+1)] = randb(n, [lb[mu], ub[mu]])
        return x

    def index(self, x):
        '''
        Finds corresponding index of the box that contains vector x.
        '''
        mind = self.mindex(x)
        if _np.any(mind == -1): return -1 # invalid index
        return indexM2S(mind, self._boxes)

    def mindex(self, x):
        '''
        Finds corresponding multi-index of the box that contains x.
        '''
        mind = -1*_np.ones(self._d, _np.int64)
        for i in range(self._d):
            if x[i] < self._bounds[i, 0] or x[i] >= self._bounds[i, 1]:
                print('Value out of bounds! Invalid box returned.')
                return mind
            mind[i] = _np.floor((x[i] - self._bounds[i, 0]) / self._h[i])
        return mind

    def midpointGrid(self):
        '''
        Returns a grid given by the midpoints of the boxes.
        '''
        b = self._bounds
        h = self._h
        d = self._d
        n = self.numBoxes()
        x = []
        for i in range(d):
            x.append( _np.linspace(b[i, 0] + h[i]/2, b[i, 1] - h[i]/2, self._boxes[i]) )
        X = _sp.meshgrid(*x, indexing='ij')
        c = _np.zeros([d, n])
        for i in range(d):
            c[i, :] = X[i].reshape(n)
        return c
    
    def vertexGrid(self):
        '''
        Returns a grid given by the vertices.
        '''
        b = self._bounds
        d = self._d
        n = self.numVertices()
        x = []
        for i in range(d):
            x.append( _np.linspace(b[i, 0], b[i, 1], self._boxes[i]+1) )
        X = _sp.meshgrid(*x, indexing='ij')
        c = _np.zeros([d, n])
        for i in range(d):
            c[i, :] = X[i].reshape(n)
        isBoundary = _np.zeros(X[i].shape, dtype=bool)
        for i in range(d):
            ind = d*[slice(None)]
            ind[i] = [0, self._boxes[i]]
            isBoundary[tuple(ind)] = True
        isBoundary = isBoundary.reshape(n)
        return c, isBoundary

    def plot(self, v, mode='2D', grid='midpoint'):
        '''
        Plots v, where v_i is the value at grid point i.
        
        :param mode: select plot type for two-dimensional domains.
        :param grid: select grid type (midpoint or vertex)
        '''
        d = self._d
        if d > 3:
            print('Not defined for d > 3.')
            return
        if grid == 'midpoint':
            c = self.midpointGrid()
            dims = self._boxes
        elif grid == 'vertex':
            c, _ = self.vertexGrid()
            dims = self._boxes + 1
        else:
            print('Grid type not defined.')
            return
        getattr(self, '_plot_%s' % d)(c, v, dims, mode)

    def _plot_1(self, c, v, dims, mode):
        c = c.squeeze()
        matplotlib.pyplot.plot(c, v)

    def _plot_2(self, c, v, dims, mode):
        X = c[0, :].reshape(dims)
        Y = c[1, :].reshape(dims)
        Z = v.reshape(dims)

        if mode=='2D':
            matplotlib.pyplot.pcolor(X, Y, Z)
        else:
            fig = matplotlib.pyplot.gcf()
            ax = fig.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.coolwarm)
            ax.set_xlabel('x_1')
            ax.set_ylabel('x_2')

    def _plot_3(self, c, v, dims, mode):
        X = c[0, :].reshape(dims)
        Y = c[1, :].reshape(dims)
        Z = c[2, :].reshape(dims)
        V = v.reshape(dims)
        
        fig = matplotlib.pyplot.gcf()
        ax = fig.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(X, Y, Z, c=V)
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        ax.set_zlabel('x_3')

# auxiliary functions
def randb(n, b):
    '''
    Returns an array of n uniformly distributed random values in the interval b.
    '''
    return b[0] + (b[1] - b[0])*_np.random.rand(1, n)
