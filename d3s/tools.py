#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as _np
import os as _os
import time as _time
import scipy.io as _sio


def indexS2M(sInd, dims):
    '''Single- to multi-index.'''
    return _np.array(_np.unravel_index(sInd, dims))


def indexM2S(mInd, dims):
    '''Multi- to single-index.'''
    return _np.ravel_multi_index(mInd, dims)


def printVector(x, name = None, k = 8):
    '''Prints the vector like Matlab.'''
    n = x.size
    c = 0
    isReal = ~_np.any(_np.iscomplex(x))
    if name != None:
        print(name + ' = ')
    while c < n:
        print('\033[94m  (columns %s through %s)\033[0m' % (c, min(c+k, n)-1))
        for j in range(c, min(c+k, n)):
            if isReal:
                print(f'  {x[j]: 10.5f}', end = '')
            else:
                re = _np.real(x[j])
                im = _np.imag(x[j])
                print(f'  {re: 8.3f} + {im:.3f}i' if im >= 0 else f'  {re: 8.3f} - {abs(im):.3f}i', end = '')
        print('')
        c += k


def printMatrix(x, name = None, k = 8):
    '''Prints the matrix like Matlab.'''
    m, n = x.shape
    c = 0
    isReal = ~_np.any(_np.iscomplex(x))
    if name != None:
        print(name + ' = ')
    while c < n:
        print('\033[94m  (columns %s through %s)\033[0m' % (c, min(c+k, n)-1))
        for i in range(m):
            for j in range(c, min(c+k, n)):
                if isReal:
                    print(f'  {x[i, j]: 10.5f}', end = '')
                else:
                    re = _np.real(x[i, j])
                    im = _np.imag(x[i, j])
                    print(f'  {re: 8.3f} + {im:.3f}i' if im >= 0 else f'  {re: 8.3f} - {abs(im):.3f}i', end = '')
            print('')
        c += k


class Timer(object):
    '''Simple timer class.'''
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = _time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('%s: ' % self.name, end = '')
        print('%s s' % (_time.time() - self.tstart))


class matmux(object):
    '''
    Communicate with a running Matlab session, which needs to be started via:
    $ tmux new -s matlab "matlab -nodesktop"
    '''
    def __init__(self):
        self.cmd     = 'tmux send-keys -t matlab "%s\n"' # tmux command to be augmented by Matlab code
        self.tmpFile = '/tmp/matmux.mat'                 # for exchanging variables between Python and Matlab

    def __call__(self, s):
        '''
        Execute Matlab code contained in the string s, e.g.:
        this('X = rand(2, 4)')
        '''
        _os.system(self.cmd % s)

    def __repr__(self):
        return 'Matlab communicator.'

    def _loadmat(self):
        self('load %s;\ndelete %s;' % (self.tmpFile, self.tmpFile))
        # wait until Matlab deleted the temporary file to make sure that the variables have been
        # read and the file can be written again
        while _os.path.isfile(self.tmpFile):
            _time.sleep(0.05)

    def exportVars(self, *args):
        '''
        Export variables to Matlab, e.g.:
        x = 1
        y = 2
        this.exportVars('x', x, 'y', y)
        '''
        variableDict = dict(zip(args[::2], args[1::2]))
        _sio.savemat(self.tmpFile, variableDict, do_compression=True)
        self._loadmat()

    def importVars(self, *args):
        '''
        Import variables from Matlab, e.g.:
        x, y = this.importVars('x', 'y')
        '''
        self('save(\'%s\', %s)' % (self.tmpFile, ', '.join(['\'%s\'' % arg for arg in args])))
        _time.sleep(10) # tmp. fix, othe
        while not _os.path.isfile(self.tmpFile): # it might take a while until the file is written
            _time.sleep(0.05)
        data = _sio.loadmat(self.tmpFile, squeeze_me=True)
        _os.remove(self.tmpFile) # delete file again
        return tuple(data[arg] for arg in args)

    def figure(self, i=-1):
        if i == -1:
            self('figure;')
        else:
            self('figure(%d);' % i)

    def close(self, i=-1):
        if i == -1:
            self('close all;')
        else:
            self('close(%d);' % i)

    def plot(self, x, y):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y})
        self._loadmat()
        self('plot(x, y);')

    def surf(self, x, y, z):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'z':z})
        self._loadmat()
        self("surf(x, y, z); xlabel('x'); ylabel('y'); zlabel('z');")
    
    def scatter(self, x, y, c):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'c':c})
        self._loadmat()
        self("scatter(x, y, 100, c, '.');")
        
    def scatter3(self, x, y, z, c):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'z':z, 'c':c})
        self._loadmat()
        self("scatter3(x, y, z, 100, c, '.'); xlabel('x'); ylabel('y'); zlabel('z');")

    def pcolor(self, x, y, z):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'z':z})
        self._loadmat()
        self('pcolor(x, y, z);')

    def imagesc(self, x):
        _sio.savemat(self.tmpFile, {'x':x})
        self._loadmat()
        self('imagesc(x);')

    def plotDomain(self, Omega, x):
        d = Omega._d
        if d == 1:
            self.plot(Omega.midpointGrid().squeeze(), x)
        elif d == 2:
            c = Omega.midpointGrid()
            cx = c[0, :].reshape(Omega._boxes)
            cy = c[1, :].reshape(Omega._boxes)
            cz  = x.reshape(Omega._boxes)
            self.surf(cx, cy, cz)
        else:
            print('Not defined for d > 2.')


class octmux(object):
    '''
    Communicate with a running Octave session, which needs to be started via:
    $ tmux new -s octave "octave --no-gui"
    '''
    def __init__(self):
        self.cmd     = 'tmux send-keys -t octave "%s" Enter' # tmux command to be augmented by Octave code
        self.tmpFile = '/tmp/octmux.mat'                     # for exchanging variables between Python and Octave

    def __call__(self, s):
        '''
        Execute Octave code contained in the string s, e.g.:
        this('X = rand(2, 4)')
        '''
        _os.system(self.cmd % s.replace(';', '\;'))

    def __repr__(self):
        return 'Octave communicator.'

    def _loadmat(self):
        self('load %s;' % self.tmpFile)
        self('delete %s;' % self.tmpFile)
        # wait until Octave deleted the temporary file to make sure that the variables have been
        # read and the file can be written again
        while _os.path.isfile(self.tmpFile):
            _time.sleep(0.05)

    def exportVars(self, *args):
        '''
        Export variables to Octave, e.g.:
        x = 1
        y = 2
        this.exportVars('x', x, 'y', y)
        '''
        variableDict = dict(zip(args[::2], args[1::2]))
        _sio.savemat(self.tmpFile, variableDict, do_compression=True)
        self._loadmat()

    def importVars(self, *args):
        '''
        Import variables from Octave, e.g.:
        x, y = this.importVars('x', 'y')
        '''
        self('save(\'%s\', %s, \'-v7\')' % (self.tmpFile, ', '.join(['\'%s\'' % arg for arg in args])))
        while not _os.path.isfile(self.tmpFile): # it might take a while until the file is written
            _time.sleep(0.05)
        data = _sio.loadmat(self.tmpFile, squeeze_me=True)
        _os.remove(self.tmpFile) # delete file again
        return tuple(data[arg] for arg in args)

    def figure(self, i=-1):
        if i == -1:
            self('figure;')
        else:
            self('figure(%d);' % i)

    def close(self, i=-1):
        if i == -1:
            self('close all;')
        else:
            self('close(%d);' % i)

    def plot(self, x, y):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y})
        self._loadmat()
        self('plot(x, y);')

    def surf(self, x, y, z):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'z':z})
        self._loadmat()
        self("surf(x, y, z); xlabel('x'); ylabel('y'); zlabel('z');")
    
    def scatter(self, x, y, c):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'c':c})
        self._loadmat()
        self("scatter(x, y, 100, c, '.');")
        
    def scatter3(self, x, y, z, c):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'z':z, 'c':c})
        self._loadmat()
        self("scatter3(x, y, z, 100, c, '.'); xlabel('x'); ylabel('y'); zlabel('z');")

    def pcolor(self, x, y, z):
        _sio.savemat(self.tmpFile, {'x':x, 'y':y, 'z':z})
        self._loadmat()
        self('pcolor(x, y, z);')

    def imagesc(self, x):
        _sio.savemat(self.tmpFile, {'x':x})
        self._loadmat()
        self('imagesc(x);')

    def plotDomain(self, Omega, x):
        d = Omega._d
        if d == 1:
            self.plot(Omega.midpointGrid().squeeze(), x)
        elif d == 2:
            c = Omega.midpointGrid()
            cx = c[0, :].reshape(Omega._boxes)
            cy = c[1, :].reshape(Omega._boxes)
            cz  = x.reshape(Omega._boxes)
            self.surf(cx, cy, cz)
        else:
            print('Not defined for d > 2.')
