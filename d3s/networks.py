# -*- coding: utf-8 -*-
import numpy as _np
import scipy as _sp
import scipy.cluster
import networkx as _nx

from d3s.algorithms import dinv, sortEig

class graph(object):
    '''
    Simple graph class.
    '''
    
    colors = ('aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellow', 'yellowgreen')
    
    def __init__(self, x):
        if _np.isscalar(x):
            self.n = x            # number of vertices
            self.A = _np.zeros(x) # adjacency matrix
        else:
            self.n = x.shape[0]  # number of vertices
            self.A = x           # adjacency matrix
    
    def addEdge(self, i, j, w=1):
        self.A[i, j] = w
        
    def isSymmetric(self, atol=1e-8):
        return _np.allclose(self.A, self.A.T, atol=atol)
    
    def randomWalk(self, x0, m):
        '''
        Generate random walk of length m.
        '''
        P = self.transitionMatrix('rw')
        v = _np.arange(self.n) # vertices
        x = _np.zeros(m, dtype=_np.uint64)
        x[0] = x0
        for i in range(1, m):
            x[i] = _np.random.choice(v, p=P[x[i-1], :])
        return x
    
    def laplacian(self, variant='rw'):
        '''
        Compute graph Laplacian.
        
        :param variant: Choose 'un' (unnormalized),
                               'rw' (random-walk),
                               'fb' (forward-backward).
        '''
        if variant == 'un':
            return _np.diag(_np.sum(self.A, 1)) - self.A
        return _np.eye(self.n) - self.transitionMatrix(variant)
        
    def transitionMatrix(self, variant='rw'):
        '''
        Compute transition probability matrix.
        
        :param variant: Choose 'rw' (random-walk),
                               'fb' (forward-backward).
        '''
        D = _np.diag(_np.sum(self.A, 1))
        P = dinv(D) @ self.A
        
        if variant == 'rw':
            return P
        elif variant == 'fb':
            D_nu = _np.diag(_np.sum(P, 0)) # uniform density mapped forward
            Q = P @ dinv(D_nu) @ P.T
            return Q
        else:
            print('Unknown type.')
    
    def draw(self, c=None, pos=None):
        A = self.A - _np.diag(_np.diag(self.A)) # remove self-loops
        
        if self.isSymmetric():
            G = _nx.from_numpy_matrix(A)
        else:
            G = _nx.from_numpy_matrix(A, create_using=_nx.DiGraph)
            
        if pos is None:
            # pos = nx.spring_layout(G)
            pos = _nx.nx_agraph.graphviz_layout(G, prog='neato')
            print(pos)
        
        if c is None:
            _nx.draw(G, pos, node_size=1000, with_labels=True, font_size=15)
        else:
            col = [graph.colors[i] for i in c]
            _nx.draw(G, pos, node_color=col, node_size=1000, with_labels=True, font_size=15)


def spectralClustering(G, nc, variant='rw'):
    P = G.transitionMatrix(variant)
    d, V = sortEig(P, evs=nc, which='LR')
    _, c = _sp.cluster.vq.kmeans2(_np.real(V), nc, iter=100, minit='++')
    return (d, V, c)