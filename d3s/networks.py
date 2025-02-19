import numpy as _np
import scipy as _sp
import scipy.cluster
import networkx as _nx

from d3s.algorithms import dinv, sortEig

class graph(object):
    '''
    Simple directed or undirected graph.
    '''
    
    colors = ('aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellow', 'yellowgreen')
    
    def __init__(self, A):
        self.n = A.shape[0]  # number of vertices
        self.A = A           # adjacency matrix
    
    def addEdge(self, i, j, w=1):
        self.A[i, j] = w
        
    def isSymmetric(self, atol=1e-8):
        return _np.allclose(self.A, self.A.T, atol=atol)
    
    def randomWalk(self, x0, m):
        '''
        Generate random walk of length m starting in x0.
        '''
        P = self.transitionMatrix('rw')
        v = _np.arange(self.n) # vertices
        x = _np.zeros(m+1, dtype=_np.uint64)
        x[0] = x0
        for i in range(1, m+1):
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
            G = _nx.from_numpy_array(A)
        else:
            G = _nx.from_numpy_array(A, create_using=_nx.DiGraph)
            
        if pos is None:
            # pos = nx.spring_layout(G)
            pos = _nx.nx_agraph.graphviz_layout(G, prog='neato')
        
        if c is None:
            _nx.draw_networkx(G, pos, node_size=500, with_labels=True, font_size=10)
        else:
            col = [graph.colors[i] for i in c]
            _nx.draw_networkx(G, pos, node_color=col, node_size=500, with_labels=True, font_size=10)
            
    def spectralClustering(self, nc, variant='rw'):
        P = self.transitionMatrix(variant)
        d, V = sortEig(P, evs=nc, which='LR')
        _, c = _sp.cluster.vq.kmeans2(_np.real(V), nc, iter=100, minit='++')
        return (d, V, c)

class tgraph(object):
    '''
    Simple time-evolving directed or undirected graph.
    '''
    def __init__(self, A):
        self.n = A.shape[0]  # number of vertices
        self.T = A.shape[2]  # number of graphs
        self.A = A           # adjacency matrix
    
    def __getitem__(self, t):
        return graph(self.A[:, :, t])
    
    def randomWalk(self, x0, m):
        '''
        Generate random walk of length m in each graph.
        '''
        for t in range(self.T):
            x = self[t].randomWalk(x0, m)
            if t == 0:
                y = x[:-1]
            elif t == self.T-1:
                y = _np.hstack([y, x])
            else:
                y = _np.hstack([y, x[:-1]])
            x0 = x[-1]
        return y
    
    def spectralClustering(self, nc):
        P = _np.eye(self.n)
        for t in range(self.T):
            P = P @ self[t].transitionMatrix('rw')
        D_nu = _np.diag(_np.sum(P, 0)) # uniform density mapped forward
        Q = P @ dinv(D_nu) @ P.T
        
        d, V = sortEig(Q, evs=nc, which='LR')
        _, c = _sp.cluster.vq.kmeans2(_np.real(V), nc, iter=100, minit='++')
        return (d, V, c)
