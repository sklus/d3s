import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import d3s.algorithms as algorithms

plt.ion()

#%% load variables from mat file into main scope
data = sp.io.loadmat('data/tica.mat', squeeze_me=True)
for s in data.keys():
    if s[:2] == '__' and s[-2:] == '__': continue
    exec('%s = data["%s"]' % (s, s))

#%% apply TICA
d1, V1 = algorithms.tica(X, Y)
d2, V2 = algorithms.amuse(X, Y)

Xn = V1.T @ X

#%% plot original data set
for i in range(4):
    plt.figure()
    plt.plot(X[i, :])
    plt.title('X_%d' % i)

#%% plot transformed data set
for i in range(4):
    plt.figure()
    plt.plot(Xn[i, :])
    plt.title('Xn_%d' % i)
