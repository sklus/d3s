import numpy as np
import matplotlib.pyplot as plt

import d3s.domain as domain
import d3s.observables as observables
import d3s.tools as tools

mat = tools.matmux()

#%% monomials in 1D
bounds = np.array([[-1, 1]])
boxes = np.array([100])
Omega = domain.discretization(bounds, boxes)

psi = observables.monomials(5)
PsiC = psi(Omega.midpointGrid())

plt.figure()
for i in range(PsiC.shape[0]):
    Omega.plot(PsiC[i, :])
plt.show()

#%% Gaussian basis functions in 1D
psi = observables.gaussians(Omega, sigma=0.1)
PsiC = psi(Omega.midpointGrid())

plt.figure()
for i in range(0, PsiC.shape[0], 20):
    Omega.plot(PsiC[i, :])
plt.show()

#%% use Matlab for plotting
for i in range(0, PsiC.shape[0], 20):
    mat.figure()
    mat.plotDomain(Omega, PsiC[i, :])

#%% monomials in 2D
bounds = np.array([[-2, 2], [-1, 2]])
boxes = np.array([10, 10])
Omega = domain.discretization(bounds, boxes)

psi = observables.monomials(2)
PsiC = psi(Omega.midpointGrid())

for i in range(PsiC.shape[0]):
    plt.figure()
    Omega.plot(PsiC[i, :], '3D')

#%% Gaussian basis functions in 2D
psi = observables.gaussians(Omega)
PsiC = psi(Omega.midpointGrid())

for i in range(0, PsiC.shape[0], 20):
    plt.figure()
    Omega.plot(PsiC[i, :], '3D')

#%% use Matlab for plotting
for i in range(0, PsiC.shape[0], 20):
    mat.figure()
    mat.plotDomain(Omega, PsiC[i, :])
