import numpy
import matplotlib

import d3s.domain as domain
import d3s.observables as observables
import d3s.tools as tools

mat = tools.matmux()

#%% monomials in 1D
bounds = numpy.array([[-1, 1]])
boxes = numpy.array([100])
Omega = domain.discretization(bounds, boxes)

psi = observables.monomials(5)
PsiC = psi(Omega.midpointGrid())

matplotlib.pyplot.figure()
for i in range(PsiC.shape[0]):
    Omega.plot(PsiC[i, :])
matplotlib.pyplot.show()

#%% Gaussian basis functions in 1D
psi = observables.gaussians(Omega, sigma=0.1)
PsiC = psi(Omega.midpointGrid())

matplotlib.pyplot.figure()
for i in range(0, PsiC.shape[0], 20):
    Omega.plot(PsiC[i, :])
matplotlib.pyplot.show()

#%% use Matlab for plotting
for i in range(0, PsiC.shape[0], 20):
    mat.figure()
    mat.plotDomain(Omega, PsiC[i, :])

#%% monomials in 2D
bounds = numpy.array([[-2, 2], [-1, 2]])
boxes = numpy.array([10, 10])
Omega = domain.discretization(bounds, boxes)

psi = observables.monomials(2)
PsiC = psi(Omega.midpointGrid())

for i in range(PsiC.shape[0]):
    matplotlib.pyplot.figure()
    Omega.plot(PsiC[i, :], '3D')

#%% Gaussian basis functions in 2D
psi = observables.gaussians(Omega)
PsiC = psi(Omega.midpointGrid())

for i in range(0, PsiC.shape[0], 20):
    matplotlib.pyplot.figure()
    Omega.plot(PsiC[i, :], '3D')

#%% use Matlab for plotting
for i in range(0, PsiC.shape[0], 20):
    mat.figure()
    mat.plotDomain(Omega, PsiC[i, :])
