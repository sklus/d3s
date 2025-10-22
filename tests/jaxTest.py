import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

import d3s.domain as domain
import d3s.observables as observables

class myBasis(observables.jaxBasis):
    def __call__(self, x):
        return jnp.array([1 - x[0]**2 - 1/2*x[1]**2,
                         (1 - x[0]**2 - 1/2*x[1]**2)*x[0],
                         (1 - x[0]**2 - 1/2*x[1]**2)*x[1]])

bounds = np.array([[-2, 2], [-2, 2]])
boxes = np.array([50, 50])
Omega = domain.discretization(bounds, boxes)

X = Omega.midpointGrid()
psi = myBasis()

PsiX = psi(X)
dPsiX = psi.diff(X)
ddPsiX = psi.ddiff(X)

for i in range(3):
    plt.figure(1)
    plt.clf()
    Omega.plot(PsiX[i, :], '3D')
    plt.title(f'$\psi_{i}$')
    plt.draw()
    
    plt.figure(2)
    plt.clf()
    Omega.plot(dPsiX[i, 0, :], '3D')
    plt.title(f'$\partial \psi_{i} / \partial x_0$')
    plt.draw()
    
    plt.figure(3)
    plt.clf()
    Omega.plot(dPsiX[i, 1, :], '3D')
    plt.title(f'$\partial \psi_{i} / \partial x_1$')
    plt.draw()
    
    plt.figure(4)
    plt.clf()
    Omega.plot(ddPsiX[i, 0, 0, :], '3D')
    plt.title(f'$\partial^2 \psi_{i} / \partial x_0 \partial x_0$')
    plt.draw()
    
    plt.figure(5)
    plt.clf()
    Omega.plot(ddPsiX[i, 0, 1, :], '3D')
    plt.title(f'$\partial^2 \psi_{i} / \partial x_0 \partial x_1$')
    plt.draw()
    
    plt.figure(6)
    plt.clf()
    Omega.plot(ddPsiX[i, 1, 0, :], '3D')
    plt.title(f'$\partial^2 \psi_{i} / \partial x_1 \partial x_0$')
    plt.draw()
    
    plt.figure(7)
    plt.clf()
    Omega.plot(ddPsiX[i, 1, 1, :], '3D')
    plt.title(f'$\partial^2 \psi_{i} / \partial x_1 \partial x_1$')
    plt.draw()
    
    plt.waitforbuttonpress() # press button in Figure 7
    