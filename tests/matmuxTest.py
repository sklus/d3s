import d3s.tools as tools
import numpy as np

#%% create matmux object
mat = tools.matmux() # Matlab must be started before via: $ tmux new -s matlab "matlab -nodesktop"

#%% call plot function directly
x = np.linspace(1, 5, 10)
y = np.random.rand(len(x))
mat.plot(x, y)

#%% export to Matlab
mat.exportVars("x", x)
mat("y = x.^2\nz = x.^3")

#%% import from Matlab
y, z = mat.importVars('y', 'z')
