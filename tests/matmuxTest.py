#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import d3s.tools as tools
import scipy

#%% create matmux object
mat = tools.matmux() # Matlab must be started before via: $ tmux new -s matlab "matlab -nodesktop"

#%% call plot function directly
x = scipy.linspace(1, 5, 10)
y = scipy.rand(len(x))
mat.plot(x, y)

#%% export to Matlab
mat.exportVars("x", x)
mat("y = x.^2\nz = x.^3")

#%% import from Matlab
y, z = mat.importVars('y', 'z')
