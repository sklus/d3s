Data-driven dynamical systems toolbox.

This toolbox contains methods for the approximation of transfer operators and their eigenfunctions as well as methods for learning the governing equations from data:
 - DMD/TICA/AMUSE
 - Ulam's method
 - EDMD
 - kernel EDMD
 - SINDy
 - ...

====

Conda Environment
-----------------
This conda environment handles boost dependencies for the user.
::

    conda env create -f environment.yml
    source activate d3s


Develop/Install
---------------
::

    python setup.py install [--user]

or::

    python setup.py develop [--user]
