Data-driven dynamical systems toolbox
-------------------------------------

This toolbox contains methods for the approximation of transfer operators and their eigenfunctions as well as methods for learning the governing equations from data:

 - DMD/TICA/AMUSE
 - Ulam's method
 - EDMD
 - kernel EDMD
 - SINDy
 - ...
 
The algorithms are implemented based on the following publications:

 - \ S. Brunton, J. Proctor, J. Kutz: *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*
 - \ M. Williams, I. Kevrekidis, C. Rowley: *A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition.*
 - \ M. Williams, C. Rowley, I. Kevrekidis: *A kernel-based method for data-driven Koopman spectral analysis.*
 - \ S. Klus, P. Koltai, C. Schütte: *On the numerical approximation of the Perron-Frobenius and Koopman operator.*
 - \ S. Klus, F. Nüske, P. Koltai, H. Wu, I. Kevrekidis, C. Schütte, F. Noé: *Data-driven model reduction and transfer operator approximation.*
 - \ S. Klus, I. Schuster, K. Muandet: *Eigendecompositions of transfer operators in reproducing kernel Hilbert spaces.*

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
