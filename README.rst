Data-driven dynamical systems toolbox
-------------------------------------

This toolbox contains methods for the approximation of transfer operators and their eigenfunctions as well as methods for learning the governing equations from data:

- DMD, TICA, AMUSE
- Ulam's method
- EDMD, kernel EDMD, generator EDMD
- SINDy
- QENDy
- kernel PCA, kernel CCA
- CMD
- SEBA

The algorithms are implemented based on the following publications:

- \ F. Bach and M. Jordan. *Kernel Independent Component Analysis.*
- \ S. Brunton, J. Proctor, and J. Kutz. *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*
- \ G. Froyland, C. Rock, and K. Sakellariou. *Sparse eigenbasis approximation: Multiple feature extraction across spatiotemporal scales with application to coherent set identification.*
- \ F. Noé and F. Nüske. *A variational approach to modeling slow processes in stochastic dynamical systems.*
- \ B. Schölkopf, A. Smola, and K.-R. Müller. *Nonlinear component analysis as a kernel eigenvalue problem.*
- \ C. Schwantes and V. Pande. *Modeling Molecular Kinetics with tICA and the Kernel Trick.*
- \ L. Tong, V. Soon, Y. Huang, and R. Liu. *AMUSE: a new blind identification algorithm.*
- \ J. Tu, C. Rowley, D. Luchtenburg, S. Brunton, and J. Kutz. *On dynamic mode decomposition: Theory and applications.*
- \ M. Williams, I. Kevrekidis, and C. Rowley. *A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition.*
- \ M. Williams, C. Rowley, and I. Kevrekidis. *A kernel-based method for data-driven Koopman spectral analysis.*
- \ S. Klus, P. Koltai, and C. Schütte. *On the numerical approximation of the Perron-Frobenius and Koopman operator.*
- \ S. Klus, F. Nüske, P. Koltai, H. Wu, I. Kevrekidis, C. Schütte, and F. Noé. *Data-driven model reduction and transfer operator approximation.*
- \ S. Klus, I. Schuster, and K. Muandet. *Eigendecompositions of transfer operators in reproducing kernel Hilbert spaces.*
- \ S. Klus, B. E. Husic, and M. Mollenhauer: *Kernel canonical correlation analysis approximates operators for the detection of coherent structures in dynamical data.*
- \ S. Klus and J.-P. N'Konzi: *Data-driven system identification using quadratic embeddings of nonlinear dynamics.*

====

Develop/Install
---------------

The ODE/SDE solvers required by some examples to generate trajectory data are implemented in C++. In order to create Python bindings, first install pybind11, then go to the cpp directory and compile the code by executing the following command::

    Linux: g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` systems.cpp -o ../d3s/systems`python3-config --extension-suffix`
    MAC:   c++ -arch x86_64 -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` systems.cpp -o ../d3s/systems`python3-config --extension-suffix`

