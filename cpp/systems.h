#ifndef D3S_SYSTEM_H
#define D3S_SYSTEM_H

#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/python/numpy.hpp>

typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;
typedef boost::python::numpy::ndarray ndarray;

typedef boost::mt19937 MersenneTwister;
typedef boost::normal_distribution<> NormalDistribution;
typedef boost::variate_generator<MersenneTwister, NormalDistribution> NormalDistributionGenerator;

// forward declarations
class ODE;
class SDE;

//------------------------------------------------------------------------------
// Helper class for accessing numpy matrices
//------------------------------------------------------------------------------
class NumpyWrapper
{
public:
    NumpyWrapper(ndarray const& x);
    double& operator()(size_t i, size_t j);
    
    Py_intptr_t shape[2]; ///< number of rows and columns
    size_t strides[2];    ///< strides in row and column directions
    double* data;         ///< pointer to raw data
};

//------------------------------------------------------------------------------
// Runge-Kutta integrator for ordinary differential equations
//------------------------------------------------------------------------------
class RungeKutta
{
public:
    RungeKutta(ODE* ode, size_t d, double h, size_t nSteps);
    
    void eval(Vector& x, Vector& y);
    
private:
    ODE* ode_;                      ///< ODE
    const double h_;                ///< step size
    const size_t nSteps_;           ///< number of integration steps
    Vector k1_, k2_, k3_, k4_, yt_; ///< temporary variables
};

//------------------------------------------------------------------------------
// Euler-Maruyama integrator for stochastic differential equations
//------------------------------------------------------------------------------
class EulerMaruyama
{
public:
    EulerMaruyama(SDE* sde, size_t d, double h, size_t nSteps);
    
    void eval(Vector& x, Vector& y);
    void updateNoise(Vector& w);
    
private:
    SDE* sde_;                            ///< drift term of SDE
    NormalDistributionGenerator normrnd_; ///< normally distributed random values
    const double h_;                      ///< step size of the integrator
    const double sqrt_h_;                 ///< precomputed square root of h for efficiency
    const size_t nSteps_;                 ///< number of integration steps
    Vector mu_;                           ///< temporary vector
};

//------------------------------------------------------------------------------
// Virtual base class for all dynamical systems
//------------------------------------------------------------------------------
class DynamicalSystemInterface
{
public:
    virtual ~DynamicalSystemInterface() {};
    
    virtual ndarray operator()(ndarray const& x); ///< Evaluates the dynamical system for all data points contained in x = [x_1, x_2, ..., x_m].
    virtual ndarray getTrajectory(ndarray const& x, size_t length); ///< Generate a trajectory for one data point x.
    virtual void eval(Vector& x, Vector& y) = 0; ///< Evaluates the dynamical system for one test point x. Must be implemented by derived classes.
    virtual size_t getDimension() const = 0; ///< Returns the number of dimensions d of the dynamical system.
};

//------------------------------------------------------------------------------
// Virtual base class for ordinary differential equations
//------------------------------------------------------------------------------
class ODE : public DynamicalSystemInterface
{
public:
    ODE(size_t d, double h, size_t nSteps);
    
    virtual void f(Vector& x, Vector& y) = 0;
    virtual void eval(Vector& x, Vector& y); // implementation of the pure virtual function inherited from DynamicalSystemInterface, calls Runge-Kutta integrator
    
private:
    RungeKutta integrator_;
};

//------------------------------------------------------------------------------
// Virtual base class for stochastic differential equations with constant sigma
//------------------------------------------------------------------------------
class SDE : public DynamicalSystemInterface
{
public:
    SDE(size_t d, double h, size_t nSteps);
    
    virtual void f(Vector& x, Vector& y) = 0;
    virtual void getSigma(Matrix& sigma) = 0;
    // TODO: add ndarray V(ndarray& x) // potential function
    
    virtual void eval(Vector& x, Vector& y); // implementation of the pure virtual function inherited from DynamicalSystemInterface, calls Euler-Maruyama integrator
    
private:
    EulerMaruyama integrator_;
};

//------------------------------------------------------------------------------
// Henon map
//------------------------------------------------------------------------------
class HenonMap : public DynamicalSystemInterface
{
public:
    HenonMap();
    void eval(Vector& x, Vector& y);
    size_t getDimension() const;
    
private:
    const double a_;
    const double b_;
};

//------------------------------------------------------------------------------
// Simple example from "Koopman Invariant Subspaces and Finite Linear
// Representations of Nonlinear Dynamical Systems for Control"
//------------------------------------------------------------------------------
class SimpleODE : public ODE
{
public:
    SimpleODE(double h = 1e-3, size_t nSteps = 1000);
    void f(Vector& x, Vector& y);
    size_t getDimension() const;
    
    static const size_t d = 2;
    
private:
    const double lambda_;
    const double mu_;
};

//------------------------------------------------------------------------------
// ABC flow
//------------------------------------------------------------------------------
class ABCFlow : public ODE
{
public:
    ABCFlow(double h = 1e-3, size_t nSteps = 1000);
    void f(Vector& x, Vector& y);
    size_t getDimension() const;
    
    static const size_t d = 3;
    
private:
    const double a_;
    const double b_;
    const double c_;
};

//------------------------------------------------------------------------------
// Chua's circuit
//------------------------------------------------------------------------------
class ChuaCircuit : public ODE
{
public:
    ChuaCircuit(double h = 1e-3, size_t nSteps = 2000);
    void f(Vector& x, Vector& y);
    size_t getDimension() const;
    
    static const size_t d = 3;
    
private:
    const double alpha_;
    const double beta_;
};

//------------------------------------------------------------------------------
// Ornstein-Uhlenbeck process
//------------------------------------------------------------------------------
class OrnsteinUhlenbeck : public SDE
{
public:
    OrnsteinUhlenbeck(double h = 1e-3, size_t nSteps = 500);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 1;
    
private:
    double alpha_;
    double beta_;
};
 
//------------------------------------------------------------------------------
// Simple triple-well in one dimension, use interval [0, 6]
//------------------------------------------------------------------------------
class TripleWell1D : public SDE
{
public:
    TripleWell1D(double h = 1e-3, size_t nSteps = 500);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 1;
};

//------------------------------------------------------------------------------
// Double well problem
//------------------------------------------------------------------------------
class DoubleWell2D : public SDE
{
public:
    DoubleWell2D(double h = 1e-3, size_t nSteps = 10000);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 2;
};

//------------------------------------------------------------------------------
// Quadruple well problem
//------------------------------------------------------------------------------
class QuadrupleWell2D : public SDE
{
public:
    QuadrupleWell2D(double h = 1e-3, size_t nSteps = 10000);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 2;
};

//------------------------------------------------------------------------------
// Triple well problem
//------------------------------------------------------------------------------
class TripleWell2D : public SDE
{
public:
    TripleWell2D(double h = 1e-5, size_t nSteps = 10000);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 2;
};

//------------------------------------------------------------------------------
// n-well on circle a.k.a. "lemon-slice potential"
//------------------------------------------------------------------------------
class LemonSlice2D : public SDE
{
public:
    LemonSlice2D(double h = 1e-3, size_t nSteps = 200);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 2;
};

//------------------------------------------------------------------------------
// System with banana-shaped potential from the Transition Manifolds paper
//------------------------------------------------------------------------------
class BananaSystem : public SDE
{
public:
    BananaSystem(double h = 1e-2, size_t nSteps = 100);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 2;
};

//------------------------------------------------------------------------------
// System 4.4/4.5 from "A Computational Method to Extract Macroscopic Variables
// and Their Dynamics in Multiscale Systems"
//------------------------------------------------------------------------------
class FastSlowSDE : public SDE
{
public:
    FastSlowSDE(double h = 2e-6, size_t nSteps = 20000);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 2;
    
private:
    const double epsilon_;
    const double a_;
};

//------------------------------------------------------------------------------
// Double well SDE in three dimensions
//------------------------------------------------------------------------------
class DoubleWell3D : public SDE
{
public:
    DoubleWell3D(double h = 1e-3, size_t nSteps = 10000);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 3;
};

//------------------------------------------------------------------------------
// Triple well SDE
//------------------------------------------------------------------------------
class TripleWell3D : public SDE
{
public:
    TripleWell3D(double h = 1e-3, size_t nSteps = 10000);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 3;
};

//------------------------------------------------------------------------------
// Double well SDE in six dimensions
//------------------------------------------------------------------------------
class DoubleWell6D : public SDE
{
public:
    DoubleWell6D(double h = 1e-3, size_t nSteps = 10000);
    void f(Vector& x, Vector& y);
    void getSigma(Matrix& sigma);
    size_t getDimension() const;
    
    static const size_t d = 6;
};

#endif // D3S_SYSTEM_H
