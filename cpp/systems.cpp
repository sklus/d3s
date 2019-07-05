#include "systems.h"

//------------------------------------------------------------------------------
// Helper class for accessing numpy matrices
//------------------------------------------------------------------------------
NumpyWrapper::NumpyWrapper(ndarray const& x)
    : shape {x.shape(0), x.shape(1)},
        strides {x.strides(0)/sizeof(double), x.strides(1)/sizeof(double)}
{
    data = reinterpret_cast<double*>(x.get_data());
}

double& NumpyWrapper::operator()(size_t i, size_t j)
{
    return data[strides[0]*i + strides[1]*j];
}

//------------------------------------------------------------------------------
// Runge-Kutta integrator for ordinary differential equations
//------------------------------------------------------------------------------
RungeKutta::RungeKutta(ODE* ode, size_t d, double h, size_t nSteps)
    : ode_(ode),
      h_(h),
      nSteps_(nSteps)
{
    k1_.resize(d); // cannot use ode->getDimension() since derived object might not be fully constructed yet
    k2_.resize(d);
    k3_.resize(d);
    k4_.resize(d);
    yt_.resize(d);
}

void RungeKutta::eval(Vector& x, Vector& y)
{
    const size_t d = ode_->getDimension();
    
    y = x; // copy initial value to y
    
    for (size_t i = 0; i < nSteps_; ++i)
    {
        ode_->f(y, k1_); // compute k1 = f(y)
        
        for (size_t j = 0; j < d; ++j)
            yt_[j] = y[j] + h_/2*k1_[j];
        ode_->f(yt_, k2_); // compute k2 = f(y+h/2*k1)
        
        for (size_t j = 0; j < d; ++j)
            yt_[j] = y[j] + h_/2*k2_[j];
        ode_->f(yt_, k3_); // compute k3 = f(y+h/2*k2)
        
        for (size_t j = 0; j < d; ++j)
            yt_[j] = y[j] + h_*k3_[j];
        ode_->f(yt_, k4_); // compute k4 = f(y+h*k3)
        
        for (size_t j = 0; j < d; ++j) // compute x_{k+1} = x_k + h/6*(k1 + 2*k2 + 2*k3 + k4)
            y[j] = y[j] + h_/6.0*(k1_[j] + 2*k2_[j] + 2*k3_[j] + k4_[j]);
    }
}

//------------------------------------------------------------------------------
// Euler-Maruyama integrator for stochastic differential equations
//------------------------------------------------------------------------------
EulerMaruyama::EulerMaruyama(SDE* sde, size_t d, double h, size_t nSteps)
    : sde_(sde),
      h_(h),
      nSteps_(nSteps),
      normrnd_(MersenneTwister(std::time(0)), NormalDistribution()), // create random number generator with underlying integer random number generator and
      sqrt_h_(std::sqrt(h_))                                         // normal distribution with mean = 0.0 and sigma = 1.0)
{
    mu_.resize(d);
}

void EulerMaruyama::eval(Vector& x, Vector& y)
{
    const size_t d = sde_->getDimension();
    
    Matrix sigma(d); // sigma is required to be constant here
    for (size_t i = 0; i < d; ++i)
        sigma[i].resize(d);
    sde_->getSigma(sigma);
    
    Vector w(d); // for the Wiener processes
    
    y = x; // copy initial value to y
    
    for (size_t i = 0; i < nSteps_; ++i)
    {
        sde_->f(y, mu_); // compute drift term mu
        updateNoise(w);  // evaluate Wiener processes
        
        for (size_t j = 0; j < d; ++j)
        {
            y[j] = y[j] + h_*mu_[j];
            
            for (size_t k = 0; k < d; ++k)
                y[j] += sigma[j][k]*sqrt_h_*w[k];
        }
    }
}

void EulerMaruyama::updateNoise(Vector& w)
{
    const size_t d = sde_->getDimension();
    for (size_t i = 0; i < d; ++i)
        w[i] = normrnd_();
}

//------------------------------------------------------------------------------
// Virtual base class for all dynamical systems
//------------------------------------------------------------------------------
ndarray DynamicalSystemInterface::operator()(ndarray const& x)
{
    NumpyWrapper xw(x); // wrapper for x matrix
    const size_t d = xw.shape[0];
    const size_t m = xw.shape[1];
    
    ndarray y = zeros(x.get_nd(), x.get_shape(), boost::python::numpy::dtype::get_builtin<double>()); // create output matrix y
    NumpyWrapper yw(y);
    
    Vector xi(d), yi(d);
    for (size_t i = 0; i < m; ++i) // for all test points
    {
        for (size_t k = 0; k < d; ++k) // copy new test point into x vector
            xi[k] = xw(k, i);
        
        eval(xi, yi); // evaluate dynamical system
        
        for (size_t k = 0; k < d; ++k) // copy result into y vector
            yw(k, i) = yi[k];
    }
    
    return y;
}

ndarray DynamicalSystemInterface::getTrajectory(ndarray const& x, size_t length)
{
    NumpyWrapper xw(x); // wrapper for x matrix
    const size_t d = xw.shape[0];
    const size_t m = xw.shape[1];
    
    boost::python::tuple shape = boost::python::make_tuple(d, length);
    ndarray y = zeros(shape, boost::python::numpy::dtype::get_builtin<double>());
    NumpyWrapper yw(y);
    
    for (size_t k = 0; k < d; ++k) // copy initial condition
        yw(k, 0) = xw(k, 0);
    
    Vector xi(d), yi(d);
    for (size_t i = 1; i < length; ++i)
    {
        for (size_t k = 0; k < d; ++k) // copy new test point into x vector
            xi[k] = yw(k, i-1);
        
        eval(xi, yi); // evaluate dynamical system
        
        for (size_t k = 0; k < d; ++k) // copy result into y vector
            yw(k, i) = yi[k];
    }
    
    return y;
}

//------------------------------------------------------------------------------
// Virtual base class for ordinary differential equations
//------------------------------------------------------------------------------

ODE::ODE(size_t d, double h, size_t nSteps)
    : integrator_(this, d, h, nSteps)
{ }

void ODE::eval(Vector& x, Vector& y) // implementation of the pure virtual function inherited from DynamicalSystemInterface
{
    integrator_.eval(x, y);
}

//------------------------------------------------------------------------------
// Virtual base class for stochastic differential equations with constant sigma
//------------------------------------------------------------------------------

SDE::SDE(size_t d, double h, size_t nSteps)
    : integrator_(this, d, h, nSteps)
{ }

void SDE::eval(Vector& x, Vector& y) // implementation of the pure virtual function inherited from DynamicalSystemInterface
{
    integrator_.eval(x, y);
}

//------------------------------------------------------------------------------
// Henon map
//------------------------------------------------------------------------------
HenonMap::HenonMap()
    : a_(1.4),
      b_(0.3)
{ }

void HenonMap::eval(Vector& x, Vector& y)
{
    y[0] = 1 - a_*x[0]*x[0] + x[1];
    y[1] = b_*x[0];
}

size_t HenonMap::getDimension() const
{
    return 2;
}

//------------------------------------------------------------------------------
// Simple example from "Koopman Invariant Subspaces and Finite Linear
// Representations of Nonlinear Dynamical Systems for Control"
//------------------------------------------------------------------------------
SimpleODE::SimpleODE(double h, size_t nSteps)
    : ODE(d, h, nSteps),
      lambda_(-0.75),
      mu_(-0.5)
{ }

void SimpleODE::f(Vector& x, Vector& y)
{
    y[0] = mu_*x[0];
    y[1] = lambda_*(x[1] - x[0]*x[0]);
}

size_t SimpleODE::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// ABC flow
//------------------------------------------------------------------------------
ABCFlow::ABCFlow(double h, size_t nSteps)
    : ODE(d, h, nSteps),
      a_(sqrt(3)),
      b_(sqrt(2)),
      c_(1)
{ }

void ABCFlow::f(Vector& x, Vector& y)
{
    y[0] = a_*sin(2*M_PI*x[2]) + c_*cos(2*M_PI*x[1]);
    y[1] = b_*sin(2*M_PI*x[0]) + a_*cos(2*M_PI*x[2]);
    y[2] = c_*sin(2*M_PI*x[1]) + b_*cos(2*M_PI*x[0]);
}

size_t ABCFlow::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Chua's circuit
//------------------------------------------------------------------------------
ChuaCircuit::ChuaCircuit(double h, size_t nSteps)
    : ODE(d, h, nSteps),
      alpha_(10.0),
      beta_(14.87)
{ }

void ChuaCircuit::f(Vector& x, Vector& y)
{
    double f = -1.27*x[0] + 0.0157*x[0]*x[0]*x[0];
    // double f = -8.0/7.0*x[0] + 4.0/63.0*x[0]*std::abs(x[0]);
    
    y[0] = alpha_*(x[1] - x[0] - f);
    y[1] = x[0] - x[1] + x[2];
    y[2] = -beta_*x[1];
}

size_t ChuaCircuit::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Ornstein-Uhlenbeck process
//------------------------------------------------------------------------------
OrnsteinUhlenbeck::OrnsteinUhlenbeck(double h, size_t nSteps)
    : SDE(d, h, nSteps),
      alpha_(1), beta_(4)
{}

void OrnsteinUhlenbeck::f(Vector& x, Vector& y)
{
    y[0] = -alpha_*x[0];
}

void OrnsteinUhlenbeck::getSigma(Matrix& sigma)
{
    sigma[0][0] = sqrt(2/beta_);
}

size_t OrnsteinUhlenbeck::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Simple triple-well in one dimension, use interval [0, 6]
//------------------------------------------------------------------------------
TripleWell1D::TripleWell1D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void TripleWell1D::f(Vector& x, Vector& y)
{
    y[0] = -1*(-24.82002100 + 82.85029600*x[0] - 82.6031550*x[0]*x[0]
            + 34.125104*std::pow(x[0], 3) - 6.20030*std::pow(x[0], 4) + 0.4104*std::pow(x[0], 5));
}

void TripleWell1D::getSigma(Matrix& sigma)
{
    sigma[0][0] = 0.75;
}

size_t TripleWell1D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Double well problem
//------------------------------------------------------------------------------
DoubleWell2D::DoubleWell2D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void DoubleWell2D::f(Vector& x, Vector& y)
{
    // Double well potential: V = (x(1, :).^2 - 1).^2 + x(2, :).^2
    y[0] = -4*x[0]*x[0]*x[0] + 4*x[0];
    y[1] = -2*x[1];
}

void DoubleWell2D::getSigma(Matrix& sigma)
{
    sigma[0][0] = 0.7; sigma[0][1] = 0.0;
    sigma[1][0] = 0.0; sigma[1][1] = 0.7;
}

size_t DoubleWell2D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Quadruple well problem
//------------------------------------------------------------------------------
QuadrupleWell2D::QuadrupleWell2D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{ }

void QuadrupleWell2D::f(Vector& x, Vector& y)
{
    // Quadruple well potential: V = (x(1, :).^2 - 1).^2 + (x(2, :).^2 - 1).^2
    y[0] = -4*x[0]*x[0]*x[0] + 4*x[0];
    y[1] = -4*x[1]*x[1]*x[1] + 4*x[1];
}

void QuadrupleWell2D::getSigma(Matrix& sigma)
{
    sigma[0][0] = 0.7; sigma[0][1] = 0.0;
    sigma[1][0] = 0.0; sigma[1][1] = 0.7;
}

size_t QuadrupleWell2D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Triple well problem
//------------------------------------------------------------------------------
TripleWell2D::TripleWell2D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void TripleWell2D::f(Vector& x, Vector& y)
{
    y[0] = -(3*exp(-x[0]*x[0] - (x[1]-1.0/3)*(x[1]-1.0/3))*(-2*x[0])
            -3*exp(-x[0]*x[0] - (x[1]-5.0/3)*(x[1]-5.0/3))*(-2*x[0])
            -5*exp(-(x[0]-1.0)*(x[0]-1.0) - x[1]*x[1])*(-2*(x[0]-1.0))
            -5*exp(-(x[0]+1.0)*(x[0]+1.0) - x[1]*x[1])*(-2*(x[0]+1.0))
            + 8.0/10*std::pow(x[0], 3));
    y[1] = -(3*exp(-x[0]*x[0] - (x[1]-1.0/3)*(x[1]-1.0/3))*(-2*(x[1]-1.0/3))
            -3*exp(-x[0]*x[0] - (x[1]-5.0/3)*(x[1]-5.0/3))*(-2*(x[1]-5.0/3))
            -5*exp(-(x[0]-1.0)*(x[0]-1.0) - x[1]*x[1])*(-2*x[1])
            -5*exp(-(x[0]+1.0)*(x[0]+1.0) - x[1]*x[1])*(-2*x[1])
            + 8.0/10*std::pow(x[1]-1.0/3, 3));
}

void TripleWell2D::getSigma(Matrix& sigma)
{
    sigma[0][0] = 1.09; sigma[0][1] = 0.0;
    sigma[1][0] = 0.0;  sigma[1][1] = 1.09;
}

size_t TripleWell2D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// n-well on circle a.k.a. "lemon-slice potential"
//------------------------------------------------------------------------------
LemonSlice2D::LemonSlice2D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void LemonSlice2D::f(Vector& x, Vector& y)
{
    // Potential: V = cos(n*atan2(x(2, :), x(1, :))) + 10*(sqrt(x(1, :).^2 + x(2, :).^2) - 1).^2;
    const int n = 5;
    y[0] = -( n*sin(n*atan2(x[1], x[0]))*x[1]/(x[0]*x[0]+x[1]*x[1]) + (20*(sqrt(x[0]*x[0]+x[1]*x[1]) - 1))*x[0]/sqrt(x[0]*x[0]+x[1]*x[1]));
    y[1] = -(-n*sin(n*atan2(x[1], x[0]))*x[0]/(x[0]*x[0]+x[1]*x[1]) + (20*(sqrt(x[0]*x[0]+x[1]*x[1]) - 1))*x[1]/sqrt(x[0]*x[0]+x[1]*x[1]));
}

void LemonSlice2D::getSigma(Matrix& sigma)
{
    sigma[0][0] = 1; sigma[0][1] = 0;
    sigma[1][0] = 0; sigma[1][1] = 1;
}

size_t LemonSlice2D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// System with banana-shaped potential from the Transition Manifolds paper
//------------------------------------------------------------------------------
BananaSystem::BananaSystem(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void BananaSystem::f(Vector& x, Vector& y)
{
    // Potential: V = (x(1,:).^2 - 1).^2 + (x(1,:).^2 + x(2,:)-1).^2 / epsilon;
    const double epsilon = 0.5;
    y[0] = -(2*(x[0]*x[0] - 1)*2*x[0] + 2*(x[0]*x[0] + x[1]-1)*2*x[0]/epsilon);
    y[1] = -(2*(x[0]*x[0] + x[1] - 1)/epsilon);
}

void BananaSystem::getSigma(Matrix& sigma)
{
    // isotropic
    // sigma[0][0] = 1; sigma[0][1] = 0;
    // sigma[1][0] = 0; sigma[1][1] = 1;
    
    // anisotropic
    sigma[0][0] = 3; sigma[0][1] = 0;
    sigma[1][0] = 0; sigma[1][1] = 1;
}

size_t BananaSystem::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// System 4.4/4.5 from "A Computational Method to Extract Macroscopic Variables
// and Their Dynamics in Multiscale Systems"
//------------------------------------------------------------------------------
FastSlowSDE::FastSlowSDE(double h, size_t nSteps)
    : SDE(d, h, nSteps),
      epsilon_(0.01),
      a_(0.02)
{}

void FastSlowSDE::f(Vector& x, Vector& y)
{
    y[0] = x[0] - x[0]*x[0]*x[0] + a_/epsilon_*x[1];
    y[1] = 1.0/(epsilon_*epsilon_)*(x[1] - x[1]*x[1]*x[1]);
}

void FastSlowSDE::getSigma(Matrix& sigma)
{
    sigma[0][0] = 0; sigma[0][1] = 0;
    sigma[1][0] = 0; sigma[1][1] = std::sqrt(0.113)/epsilon_;
}

size_t FastSlowSDE::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Double well SDE in three dimensions
//------------------------------------------------------------------------------
DoubleWell3D::DoubleWell3D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void DoubleWell3D::f(Vector& x, Vector& y)
{
    // Double well potential
    y[0] = -4*x[0]*x[0]*x[0] + 4*x[0];
    y[1] = -2*x[1];
    y[2] = -2*x[2];
}

void DoubleWell3D::getSigma(Matrix& sigma)
{
    sigma[0][0] = 0.7; sigma[0][1] = 0.0; sigma[0][2] = 0.0;
    sigma[1][0] = 0.0; sigma[1][1] = 0.7; sigma[1][2] = 0.0;
    sigma[2][0] = 0.0; sigma[2][1] = 0.0; sigma[2][2] = 0.7;
}

size_t DoubleWell3D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Triple well SDE
//------------------------------------------------------------------------------
TripleWell3D::TripleWell3D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void TripleWell3D::f(Vector& x, Vector& y)
{
    y[0] = -(3*exp(-x[0]*x[0] - (x[1]-1.0/3)*(x[1]-1.0/3))*(-2*x[0])
            -3*exp(-x[0]*x[0] - (x[1]-5.0/3)*(x[1]-5.0/3))*(-2*x[0])
            -5*exp(-(x[0]-1.0)*(x[0]-1.0) - x[1]*x[1])*(-2*(x[0]-1.0))
            -5*exp(-(x[0]+1.0)*(x[0]+1.0) - x[1]*x[1])*(-2*(x[0]+1.0))
            + 8.0/10*std::pow(x[0], 3));
    
    y[1] = -(3*exp(-x[0]*x[0] - (x[1]-1.0/3)*(x[1]-1.0/3))*(-2*(x[1]-1.0/3))
            -3*exp(-x[0]*x[0] - (x[1]-5.0/3)*(x[1]-5.0/3))*(-2*(x[1]-5.0/3))
            -5*exp(-(x[0]-1.0)*(x[0]-1.0) - x[1]*x[1])*(-2*x[1])
            -5*exp(-(x[0]+1.0)*(x[0]+1.0) - x[1]*x[1])*(-2*x[1])
                + 8.0/10*std::pow(x[1]-1.0/3, 3));
    
    y[2] = -1.55*2*x[2];
}

void TripleWell3D::getSigma(Matrix& sigma)
{
    sigma[0][0] = 1.09; sigma[0][1] = 0.0;  sigma[0][2] = 0.0;
    sigma[1][0] = 0.0;  sigma[1][1] = 1.09; sigma[1][2] = 0.0;
    sigma[2][0] = 0.0;  sigma[2][1] = 0.0;  sigma[2][2] = 1.09;
}

size_t TripleWell3D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// Double well SDE in six dimensions
//------------------------------------------------------------------------------
DoubleWell6D::DoubleWell6D(double h, size_t nSteps)
    : SDE(d, h, nSteps)
{}

void DoubleWell6D::f(Vector& x, Vector& y)
{
    y[0] = -4*x[0]*x[0]*x[0] + 4*x[0];
    y[1] = -2*x[1];
    y[2] = -8*x[2]*x[2]*x[2] + 8*x[2];
    y[3] = -2*x[3];
    y[4] = -6*x[4]*x[4]*x[4] + 6*x[4];
    y[5] = -2*x[5];
}

void DoubleWell6D::getSigma(Matrix& sigma)
{
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j)
            sigma[i][j] = (i == j ? 0.7 : 0.0);
}

size_t DoubleWell6D::getDimension() const
{
    return d;
}

//------------------------------------------------------------------------------
// class export to python
//------------------------------------------------------------------------------

#define S(x) #x
#define EXPORT_DISC(name)                                  \
    class_<name>(S(name))                                  \
            .def("getDimension", &name::getDimension)      \
            .def("__call__", &name::operator())            \
            .def("getTrajectory", &name::getTrajectory);
#define EXPORT_CONT(name)                                  \
    class_<name>(S(name), init<double, size_t>())          \
            .def("getDimension", &name::getDimension)      \
            .def("__call__", &name::operator())            \
            .def("getTrajectory", &name::getTrajectory);


using boost::python::class_;
using boost::python::init;

BOOST_PYTHON_MODULE(systems)
{
    boost::python::numpy::initialize();
    
    EXPORT_DISC(HenonMap);
    EXPORT_CONT(SimpleODE);
    EXPORT_CONT(ABCFlow);
    EXPORT_CONT(ChuaCircuit);
    EXPORT_CONT(OrnsteinUhlenbeck);
    EXPORT_CONT(TripleWell1D);
    EXPORT_CONT(DoubleWell2D);
    EXPORT_CONT(QuadrupleWell2D);
    EXPORT_CONT(TripleWell2D);
    EXPORT_CONT(LemonSlice2D);
    EXPORT_CONT(BananaSystem);
    EXPORT_CONT(FastSlowSDE);
    EXPORT_CONT(DoubleWell3D);
    EXPORT_CONT(TripleWell3D);
    EXPORT_CONT(DoubleWell6D);
}
