from info import __doc__, __all__
from mtrand import *

# Some aliases:
ranf = random = sample = random_sample
__all__.extend(['ranf','random','sample'])

def __RandomState_ctor():
    """Return a RandomState instance.
    
    This function exists solely to assist (un)pickling.
    """
    return RandomState()

from numpy.testing import ScipyTest 
test = ScipyTest().test


# mtrand.pyx -- A Pyrex wrapper of Jean-Sebastien Roy's RandomKit
#
# Copyright 2005 Robert Kern (robert.kern@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

include "Python.pxi"
include "numpy.pxi"

cdef extern from "math.h":
    double exp(double x)
    double log(double x)
    double floor(double x)
    double sin(double x)
    double cos(double x)

cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    char *rk_strerror[2]
    
    # 0xFFFFFFFFUL
    unsigned long RK_MAX

    void rk_seed(unsigned long seed, rk_state *state)
    rk_error rk_randomseed(rk_state *state)
    unsigned long rk_random(rk_state *state)
    long rk_long(rk_state *state)
    unsigned long rk_ulong(rk_state *state)
    unsigned long rk_interval(unsigned long max, rk_state *state)
    double rk_double(rk_state *state)
    void rk_fill(void *buffer, size_t size, rk_state *state)
    rk_error rk_devfill(void *buffer, size_t size, int strong)
    rk_error rk_altfill(void *buffer, size_t size, int strong,
            rk_state *state)
    double rk_gauss(rk_state *state)

cdef extern from "distributions.h":
    
    double rk_normal(rk_state *state, double loc, double scale)
    double rk_standard_exponential(rk_state *state)
    double rk_exponential(rk_state *state, double scale)
    double rk_uniform(rk_state *state, double loc, double scale)
    double rk_standard_gamma(rk_state *state, double shape)
    double rk_gamma(rk_state *state, double shape, double scale)
    double rk_beta(rk_state *state, double a, double b)
    double rk_chisquare(rk_state *state, double df)
    double rk_noncentral_chisquare(rk_state *state, double df, double nonc)
    double rk_f(rk_state *state, double dfnum, double dfden)
    double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc)
    double rk_standard_cauchy(rk_state *state)
    double rk_standard_t(rk_state *state, double df)
    double rk_vonmises(rk_state *state, double mu, double kappa)
    double rk_pareto(rk_state *state, double a)
    double rk_weibull(rk_state *state, double a)
    double rk_power(rk_state *state, double a)
    double rk_laplace(rk_state *state, double loc, double scale)
    double rk_gumbel(rk_state *state, double loc, double scale)
    double rk_logistic(rk_state *state, double loc, double scale)
    double rk_lognormal(rk_state *state, double mode, double sigma)
    double rk_rayleigh(rk_state *state, double mode)
    double rk_wald(rk_state *state, double mean, double scale)
    double rk_triangular(rk_state *state, double left, double mode, double right)
    
    long rk_binomial(rk_state *state, long n, double p)
    long rk_binomial_btpe(rk_state *state, long n, double p)
    long rk_binomial_inversion(rk_state *state, long n, double p)
    long rk_negative_binomial(rk_state *state, long n, double p)
    long rk_poisson(rk_state *state, double lam)
    long rk_poisson_mult(rk_state *state, double lam)
    long rk_poisson_ptrs(rk_state *state, double lam)
    long rk_zipf(rk_state *state, double a)
    long rk_geometric(rk_state *state, double p)
    long rk_hypergeometric(rk_state *state, long good, long bad, long sample)
    long rk_logseries(rk_state *state, double p)

ctypedef double (* rk_cont0)(rk_state *state)
ctypedef double (* rk_cont1)(rk_state *state, double a)
ctypedef double (* rk_cont2)(rk_state *state, double a, double b)
ctypedef double (* rk_cont3)(rk_state *state, double a, double b, double c)

ctypedef long (* rk_disc0)(rk_state *state)
ctypedef long (* rk_discnp)(rk_state *state, long n, double p)
ctypedef long (* rk_discnmN)(rk_state *state, long n, long m, long N)
ctypedef long (* rk_discd)(rk_state *state, double a)


cdef extern from "initarray.h":
   void init_by_array(rk_state *self, unsigned long *init_key, 
                      unsigned long key_length)

# Initialize numpy
