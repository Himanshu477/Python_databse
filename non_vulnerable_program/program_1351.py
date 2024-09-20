from numpy.testing import ScipyTest 
test = ScipyTest().test


"""Lite version of numpy.linalg.
"""
# This module is a lite version of LinAlg.py module which contains
# high-level Python interface to the LAPACK library.  The lite version
# only accesses the following LAPACK functions: dgesv, zgesv, dgeev,
# zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf, dpotrf.

__all__ = ['LinAlgError', 'solve_linear_equations', 'solve',
           'inverse', 'inv', 'cholesky_decomposition', 'cholesky', 'eigenvalues',
           'eigvals', 'Heigenvalues', 'eigvalsh', 'generalized_inverse', 'pinv',
           'determinant', 'det', 'singular_value_decomposition', 'svd',
           'eigenvectors', 'eig', 'Heigenvectors', 'eigh','lstsq', 'linear_least_squares'
           ]
           
