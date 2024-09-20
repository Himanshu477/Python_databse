from basic_lite import *
inv = inverse
solve = solve_linear_equations
cholesky = cholesky_decomposition
eigvals = eigenvalues
eigvalsh = Heigenvalues
eig = eigenvectors
eigh = Heigenvectors
svd = singular_value_decomposition
pinv = generalized_inverse
det = determinant
lstsq = linear_least_squares



#!/usr/bin/env python
"""

Rules for building C/API module with f2py2e.

Here is a skeleton of a new wrapper function (13Dec2001):

wrapper_function(args)
  declarations
  get_python_arguments, say, `a' and `b'

  get_a_from_python
  if (successful) {

    get_b_from_python
    if (successful) {

      callfortran
      if (succesful) {

        put_a_to_python
        if (succesful) {

          put_b_to_python
          if (succesful) {

            buildvalue = ...

          }

        }
        
      }

    }
    cleanup_b

  }
  cleanup_a

  return buildvalue
"""
"""
Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/08/30 08:58:42 $
Pearu Peterson
"""

__version__ = "$Revision: 1.129 $"[10:-1]

