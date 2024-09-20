import Numeric
from Numeric import *
import fastumath
from utility import *
from convenience import *
from polynomial import *
from scimath import *
from helpmod import help, source
from Matrix import Matrix as mat
Mat = mat  # deprecated





"""Contains basic routines of common interest.  Always imported first.
   Basically MLab minus the LinearAlgebra-dependent functions.

   But max is changed to amax (array max)
    and min is changed to amin (array min)
   so that the builtin max and min are still available.
"""


__all__ = ['logspace','linspace','round','any','all','fix','mod','fftshift',
           'ifftshift','fftfreq','cont_ft','toeplitz','hankel','real','imag',
           'iscomplex','isreal','array_iscomplex','array_isreal','isposinf',
           'isneginf','nan_to_num','eye','tri','diag','fliplr','flipud',
           'rot90','tril','triu','amax','amin','ptp','cumsum','prod','cumprod',
           'diff','squeeze','sinc','angle','unwrap','real_if_close',
           'sort_complex']

