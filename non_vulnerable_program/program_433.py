import sys as _sys
_sys.modules['umath'] = fastumath


if Numeric.__version__ < '23.5':
    matrixmultiply=dot

Inf = inf = fastumath.PINF
try:
    NAN = NaN = nan = fastumath.NAN
except AttributeError:
    NaN = NAN = nan = fastumath.PINF/fastumath.PINF

try:
