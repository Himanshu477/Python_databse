from machar import machar_float, machar_single, machar_longfloat

single_epsilon = machar_single.epsilon
single_tiny = machar_single.tiny
single_max = machar_single.huge
single_min = -single_max
single_precision = machar_single.precision
single_resolution = machar_single.resolution

float_epsilon = machar_float.epsilon
float_tiny = machar_float.tiny
float_max = machar_float.huge
float_min = -float_max
float_precision = machar_float.precision
float_resolution = machar_float.resolution

longfloat_epsilon = machar_longfloat.epsilon
longfloat_tiny = machar_longfloat.tiny
longfloat_max = machar_longfloat.huge
longfloat_min = -longfloat_max
longfloat_precision = machar_longfloat.precision
longfloat_resolution = machar_longfloat.resolution



if __name__ == '__main__':
    print 'single epsilon:',single_epsilon
    print 'single tiny:',single_tiny
    print 'float epsilon:',float_epsilon
    print 'float tiny:',float_tiny
    print 'longfloat epsilon:',longfloat_epsilon
    print 'longfloat tiny:',longfloat_tiny



#
# Machine arithmetics - determine the parameters of the
# floating-point arithmetic system
#
# Author: Pearu Peterson, September 2003
#

__all__ = ['MachAr','machar_float','machar_single','machar_longfloat']

