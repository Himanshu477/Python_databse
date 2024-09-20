    import core
    import extras
    revision = [core.__revision__.split(':')[-1][:-1].strip(),
                extras.__revision__.split(':')[-1][:-1].strip(),]
    version += '.dev%04i' % max([int(rev) for rev in revision])


# pylint: disable-msg=E1002
"""MA: a facility for dealing with missing observations
MA is generally used as a numpy.array look-alike.
by Paul F. Dubois.

Copyright 1999, 2000, 2001 Regents of the University of California.
Released for unlimited redistribution.
Adapted for numpy_core 2005 by Travis Oliphant and
(mainly) Paul Dubois.

Subclassing of the base ndarray 2006 by Pierre Gerard-Marchant.
pgmdevlist_AT_gmail_DOT_com
Improvements suggested by Reggie Dugard (reggie_AT_merfinllc_DOT_com)

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: core.py 3639 2007-12-13 03:39:17Z pierregm $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: pierregm $)"
__version__ = '1.0'
__revision__ = "$Revision: 3639 $"
__date__     = '$Date: 2007-12-13 05:39:17 +0200 (Thu, 13 Dec 2007) $'

__docformat__ = "restructuredtext en"

__all__ = ['MAError', 'MaskType', 'MaskedArray',
           'bool_', 'complex_', 'float_', 'int_', 'object_',
           'abs', 'absolute', 'add', 'all', 'allclose', 'allequal', 'alltrue',
               'amax', 'amin', 'anom', 'anomalies', 'any', 'arange',
               'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2',
               'arctanh', 'argmax', 'argmin', 'argsort', 'around',
               'array', 'asarray','asanyarray',
           'bitwise_and', 'bitwise_or', 'bitwise_xor',
           'ceil', 'choose', 'compressed', 'concatenate', 'conjugate',
               'cos', 'cosh', 'count',
           'default_fill_value', 'diagonal', 'divide', 'dump', 'dumps',
           'empty', 'empty_like', 'equal', 'exp',
           'fabs', 'fmod', 'filled', 'floor', 'floor_divide','fix_invalid',
           'getmask', 'getmaskarray', 'greater', 'greater_equal', 'hypot',
           'ids', 'inner', 'innerproduct',
               'isMA', 'isMaskedArray', 'is_mask', 'is_masked', 'isarray',
           'left_shift', 'less', 'less_equal', 'load', 'loads', 'log', 'log10',
               'logical_and', 'logical_not', 'logical_or', 'logical_xor',
           'make_mask', 'make_mask_none', 'mask_or', 'masked',
               'masked_array', 'masked_equal', 'masked_greater',
               'masked_greater_equal', 'masked_inside', 'masked_less',
               'masked_less_equal', 'masked_not_equal', 'masked_object',
               'masked_outside', 'masked_print_option', 'masked_singleton',
               'masked_values', 'masked_where', 'max', 'maximum', 'mean', 'min',
               'minimum', 'multiply',
           'negative', 'nomask', 'nonzero', 'not_equal',
           'ones', 'outer', 'outerproduct',
           'power', 'product', 'ptp', 'put', 'putmask',
           'rank', 'ravel', 'remainder', 'repeat', 'reshape', 'resize',
               'right_shift', 'round_',
           'shape', 'sin', 'sinh', 'size', 'sometrue', 'sort', 'sqrt', 'std',
               'subtract', 'sum', 'swapaxes',
           'take', 'tan', 'tanh', 'transpose', 'true_divide',
           'var', 'where',
           'zeros']

