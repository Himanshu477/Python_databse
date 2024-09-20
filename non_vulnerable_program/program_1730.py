from numpy import abs, absolute, add, arccos, arccosh, arcsin, arcsinh, \
     arctan, arctan2, arctanh, bitwise_and, bitwise_not, bitwise_or, \
     bitwise_xor, ceil, cos, cosh, cumproduct, cumsum, divide, equal, \
     exp, fabs, floor, floor_divide, fmod, greater, greater_equal, hypot, \
     ieeemask, isnan, less, less_equal, log, log10, logical_and, \
     logical_not, logical_or, logical_xor, lshift, maximum, minimum, \
     minus, multiply, negative, nonzero, not_equal, power, product, \
     remainder, rshift, sin, sinh, sqrt, subtract, sum, tan, tanh, \
     true_divide


""" This module contains a "session saver" which saves the state of a
NumPy session to a file.  At a later time, a different Python
process can be started and the saved session can be restored using
load().

The session saver relies on the Python pickle protocol to save and
restore objects.  Objects which are not themselves picklable (e.g.
modules) can sometimes be saved by "proxy",  particularly when they
are global constants of some kind.  If it's not known that proxying
will work,  a warning is issued at save time.  If a proxy fails to
reload properly (e.g. because it's not a global constant),  a warning
is issued at reload time and that name is bound to a _ProxyFailure
instance which tries to identify what should have been restored.

First, some unfortunate (probably unnecessary) concessions to doctest
to keep the test run free of warnings.

>>> del _PROXY_ALLOWED
>>> del copy
>>> del __builtins__

By default, save() stores every variable in the caller's namespace:

>>> import numpy as na
>>> a = na.arange(10)
>>> save()

Alternately,  save() can be passed a comma seperated string of variables:

>>> save("a,na")

Alternately,  save() can be passed a dictionary, typically one you already
have lying around somewhere rather than created inline as shown here:

>>> save(dictionary={"a":a,"na":na})

If both variables and a dictionary are specified, the variables to be
saved are taken from the dictionary.

>>> save(variables="a,na",dictionary={"a":a,"na":na})

Remove names from the session namespace

>>> del a, na

By default, load() restores every variable/object in the session file
to the caller's namespace.

>>> load()

load() can be passed a comma seperated string of variables to be
restored from the session file to the caller's namespace:

>>> load("a,na")

load() can also be passed a dictionary to *restore to*:

>>> d = {}
>>> load(dictionary=d)

load can be passed both a list variables of variables to restore and a
dictionary to restore to:

>>> load(variables="a,na", dictionary=d)

>>> na.all(a == na.arange(10))
1
>>> na.__name__
'numpy'

NOTE:  session saving is faked for modules using module proxy objects.
Saved modules are re-imported at load time but any "state" in the module
which is not restored by a simple import is lost.

"""

__all__ = ['load', 'save']

