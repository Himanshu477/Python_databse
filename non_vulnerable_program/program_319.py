import sys
import threading
import Queue
import traceback
import types
import inspect
import time
import atexit

class ParallelExec(threading.Thread):
    """ Create a thread of parallel execution.
    """
    def __init__(self):
        threading.Thread.__init__(self)
        self.__queue = Queue.Queue(0)
        self.__frame = sys._getframe(1)
        self.setDaemon(1)
        self.start()

    def __call__(self,code,frame=None,wait=0):
        """ Execute code in parallel thread inside given frame (default
        frame is where this instance was created).
        If wait is True then __call__ returns after code is executed,
        otherwise code execution happens in background.
        """
        if wait:
            wait_for_code = threading.Event()
        else:
            wait_for_code = None
        self.__queue.put((code,frame,wait_for_code))
        if wait:
            wait_for_code.wait()

    def shutdown(self):
        """ Shutdown parallel thread."""
        self.__queue.put((None,None))

    def run(self):
        """ Called by threading.Thread."""
        while 1:
            code, frame, wait_for_code = self.__queue.get()
            if code is None:
                break
            if frame is None:
                frame = self.__frame
            try:
                exec (code, frame.f_globals,frame.f_locals)
            except Exception:
                traceback.print_exc()
            if wait_for_code is not None:
                wait_for_code.set()

def migrate(obj, caller):
    """ Return obj wrapper that facilitates accessing object
    from another thread."""
    if inspect.isroutine(obj):
        return MigratedRoutine(obj, caller)
    raise NotImplementedError,`type(obj)`

class Attrs:
    def __init__(self,**kws):
        for k,v in kws.items():
            setattr(self,k,v)

class MigratedRoutine:
    """ Wrapper for calling routines from another thread.

    func   - function or built-in or method
    caller('<command>',<frame>) - executes command in another thread
    """
    def __init__(self, func, caller):
        self.__attrs = Attrs(func=func, caller=caller, finished=threading.Event())
        for n,v in inspect.getmembers(func):
            if n in ['__dict__','__class__','__call__','__attrs']:
                continue
            setattr(self,n,v)

    def __call__(self, *args, **kws):
        attrs = self.__attrs
        frame = sys._getframe(0)
        attrs.finished.clear()
        attrs.caller('attrs.result = attrs.func(*args, **kws)',frame)
        attrs.caller('attrs.finished.set()',frame)
        attrs.finished.wait()
        result = attrs.result
        attrs.result = None
        return result


""" Basic functions used by several sub-packages and useful to have in the
main name-space

Type handling
==============
iscomplexobj     --  Test for complex object, scalar result
isrealobj        --  Test for real object, scalar result
iscomplex        --  Test for complex elements, array result
isreal           --  Test for real elements, array result
imag             --  Imaginary part
real             --  Real part
real_if_close    --  Turns complex number with tiny imaginary part to real
isneginf         --  Tests for negative infinity ---|
isposinf         --  Tests for positive infinity    |
isnan            --  Tests for nans                 |----  array results
isinf            --  Tests for infinity             |
isfinite         --  Tests for finite numbers    ---| 
isscalar         --  True if argument is a scalar
nan_to_num       --  Replaces NaN's with 0 and infinities with large numbers
typename         --  Return english name for given typecode character
cast             --  Dictionary of functions to force cast to each type
common_type      --  Determine the 'minimum common type code' for a group
                       of arrays

Index tricks
==================
mgrid            --  Method which allows easy construction of N-d 'mesh-grids'
r_               --  Append and construct arrays -- turns slice objects into
                       ranges and concatenates them, for 2d arrays appends
                       rows.
c_               --  Append and construct arrays -- for 2d arrays appends
                       columns.

index_exp        --  Konrad Hinsen's index_expression class instance which
                     can be useful for building complicated slicing syntax.

Useful functions
==================
select           --  Extension of where to multiple conditions and choices
extract          --  Extract 1d array from flattened array according to mask
insert           --  Insert 1d array of values into Nd array according to mask
linspace         --  Evenly spaced samples in linear space
logspace         --  Evenly spaced samples in logarithmic space
fix              --  Round x to nearest integer towards zero
mod              --  Modulo mod(x,y) = x % y except keeps sign of y
amax             --  Array maximum along axis
amin             --  Array minimum along axis
ptp              --  Array max-min along axis
cumsum           --  Cumulative sum along axis
prod             --  Product of elements along axis
cumprod          --  Cumluative product along axis
diff             --  Discrete differences along axis
angle            --  Returns angle of complex argument
unwrap           --  Unwrap phase along given axis (1-d algorithm)
sort_complex     --  Sort a complex-array (based on real, then imaginary)
trim_zeros       --  trim the leading and trailing zeros from 1D array.

vectorize        -- a class that wraps a Python function taking scalar
                         arguments into a generalized function which
                         can handle arrays of arguments using the broadcast
                         rules of Numeric Python.

Shape manipulation
===================
squeeze          --  Return a with length-one dimensions removed.
atleast_1d       --  Force arrays to be > 1D
atleast_2d       --  Force arrays to be > 2D
atleast_3d       --  Force arrays to be > 3D
vstack           --  Stack arrays vertically (row on row)
hstack           --  Stack arrays horizontally (column on column)
column_stack     --  Stack 1D arrays as columns into 2D array
dstack           --  Stack arrays depthwise (along third dimension)
split            --  Divide array into a list of sub-arrays
hsplit           --  Split into columns
vsplit           --  Split into rows
dsplit           --  Split along third dimension

Matrix (2d array) manipluations
===============================
fliplr           --  2D array with columns flipped
flipud           --  2D array with rows flipped
rot90            --  Rotate a 2D array a multiple of 90 degrees
eye              --  Return a 2D array with ones down a given diagonal
diag             --  Construct a 2D array from a vector, or return a given
                       diagonal from a 2D array.                       
mat              --  Construct a Matrix

Polynomials
============
poly1d           --  A one-dimensional polynomial class

poly             --  Return polynomial coefficients from roots
roots            --  Find roots of polynomial given coefficients
polyint          --  Integrate polynomial
polyder          --  Differentiate polynomial
polyadd          --  Add polynomials
polysub          --  Substract polynomials
polymul          --  Multiply polynomials
polydiv          --  Divide polynomials
polyval          --  Evaluate polynomial at given argument

General functions
=================
vectorize -- Generalized Function class

Import tricks
=============
ppimport         --  Postpone module import until trying to use it
ppimport_attr    --  Postpone module import until trying to use its
                      attribute

Machine arithmetics
===================
machar_single    --  MachAr instance storing the parameters of system
                     single precision floating point arithmetics
machar_double    --  MachAr instance storing the parameters of system
                     double precision floating point arithmetics

Threading tricks
================
ParallelExec     --  Execute commands in parallel thread.
"""

standalone = 1


standalone = 1


standalone = 1


"""
C/C++ integration
=================

        1. inline() -- a function for including C/C++ code within Python
        2. blitz()  -- a function for compiling Numeric expressions to C++
        3. ext_tools-- a module that helps construct C/C++ extension modules.
        4. accelerate -- a module that inline accelerates Python functions
"""
postpone_import = 1
standalone = 1


#!/usr/bin/env python
#
# exec_command
#
# Implements exec_command function that is (almost) equivalent to
# commands.getstatusoutput function but on NT, DOS systems the
# returned status is actually correct (though, the returned status
# values may be different by a factor). In addition, exec_command
# takes keyword arguments for (re-)defining environment variables.
#
# Provides functions:
#   exec_command  --- execute command in a specified directory and
#                     in the modified environment.
#   splitcmdline  --- inverse of ' '.join(argv)
#   find_executable --- locate a command using info from environment
#                     variable PATH. Equivalent to posix `which`
#                     command.
#
# Author: Pearu Peterson <pearu@cens.ioc.ee>
# Created: 11 January 2003
#
# Requires: Python 2.x
#
# Succesfully tested on:
#   os.name | sys.platform | comments
#   --------+--------------+----------
#   posix   | linux2       | Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3
#                            PyCrust 0.9.3, Idle 1.0.2
#   posix   | linux2       | Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2
#   posix   | sunos5       | SunOS 5.9, Python 2.2, 2.3.2
#   posix   | darwin       | Darwin 7.2.0, Python 2.3
#   nt      | win32        | Windows Me
#                            Python 2.3(EE), Idle 1.0, PyCrust 0.7.2
#                            Python 2.1.1 Idle 0.8
#   nt      | win32        | Windows 98, Python 2.1.1. Idle 0.8
#   nt      | win32        | Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests
#                            fail i.e. redefining environment variables may
#                            not work.
#

__all__ = ['exec_command','find_executable']

