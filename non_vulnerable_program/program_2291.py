import numpy as np
from numpy.testing import *

class TestBuiltin(TestCase):
    def test_run(self):
        """Only test hash runs at all."""
        for t in [np.int, np.float, np.complex, np.int32, np.str, np.object,
                np.unicode]:
            dt = np.dtype(t)
            hash(dt)

class TestRecord(TestCase):
    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        a = np.dtype([('yo', np.int)])
        b = np.dtype([('yo', np.int)])
        self.failUnless(hash(a) == hash(b), 
                "two equivalent types do not hash to the same value !")

    def test_different_names(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype([('yo', np.int)])
        b = np.dtype([('ye', np.int)])
        self.failUnless(hash(a) != hash(b),
                "%s and %s hash the same !" % (a, b))

    def test_different_titles(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
            'titles': ['RRed pixel', 'Blue pixel']})
        self.failUnless(hash(a) != hash(b),
                "%s and %s hash the same !" % (a, b))

class TestSubarray(TestCase):
    def test_equivalent_record(self):
        """Test whether equivalent subarray dtypes hash the same."""
        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (2, 3)))
        self.failUnless(hash(a) == hash(b), 
                "two equivalent types do not hash to the same value !")

    def test_nonequivalent_record(self):
        """Test whether different subarray dtypes hash differently."""
        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (3, 2)))
        self.failUnless(hash(a) != hash(b), 
                "%s and %s hash the same !" % (a, b))

        a = np.dtype((np.int, (2, 3)))
        b = np.dtype((np.int, (2, 2)))
        self.failUnless(hash(a) != hash(b), 
                "%s and %s hash the same !" % (a, b))

        a = np.dtype((np.int, (1, 2, 3)))
        b = np.dtype((np.int, (1, 2)))
        self.failUnless(hash(a) != hash(b), 
                "%s and %s hash the same !" % (a, b))

class TestMonsterType(TestCase):
    """Test deeply nested subtypes."""
    pass

if __name__ == "__main__":
    run_module_suite()


********************
Using Python as glue
********************

|    There is no conversation more boring than the one where everybody
|    agrees.
|    --- *Michel de Montaigne*

|    Duct tape is like the force. It has a light side, and a dark side, and
|    it holds the universe together.
|    --- *Carl Zwanzig*

Many people like to say that Python is a fantastic glue language.
Hopefully, this Chapter will convince you that this is true. The first
adopters of Python for science were typically people who used it to
glue together large applicaton codes running on super-computers. Not
only was it much nicer to code in Python than in a shell script or
Perl, in addition, the ability to easily extend Python made it
relatively easy to create new classes and types specifically adapted
to the problems being solved. From the interactions of these early
contributors, Numeric emerged as an array-like object that could be
used to pass data between these applications.

As Numeric has matured and developed into NumPy, people have been able
to write more code directly in NumPy. Often this code is fast-enough
for production use, but there are still times that there is a need to
access compiled code. Either to get that last bit of efficiency out of
the algorithm or to make it easier to access widely-available codes
written in C/C++ or Fortran.

This chapter will review many of the tools that are available for the
purpose of accessing code written in other compiled languages. There
are many resources available for learning to call other compiled
libraries from Python and the purpose of this Chapter is not to make
you an expert. The main goal is to make you aware of some of the
possibilities so that you will know what to "Google" in order to learn more.

The http://www.scipy.org website also contains a great deal of useful
information about many of these tools. For example, there is a nice
description of using several of the tools explained in this chapter at
http://www.scipy.org/PerformancePython. This link provides several
ways to solve the same problem showing how to use and connect with
compiled code to get the best performance. In the process you can get
a taste for several of the approaches that will be discussed in this
chapter.


Calling other compiled libraries from Python
============================================

While Python is a great language and a pleasure to code in, its
dynamic nature results in overhead that can cause some code ( *i.e.*
raw computations inside of for loops) to be up 10-100 times slower
than equivalent code written in a static compiled language. In
addition, it can cause memory usage to be larger than necessary as
temporary arrays are created and destroyed during computation. For
many types of computing needs the extra slow-down and memory
consumption can often not be spared (at least for time- or memory-
critical portions of your code). Therefore one of the most common
needs is to call out from Python code to a fast, machine-code routine
(e.g. compiled using C/C++ or Fortran). The fact that this is
relatively easy to do is a big reason why Python is such an excellent
high-level language for scientific and engineering programming.

Their are two basic approaches to calling compiled code: writing an
extension module that is then imported to Python using the import
command, or calling a shared-library subroutine directly from Python
using the ctypes module (included in the standard distribution with
Python 2.5). The first method is the most common (but with the
inclusion of ctypes into Python 2.5 this status may change).

.. warning::

    Calling C-code from Python can result in Python crashes if you are not
    careful. None of the approaches in this chapter are immune. You have
    to know something about the way data is handled by both NumPy and by
    the third-party library being used.


Hand-generated wrappers
=======================

Extension modules were discussed in Chapter `1
<#sec-writing-an-extension>`__ . The most basic way to interface with
compiled code is to write an extension module and construct a module
method that calls the compiled code. For improved readability, your
method should take advantage of the PyArg_ParseTuple call to convert
between Python objects and C data-types. For standard C data-types
there is probably already a built-in converter. For others you may
need to write your own converter and use the "O&" format string which
allows you to specify a function that will be used to perform the
conversion from the Python object to whatever C-structures are needed.

Once the conversions to the appropriate C-structures and C data-types
have been performed, the next step in the wrapper is to call the
underlying function. This is straightforward if the underlying
function is in C or C++. However, in order to call Fortran code you
must be familiar with how Fortran subroutines are called from C/C++
using your compiler and platform. This can vary somewhat platforms and
compilers (which is another reason f2py makes life much simpler for
interfacing Fortran code) but generally involves underscore mangling
of the name and the fact that all variables are passed by reference
(i.e. all arguments are pointers).

The advantage of the hand-generated wrapper is that you have complete
control over how the C-library gets used and called which can lead to
a lean and tight interface with minimal over-head. The disadvantage is
that you have to write, debug, and maintain C-code, although most of
it can be adapted using the time-honored technique of
"cutting-pasting-and-modifying" from other extension modules. Because,
the procedure of calling out to additional C-code is fairly
regimented, code-generation procedures have been developed to make
this process easier. One of these code- generation techniques is
distributed with NumPy and allows easy integration with Fortran and
(simple) C code. This package, f2py, will be covered briefly in the
next session.


f2py
====

F2py allows you to automatically construct an extension module that
interfaces to routines in Fortran 77/90/95 code. It has the ability to
parse Fortran 77/90/95 code and automatically generate Python
signatures for the subroutines it encounters, or you can guide how the
subroutine interfaces with Python by constructing an interface-
defintion-file (or modifying the f2py-produced one).

.. index::
   single: f2py

Creating source for a basic extension module
--------------------------------------------

Probably the easiest way to introduce f2py is to offer a simple
example. Here is one of the subroutines contained in a file named
:file:`add.f`:

.. code-block:: none

    C
          SUBROUTINE ZADD(A,B,C,N)
    C
          DOUBLE COMPLEX A(*)
          DOUBLE COMPLEX B(*)
          DOUBLE COMPLEX C(*)
          INTEGER N
          DO 20 J = 1, N
             C(J) = A(J)+B(J)
     20   CONTINUE
          END

This routine simply adds the elements in two contiguous arrays and
places the result in a third. The memory for all three arrays must be
provided by the calling routine. A very basic interface to this
routine can be automatically generated by f2py::

    f2py -m add add.f

You should be able to run this command assuming your search-path is
set-up properly. This command will produce an extension module named
addmodule.c in the current directory. This extension module can now be
compiled and used from Python just like any other extension module.


Creating a compiled extension module
------------------------------------

You can also get f2py to compile add.f and also compile its produced
extension module leaving only a shared-library extension file that can
be imported from Python::

    f2py -c -m add add.f

This command leaves a file named add.{ext} in the current directory
(where {ext} is the appropriate extension for a python extension
module on your platform --- so, pyd, *etc.* ). This module may then be
