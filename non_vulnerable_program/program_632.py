from cPickle import load, loads
_cload = load
_file = file

def load(file):
    if isinstance(file, type("")):
        file = _file(file,"rb")
    return _cload(file)

# These are all essentially abbreviations
# These might wind up in a special abbreviations module

def ones(shape, dtype=intp, fortran=0):
    """ones(shape, dtype=intp) returns an array of the given
    dimensions which is initialized to all ones. 
    """
    a=zeros(shape, dtype, fortran)
    a+=1
    ### a[...]=1  -- slower
    return a
 
def identity(n,dtype=intp):
    """identity(n) returns the identity matrix of shape n x n.
    """
    a = array([1]+n*[0],dtype=dtype)
    b = empty((n,n),dtype=dtype)
    b.flat = a
    return b

def allclose (a, b, rtol=1.e-5, atol=1.e-8):
    """ allclose(a,b,rtol=1.e-5,atol=1.e-8)
        Returns true if all components of a and b are equal
        subject to given tolerances.
        The relative error rtol must be positive and << 1.0
        The absolute error atol comes into play for those elements
        of y that are very small or zero; it says how small x must be also.
    """
    x = array(a, copy=0)
    y = array(b, copy=0)
    d = less(absolute(x-y), atol + rtol * absolute(y))
    return alltrue(ravel(d))
            

# Now a method....
##def setflags(arr, write=None, swap=None, uic=None, align=None):
##    if not isinstance(arr, ndarray):
##        raise ValueError, "first argument must be an array"
##    sdict = {}
##    if write is not None:
##        sdict['WRITEABLE'] = not not write
##    if swap is not None:
##        sdict['NOTSWAPPED'] = not swap
##    if uic is not None:
##        if (uic):
##            raise ValueError, "Can only set UPDATEIFCOPY flag to False"
##        sdict['UPDATEIFCOPY'] = False
##    if align is not None:
##        sdict['ALIGNED'] = not not align
##    arr.flags = sdict

_errdict = {"ignore":ERR_IGNORE,
            "warn":ERR_WARN,
            "raise":ERR_RAISE,
            "call":ERR_CALL}

_errdict_rev = {}
for key in _errdict.keys():
    _errdict_rev[_errdict[key]] = key

def seterr(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
    maskvalue = (_errdict[divide] << SHIFT_DIVIDEBYZERO) + \
                (_errdict[over] << SHIFT_OVERFLOW ) + \
                (_errdict[under] << SHIFT_UNDERFLOW) + \
                (_errdict[invalid] << SHIFT_INVALID)
    frame = sys._getframe().f_back
    frame.f_locals[UFUNC_ERRMASK_NAME] = maskvalue
    return

def geterr():
    frame = sys._getframe().f_back
    try:
        maskvalue = frame.f_locals[UFUNC_ERRMASK_NAME]
    except KeyError:
        maskvalue = ERR_DEFAULT

    mask = 3
    res = {}
    val = (maskvalue >> SHIFT_DIVIDEBYZERO) & mask
    res['divide'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_OVERFLOW) & mask
    res['over'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_UNDERFLOW) & mask
    res['under'] = _errdict_rev[val]
    val = (maskvalue >> SHIFT_INVALID) & mask
    res['invalid'] = _errdict_rev[val]
    return res

    
    


#!/usr/bin/env python
"""
This file defines a set of system_info classes for getting
information about various resources (libraries, library directories,
include directories, etc.) in the system. Currently, the following
classes are available:

  atlas_info
  atlas_threads_info
  atlas_blas_info
  atlas_blas_threads_info
  lapack_atlas_info
  blas_info
  lapack_info
  blas_opt_info       # usage recommended
  lapack_opt_info     # usage recommended
  fftw_info,dfftw_info,sfftw_info
  fftw_threads_info,dfftw_threads_info,sfftw_threads_info
  djbfft_info
  x11_info
  lapack_src_info
  blas_src_info
  numpy_info
  numarray_info
  boost_python_info
  agg2_info
  wx_info
  gdk_pixbuf_xlib_2_info
  gdk_pixbuf_2_info
  gdk_x11_2_info
  gtkp_x11_2_info
  gtkp_2_info
  xft_info
  freetype2_info

Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw','lapack','blas',
  'lapack_src', 'blas_src', etc. For a complete list of allowed names,
  see the definition of get_info() function below.

  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (system_info could not find it).

  Several *_info classes specify an environment variable to specify
  the locations of software. When setting the corresponding environment
  variable to 'None' then the software will be ignored, even when it
  is available in system.

Global parameters:
  system_info.search_static_first - search static libraries (.a)
             in precedence to shared ones (.so, .sl) if enabled.
  system_info.verbosity - output the results to stdout if enabled.

The file 'site.cfg' in the same directory as this module is read
for configuration options. The format is that used by ConfigParser (i.e.,
Windows .INI style). The section DEFAULT has options that are the default
for each section. The available sections are fftw, atlas, and x11. Appropiate
defaults are used if nothing is specified.

The order of finding the locations of resources is the following:
 1. environment variable
 2. section in site.cfg
 3. DEFAULT section in site.cfg
Only the first complete match is returned.

Example:
----------
[DEFAULT]
library_dirs = /usr/lib:/usr/local/lib:/opt/lib
include_dirs = /usr/include:/usr/local/include:/opt/include
src_dirs = /usr/local/src:/opt/src
# search static libraries (.a) in preference to shared ones (.so)
search_static_first = 0

[fftw]
fftw_libs = rfftw, fftw
fftw_opt_libs = rfftw_threaded, fftw_threaded
# if the above aren't found, look for {s,d}fftw_libs and {s,d}fftw_opt_libs

[atlas]
library_dirs = /usr/lib/3dnow:/usr/lib/3dnow/atlas
# for overriding the names of the atlas libraries
atlas_libs = lapack, f77blas, cblas, atlas

[x11]
library_dirs = /usr/X11R6/lib
include_dirs = /usr/X11R6/include
----------

Authors:
  Pearu Peterson <pearu@cens.ioc.ee>, February 2002
  David M. Cooke <cookedm@physics.mcmaster.ca>, April 2002

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the 
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

__revision__ = '$Id: system_info.py,v 1.1 2005/04/09 19:29:35 pearu Exp $'
