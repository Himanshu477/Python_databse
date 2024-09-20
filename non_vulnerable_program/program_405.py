    from fcompiler import new_fcompiler
    compiler = new_fcompiler(compiler='ibm')
    compiler.customize()
    print compiler.get_version()


"""
This module allows one to use SWIG2 (SWIG version >= 1.3) wrapped
objects from Weave.  SWIG-1.3 wraps objects differently from SWIG-1.1.

The code here is based on wx_spec.py.  However, this module is more
like a template for any SWIG2 wrapped converter.  To wrap specific
code that uses SWIG the user simply needs to override the defaults in
the swig2_converter class.

By default this code assumes that the user will not link with the SWIG
runtime library (libswigpy under *nix).  In this case no type checking
will be performed by SWIG.

To turn on type checking and link with the SWIG runtime library, there
are two approaches.

 1. If you are writing a customized converter based on this code then
    in the overloaded init_info, just call swig2_converter.init_info
    with runtime=1 and add the swig runtime library to the libraries
    loaded.

 2. If you are using the default swig2_converter you need to add two
    keyword arguments to your weave.inline call:

     a. Add a define_macros=[('SWIG_NOINCLUDE', None)]

     b. Add the swigpy library to the libraries like so:
        libraries=['swigpy']

Prabhu Ramachandran <prabhu@aero.iitm.ernet.in>
"""

