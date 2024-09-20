from distutils.extension import Extension as old_Extension

class Extension(old_Extension):
    def __init__ (self, name, sources,
                  include_dirs=None,
                  define_macros=None,
                  undef_macros=None,
                  library_dirs=None,
                  libraries=None,
                  runtime_library_dirs=None,
                  extra_objects=None,
                  extra_compile_args=None,
                  extra_link_args=None,
                  export_symbols=None,
                  f2py_options=None
                 ):
        old_Extension.__init__(self,name, sources,
                               include_dirs,
                               define_macros,
                               undef_macros,
                               library_dirs,
                               libraries,
                               runtime_library_dirs,
                               extra_objects,
                               extra_compile_args,
                               extra_link_args,
                               export_symbols)
        self.f2py_options = f2py_options or []
        
# class Extension


"""distutils.command.run_f2py

Implements the Distutils 'run_f2py' command.
"""

# created 2002/01/09, Pearu Peterson 

__revision__ = "$Id$"

