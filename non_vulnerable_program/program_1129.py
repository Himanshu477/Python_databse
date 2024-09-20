         import numpy
         numpy.pkgload('linalg','fft',...)

       or

         numpy.pkgload()

       to load all of them in one call.

       If a name which doesn't exist in numpy's namespace is
       given, an exception [[WHAT? ImportError, probably?]] is raised.
       [NotImplemented]

     Inputs:

       - the names (one or more strings) of all the numpy modules one wishes to
       load into the top-level namespace.

     Optional keyword inputs:

       - verbose - integer specifying verbosity level [default: 0].
       - force   - when True, force reloading loaded packages [default: False].
       - postpone - when True, don't load packages [default: False]

     If no input arguments are given, then all of numpy's subpackages are
