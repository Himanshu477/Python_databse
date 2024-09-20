    from numpy.distutils.core import setup
    setup(maintainer = "SciPy Developers",
          maintainer_email = "numpy-dev@numpy.org",
          description = "SciPy test module",
          url = "http://www.numpy.org",
          license = "SciPy License (BSD Style)",
          **configuration(top_path='').todict()
          )


#
# Machine arithmetics - determine the parameters of the
# floating-point arithmetic system
#
# Author: Pearu Peterson, September 2003
#

__all__ = ['MachAr']

