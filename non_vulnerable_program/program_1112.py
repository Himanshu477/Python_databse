    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())



# This module converts code written for Numeric to run with scipy.base

# Makes the following changes:
#  * Converts typecharacters
#  * Changes import statements (warns of use of from Numeric import *)
#  * Changes import statements (using numerix) ...
#  * Makes search and replace changes to:
#    - .typecode()
#    - .iscontiguous()
#    - .byteswapped()
#    - .itemsize()
#  * Converts .flat to .ravel() except for .flat = xxx or .flat[xxx]
#  * Change typecode= to dtype=
#  * Eliminates savespace=xxx
#  * Replace xxx.spacesaver() with True
#  * Convert xx.savespace(?) to pass + ## xx.savespace(?)
#  #### -- not * Convert a.shape = ? to a.reshape(?) 
#  * Prints warning for use of bool, int, float, copmlex, object, and unicode
#

__all__ = ['fromfile', 'fromstr']

