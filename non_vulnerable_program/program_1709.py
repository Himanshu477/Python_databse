    fromargs(sys.argv)


"""
This module converts code written for numpy.oldnumeric to work
with numpy

Makes the following changes:
 * Converts typecharacters '1swu' to 'bhHI' respectively
   when used as typecodes
 * Changes import statements
 * Change typecode= to dtype=
 * Eliminates savespace=xxx keyword arguments
 *  Removes it when keyword is not given as well
 * replaces matrixmultiply with dot
 * converts functions that don't give axis= keyword that have changed
 * converts functions that don't give typecode= keyword that have changed
 * converts use of capitalized type-names
 * converts old function names in oldnumeric.linear_algebra,
   oldnumeric.random_array, and oldnumeric.fft

"""
#__all__ = ['fromfile', 'fromstr']
__all__ = []

