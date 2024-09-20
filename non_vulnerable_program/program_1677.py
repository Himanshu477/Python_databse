    fromargs(sys.argv)


"""
This module converts code written for numpy.oldnumeric to work
with numpy

Makes the following changes:
 * Converts typecharacters
 * Changes import statements
 * Change typecode= to dtype=
 * Eliminates savespace=xxx
 * replaces matrixmultiply with dot
 * converts functions that don't give axis= keyword that have changed
 * converts functions that don't give typecode= keyword that have changed
 * converts use of capitalized type-names

 * converts old function names in linalg.old, random.old, dft.old

"""
__all__ = ['fromfile', 'fromstr']

