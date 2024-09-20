import sys
import os
import re
import warnings

flatindex_re = re.compile('([.]flat(\s*?[[=]))')
int_re = re.compile('int\s*[(][^)]*[)]')
bool_re = re.compile('bool\s*[(][^)]*[)]')
float_re = re.compile('float\s*[(][^)]*[)]')
complex_re = re.compile('complex\s*[(][^)]*[)]')
unicode_re = re.compile('unicode\s*[(][^)]*[)]')

def replacetypechars(astr):
    astr = astr.replace("'s'","'h'")
    astr = astr.replace("'c'","'S1'")
    astr = astr.replace("'b'","'B'")
    astr = astr.replace("'1'","'b'")
    astr = astr.replace("'s'","'h'")
    astr = astr.replace("'w'","'H'")
    astr = astr.replace("'u'","'I'")
    return astr

# This function replaces
#  import x1, x2, x3
#
#with
#  import x1
#  import x2
#  import x3
