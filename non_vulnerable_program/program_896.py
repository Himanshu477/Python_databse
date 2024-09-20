import re
import sys
import os
import string

__doc__ = """This module generates a DEF file from the symbols in
an MSVC-compiled DLL import library.  It correctly discriminates between
data and functions.  The data is collected from the output of the program
nm(1).

Usage:
    python lib2def.py [libname.lib] [output.def]
or
    python lib2def.py [libname.lib] > output.def

libname.lib defaults to python<py_ver>.lib and output.def defaults to stdout

Author: Robert Kern <kernr@mail.ncifcrf.gov>
Last Update: April 30, 1999
"""

__version__ = '0.1a'

