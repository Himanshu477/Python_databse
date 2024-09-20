import os
import sys
import imp
from glob import glob

class PackageImport:
    """ Import packages from the current directory that implement
    info.py. See numpy/doc/DISTUTILS.txt for more info.
    """

