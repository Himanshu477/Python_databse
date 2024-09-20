import sys

def get_exception():
    return sys.exc_info()[1]


#!/usr/bin/env python3
# -*- python -*-
"""
%prog SUBMODULE...

Hack to pipe submodules of Numpy through 2to3 and build them in-place
one-by-one.

Example usage:

    python3 tools/py3tool.py testing distutils core

This will copy files to _py3k/numpy, add a dummy __init__.py and
version.py on the top level, and copy and 2to3 the files of the three
submodules.

When running py3tool again, only changed files are re-processed, which
makes the test-bugfix cycle faster.

"""
