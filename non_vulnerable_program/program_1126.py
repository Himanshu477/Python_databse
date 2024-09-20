import os, sys
NO_SCIPY_IMPORT = os.environ.get('NO_SCIPY_IMPORT',None)
SCIPY_IMPORT_VERBOSE = int(os.environ.get('SCIPY_IMPORT_VERBOSE','0'))

try:
