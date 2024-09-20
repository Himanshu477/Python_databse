from optparse import OptionParser
import shutil
import os
import sys
import re
import subprocess

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
TEMP = os.path.normpath(os.path.join(BASE, '_py3k'))

SCRIPTS = [
    'generate_umath.py',
    'setup.py',
    'generate_numpy_api.py',
    'generate_ufunc_api.py',
]

def main():
    p = OptionParser(usage=__doc__.strip())
    p.add_option("--clean", "-c", action="store_true",
                 help="clean source directory")
    options, args = p.parse_args()

    if not args:
        p.error('no submodules given')
    else:
        dirs = ['numpy/%s' % x for x in map(os.path.basename, args)]

    # Prepare
    if not os.path.isdir(TEMP):
        os.makedirs(TEMP)

    # Set up dummy files (for building only submodules)
    dummy_files = {
        '__init__.py': 'from numpy.version import version as __version__',
        'version.py': 'version = "1.4.0.dev"'
    }

    for fn, content in dummy_files.items():
        fn = os.path.join(TEMP, 'numpy', fn)
        if not os.path.isfile(fn):
            try:
                os.makedirs(os.path.dirname(fn))
            except OSError:
                pass
            f = open(fn, 'wb+')
            f.write(content.encode('ascii'))
            f.close()

    # Environment
    pp = [os.path.abspath(TEMP)]
    def getenv():
        env = dict(os.environ)
        env.update({'PYTHONPATH': ':'.join(pp)})
        return env

    # Copy
    for d in dirs:
        src = os.path.join(BASE, d)
        dst = os.path.join(TEMP, d)

        # Run 2to3
        sync_2to3(dst=dst,
                  src=src,
                  patchfile=os.path.join(TEMP, os.path.basename(d) + '.patch'),
                  clean=options.clean)

        # Run setup.py, falling back to Pdb post-mortem on exceptions
        setup_py = os.path.join(dst, 'setup.py')
        if os.path.isfile(setup_py):
            code = """\
