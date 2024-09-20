    from core_version import version as __core_version__

    pkgload('testing','base','corefft','corelinalg','random',
            verbose=SCIPY_IMPORT_VERBOSE)


    test = ScipyTest('numpy').test
    __all__.append('test')

__numpy_doc__ = """

SciPy: A scientific computing package for Python
================================================

Available subpackages
---------------------
"""

if NO_SCIPY_IMPORT is not None:
    print >> sys.stderr, 'Skip importing numpy packages (NO_SCIPY_IMPORT=%s)' % (NO_SCIPY_IMPORT)
    show_numpy_config = None
elif show_core_config is None:
    show_numpy_config = None
else:
    try:
