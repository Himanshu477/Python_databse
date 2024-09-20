    from Cython.Distutils import build_ext
    has_cython = True
except ImportError:
    has_cython = False

# Define a cython-based extension module, using the generated sources if cython
# is not available.
if has_cython:
    pyx_sources = ['numpyx.pyx']
    cmdclass    = {'build_ext': build_ext}
else:
    # In production work, you can ship the auto-generated C source yourself to
    # your users.  In this case, we do NOT ship the .c file as part of numpy,
    # so you'll need to actually have cython installed at least the first
    # time.  Since this is really just an example to show you how to use
    # *Cython*, it makes more sense NOT to ship the C sources so you can edit
    # the pyx at will with less chances for source update conflicts when you
    # update numpy.
    pyx_sources = ['numpyx.c']
    cmdclass    = {}


# Declare the extension object
pyx_ext = Extension('numpyx',
                    pyx_sources,
                    include_dirs = [numpy.get_include()])

# Call the routine which does the real work
setup(name        = 'numpyx',
      description = 'Small example on using Cython to write a Numpy extension',
      ext_modules = [pyx_ext],
      cmdclass    = cmdclass,
      )


