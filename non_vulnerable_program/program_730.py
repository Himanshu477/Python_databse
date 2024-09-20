import f2py2e
f2py2e.main()
'''%(os.path.basename(sys.executable))

f2py_exe = 'f2py'+os.path.basename(sys.executable)[6:]
if f2py_exe[-4:]=='.exe':
    f2py_exe = f2py_exe[:-4] + '.py'
if 'bdist_wininst' in sys.argv and f2py_exe[-3:] != '.py':
    f2py_exe = f2py_exe + '.py'

if not os.path.exists(f2py_exe):
    f = open(f2py_exe,'w')
    f.write(f2py_py())
    f.close()

if __name__ == "__main__":

    print 'F2PY Version',version

    config = {}
    if sys.version[:3]>='2.3':
        config['download_url'] = "http://cens.ioc.ee/projects/f2py2e/2.x"\
                                 "/F2PY-2-latest.tar.gz"
        config['classifiers'] = [
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: C',
            'Programming Language :: Fortran',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Software Development :: Code Generators',
            ]
    setup(name="F2PY",
          version=version,
          description       = "F2PY - Fortran to Python Interface Generaton",
          author            = "Pearu Peterson",
          author_email      = "pearu@cens.ioc.ee",
          maintainer        = "Pearu Peterson",
          maintainer_email  = "pearu@cens.ioc.ee",
          license           = "LGPL",
          platforms         = "Unix, Windows (mingw|cygwin), Mac OSX",
          long_description  = """\
The Fortran to Python Interface Generator, or F2PY for short, is a
command line tool (f2py) for generating Python C/API modules for
wrapping Fortran 77/90/95 subroutines, accessing common blocks from
Python, and calling Python functions from Fortran (call-backs).
Interfacing subroutines/data from Fortran 90/95 modules is supported.""",
          url               = "http://cens.ioc.ee/projects/f2py2e/",
          cmdclass          = {'install_data': my_install_data},
          scripts           = [f2py_exe],
          packages          = ['f2py2e'],
          package_dir       = {'f2py2e':'.'},
          data_files        = [('f2py2e/src',
                                ['src/fortranobject.c','src/fortranobject.h'])],
          keywords          = ['Fortran','f2py'],
          **config
          )


#!/usr/bin/env python
"""
Usage:
   Run runme.py instead.

Copyright 1999 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2002/01/02 22:43:51 $
Pearu Peterson
"""

__version__ = "$Revision: 1.10 $[10:-1]"

ffname='iotestrout.f'
hfname='iotest.pyf'
pyname='runiotest.py'
