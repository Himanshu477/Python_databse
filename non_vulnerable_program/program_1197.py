from numpy.distutils.extension import Extension
from numpy.distutils.command import config
from numpy.distutils.command import build
from numpy.distutils.command import build_py
from numpy.distutils.command import config_compiler
from numpy.distutils.command import build_ext
from numpy.distutils.command import build_clib
from numpy.distutils.command import build_src
from numpy.distutils.command import build_scripts
from numpy.distutils.command import sdist
from numpy.distutils.command import install_data
from numpy.distutils.command import install_headers
from numpy.distutils.command import install
from numpy.distutils.command import bdist_rpm
from numpy.distutils.misc_util import get_data_files

numpy_cmdclass = {'build':            build.build,
                  'build_src':        build_src.build_src,
                  'build_scripts':    build_scripts.build_scripts,
                  'config_fc':        config_compiler.config_fc,
                  'config':           config.config,
                  'build_ext':        build_ext.build_ext,
                  'build_py':         build_py.build_py,
                  'build_clib':       build_clib.build_clib,
                  'sdist':            sdist.sdist,
                  'install_data':     install_data.install_data,
                  'install_headers':  install_headers.install_headers,
                  'install':          install.install,
                  'bdist_rpm':        bdist_rpm.bdist_rpm,
                  }
if have_setuptools:
