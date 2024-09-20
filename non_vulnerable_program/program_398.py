import distutils.cygwinccompiler
from distutils.version import StrictVersion
from scipy_distutils.ccompiler import gen_preprocess_options, gen_lib_options
from distutils.errors import DistutilsExecError, CompileError, UnknownFileError

