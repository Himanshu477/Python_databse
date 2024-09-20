import os
import sys
import re
from types import StringType,NoneType
from distutils.sysconfig import get_config_var
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError,DistutilsArgError,\
     DistutilsExecError,CompileError,LinkError,DistutilsPlatformError
