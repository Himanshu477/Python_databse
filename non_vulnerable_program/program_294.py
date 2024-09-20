import distutils
from distutils.core import Extension, setup
from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler
from distutils.ccompiler import new_compiler

