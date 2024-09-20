import os
import sys
from pprint import pformat

__all__ = ['interactive_sys_argv']

def show_information(*args):
    print 'Python',sys.version
    for a in ['platform','prefix','byteorder','path']:
        print 'sys.%s = %s' % (a,pformat(getattr(sys,a)))
    for a in ['name']:
        print 'os.%s = %s' % (a,pformat(getattr(os,a)))
    if hasattr(os,'uname'):
        print 'system,node,release,version,machine = ',os.uname()    

def show_environ(*args):
    for k,i in os.environ.items():
        print '  %s = %s' % (k, i)

def show_fortran_compilers(*args):
    from fcompiler import show_fcompilers
    show_fcompilers()

def show_compilers(*args):
    from distutils.ccompiler import show_compilers
    show_compilers()

def show_tasks(argv,ccompiler,fcompiler):
    print """\

Tasks: 
  i       - Show python/platform/machine information
  ie      - Show environment information
  c       - Show C compilers information
  c<name> - Set C compiler (current:%s)
  f       - Show Fortran compilers information
  f<name> - Set Fortran compiler (current:%s)
  e       - Edit proposed sys.argv[1:].

Task aliases:
  0         - Configure
  1         - Build
  2         - Install
  2<prefix> - Install with prefix.
  3         - Inplace build
  4         - Source distribution
  5         - Binary distribution

Proposed sys.argv = %s
    """ % (ccompiler, fcompiler, argv)


