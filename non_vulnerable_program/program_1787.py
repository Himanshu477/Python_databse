import os
import sys
from numpy.testing import *

def build(fortran_code, rebuild=True):
    modulename = os.path.splitext(os.path.basename(__file__))[0]+'_ext'
    try:
        exec ('import %s as m' % (modulename))
        if rebuild and os.stat(m.__file__)[8] < os.stat(__file__)[8]:
            del sys.modules[m.__name__] # soft unload extension module
            os.remove(m.__file__)
            raise ImportError,'%s is newer than %s' % (__file__, m.__file__)
    except ImportError,msg:
        print msg, ', recompiling %s.' % (modulename)
        import tempfile
        fname = tempfile.mktemp() + '.f'
        f = open(fname,'w')
        f.write(fortran_code)
        f.close()
        sys_argv = ['--build-dir','foo']
        #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
        from main import build_extension
        sys_argv.extend(['-m',modulename, fname])
        build_extension(sys_argv)
        os.remove(fname)
        os.system(' '.join([sys.executable] + sys.argv))
        sys.exit(0)
    return m

fortran_code = '''
      subroutine fooint1(a)
      integer*1 a
!f2py intent(in,out) a
      a = a + 1
      end
      subroutine fooint2(a)
      integer*2 a
!f2py intent(in,out) a
      a = a + 1
      end
      subroutine fooint4(a)
      integer*4 a
!f2py intent(in,out) a
      a = a + 1
      end
      subroutine fooint8(a)
      integer*8 a
!f2py intent(in,out) a
      a = a + 1
      end
      subroutine foofloat4(a)
      real*4 a
!f2py intent(in,out) a
      a = a + 1.0e0
      end
      subroutine foofloat8(a)
      real*8 a
!f2py intent(in,out) a
      a = a + 1.0d0
      end
      subroutine foocomplex8(a)
      complex*8 a
!f2py intent(in,out) a
      a = a + 1.0e0
      end
      subroutine foocomplex16(a)
      complex*16 a
!f2py intent(in,out) a
      a = a + 1.0d0
      end
'''

# tester note: set rebuild=True when changing fortan_code and for SVN
m = build(fortran_code, rebuild=True)

