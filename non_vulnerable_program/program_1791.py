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
        assert str(msg).startswith('No module named'),str(msg)
        print msg, ', recompiling %s.' % (modulename)
        import tempfile
        fname = tempfile.mktemp() + '.f'
        f = open(fname,'w')
        f.write(fortran_code)
        f.close()
        sys_argv = ['--build-dir','tmp']
        #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
        from main import build_extension
        sys_argv.extend(['-m',modulename, fname])
        build_extension(sys_argv)
        os.remove(fname)
        os.system(' '.join([sys.executable] + sys.argv))
        sys.exit(0)
    return m

fortran_code = '''
      function fooint1(a)
      integer*1 a
      integer*1 fooint1
      fooint1 = a + 1
      end
      function fooint2(a)
      integer*2 a
      integer*2 fooint2
      fooint2 = a + 1
      end
      function fooint4(a)
      integer*4 a
      integer*4 fooint4
      fooint4 = a + 1
      end
      function fooint8(a)
      integer*8 a
      integer*8 fooint8
      fooint8 = a + 1
      end
      function foofloat4(a)
      real*4 a
      real*4 foofloat4
      foofloat4 = a + 1.0e0
      end
      function foofloat8(a)
      real*8 a
      real*8 foofloat8
      foofloat8 = a + 1.0d0
      end
      function foocomplex8(a)
      complex*8 a
      complex*8 foocomplex8
      foocomplex8 = a + 1.0e0
      end
      function foocomplex16(a)
      complex*16 a
      complex*16 foocomplex16
      foocomplex16 = a + 1.0d0
      end
      function foobool1(a)
      logical*1 a
      logical*1 foobool1
      foobool1 = .not. a
      end
      function foobool2(a)
      logical*2 a
      logical*2 foobool2
      foobool2 = .not. a
      end
      function foobool4(a)
      logical*4 a
      logical*4 foobool4
      foobool4 = .not. a
      end
      function foobool8(a)
      logical*8 a
      logical*8 foobool8
      foobool8 = .not. a
      end
      function foostring1(a)
      character*1 a
      character*1 foostring1
      foostring1 = "1"
      end
      function foostring5(a)
      character*5 a
      character*5 foostring5
      foostring5 = a
      foostring5(1:2) = "12"
      end
!      function foostringstar(a)
!      character*(*) a
!      character*(*) foostringstar
!      if (len(a).gt.0) then
!        foostringstar = a
!        foostringstar(1:1) = "1"
!      endif
!      end
'''

# tester note: set rebuild=True when changing fortan_code and for SVN
m = build(fortran_code, rebuild=True)

