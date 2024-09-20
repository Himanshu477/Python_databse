import os
import sys
from numpy.testing import *

def build(fortran_code, rebuild=True):
    modulename = os.path.splitext(os.path.basename(__file__))[0] + '_ext'
    try:
        exec ('import %s as m' % (modulename))
        if rebuild and os.stat(m.__file__)[8] < os.stat(__file__)[8]:
            del sys.modules[m.__name__] # soft unload extension module
            os.remove(m.__file__)
            raise ImportError,'%s is newer than %s' % (__file__, m.__file__)
    except ImportError,msg:
        assert str(msg)==('No module named %s' % (modulename)),str(msg)
        print msg, ', recompiling %s.' % (modulename)
        import tempfile
        fname = tempfile.mktemp() + '.f90'
        f = open(fname,'w')
        f.write(fortran_code)
        f.close()
        sys_argv = []
        sys_argv.extend(['--build-dir','tmp'])
        #sys_argv.extend(['-DF2PY_DEBUG_PYOBJ_TOFROM'])
        from main import build_extension
        sys_argv.extend(['-m',modulename, fname])
        build_extension(sys_argv)
        os.remove(fname)
        status = os.system(' '.join([sys.executable] + sys.argv))
        sys.exit(status)
    return m

fortran_code = '''
module test_module_scalar_ext

  contains
    subroutine foo(a)
    integer a
!f2py intent(in,out) a
    a = a + 1
    end subroutine foo
    function foo2(a)
    integer a
    integer foo2
    foo2 = a + 2
    end function foo2
end module test_module_scalar_ext
'''

# tester note: set rebuild=True when changing fortan_code and for SVN
m = build(fortran_code, rebuild=True)

