import f2py2e
import math
import sys
from Numeric import array

def build(f2py_opts):
    try:
        import f77_ext_callback
    except ImportError:
        assert not f2py2e.compile('''\
       subroutine t(fun,a)
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine func(a)
cf2py  intent(in,out) a
       integer a
       a = a + 11
       end

       subroutine func0(a)
cf2py  intent(out) a
       integer a
       a = 11
       end

       subroutine t2(a)
cf2py  intent(callback) fun
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

''','f77_ext_callback',f2py_opts,source_fn='f77_callback.f')

