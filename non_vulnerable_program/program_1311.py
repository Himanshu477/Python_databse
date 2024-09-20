import numpy.f2py as f2py2e
from numpy.base import array

def build(f2py_opts):
    try:
        import f77_ext_return_real
    except ImportError:
        assert not f2py2e.compile('''\
       function t0(value)
         real value
         real t0
         t0 = value
       end
       function t4(value)
         real*4 value
         real*4 t4
         t4 = value
       end
       function t8(value)
         real*8 value
         real*8 t8
         t8 = value
       end
       function td(value)
         double precision value
         double precision td
         td = value
       end

       subroutine s0(t0,value)
         real value
         real t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s4(t4,value)
         real*4 value
         real*4 t4
cf2py    intent(out) t4
         t4 = value
       end
       subroutine s8(t8,value)
         real*8 value
         real*8 t8
cf2py    intent(out) t8
         t8 = value
       end
       subroutine sd(td,value)
         double precision value
         double precision td
cf2py    intent(out) td
         td = value
       end
''','f77_ext_return_real',f2py_opts,source_fn='f77_ret_real.f')

