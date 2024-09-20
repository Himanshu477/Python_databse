import f2py2e
from Numeric import array

def build(f2py_opts):
    try:
        import f77_ext_return_integer
    except ImportError:
        assert not f2py2e.compile('''\
       function t0(value)
         integer value
         integer t0
         t0 = value
       end
       function t1(value)
         integer*1 value
         integer*1 t1
         t1 = value
       end
       function t2(value)
         integer*2 value
         integer*2 t2
         t2 = value
       end
       function t4(value)
         integer*4 value
         integer*4 t4
         t4 = value
       end
       function t8(value)
         integer*8 value
         integer*8 t8
         t8 = value
       end

       subroutine s0(t0,value)
         integer value
         integer t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s1(t1,value)
         integer*1 value
         integer*1 t1
cf2py    intent(out) t1
         t1 = value
       end
       subroutine s2(t2,value)
         integer*2 value
         integer*2 t2
cf2py    intent(out) t2
         t2 = value
       end
       subroutine s4(t4,value)
         integer*4 value
         integer*4 t4
cf2py    intent(out) t4
         t4 = value
       end
       subroutine s8(t8,value)
         integer*8 value
         integer*8 t8
cf2py    intent(out) t8
         t8 = value
       end

''','f77_ext_return_integer',f2py_opts,source_fn='f77_ret_int.f')

