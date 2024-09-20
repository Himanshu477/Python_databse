import f2py2e
from Numeric import array
try: True
except NameError:
    True = 1
    False = 0

def build(f2py_opts):
    try:
        import f77_ext_return_logical
    except ImportError:
        assert not f2py2e.compile('''\
       function t0(value)
         logical value
         logical t0
         t0 = value
       end
       function t1(value)
         logical*1 value
         logical*1 t1
         t1 = value
       end
       function t2(value)
         logical*2 value
         logical*2 t2
         t2 = value
       end
       function t4(value)
         logical*4 value
         logical*4 t4
         t4 = value
       end
c       function t8(value)
c         logical*8 value
c         logical*8 t8
c         t8 = value
c       end

       subroutine s0(t0,value)
         logical value
         logical t0
cf2py    intent(out) t0
         t0 = value
       end
       subroutine s1(t1,value)
         logical*1 value
         logical*1 t1
cf2py    intent(out) t1
         t1 = value
       end
       subroutine s2(t2,value)
         logical*2 value
         logical*2 t2
cf2py    intent(out) t2
         t2 = value
       end
       subroutine s4(t4,value)
         logical*4 value
         logical*4 t4
cf2py    intent(out) t4
         t4 = value
       end
c       subroutine s8(t8,value)
c         logical*8 value
c         logical*8 t8
cf2py    intent(out) t8
c         t8 = value
c       end
''','f77_ext_return_logical',f2py_opts)

    #from f77_ext_return_logical import t0,t1,t2,t4,t8,s0,s1,s2,s4,s8
    #test_functions = [t0,t1,t2,t4,t8,s0,s1,s2,s4,s8]
