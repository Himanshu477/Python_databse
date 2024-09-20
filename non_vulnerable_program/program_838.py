import sys
import f2py2e
from Numeric import array

def build(f2py_opts):
    try:
        import f90_ext_return_real
    except ImportError:
        assert not f2py2e.compile('''\
module f90_return_real
  contains
       function t0(value)
         real :: value
         real :: t0
         t0 = value
       end function t0
       function t4(value)
         real(kind=4) :: value
         real(kind=4) :: t4
         t4 = value
       end function t4
       function t8(value)
         real(kind=8) :: value
         real(kind=8) :: t8
         t8 = value
       end function t8
       function td(value)
         double precision :: value
         double precision :: td
         td = value
       end function td

       subroutine s0(t0,value)
         real :: value
         real :: t0
!f2py    intent(out) t0
         t0 = value
       end subroutine s0
       subroutine s4(t4,value)
         real(kind=4) :: value
         real(kind=4) :: t4
!f2py    intent(out) t4
         t4 = value
       end subroutine s4
       subroutine s8(t8,value)
         real(kind=8) :: value
         real(kind=8) :: t8
!f2py    intent(out) t8
         t8 = value
       end subroutine s8
       subroutine sd(td,value)
         double precision :: value
         double precision :: td
!f2py    intent(out) td
         td = value
       end subroutine sd
end module f90_return_real
''','f90_ext_return_real',f2py_opts,source_fn='f90_ret_real.f90')

