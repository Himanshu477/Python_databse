from lib.main import build_extension, compile
restore_path()

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
      subroutine foobool1(a)
      logical*1 a
!f2py intent(in,out) a
      a = .not. a
      end
      subroutine foobool2(a)
      logical*2 a
!f2py intent(in,out) a
      a = .not. a
      end
      subroutine foobool4(a)
      logical*4 a
!f2py intent(in,out) a
      a = .not. a
      end
      subroutine foobool8(a)
      logical*8 a
!f2py intent(in,out) a
      a = .not. a
      end
      subroutine foostring1(a)
      character*1 a
!f2py intent(in,out) a
      a = "1"
      end
      subroutine foostring5(a)
      character*5 a
!f2py intent(in,out) a
      a(1:2) = "12"
      end
      subroutine foostringstar(a)
      character*(*) a
!f2py intent(in,out) a
      if (len(a).gt.0) then
        a(1:1) = "1"
      endif
      end
'''

m, = compile(fortran_code, 'test_scalar_in_out_ext', source_ext = '.f')

