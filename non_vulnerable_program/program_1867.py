from lib.main import build_extension, compile
restore_path()

fortran_code = '''\
! -*- f77 -*-
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

m, = compile(fortran_code, 'test_scalar_function_in_ext')

