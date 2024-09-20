import Numeric
def foo(a):
    a = Numeric.array(a)
    m,n = a.shape
    for i in range(m):
        for j in range(n):
            a[i,j] = a[i,j] + 10*(i+1) + (j+1)
    return a
#eof


!    -*- f90 -*-
python module __user__routines 
    interface
        function fun(i) result (r)
            integer :: i
            real*8 :: r
        end function fun
    end interface
end python module __user__routines

python module callback2
    interface
        subroutine foo(f,r)
            use __user__routines, f=>fun
            external f
            real*8 intent(out) :: r
        end subroutine foo
    end interface 
end python module callback2


!    -*- f90 -*-
python module fib2 ! in 
    interface  ! in :fib2
        subroutine fib(a,n) ! in :fib2:fib1.f
            real*8 dimension(n) :: a
            integer optional,check(len(a)>=n),depend(a) :: n=len(a)
        end subroutine fib
    end interface 
end python module fib2

! This file was auto-generated with f2py (version:2.28.198-1366).
! See http://cens.ioc.ee/projects/f2py2e/


!    -*- f90 -*-
python module fib2 
    interface
        subroutine fib(a,n)
            real*8 dimension(n),intent(out),depend(n) :: a
            integer intent(in) :: n
        end subroutine fib
    end interface 
end python module fib2


#!/usr/bin/env python
# File: setup_example.py

