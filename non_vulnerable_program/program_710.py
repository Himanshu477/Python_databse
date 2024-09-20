import sys,os,string,fileinput,re,commands

try: fn=sys.argv[2]
except:
    try: fn='inputless_'+sys.argv[1]
    except: stdoutflag=1
try: fi=sys.argv[1]
except: fi=()
if not stdoutflag:
    sys.stdout=open(fn,'w')

nonverb=r'[\w\s\\&=\^\*\.\{\(\)\[\?\+\$/]*(?!\\verb.)'
input=re.compile(nonverb+r'\\(input|include)\*?\s*\{?.*}?')
comment=re.compile(r'[^%]*%')

for l in fileinput.input(fi):
    l=l[:-1]
    l1=''
    if comment.match(l):
        m=comment.match(l)
        l1=l[m.end()-1:]
        l=l[:m.end()-1]
    m=input.match(l)
    if m:
        l=string.strip(l)
        if l[-1]=='}': l=l[:-1]
        i=m.end()-2
        sys.stderr.write('>>>>>>')
        while i>-1 and (l[i] not in [' ','{']): i=i-1
        if i>-1:
            fn=l[i+1:]
            try: f=open(fn,'r'); flag=1; f.close()
            except:
                try: f=open(fn+'.tex','r'); flag=1;fn=fn+'.tex'; f.close()
                except: flag=0
            if flag==0:
                sys.stderr.write('Could not open a file: '+fn+'\n')
                print l+l1
                continue
            elif flag==1:
                sys.stderr.write(fn+'\n')
                print '%%%%% Begin of '+fn
                print commands.getoutput(sys.argv[0]+' < '+fn)
                print '%%%%% End of '+fn
        else:
            sys.stderr.write('Could not extract a file name from: '+l)
            print l+l1
    else:
        print l+l1
sys.stdout.close()





!%f90 -*- f90 -*-
python module foo
    interface
        subroutine exp1(l,u,n)
            real*8 dimension(2) :: l
            real*8 dimension(2) :: u
            integer*4 :: n
        end subroutine exp1
    end interface 
end python module foo
! This file was auto-generated with f2py 
! (version:2.298).
! See http://cens.ioc.ee/projects/f2py2e/


!%f90 -*- f90 -*-
python module foo
    interface
        subroutine exp1(l,u,n)
            real*8 dimension(2) :: l
            real*8 dimension(2) :: u
            intent(out) l,u
            integer*4 optional :: n = 1
        end subroutine exp1
    end interface 
end python module foo
! This file was auto-generated with f2py 
! (version:2.298) and modified by pearu.
! See http://cens.ioc.ee/projects/f2py2e/


!%f90 -*- f90 -*-

!  Example:
!    Using f2py for wrapping multi-dimensional Fortran and C arrays
!    [NEW APPROACH, use it with f2py higher than 2.8.x]
!  $Id: fun.pyf,v 1.3 2002/01/18 10:06:50 pearu Exp $

! Usage (with gcc compiler):
!   f2py -c fun.pyf foo.f bar.c

python module fun ! in 
    interface  ! in :fun

! >>> from Numeric import *
! >>> import fun
! >>> a=array([[1,2,3],[4,5,6]])

        subroutine foo(a,m,n) ! in :fun:foo.f
          integer dimension(m,n) :: a
          intent(in,out,copy) :: a
          integer optional,check(shape(a,0)==m),depend(a) :: m=shape(a,0)
          integer optional,check(shape(a,1)==n),depend(a) :: n=shape(a,1)
        end subroutine foo

! >>> print fun.foo.__doc__
! foo - Function signature:
!   a = foo(a,[m,n])
! Required arguments:
!   a : input rank-2 array('i') with bounds (m,n)
! Optional arguments:
!   m := shape(a,0) input int
!   n := shape(a,1) input int
! Return objects:
!   a : rank-2 array('i') with bounds (m,n)

! >>> print fun.foo(a)
!  F77:
!  m= 2, n= 3
!  Row  1:
!  a(i= 1,j= 1) =  1
!  a(i= 1,j= 2) =  2
!  a(i= 1,j= 3) =  3
!  Row  2:
!  a(i= 2,j= 1) =  4
!  a(i= 2,j= 2) =  5
!  a(i= 2,j= 3) =  6
! [[77777     2     3]
!  [    4     5     6]]


        subroutine bar(a,m,n)
          intent(c)
          intent(c) bar
          integer dimension(m,n) :: a
          intent(in,out) :: a
          integer optional,check(shape(a,0)==m),depend(a) :: m=shape(a,0)
          integer optional,check(shape(a,1)==n),depend(a) :: n=shape(a,1)
          intent(in) m,n
        end subroutine bar

! >>> print fun.bar.__doc__
! bar - Function signature:
!   a = bar(a,[m,n])
! Required arguments:
!   a : input rank-2 array('i') with bounds (m,n)
! Optional arguments:
!   m := shape(a,0) input int
!   n := shape(a,1) input int
! Return objects:
!   a : rank-2 array('i') with bounds (m,n)

! >>> print fun.bar(a)
! C:m=2, n=3
! Row 1:
! a(i=0,j=0)=1
! a(i=0,j=1)=2
! a(i=0,j=2)=3
! Row 2:
! a(i=1,j=0)=4
! a(i=1,j=1)=5
! a(i=1,j=2)=6
! [[7777    2    3]
!  [   4    5    6]]

    end interface 
end python module fun

! This file was auto-generated with f2py (version:2.9.166).
! See http://cens.ioc.ee/projects/f2py2e/


!%f90 -*- f90 -*-

!  Example:
!    Using f2py for wrapping multi-dimensional Fortran and C arrays
!    [OLD APPROACH, do not use it with f2py higher than 2.8.x]
!  $Id: run.pyf,v 1.1 2002/01/14 15:49:46 pearu Exp $

! Usage (with gcc compiler):
!   f2py -c run.pyf foo.f bar.c -DNO_APPEND_FORTRAN

python module run ! in 
    interface  ! in :run

! >>> from Numeric import *
! >>> import run
! >>> a=array([[1,2,3],[4,5,6]],'i')

        subroutine foo(a,m,n)
          fortranname foo_
          integer dimension(m,n) :: a
          integer optional,check(shape(a,1)==m),depend(a) :: m=shape(a,1)
          integer optional,check(shape(a,0)==n),depend(a) :: n=shape(a,0)
        end subroutine foo

! >>> print run.foo.__doc__
! foo - Function signature:
!   foo(a,[m,n])
! Required arguments:
!   a : input rank-2 array('i') with bounds (n,m)
! Optional arguments:
!   m := shape(a,1) input int
!   n := shape(a,0) input int

! >>> run.foo(a)
!  F77:
!  m= 3, n= 2
!  Row  1:
!  a(i= 1,j= 1) =  1
!  a(i= 1,j= 2) =  4
!  Row  2:
!  a(i= 2,j= 1) =  2
!  a(i= 2,j= 2) =  5
!  Row  3:
!  a(i= 3,j= 1) =  3
!  a(i= 3,j= 2) =  6

! >>> run.foo(transpose(a))
!  F77:
!  m= 2, n= 3
!  Row  1:
!  a(i= 1,j= 1) =  1
!  a(i= 1,j= 2) =  2
!  a(i= 1,j= 3) =  3
!  Row  2:
!  a(i= 2,j= 1) =  4
!  a(i= 2,j= 2) =  5
!  a(i= 2,j= 3) =  6

        subroutine bar(a,m,n)
          intent(c)
          integer dimension(m,n) :: a
          integer optional,check(shape(a,0)==m),depend(a) :: m=shape(a,0)
          integer optional,check(shape(a,1)==n),depend(a) :: n=shape(a,1)
        end subroutine bar

! >>> print run.bar.__doc__
! bar - Function signature:
!   bar(a,[m,n])
! Required arguments:
!   a :  rank-2 array('i') with bounds (m,n)
! Optional arguments:
!   m := shape(a,0)  int
!   n := shape(a,1)  int

! >>> run.bar(a)
! C:m=2, n=3
! Row 1:
! a(i=0,j=0)=1
! a(i=0,j=1)=2
! a(i=0,j=2)=3
! Row 2:
! a(i=1,j=0)=4
! a(i=1,j=1)=5
! a(i=1,j=2)=6


    end interface 
end python module run

! This file was auto-generated with f2py (version:2.8.172).
! See http://cens.ioc.ee/projects/f2py2e/


subroutine foo(a,m,n)
integer m = size(a,1)
integer n = size(a,2)
real, intent(inout) :: a(m,n)
end subroutine foo


#File: pytest.py
