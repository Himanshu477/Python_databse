from Numeric import *
import simple

print simple.__doc__
for function in dir(simple):
    print function


simple.test()

print simple


!%f90 -*- f90 -*-
module data_mod ! in :simple:data_mod.f
    real allocatable,dimension(*,*) :: data
    integer :: nj
    integer :: ni
end module data_mod
module simple ! in 
  use data_mod,only: n=>ni,x,m=>ni
  interface  ! in :simple
     subroutine test ! in :simple:test.f
     end subroutine test
  end interface
end module simple

! This file was auto-generated with f2py (version:1.218).
! See http://cens.ioc.ee/projects/f2py2e/


#!/usr/bin/env python
"""

Copyright 2001 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision: 1.1 $
$Date: 2001/12/13 16:56:10 $
Pearu Peterson
"""

__version__ = "$Id: test_array.py,v 1.1 2001/12/13 16:56:10 pearu Exp $"


