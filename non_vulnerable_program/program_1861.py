from lib.main import build_extension, compile
restore_path()

fortran_code = '''
module test_module_module_ext2
  type rat
    integer n,d
  end type rat
  contains
    subroutine foo2()
      print*,"In foo2"
    end subroutine foo2
end module
module test_module_module_ext
  contains
    subroutine foo
      use test_module_module_ext2
      print*,"In foo"
      call foo2
    end subroutine foo
    subroutine bar(a)
      use test_module_module_ext2
      type(rat) a
      print*,"In bar,a=",a
    end subroutine bar
end module test_module_module_ext 
'''

m,m2 = compile(fortran_code, modulenames=['test_module_module_ext',
                                          'test_module_module_ext2',
                                          ])

