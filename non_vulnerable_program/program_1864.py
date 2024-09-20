from lib.main import build_extension, compile
restore_path()

fortran_code = '''
module test_module_scalar_ext

  contains
    subroutine foo(a)
    integer a
!f2py intent(in,out) a
    a = a + 1
    end subroutine foo
    function foo2(a)
    integer a
    integer foo2
    foo2 = a + 2
    end function foo2
end module test_module_scalar_ext
'''

m, = compile(fortran_code, modulenames = ['test_module_scalar_ext'])

