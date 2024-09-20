from lib.main import build_extension, compile
restore_path()

fortran_code = '''
subroutine foo(a)
  type myt
    integer flag
  end type myt
  type(myt) a
!f2py intent(in,out) a
  a % flag = a % flag + 1
end
function foo2(a)
  type myt
    integer flag
  end type myt
  type(myt) a
  type(myt) foo2
  foo2 % flag = a % flag + 2
end
'''

m, = compile(fortran_code, 'test_derived_scalar_ext')

