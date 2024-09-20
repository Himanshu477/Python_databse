import f2py2e
from Numeric import array

def build(f2py_opts):
    try:
        import c_ext_return_real
    except ImportError:
        assert not f2py2e.compile('''\
python module c_ext_return_real
usercode \'\'\'
float t4(float value) { return value; }
void s4(float *t4, float value) { *t4 = value; }
double t8(double value) { return value; }
void s8(double *t8, double value) { *t8 = value; }
\'\'\'
interface
  function t4(value)
    real*4 intent(c) :: t4,value
  end
  function t8(value)
    real*8 intent(c) :: t8,value
  end
  subroutine s4(t4,value)
    intent(c) s4
    real*4 intent(out) :: t4
    real*4 intent(c) :: value
  end
  subroutine s8(t8,value)
    intent(c) s8
    real*8 intent(out) :: t8
    real*8 intent(c) :: value
  end
end interface
end python module c_ext_return_real
''','c_ext_return_real',f2py_opts,source_fn='c_ret_real.pyf')

