    from utils import str2stmt, get_char_bit

    stmt = str2stmt("""
    module rat
      integer :: i
      type rational
        integer n
        integer*8 d
      end type rational
    end module rat
    subroutine foo(a)
    use rat
    type(rational) a
    end
    """)
    #stmt = stmt.content[-1].content[1]
    #print stmt
    #wrapgen = TypeWrapper(stmt)
    #print wrapgen.fortran_code()
    #print wrapgen.c_code()

    foo_code = """! -*- f90 -*-
      module rat
        type rational
          integer d,n
        end type rational
      end module rat
      subroutine foo(a,b)
        use rat
        integer a
        character*5 b
        type(rational) c
        print*,'a=',a,b,c
      end
"""

    wm = PythonWrapperModule('foo')
    wm.add(str2stmt(foo_code))
    #wm.add_fortran_code(foo_code)
    #wm.add_subroutine(str2stmt(foo_code))
    #print wm.c_code()

    c_code = wm.c_code()
    f_code = wm.fortran_code()

    f = open('foomodule.c','w')
    f.write(c_code)
    f.close()
    f = open('foo.f','w')
    f.write(foo_code)
    f.close()
    f = open('foo_wrap.f','w')
    f.write(f_code)
    f.close()
    f = open('foo_setup.py','w')
    f.write('''\
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('foopack',parent_package,top_path)
    config.add_library('foolib',
                       sources = ['foo.f','foo_wrap.f'])
    config.add_extension('foo',
                         sources=['foomodule.c'],
                         libraries = ['foolib'],
                         )
    return config
if __name__ == '__main__':
