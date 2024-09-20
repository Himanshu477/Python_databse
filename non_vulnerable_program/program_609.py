    from scipy.distutils.command.build_src import appendpath
    print "****%s****" % config.local_path
    config.add_include_dirs(appendpath('build/src',join(config.local_path,'Src')))

    deps = [join('src','umathmodule.c.src'),
            join('src','arrayobject.c'),
            join('src','arraymethods.c'),
            join('src','scalartypes.inc.src'),
            join('src','arraytypes.inc.src'),
            ]

    config.add_extension('multiarray',
                         sources = [join('src','multiarraymodule.c'),
                                    generate_config_h,
                                    generate_array_api,
                                    join('src','scalartypes.inc.src'),
                                    join('src','arraytypes.inc.src'),
                                    join(codegen_dir,'generate_array_api.py'),
                                    join('*.py')
                                    ],
                         depends = deps
                         )

    config.add_extension('umath',
                         sources = [generate_config_h,
                                    join('src','umathmodule.c.src'),
                                    generate_umath_c,
                                    generate_ufunc_api,
                                    join('src','scalartypes.inc.src'),
                                    join('src','arraytypes.inc.src'),
                                    ],
                         depends = [join('src','ufuncobject.c'),
                                    generate_umath_py,
                                    join(codegen_dir,'generate_ufunc_api.py')
                                    ]+deps,
                         )
    config.add_include_dirs(appendpath('build/src',config.local_path))
    config.add_extension('_compiled_base',
                         sources=['_compiled_base.c'],
                         )
                         

    return config

def testcode_mathlib():
    return """\
/* check whether libm is broken */
#include <math.h>
int main(int argc, char *argv[])
{
  return exp(-720.) > 1.0;  /* typically an IEEE denormal */
}
"""


def generate_testcode(target):
    testcode = [r'''
#include <Python.h>
#include <limits.h>
#include <stdio.h>

int main(int argc, char **argv)
{

        FILE *fp;

        fp = fopen("'''+target+'''","w");
        ''']

    c_size_test = r'''
#ifndef %(sz)s
          fprintf(fp,"#define %(sz)s %%d\n", sizeof(%(type)s));
#else
          fprintf(fp,"/* #define %(sz)s %%d */\n", %(sz)s);
#endif
'''
    for sz, t in [('SIZEOF_SHORT', 'short'),
                  ('SIZEOF_INT', 'int'),
                  ('SIZEOF_LONG', 'long'),
                  ('SIZEOF_FLOAT', 'float'),
                  ('SIZEOF_DOUBLE', 'double'),
                  ('SIZEOF_LONG_DOUBLE', 'long double'),
                  ('SIZEOF_PY_INTPTR_T', 'Py_intptr_t'),
                  ]:
        testcode.append(c_size_test % {'sz' : sz, 'type' : t})

    testcode.append('#ifdef PY_LONG_LONG')
    testcode.append(c_size_test % {'sz' : 'SIZEOF_LONG_LONG',
                                   'type' : 'PY_LONG_LONG'})
    testcode.append(c_size_test % {'sz' : 'SIZEOF_PY_LONG_LONG',
                                   'type' : 'PY_LONG_LONG'})
    

    testcode.append(r'''
#else
        fprintf(fp, "/* PY_LONG_LONG not defined */\n");
#endif
#ifndef CHAR_BIT
          {
             unsigned char var = 2;
             int i=0;
             while (var >= 2) {
                     var = var << 1;
                     i++;
             }
             fprintf(fp,"#define CHAR_BIT %d\n", i+1);
          }
#else
          fprintf(fp, "/* #define CHAR_BIT %d */\n", CHAR_BIT);
#endif
          fclose(fp);
          return 0;
}
''')
    testcode = '\n'.join(testcode)    
    return testcode




version='0.4.0'


#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    config = Configuration('scipy',parent_package,top_path)
    config.add_subpackage('distutils')
    config.add_subpackage('base')
    return config.todict()

if __name__ == '__main__':
    # Remove current working directory from sys.path
    # to avoid importing scipy.distutils as Python std. distutils:
