import sys
def generate_testcode(target):
    if sys.platform == 'win32':
        target = target.replace('\\','\\\\')
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

if __name__=='__main__':
