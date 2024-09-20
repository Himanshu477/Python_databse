import os
from glob import glob
from scipy.distutils.misc_util import get_path, Configuration, dot_join

def configuration(parent_package='',parent_path=None):
    parent_path2 = parent_path
    parent_path = parent_package
    local_path = get_path(__name__,parent_path2)
    config = Configuration('weave',parent_package)
    config.add_subpackage('tests')
    scxx_files = glob(os.path.join(local_path,'scxx','*.*'))
    install_path = os.path.join(parent_path,'weave','scxx')
    config.add_data_dir('scxx')
    config.add_data_dir(os.path.join('blitz','blitz'))
    config.add_data_dir(os.path.join('blitz','blitz','array'))
    config.add_data_dir(os.path.join('blitz','blitz','meta'))
    config.add_data_files(*glob(os.path.join(local_path,'doc','*.html')))
    config.add_data_files(*glob(os.path.join(local_path,'examples','*.py')))    
    return config

if __name__ == '__main__':    
    from scipy.distutils.core import setup
    from weave_version import weave_version
    setup(version = weave_version,
          description = "Tools for inlining C/C++ in Python",
          author = "Eric Jones",
          author_email = "eric@enthought.com",
          licence = "SciPy License (BSD Style)",
          url = 'http://www.scipy.org',
          **configuration(parent_path=''))



#  doc is comment_documentation

# use list so order is preserved.
ufunc_api_list = [
    (r"""
    """,
     'FromFuncAndData', 'PyUFuncGenericFunction *, void **, char *, int, int, int, int, char *, char *, int', 'PyObject *'),

    (r"""
    """,
     'RegisterLoopForType','PyUFuncObject *, int, PyUFuncGenericFunction, void *', 'int'),
    
    (r"""
    """,
     'GenericFunction', 'PyUFuncObject *, PyObject *, PyArrayObject **', 'int'),

    (r"""
    """,
     'f_f_As_d_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'd_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'f_f','char **, intp *, intp *, void *','void'),    

    (r"""
    """,
     'g_g','char **, intp *, intp *, void *','void'),     

    (r"""
    """,
     'F_F_As_D_D','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'F_F','char **, intp *, intp *, void *','void'),     

    (r"""
    """,
     'D_D','char **, intp *, intp *, void *','void'),     

    (r"""
    """,
     'G_G','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'O_O','char **, intp *, intp *, void *','void'), 

    (r"""
    """,
     'ff_f_As_dd_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'ff_f','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'dd_d','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'gg_g','char **, intp *, intp *, void *','void'),     

    (r"""
    """,
     'FF_F_As_DD_D','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'DD_D','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'FF_F','char **, intp *, intp *, void *','void'),         

    (r"""
    """,
     'GG_G','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'OO_O','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'O_O_method','char **, intp *, intp *, void *','void'),

    (r"""
    """,
     'checkfperr', 'int, PyObject *', 'int'),

    (r"""
    """,
     'clearfperr', 'void', 'void')

]
    
# API fixes for __arrayobject_api.h

fixed = 1
nummulti = len(ufunc_api_list)
numtotal = fixed + nummulti


module_list = []
extension_list = []
init_list = []

#setup object API
for k, item in enumerate(ufunc_api_list):
    num = fixed + k
    astr = "static %s PyUFunc_%s \\\n       (%s);" % \
           (item[3],item[1],item[2])
    module_list.append(astr)
    astr = "#define PyUFunc_%s \\\n        (*(%s (*)(%s)) \\\n"\
           "         PyUFunc_API[%d])" % (item[1],item[3],item[2],num)
    extension_list.append(astr)
    astr = "        (void *) PyUFunc_%s," % item[1]
    init_list.append(astr)


outstr = r"""
#ifdef _UMATHMODULE

static PyTypeObject PyUFunc_Type;

%s

#else

static void **PyUFunc_API;

#define PyUFunc_Type (*(PyTypeObject *)PyUFunc_API[0])

%s

static int
