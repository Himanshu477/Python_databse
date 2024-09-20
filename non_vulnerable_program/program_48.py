import os,sys,string
import shelve

def getmodule(object):
    """ Discover the name of the module where object was defined.
    
        This is an augmented version of inspect.getmodule that can discover 
        the parent module for extension functions.
    """
    import inspect
    value = inspect.getmodule(object)
    if value is None:
        #walk trough all modules looking for function
        for name,mod in sys.modules.items():
            # try except used because of some comparison failures
            # in wxPoint code.  Need to review this
            try:
                if mod and object in mod.__dict__.values():
                    value = mod
                    # if it is a built-in module, keep looking to see
                    # if a non-builtin also has it.  Otherwise quit and
                    # consider the module found. (ain't perfect, but will 
                    # have to do for now).
                    if string.find('(built-in)',str(mod)) is -1:
                        break
                    
            except TypeError:
                pass        
    return value

def expr_to_filename(expr):
    """ Convert an arbitrary expr string to a valid file name.
    
        The name is based on the md5 check sum for the string and
        Something that was a little more human readable would be 
        nice, but the computer doesn't seem to care.
    """
    import md5
    base = 'sc_'
    return base + md5.new(expr).hexdigest()

def unique_file(d,expr):
    """ Generate a unqiue file name based on expr in directory d
    
        This is meant for use with building extension modules, so
        a file name is considered unique if none of the following
        extension '.cpp','.o','.so','module.so','.py', or '.pyd'
        exists in directory d.  The fully qualified path to the
        new name is returned.  You'll need to append your own
        extension to it before creating files.
    """
    files = os.listdir(d)
    #base = 'scipy_compile'
    base = expr_to_filename(expr)
    for i in range(1000000):
        fname = base + `i`
        if not (fname+'.cpp' in files or
                fname+'.o' in files or
                fname+'.so' in files or
                fname+'module.so' in files or
                fname+'.py' in files or
                fname+'.pyd' in files):
            break
    return os.path.join(d,fname)
    
def default_dir():
    """ Return a default location to store compiled files and catalogs.
        
        XX is the Python version number in all paths listed below
        On windows, the default location is the temporary directory
        returned by gettempdir()/pythonXX.
        
        On Unix, ~/.pythonXX_compiled is the default location.  If it doesn't
        exist, it is created.  The directory is marked rwx------.
        
        If for some reason it isn't possible to build a default directory
        in the user's home, /tmp/<uid>_pythonXX_compiled is used.  If it 
        doesn't exist, it is created.  The directory is marked rwx------
        to try and keep people from being able to sneak a bad module
        in on you.        
    """
    import tempfile        
    python_name = "python%d%d_compiled" % tuple(sys.version_info[:2])    
    if sys.platform != 'win32':
        try:
            path = os.path.join(os.environ['HOME'],'.' + python_name)
        except KeyError:
            temp_dir = `os.getuid()` + '_' + python_name
            path = os.path.join(tempfile.gettempdir(),temp_dir)        
    else:
        path = os.path.join(tempfile.gettempdir(),python_name)
        
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path,0700) # make it only accessible by this user.
    if not os.access(path,os.W_OK):
        print 'warning: default directory is not write accessible.'
        print 'defualt:', path
    return path

def default_temp_dir():
    path = os.path.join(default_dir(),'temp')
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path,0700) # make it only accessible by this user.
    if not os.access(path,os.W_OK):
        print 'warning: default directory is not write accessible.'
        print 'defualt:', path
    return path

    
def os_dependent_catalog_name():
    """ Generate catalog name dependent on OS and Python version being used.
    
        This allows multiple platforms to have catalog files in the
        same directory without stepping on each other.  For now, it 
        bases the name of the value returned by sys.platform and the
        version of python being run.  If this isn't enough to descriminate
        on some platforms, we can try to add other info.  It has 
        occured to me that if we get fancy enough to optimize for different
        architectures, then chip type might be added to the catalog name also.
    """
    version = '%d%d' % sys.version_info[:2]
    return sys.platform+version+'compiled_catalog'
    
def catalog_path(module_path):
    """ Return the full path name for the catalog file in the given directory.
    
        module_path can either be a file name or a path name.  If it is a 
        file name, the catalog file name in its parent directory is returned.
        If it is a directory, the catalog file in that directory is returned.

        If module_path doesn't exist, None is returned.  Note though, that the
        catalog file does *not* have to exist, only its parent.  '~', shell
        variables, and relative ('.' and '..') paths are all acceptable.
        
        catalog file names are os dependent (based on sys.platform), so this 
        should support multiple platforms sharing the same disk space 
        (NFS mounts). See os_dependent_catalog_name() for more info.
    """
    module_path = os.path.expanduser(module_path)
    module_path = os.path.expandvars(module_path)
    module_path = os.path.abspath(module_path)
    if not os.path.exists(module_path):
        catalog_file = None
    elif not os.path.isdir(module_path):
        module_path,dummy = os.path.split(module_path)
        catalog_file = os.path.join(module_path,os_dependent_catalog_name())
    else:    
        catalog_file = os.path.join(module_path,os_dependent_catalog_name())
    return catalog_file

def get_catalog(module_path,mode='r'):
    """ Return a function catalog (shelve object) from the path module_path

        If module_path is a directory, the function catalog returned is
        from that directory.  If module_path is an actual module_name,
        then the function catalog returned is from its parent directory.
        mode uses the standard 'c' = create, 'r' = read, 'w' = write
        file open modes.
        
        See catalog_path() for more information on module_path.
    """
    catalog_file = catalog_path(module_path)
    #print catalog_file
    #sh = shelve.open(catalog_file,mode)
    try:
        sh = shelve.open(catalog_file,mode)
    except: # not sure how to pin down which error to catch yet
        sh = None
    return sh

class catalog:
    """ Stores information about compiled functions both in cache and on disk.
    
        catalog stores (code, list_of_function) pairs so that all the functions
        that have been compiled for code are available for calling (usually in
        inline or blitz).
        
        catalog keeps a dictionary of previously accessed code values cached 
        for quick access.  It also handles the looking up of functions compiled 
        in previously called Python sessions on disk in function catalogs. 
        catalog searches the directories in the PYTHONCOMPILED environment 
        variable in order loading functions that correspond to the given code 
        fragment.  A default directory is also searched for catalog functions. 
        On unix, the default directory is usually '~/.pythonxx_compiled' where 
        xx is the version of Python used. On windows, it is the directory 
        returned by temfile.tempdir().  Functions closer to the front are of 
        the variable list are guaranteed to be closer to the front of the 
        function list so that they will be called first.  See 
        get_cataloged_functions() for more info on how the search order is 
        traversed.
        
        Catalog also handles storing information about compiled functions to
        a catalog.  When writing this information, the first writable catalog
        file in PYTHONCOMPILED path is used.  If a writable catalog is not
        found, it is written to the catalog in the default directory.  This
        directory should always be writable.
    """
    def __init__(self,user_path_list=None):
        """ Create a catalog for storing/searching for compiled functions. 
        
            user_path_list contains directories that should be searched 
            first for function catalogs.  They will come before the path
            entries in the PYTHONCOMPILED environment varilable.
        """
        if type(user_path_list) == type('string'):
            self.user_path_list = [user_path_list]
        elif user_path_list:
            self.user_path_list = user_path_list
        else:
            self.user_path_list = []
        self.cache = {}
        self.module_dir = None
        self.paths_added = 0
        
    def set_module_directory(self,module_dir):
        """ Set the path that will replace 'MODULE' in catalog searches.
        
            You should call clear_module_directory() when your finished
            working with it.
        """
        self.module_dir = module_dir
    def get_module_directory(self):
        """ Return the path used to replace the 'MODULE' in searches.
        """
        return self.module_dir
    def clear_module_directory(self):
        """ Reset 'MODULE' path to None so that it is ignored in searches.        
        """
        self.module_dir = None
        
    def get_environ_path(self):
        """ Return list of paths from 'PYTHONCOMPILED' environment variable.
        
            On Unix the path in PYTHONCOMPILED is a ':' separated list of
            directories.  On Windows, a ';' separated list is used. 
        """
        paths = []
        if os.environ.has_key('PYTHONCOMPILED'):
            path_string = os.environ['PYTHONCOMPILED'] 
            if sys.platform == 'win32':
                #probably should also look in registry
                paths = path_string.split(';')
            else:    
                paths = path_string.split(':')
        return paths    

    def build_search_order(self):
        """ Returns a list of paths that are searched for catalogs.  
        
            Values specified in the catalog constructor are searched first,
            then values found in the PYTHONCOMPILED environment variable.
            The directory returned by default_dir() is always returned at
            the end of the list.
            
            There is a 'magic' path name called 'MODULE' that is replaced
            by the directory defined by set_module_directory().  If the
            module directory hasn't been set, 'MODULE' is ignored.
        """
        
        paths = self.user_path_list + self.get_environ_path()
        search_order = []
        for path in paths:
            if path == 'MODULE':
                if self.module_dir:
                    search_order.append(self.module_dir)
            else:
                search_order.append(path)
        search_order.append(default_dir())
        return search_order

    def get_catalog_files(self):
        """ Returns catalog file list in correct search order.
          
            Some of the catalog files may not currently exists.
            However, all will be valid locations for a catalog
            to be created (if you have write permission).
        """
        files = map(catalog_path,self.build_search_order())
        files = filter(lambda x: x is not None,files)
        return files

    def get_existing_files(self):
        """ Returns all existing catalog file list in correct search order.
        """
        files = self.get_catalog_files()
        existing_files = filter(os.path.exists,files)
        return existing_files

    def get_writable_file(self,existing_only=0):
        """ Return the name of the first writable catalog file.
        
            Its parent directory must also be writable.  This is so that
            compiled modules can be written to the same directory.
        """
        # note: both file and its parent directory must be writeable
        if existing_only:
            files = self.get_existing_files()
        else:
            files = self.get_catalog_files()
        # filter for (file exists and is writable) OR directory is writable
        def file_test(x):
            from os import access, F_OK, W_OK
            return (access(x,F_OK) and access(x,W_OK) or
                    access(os.path.dirname(x),W_OK))
        writable = filter(file_test,files)
        if writable:
            file = writable[0]
        else:
            file = None
        return file
        
    def get_writable_dir(self):
        """ Return the parent directory of first writable catalog file.
        
            The returned directory has write access.
        """
        return os.path.dirname(self.get_writable_file())
        
    def unique_module_name(self,code,module_dir=None):
        """ Return full path to unique file name that in writable location.
        
            The directory for the file is the first writable directory in 
            the catalog search path.  The unique file name is derived from
            the code fragment.  If, module_dir is specified, it is used
            to replace 'MODULE' in the search path.
        """
        if module_dir is not None:
            self.set_module_directory(module_dir)
        try:
            d = self.get_writable_dir()
        finally:
            if module_dir is not None:
                self.clear_module_directory()
        return unique_file(d,code)

    def path_key(self,code):
        """ Return key for path information for functions associated with code.
        """
        return '__path__' + code
        
    def configure_path(self,cat,code):
        """ Add the python path for the given code to the sys.path
        
            unconfigure_path() should be called as soon as possible after
            imports associated with code are finished so that sys.path 
            is restored to normal.
        """
        try:
            paths = cat[self.path_key(code)]
            self.paths_added = len(paths)
            sys.path = paths + sys.path
        except:
            self.paths_added = 0            
                    
    def unconfigure_path(self):
        """ Restores sys.path to normal after calls to configure_path()
        
            Remove the previously added paths from sys.path
        """
        sys.path = sys.path[self.paths_added:]
        self.paths_added = 0

    def get_cataloged_functions(self,code):
        """ Load all functions associated with code from catalog search path.
        
            Sometimes there can be trouble loading a function listed in a
            catalog file because the actual module that holds the function 
            has been moved or deleted.  When this happens, that catalog file
            is "repaired", meaning the entire entry for this function is 
            removed from the file.  This only affects the catalog file that
            has problems -- not the others in the search path.
            
            The "repair" behavior may not be needed, but I'll keep it for now.
        """
        mode = 'r'
        cat = None
        function_list = []
        for path in self.build_search_order():
            cat = get_catalog(path,mode)
            if cat is not None and cat.has_key(code):
                # set up the python path so that modules for this
                # function can be loaded.
                self.configure_path(cat,code)
                try:                    
                    function_list += cat[code]
                except: #SystemError and ImportError so far seen                        
                    # problems loading a function from the catalog.  Try to
                    # repair the cause.
                    cat.close()
                    self.repair_catalog(path,code)
                self.unconfigure_path()             
        return function_list


    def repair_catalog(self,catalog_path,code):
        """ Remove entry for code from catalog_path
        
            Occasionally catalog entries could get corrupted. An example
            would be when a module that had functions in the catalog was
            deleted or moved on the disk.  The best current repair method is 
            just to trash the entire catalog entry for this piece of code.  
            This may loose function entries that are valid, but thats life.
            
            catalog_path must be writable for repair.  If it isn't, the
            function exists with a warning.            
        """
        writable_cat = None
        if not os.path.exists(catalog_path):
            return
        try:
            writable_cat = get_catalog(catalog_path,'w')
        except:
            print 'warning: unable to repair catalog entry\n %s\n in\n %s' % \
                  (code,catalog_path)
            return          
        if writable_cat.has_key(code):
            print 'repairing catalog by removing key'
            del writable_cat[code]   
            del writable_cat[self.path_key(code)]   
            
    def get_functions_fast(self,code):
        """ Return list of functions for code from the cache.
        
            Return an empty list if the code entry is not found.
        """
        return self.cache.get(code,[])
                
    def get_functions(self,code,module_dir=None):
        """ Return the list of functions associated with this code fragment.
        
            The cache is first searched for the function.  If an entry
            in the cache is not found, then catalog files on disk are 
            searched for the entry.  This is slooooow, but only happens
            once per code object.  All the functions found in catalog files
            on a cache miss are loaded into the cache to speed up future calls.
            The search order is as follows:
            
                1. user specified path (from catalog initialization)
                2. directories from the PYTHONCOMPILED environment variable
                3. The temporary directory on your platform.

            The path specified by module_dir will replace the 'MODULE' 
            place holder in the catalog search path. See build_search_order()
            for more info on the search path. 
        """        
        # Fast!! try cache first.
        if self.cache.has_key(code):
            return self.cache[code]
        
        # 2. Slow!! read previously compiled functions from disk.
        try:
            self.set_module_directory(module_dir)
            function_list = self.get_cataloged_functions(code)
            if code:
                function_list
            # put function_list in cache to save future lookups.
            if function_list:
                self.cache[code] = function_list
            # return function_list, empty or otherwise.
        finally:
            self.clear_module_directory()
        return function_list

    def add_function(self,code,function,module_dir=None):
        """ Adds a function to the catalog.
        
            The function is added to the cache as well as the first
            writable file catalog found in the search path.  If no
            code entry exists in the cache, the on disk catalogs
            are loaded into the cache and function is added to the
            beginning of the function list.
            
            The path specified by module_dir will replace the 'MODULE' 
            place holder in the catalog search path. See build_search_order()
            for more info on the search path. 
        """    

        # 1. put it in the cache.
        if self.cache.has_key(code):
            if function not in self.cache[code]:
                self.cache[code].insert(0,function)
        else:           
            # Load functions and put this one up front
            self.cache[code] = self.get_functions(code)          
            self.fast_cache(code,function)
        
        # 2. Store the function entry to disk.    
        try:
            self.set_module_directory(module_dir)
            self.add_function_persistent(code,function)
        finally:
            self.clear_module_directory()
        
    def add_function_persistent(self,code,function):
        """ Store the code->function relationship to disk.
        
            Two pieces of information are needed for loading functions
            from disk -- the function pickle (which conveniently stores
            the module name, etc.) and the path to its module's directory.
            The latter is needed so that the function can be loaded no
            matter what the user's Python path is.
        """       
        # add function to data in first writable catalog
        mode = 'cw' # create, write
        cat_file = self.get_writable_dir()
        cat = get_catalog(cat_file,mode)
        if cat is None:
            cat_dir = default_dir()
            cat = get_catalog(cat_file,mode)
        if cat is None:
            cat_dir = default_dir()                            
            cat_file = catalog_path(cat_dir)
            print 'problems with default catalog -- removing'
            os.remove(cat_file)
            cat = get_catalog(cat_dir,mode)
        if cat is None:
            raise ValueError, 'Failed to access a catalog for storing functions'    
        function_list = [function] + cat.get(code,[])
        cat[code] = function_list
        
        # now add needed path information for loading function
        module = getmodule(function)
        try:
            # built in modules don't have the __file__ extension, so this
            # will fail.  Just pass in this case since path additions aren't
            # needed for built-in modules.
            mod_path,f=os.path.split(os.path.abspath(module.__file__))
            pkey = self.path_key(code)
            cat[pkey] = [mod_path] + cat.get(pkey,[])
        except:
            pass

    def fast_cache(self,code,function):
        """ Move function to the front of the cache entry for code
        
            If future calls to the function have the same type signature,
            this will speed up access significantly because the first
            function call is correct.
            
            Note:  The cache added to the inline_tools module is significantly
                   faster than always calling get_functions, so this isn't
                   as necessary as it used to be.  Still, it's probably worth
                   doing.              
        """
        try:
            if self.cache[code][0] == function:
                return
        except: # KeyError, IndexError   
            pass
        try:
            self.cache[code].remove(function)
        except ValueError:
            pass
        # put new function at the beginning of the list to search.
        self.cache[code].insert(0,function)
        
def test():
    from scipy_test import module_test
    module_test(__name__,__file__)

def test_suite():
    from scipy_test import module_test_suite
    return module_test_suite(__name__,__file__)    


module_header = \
"""    
// blitz must be first, or you get have issues with isnan defintion.
#include <blitz/array.h> 

#include "Python.h" 
#include "Numeric/arrayobject.h" 

// Use Exception stuff from SCXX
#include "PWOBase.h" 

#include <stdio.h> 
#include <math.h> 
#include <complex>

static PyArrayObject* obj_to_numpy(PyObject* py_obj, char* name, 
                                  int Ndims, int numeric_type )
{
    PyArrayObject* arr_obj = NULL;

    // Make sure input is an array.
    if (!PyArray_Check(py_obj))
        throw PWException(PyExc_TypeError,
                          "Input array *name* must be an array.");

    arr_obj = (PyArrayObject*) py_obj;
    
    // Make sure input has correct numeric type.
    if (arr_obj->descr->type_num != numeric_type)
    {
        // This should be more explicit:
        // Put the desired and actual type in the message.
        // printf("%d,%d",arr_obj->descr->type_num,numeric_type);
        throw PWException(PyExc_TypeError,
                          "Input array *name* is the wrong numeric type.");
    }
    
    // Make sure input has correct rank (defined as number of dimensions).
    // Currently, all arrays must have the same shape.
    // Broadcasting is not supported.
    // ...
    if (arr_obj->nd != Ndims)
    {
        // This should be more explicit:
        // Put the desired and actual dimensionality in message.
        throw PWException(PyExc_TypeError,
                         "Input array *name* has wrong number of dimensions.");
    }    
    // check the size of arrays.  Acutally, the size of the "views" really
    // needs checking -- not the arrays.
    // ...
    
    // Any need to deal with INC/DEC REFs?
    Py_INCREF(py_obj);
    return arr_obj;
}

// simple meta-program templates to specify python typecodes
// for each of the numeric types.
template<class T>
class py_type{public: enum {code = 100};};
class py_type<char>{public: enum {code = PyArray_CHAR};};
class py_type<unsigned char>{public: enum { code = PyArray_UBYTE};};
class py_type<short>{public:  enum { code = PyArray_SHORT};};
class py_type<int>{public: enum { code = PyArray_INT};};
class py_type<long>{public: enum { code = PyArray_LONG};};
class py_type<float>{public: enum { code = PyArray_FLOAT};};
class py_type<double>{public: enum { code = PyArray_DOUBLE};};
class py_type<complex<float> >{public: enum { code = PyArray_CFLOAT};};
class py_type<complex<double> >{public: enum { code = PyArray_CDOUBLE};};

template<class T, int N>
static blitz::Array<T,N> py_to_blitz(PyObject* py_obj,char* name)
{

    PyArrayObject* arr_obj = obj_to_numpy(py_obj,name,N,py_type<T>::code);
    
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    int stride_acc = 1;
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,        
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}

template<class T> 
static T py_to_scalar(PyObject* py_obj,char* name)
{
    //never used.
    return (T) 0;
}
template<>
static int py_to_scalar<int>(PyObject* py_obj,char* name)
{
    return (int) PyLong_AsLong(py_obj);
}

template<>
static long py_to_scalar<long>(PyObject* py_obj,char* name)
{
    return (long) PyLong_AsLong(py_obj);
}
template<> 
static float py_to_scalar<float>(PyObject* py_obj,char* name)
{
    return (float) PyFloat_AsDouble(py_obj);
}
template<> 
static double py_to_scalar<double>(PyObject* py_obj,char* name)
{
    return PyFloat_AsDouble(py_obj);
}

// complex not checked.
template<> 
static std::complex<float> py_to_scalar<std::complex<float> >(PyObject* py_obj,
                                                              char* name)
{
    return std::complex<float>((float) PyComplex_RealAsDouble(py_obj),
                               (float) PyComplex_RealAsDouble(py_obj));    
}
template<> 
static std::complex<double> py_to_scalar<std::complex<double> >(
                                            PyObject* py_obj,char* name)
{
    return std::complex<double>(PyComplex_RealAsDouble(py_obj),
                                PyComplex_RealAsDouble(py_obj));    
}
"""    

""" Generic support code for handling standard Numeric arrays      
"""

