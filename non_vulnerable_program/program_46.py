import sys,os,string,time
import tempfile
import exceptions

class CompileError(exceptions.Exception):
    pass
    
def build_extension(module_path,compiler_name = '',build_dir = None,
                    temp_dir = None, verbose = 0, **kw):
    """ Build the file given by module_path into a Python extension module.
    
        build_extensions uses distutils to build Python extension modules.
        kw arguments not used are passed on to the distutils extension
        module.  Directory settings can handle absoulte settings, but don't
        currently expand '~' or environment variables.
        
        module_path   -- the full path name to the c file to compile.  
                         Something like:  /full/path/name/module_name.c 
                         The name of the c/c++ file should be the same as the
                         name of the module (i.e. the initmodule() routine)
        compiler_name -- The name of the compiler to use.  On Windows if it 
                         isn't given, MSVC is used if it exists (is found).
                         gcc is used as a second choice. If neither are found, 
                         the default distutils compiler is used. Acceptable 
                         names are 'gcc', 'msvc' or any of the compiler names 
                         shown by distutils.ccompiler.show_compilers()
        build_dir     -- The location where the resulting extension module 
                         should be placed. This location must be writable.  If
                         it isn't, several default locations are tried.  If the 
                         build_dir is not in the current python path, a warning
                         is emitted, and it is added to the end of the path.
                         build_dir defaults to the current directory.
        temp_dir      -- The location where temporary files (*.o or *.obj)
                         from the build are placed. This location must be 
                         writable.  If it isn't, several default locations are 
                         tried.  It defaults to tempfile.gettempdir()
        verbose       -- 0, 1, or 2.  0 is as quiet as possible. 1 prints
                         minimal information.  2 is noisy.                 
        **kw          -- keyword arguments. These are passed on to the 
                         distutils extension module.  Most of the keywords
                         are listed below.

        Distutils keywords.  These are cut and pasted from Greg Ward's
        distutils.extension.Extension class for convenience:
        
        sources : [string]
          list of source filenames, relative to the distribution root
          (where the setup script lives), in Unix form (slash-separated)
          for portability.  Source files may be C, C++, SWIG (.i),
          platform-specific resource files, or whatever else is recognized
          by the "build_ext" command as source for a Python extension.
          Note: The module_path file is always appended to the front of this
                list                
        include_dirs : [string]
          list of directories to search for C/C++ header files (in Unix
          form for portability)          
        define_macros : [(name : string, value : string|None)]
          list of macros to define; each macro is defined using a 2-tuple,
          where 'value' is either the string to define it to or None to
          define it without a particular value (equivalent of "#define
          FOO" in source or -DFOO on Unix C compiler command line)          
        undef_macros : [string]
          list of macros to undefine explicitly
        library_dirs : [string]
          list of directories to search for C/C++ libraries at link time
        libraries : [string]
          list of library names (not filenames or paths) to link against
        runtime_library_dirs : [string]
          list of directories to search for C/C++ libraries at run time
          (for shared extensions, this is when the extension is loaded)
        extra_objects : [string]
          list of extra files to link with (eg. object files not implied
          by 'sources', static library that must be explicitly specified,
          binary resource files, etc.)
        extra_compile_args : [string]
          any extra platform- and compiler-specific information to use
          when compiling the source files in 'sources'.  For platforms and
          compilers where "command line" makes sense, this is typically a
          list of command-line arguments, but for other platforms it could
          be anything.
        extra_link_args : [string]
          any extra platform- and compiler-specific information to use
          when linking object files together to create the extension (or
          to create a new static Python interpreter).  Similar
          interpretation as for 'extra_compile_args'.
        export_symbols : [string]
          list of symbols to be exported from a shared extension.  Not
          used on all platforms, and not generally necessary for Python
          extensions, which typically export exactly one symbol: "init" +
          extension_name.
    """
    success = 0
    from distutils.core import setup, Extension
    
    # this is a screwy trick to get rid of a ton of warnings on Unix
    import distutils.sysconfig
    distutils.sysconfig.get_config_vars()
    if distutils.sysconfig._config_vars.has_key('OPT'):
        flags = distutils.sysconfig._config_vars['OPT']        
        flags = flags.replace('-Wall','')
        distutils.sysconfig._config_vars['OPT'] = flags
    
    # get the name of the module and the extension directory it lives in.  
    module_dir,cpp_name = os.path.split(os.path.abspath(module_path))
    module_name,ext = os.path.splitext(cpp_name)    
    
    # configure temp and build directories
    temp_dir = configure_temp_dir(temp_dir)    
    build_dir = configure_build_dir(module_dir)
    
    compiler_name = choose_compiler(compiler_name)
    configure_sys_argv(compiler_name,temp_dir,build_dir)
    
    # the business end of the function
    try:
        if verbose == 1:
            print 'Compiling code...'
            
        # set compiler verboseness 2 or more makes it output results
        if verbose > 1:  verb = 1                
        else:            verb = 0
        
        t1 = time.time()        
        # add module to the needed source code files and build extension
        sources = kw.get('sources',[])
        kw['sources'] = [module_path] + sources        
        
        # add module to the needed source code files and build extension
        # FIX this is g++ specific. It probably should be fixed for other
        # Unices/compilers.  Find a generic solution
        if sys.platform[:-1] == 'linux':
            libraries = kw.get('libraries',[])
            kw['libraries'] = ['stdc++'] +  libraries        
        ext = Extension(module_name, **kw)
        
        # the switcheroo on SystemExit here is meant to keep command line
        # sessions from exiting when compiles fail.
        builtin = sys.modules['__builtin__']
        old_SysExit = builtin.__dict__['SystemExit']
        builtin.__dict__['SystemExit'] = CompileError
        
        # distutils for MSVC messes with the environment, so we save the
        # current state and restore them afterward.
        import copy
        environ = copy.deepcopy(os.environ)
        try:
            setup(name = module_name, ext_modules = [ext],verbose=verb)
        finally:
            # restore state
            os.environ = environ        
            # restore SystemExit
            builtin.__dict__['SystemExit'] = old_SysExit
        t2 = time.time()
        
        if verbose == 1:
            print 'finished compiling (sec): ', t2 - t1    
        success = 1
        configure_python_path(build_dir)
    except SyntaxError: #TypeError:
        success = 0    
            
    # restore argv after our trick...            
    restore_sys_argv()
    
    return success

old_argv = []
def configure_sys_argv(compiler_name,temp_dir,build_dir):
    # We're gonna play some tricks with argv here to pass info to distutils 
    # which is really built for command line use. better way??
    global old_argv
    old_argv = sys.argv[:]        
    sys.argv = ['','build_ext','--build-lib', build_dir,
                               '--build-temp',temp_dir]    
    if compiler_name == 'gcc':
        sys.argv.insert(2,'--compiler='+compiler_name)
    elif compiler_name:
        sys.argv.insert(2,'--compiler='+compiler_name)

def restore_sys_argv():
    sys.argv = old_argv
            
def configure_python_path(build_dir):    
    #make sure the module lives in a directory on the python path.
    python_paths = [os.path.abspath(x) for x in sys.path]
    if os.path.abspath(build_dir) not in python_paths:
        #print "warning: build directory was not part of python path."\
        #      " It has been appended to the path."
        sys.path.append(os.path.abspath(build_dir))

def choose_compiler(compiler_name=''):
    """ Try and figure out which compiler is gonna be used on windows.
        On other platforms, it just returns whatever value it is given.
        
        converts 'gcc' to 'mingw32' on win32
    """
    if sys.platform == 'win32':        
        if not compiler_name:
            # On Windows, default to MSVC and use gcc if it wasn't found
            # wasn't found.  If neither are found, go with whatever
            # the default is for distutils -- and probably fail...
            if msvc_exists():
                compiler_name = 'msvc'
            elif gcc_exists():
                compiler_name = 'mingw32'
        elif compiler_name == 'gcc':
                compiler_name = 'mingw32'
    else:
        # don't know how to force gcc -- look into this.
        if compiler_name == 'gcc':
                compiler_name = 'unix'                    
    return compiler_name
    
def gcc_exists():
    """ Test to make sure gcc is found 
       
        Does this return correct value on win98???
    """
    result = 0
    try:
        w,r=os.popen4('gcc -v')
        w.close()
        str_result = r.read()
        #print str_result
        if string.find(str_result,'Reading specs') != -1:
            result = 1
    except:
        # This was needed because the msvc compiler messes with
        # the path variable. and will occasionlly mess things up
        # so much that gcc is lost in the path. (Occurs in test
        # scripts)
        result = not os.system('gcc -v')
    return result

def msvc_exists():
    """ Determine whether MSVC is available on the machine.
    """
    result = 0
    try:
        w,r=os.popen4('cl')
        w.close()
        str_result = r.read()
        #print str_result
        if string.find(str_result,'Microsoft') != -1:
            result = 1
    except:
        #assume we're ok if devstudio exists
        import distutils.msvccompiler
        version = distutils.msvccompiler.get_devstudio_version()
        if version:
            result = 1
    return result

def configure_temp_dir(temp_dir=None):
    if temp_dir is None:         
        temp_dir = tempfile.gettempdir()
    elif not os.path.exists(temp_dir) or not os.access(temp_dir,os.W_OK):
        print "warning: specified temp_dir '%s' does not exist or is " \
              "or is not writable. Using the default temp directory" % \
              temp_dir
        temp_dir = tempfile.gettempdir()

    # final check that that directories are writable.        
    if not os.access(temp_dir,os.W_OK):
        msg = "Either the temp or build directory wasn't writable. Check" \
              " these locations: '%s'" % temp_dir  
        raise ValueError, msg
    return temp_dir

def configure_build_dir(build_dir=None):
    # make sure build_dir exists and is writable
    if build_dir and (not os.path.exists(build_dir) or 
                      not os.access(build_dir,os.W_OK)):
        print "warning: specified build_dir '%s' does not exist or is " \
               "or is not writable. Trying default locations" % build_dir
        build_dir = None
        
    if build_dir is None:
        #default to building in the home directory of the given module.        
        build_dir = os.curdir
        # if it doesn't work use the current directory.  This should always
        # be writable.    
        if not os.access(build_dir,os.W_OK):
            print "warning:, neither the module's directory nor the "\
                  "current directory are writable.  Using the temporary"\
                  "directory."
            build_dir = tempfile.gettempdir()

    # final check that that directories are writable.
    if not os.access(build_dir,os.W_OK):
        msg = "The build directory wasn't writable. Check" \
              " this location: '%s'" % build_dir
        raise ValueError, msg
        
    return os.path.abspath(build_dir)        
    
if sys.platform == 'win32':
