import catalog
restore_path()

def temp_catalog_files():
    d = catalog.default_dir()
    f = catalog.os_dependent_catalog_name()
    suffixes = ['.dat','.dir','']
    cat_files = [os.path.join(d,f+suffix) for suffix in suffixes]
    return cat_files

def clear_temp_catalog():
    """ Remove any catalog from the temp dir
    """
    cat_files = temp_catalog_files()
    for catalog_file in cat_files:
        if os.path.exists(catalog_file):
            if os.path.exists(catalog_file+'.bak'):
                os.remove(catalog_file+'.bak')
            os.rename(catalog_file,catalog_file+'.bak')

def restore_temp_catalog():
    """ Remove any catalog from the temp dir
    """
    cat_files = temp_catalog_files()
    for catalog_file in cat_files:
        if os.path.exists(catalog_file+'.bak'):
            if os.path.exists(catalog_file):            
                os.remove(catalog_file)
            os.rename(catalog_file+'.bak',catalog_file)

def empty_temp_dir():
    """ Create a sub directory in the temp directory for use in tests
    """
    import tempfile
    d = catalog.default_dir()
    for i in range(10000):
        new_d = os.path.join(d,tempfile.gettempprefix()[1:-1]+`i`)
        if not os.path.exists(new_d):
            os.mkdir(new_d)
            break
    return new_d

def cleanup_temp_dir(d):
    """ Remove a directory created by empty_temp_dir
        should probably catch errors
    """
    files = map(lambda x,d=d: os.path.join(d,x),os.listdir(d))
    for i in files:
        if os.path.isdir(i):
            cleanup_temp_dir(i)
        else:
            os.remove(i)
    os.rmdir(d)

def simple_module(directory,name,function_prefix,count=2):
    module_name = os.path.join(directory,name+'.py')
    func = "def %(function_prefix)s%(fid)d():\n    pass\n"
    code = ''
    for fid in range(count):
        code+= func % locals()
    open(module_name,'w').write(code)
    sys.path.append(directory)    
    exec "import " + name
    funcs = []
    for i in range(count):
        funcs.append(eval(name+'.'+function_prefix+`i`))
    sys.path = sys.path[:-1]    
    return module_name, funcs        


