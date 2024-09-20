import tempfile

def clear_temp_catalog():
    """ Remove any catalog from the temp dir
    """
    global backup_dir 
    backup_dir =tempfile.mktemp()
    os.mkdir(backup_dir)
    for file in temp_catalog_files():
        move_file(file,backup_dir)
        #d,f = os.path.split(file)
        #backup = os.path.join(backup_dir,f)
        #os.rename(file,backup)

def restore_temp_catalog():
    """ Remove any catalog from the temp dir
    """
    global backup_dir
    cat_dir = catalog.default_dir()
    for file in os.listdir(backup_dir):
        file = os.path.join(backup_dir,file)
        d,f = os.path.split(file)
        dst_file = os.path.join(cat_dir, f)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        #os.rename(file,dst_file)
        move_file(file,dst_file)
    os.rmdir(backup_dir)
    backup_dir = None
         
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
        try:
            if os.path.isdir(i):
                cleanup_temp_dir(i)
            else:
                os.remove(i)
        except OSError:
            pass # failed to remove file for whatever reason 
                 # (maybe it is a DLL Python is currently using)        
    try:
        os.rmdir(d)
    except OSError:
        pass        
        

# from distutils -- old versions had bug, so copying here to make sure 
# a working version is available.
