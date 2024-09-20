    import time
    t1 = time.time()    
    dist = dummy_dist()    
    compiler_obj = create_compiler_instance(dist)
    exe_name = compiler_exe_name(compiler_obj)
    exe_path = compiler_exe_path(exe_name)
    chk_sum = check_sum(exe_path)    
    
    t2 = time.time()
    print 'compiler exe:', exe_path
    print 'check sum:', chk_sum
    print 'time (sec):', t2 - t1
    print
    """
    path = get_compiler_dir('gcc')
    print 'gcc path:', path
    print
    try: 
        path = get_compiler_dir('msvc')
        print 'gcc path:', path
    except ValueError:
        pass    

""" Information about platform and python version and compilers

    This information is manly used to build directory names that
    keep the object files and shared libaries straight when
    multiple platforms share the same file system.
"""

