    import sys
    if '--without-dependencies' in sys.argv:
        with_dependencies = 0
        sys.argv.remove('--without-dependencies')
    else:
        with_dependencies = 1    
    stand_alone_package(with_dependencies)
    


