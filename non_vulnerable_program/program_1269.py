    import distutils.core
    old_dist = distutils.core._setup_distribution
    distutils.core._setup_distribution = None
    try:
        dist = setup(ext_modules=[ext],
                     script_name = 'get_atlas_version',
                     script_args = ['build_src','build_ext']+extra_args)
    except Exception,msg:
        print "##### msg: %s" % msg
        if not msg:
            msg = "Unknown Exception"
        log.warn(msg)
        return None
    distutils.core._setup_distribution = old_dist

