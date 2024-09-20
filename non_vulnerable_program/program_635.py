    import log
    magic = hex(hash(`config`))
    def atlas_version_c(extension, build_dir,magic=magic):
        source = os.path.join(build_dir,'atlas_version_%s.c' % (magic))
        if os.path.isfile(source):
            from distutils.dep_util import newer
            if newer(source,__file__):
                return source
        f = open(source,'w')
        f.write(atlas_version_c_text)
        f.close()
        return source
    ext = Extension('atlas_version',
                    sources=[atlas_version_c],
                    **config)
#    extra_args = ['--build-lib',get_build_temp_dir()]
    extra_args = []
    for a in sys.argv:
        if re.match('[-][-]compiler[=]',a):
            extra_args.append(a)
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

