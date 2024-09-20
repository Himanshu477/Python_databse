    from distutils.sysconfig import get_config_var
    so_ext = get_config_var('SO')
    build_ext = dist.get_command_obj('build_ext')
    target = os.path.join(build_ext.build_lib,'atlas_version'+so_ext)
    cmd = [get_pythonexe(),'-c',
           '"import imp;imp.load_dynamic(\\"atlas_version\\",\\"%s\\")"'\
           % (os.path.basename(target))]
    s,o = exec_command(cmd,execute_in=os.path.dirname(target),use_tee=0)
    atlas_version = None
    if not s:
        m = re.search(r'ATLAS version (?P<version>\d+[.]\d+[.]\d+)',o)
        if m:
            atlas_version = m.group('version')
    if atlas_version is None:
        if re.search(r'undefined symbol: ATL_buildinfo',o,re.M):
            atlas_version = '3.2.1_pre3.3.6'
        else:
            print 'Command:',' '.join(cmd)
            print 'Status:',s
            print 'Output:',o
    return atlas_version


class lapack_opt_info(system_info):
    
    def calc_info(self):

        if sys.platform=='darwin' and not os.environ.get('ATLAS',None):
            args = []
            link_args = []
            if os.path.exists('/System/Library/Frameworks/Accelerate.framework/'):
                args.extend(['-faltivec','-framework','Accelerate'])
                link_args.extend(['-Wl,-framework','-Wl,Accelerate'])
            elif os.path.exists('/System/Library/Frameworks/vecLib.framework/'):
                args.extend(['-faltivec','-framework','vecLib'])
                link_args.extend(['-Wl,-framework','-Wl,vecLib'])
            if args:
                self.set_info(extra_compile_args=args,
                              extra_link_args=link_args,
                              define_macros=[('NO_ATLAS_INFO',3)])
                return

        atlas_info = get_info('atlas_threads')
        if not atlas_info:
            atlas_info = get_info('atlas')
        #atlas_info = {} ## uncomment for testing
        atlas_version = None
        need_lapack = 0
        need_blas = 0
        info = {}
        if atlas_info:
            version_info = atlas_info.copy()
            atlas_version = get_atlas_version(**version_info)
            if not atlas_info.has_key('define_macros'):
                atlas_info['define_macros'] = []
            if atlas_version is None:
                atlas_info['define_macros'].append(('NO_ATLAS_INFO',2))
            else:
                atlas_info['define_macros'].append(('ATLAS_INFO',
                                                    '"\\"%s\\""' % atlas_version))
        if atlas_version=='3.2.1_pre3.3.6':
            atlas_info['define_macros'].append(('NO_ATLAS_INFO',4))
            l = atlas_info.get('define_macros',[])
            if ('ATLAS_WITH_LAPACK_ATLAS',None) in l \
                   or ('ATLAS_WITHOUT_LAPACK',None) in l:
                need_lapack = 1
            info = atlas_info
        else:
            warnings.warn(AtlasNotFoundError.__doc__)
            need_blas = 1
            need_lapack = 1
            dict_append(info,define_macros=[('NO_ATLAS_INFO',1)])

        if need_lapack:
            lapack_info = get_info('lapack')
            #lapack_info = {} ## uncomment for testing
            if lapack_info:
                dict_append(info,**lapack_info)
            else:
                warnings.warn(LapackNotFoundError.__doc__)
                lapack_src_info = get_info('lapack_src')
                if not lapack_src_info:
                    warnings.warn(LapackSrcNotFoundError.__doc__)
                    return
                dict_append(info,libraries=[('flapack_src',lapack_src_info)])

        if need_blas:
            blas_info = get_info('blas')
            #blas_info = {} ## uncomment for testing
            if blas_info:
                dict_append(info,**blas_info)
            else:
                warnings.warn(BlasNotFoundError.__doc__)
                blas_src_info = get_info('blas_src')
                if not blas_src_info:
                    warnings.warn(BlasSrcNotFoundError.__doc__)
                    return
                dict_append(info,libraries=[('fblas_src',blas_src_info)])

        self.set_info(**info)
        return


class blas_opt_info(system_info):
    
    def calc_info(self):

        if sys.platform=='darwin' and not os.environ.get('ATLAS',None):
            args = []
            link_args = []
            if os.path.exists('/System/Library/Frameworks/Accelerate.framework/'):
                args.extend(['-faltivec','-framework','Accelerate'])
                link_args.extend(['-Wl,-framework','-Wl,Accelerate'])
            elif os.path.exists('/System/Library/Frameworks/vecLib.framework/'):
                args.extend(['-faltivec','-framework','vecLib'])
                link_args.extend(['-Wl,-framework','-Wl,vecLib'])
            if args:
                self.set_info(extra_compile_args=args,
                              extra_link_args=link_args,
                              define_macros=[('NO_ATLAS_INFO',3)])
                return

        atlas_info = get_info('atlas_blas_threads')
        if not atlas_info:
            atlas_info = get_info('atlas_blas')
        atlas_version = None
        need_blas = 0
        info = {}
        if atlas_info:
            version_info = atlas_info.copy()
            atlas_version = get_atlas_version(**version_info)
            if not atlas_info.has_key('define_macros'):
                atlas_info['define_macros'] = []
            if atlas_version is None:
                atlas_info['define_macros'].append(('NO_ATLAS_INFO',2))
            else:
                atlas_info['define_macros'].append(('ATLAS_INFO',
                                                    '"\\"%s\\""' % atlas_version))
            info = atlas_info
        else:
            warnings.warn(AtlasNotFoundError.__doc__)
            need_blas = 1
            dict_append(info,define_macros=[('NO_ATLAS_INFO',1)])

        if need_blas:
            blas_info = get_info('blas')
            if blas_info:
                dict_append(info,**blas_info)
            else:
                warnings.warn(BlasNotFoundError.__doc__)
                blas_src_info = get_info('blas_src')
                if not blas_src_info:
                    warnings.warn(BlasSrcNotFoundError.__doc__)
                    return
                dict_append(info,libraries=[('fblas_src',blas_src_info)])

        self.set_info(**info)
        return


class blas_info(system_info):
    section = 'blas'
    dir_env_var = 'BLAS'
    _lib_names = ['blas']
    notfounderror = BlasNotFoundError

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()

        blas_libs = self.get_libs('blas_libs', self._lib_names)
        for d in lib_dirs:
            blas = self.check_libs(d,blas_libs,[])
            if blas is not None:
                info = blas                
                break
        else:
            return
        info['language'] = 'f77'  # XXX: is it generally true?
        self.set_info(**info)


class blas_src_info(system_info):
    section = 'blas_src'
    dir_env_var = 'BLAS_SRC'
    notfounderror = BlasSrcNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d,['blas']))
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d,'daxpy.f')):
                src_dir = d
                break
        if not src_dir:
            #XXX: Get sources from netlib. May be ask first.
            return
        blas1 = '''
        caxpy csscal dnrm2 dzasum saxpy srotg zdotc ccopy cswap drot
        dznrm2 scasum srotm zdotu cdotc dasum drotg icamax scnrm2
        srotmg zdrot cdotu daxpy drotm idamax scopy sscal zdscal crotg
        dcabs1 drotmg isamax sdot sswap zrotg cscal dcopy dscal izamax
        snrm2 zaxpy zscal csrot ddot dswap sasum srot zcopy zswap
        '''
        blas2 = '''
        cgbmv chpmv ctrsv dsymv dtrsv sspr2 strmv zhemv ztpmv cgemv
        chpr dgbmv dsyr lsame ssymv strsv zher ztpsv cgerc chpr2 dgemv
        dsyr2 sgbmv ssyr xerbla zher2 ztrmv cgeru ctbmv dger dtbmv
        sgemv ssyr2 zgbmv zhpmv ztrsv chbmv ctbsv dsbmv dtbsv sger
        stbmv zgemv zhpr chemv ctpmv dspmv dtpmv ssbmv stbsv zgerc
        zhpr2 cher ctpsv dspr dtpsv sspmv stpmv zgeru ztbmv cher2
        ctrmv dspr2 dtrmv sspr stpsv zhbmv ztbsv
        '''
        blas3 = '''
        cgemm csymm ctrsm dsyrk sgemm strmm zhemm zsyr2k chemm csyr2k
        dgemm dtrmm ssymm strsm zher2k zsyrk cher2k csyrk dsymm dtrsm
        ssyr2k zherk ztrmm cherk ctrmm dsyr2k ssyrk zgemm zsymm ztrsm
        '''
        sources = [os.path.join(src_dir,f+'.f') \
                   for f in (blas1+blas2+blas3).split()]
        #XXX: should we check here actual existence of source files?
        info = {'sources':sources,'language':'f77'}
        self.set_info(**info)

class x11_info(system_info):
    section = 'x11'
    notfounderror = X11NotFoundError

    def __init__(self):
        system_info.__init__(self,
                             default_lib_dirs=default_x11_lib_dirs,
                             default_include_dirs=default_x11_include_dirs)

    def calc_info(self):
        if sys.platform  in ['win32']:
            return
        lib_dirs = self.get_lib_dirs()
        include_dirs = self.get_include_dirs()
        x11_libs = self.get_libs('x11_libs', ['X11'])
        for lib_dir in lib_dirs:
            info = self.check_libs(lib_dir, x11_libs, [])
            if info is not None:
                break
        else:
            return
        inc_dir = None
        for d in include_dirs:
            if self.combine_paths(d, 'X11/X.h'):
                inc_dir = d
                break
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir])
        self.set_info(**info)

class numpy_info(system_info):
    section = 'numpy'
    modulename = 'Numeric'
    notfounderror = NumericNotFoundError

    def __init__(self):
        from distutils.sysconfig import get_python_inc
        include_dirs = []
        try:
            module = __import__(self.modulename)
            prefix = []
            for name in module.__file__.split(os.sep):
                if name=='lib':
                    break
                prefix.append(name)
            include_dirs.append(get_python_inc(prefix=os.sep.join(prefix)))
        except ImportError:
            pass
        py_incl_dir = get_python_inc()
        include_dirs.append(py_incl_dir)
        for d in default_include_dirs:
            d = os.path.join(d, os.path.basename(py_incl_dir))
            if d not in include_dirs:
                include_dirs.append(d)
        system_info.__init__(self,
                             default_lib_dirs=[],
                             default_include_dirs=include_dirs)

    def calc_info(self):
        try:
            module = __import__(self.modulename)
        except ImportError:
            return
        info = {}
        macros = [(self.modulename.upper()+'_VERSION',
                   '"\\"%s\\""' % (module.__version__)),
                  (self.modulename.upper(),None)]
##         try:
##             macros.append(
##                 (self.modulename.upper()+'_VERSION_HEX',
##                  hex(vstr2hex(module.__version__))),
##                 )
##         except Exception,msg:
##             print msg
        dict_append(info, define_macros = macros)
        include_dirs = self.get_include_dirs()
        inc_dir = None
        for d in include_dirs:
            if self.combine_paths(d,
                                  os.path.join(self.modulename,
                                               'arrayobject.h')):
                inc_dir = d
                break
        if inc_dir is not None:
            dict_append(info, include_dirs=[inc_dir])
        if info:
            self.set_info(**info)
        return

class numarray_info(numpy_info):
    section = 'numarray'
    modulename = 'numarray'

class boost_python_info(system_info):
    section = 'boost_python'
    dir_env_var = 'BOOST'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d,['boost*']))
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        from distutils.sysconfig import get_python_inc
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d,'libs','python','src','module.cpp')):
                src_dir = d
                break
        if not src_dir:
            return
        py_incl_dir = get_python_inc()
        srcs_dir = os.path.join(src_dir,'libs','python','src')
        bpl_srcs = glob(os.path.join(srcs_dir,'*.cpp'))
        bpl_srcs += glob(os.path.join(srcs_dir,'*','*.cpp'))
        info = {'libraries':[('boost_python_src',{'include_dirs':[src_dir,py_incl_dir],
                                                  'sources':bpl_srcs})],
                'include_dirs':[src_dir],
                }
        if info:
            self.set_info(**info)
        return

class agg2_info(system_info):
    section = 'agg2'
    dir_env_var = 'AGG2'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d,['agg2*']))
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d,'src','agg_affine_matrix.cpp')):
                src_dir = d
                break
        if not src_dir:
            return
        if sys.platform=='win32':
            agg2_srcs = glob(os.path.join(src_dir,'src','platform','win32','agg_win32_bmp.cpp'))
        else:
            agg2_srcs = glob(os.path.join(src_dir,'src','*.cpp'))
            agg2_srcs += [os.path.join(src_dir,'src','platform','X11','agg_platform_support.cpp')]
        
        info = {'libraries':[('agg2_src',{'sources':agg2_srcs,
                                          'include_dirs':[os.path.join(src_dir,'include')],
                                          })],
                'include_dirs':[os.path.join(src_dir,'include')],
                }
        if info:
            self.set_info(**info)
        return

class _pkg_config_info(system_info):
    section = None
    config_env_var = 'PKG_CONFIG'
    default_config_exe = 'pkg-config'
    append_config_exe = ''
    version_macro_name = None
    release_macro_name = None
    version_flag = '--modversion'
    cflags_flag = '--cflags'

    def get_config_exe(self):
        if os.environ.has_key(self.config_env_var):
            return os.environ[self.config_env_var]
        return self.default_config_exe
    def get_config_output(self, config_exe, option):
        s,o = exec_command(config_exe+' '+self.append_config_exe+' '+option,use_tee=0)
        if not s:
            return o

    def calc_info(self):
        config_exe = find_executable(self.get_config_exe())
        if not os.path.isfile(config_exe):
            print 'File not found: %s. Cannot determine %s info.' \
                  % (config_exe, self.section)
            return
        info = {}
        macros = []
        libraries = []
        library_dirs = []
        include_dirs = []
        extra_link_args = []
        extra_compile_args = []
        version = self.get_config_output(config_exe,self.version_flag)
        if version:
            macros.append((self.__class__.__name__.split('.')[-1].upper(),
                           '"\\"%s\\""' % (version)))
            if self.version_macro_name:
                macros.append((self.version_macro_name+'_%s' % (version.replace('.','_')),None))
        if self.release_macro_name:
            release = self.get_config_output(config_exe,'--release')
            if release:
                macros.append((self.release_macro_name+'_%s' % (release.replace('.','_')),None))
        opts = self.get_config_output(config_exe,'--libs')
        if opts:
            for opt in opts.split():
                if opt[:2]=='-l':
                    libraries.append(opt[2:])
                elif opt[:2]=='-L':
                    library_dirs.append(opt[2:])
                else:
                    extra_link_args.append(opt)
        opts = self.get_config_output(config_exe,self.cflags_flag)
        if opts:
            for opt in opts.split():
                if opt[:2]=='-I':
                    include_dirs.append(opt[2:])
                elif opt[:2]=='-D':
                    if '=' in opt:
                        n,v = opt[2:].split('=')
                        macros.append((n,v))
                    else:
                        macros.append((opt[2:],None))
                else:
                    extra_compile_args.append(opt)
        if macros: dict_append(info, define_macros = macros)
        if libraries: dict_append(info, libraries = libraries)
        if library_dirs: dict_append(info, library_dirs = library_dirs)
        if include_dirs: dict_append(info, include_dirs = include_dirs)
        if extra_link_args: dict_append(info, extra_link_args = extra_link_args)
        if extra_compile_args: dict_append(info, extra_compile_args = extra_compile_args)
        if info:
            self.set_info(**info)
        return

class wx_info(_pkg_config_info):
    section = 'wx'
    config_env_var = 'WX_CONFIG'
    default_config_exe = 'wx-config'
    append_config_exe = ''
    version_macro_name = 'WX_VERSION'
    release_macro_name = 'WX_RELEASE'
    version_flag = '--version'
    cflags_flag = '--cxxflags'

class gdk_pixbuf_xlib_2_info(_pkg_config_info):
    section = 'gdk_pixbuf_xlib_2'
    append_config_exe = 'gdk-pixbuf-xlib-2.0'
    version_macro_name = 'GDK_PIXBUF_XLIB_VERSION'

class gdk_pixbuf_2_info(_pkg_config_info):
    section = 'gdk_pixbuf_2'
    append_config_exe = 'gdk-pixbuf-2.0'
    version_macro_name = 'GDK_PIXBUF_VERSION'

class gdk_x11_2_info(_pkg_config_info):
    section = 'gdk_x11_2'
    append_config_exe = 'gdk-x11-2.0'
    version_macro_name = 'GDK_X11_VERSION'

class gdk_2_info(_pkg_config_info):
    section = 'gdk_2'
    append_config_exe = 'gdk-2.0'
    version_macro_name = 'GDK_VERSION'

class gdk_info(_pkg_config_info):
    section = 'gdk'
    append_config_exe = 'gdk'
    version_macro_name = 'GDK_VERSION'

class gtkp_x11_2_info(_pkg_config_info):
    section = 'gtkp_x11_2'
    append_config_exe = 'gtk+-x11-2.0'
    version_macro_name = 'GTK_X11_VERSION'


class gtkp_2_info(_pkg_config_info):
    section = 'gtkp_2'
    append_config_exe = 'gtk+-2.0'
    version_macro_name = 'GTK_VERSION'

class xft_info(_pkg_config_info):
    section = 'xft'
    append_config_exe = 'xft'
    version_macro_name = 'XFT_VERSION'

class freetype2_info(_pkg_config_info):
    section = 'freetype2'
    append_config_exe = 'freetype2'
    version_macro_name = 'FREETYPE2_VERSION'

## def vstr2hex(version):
##     bits = []
##     n = [24,16,8,4,0]
##     r = 0
##     for s in version.split('.'):
##         r |= int(s) << n[0]
##         del n[0]
##     return r

#--------------------------------------------------------------------

def combine_paths(*args,**kws):
    """ Return a list of existing paths composed by all combinations of
        items from arguments.
    """
    r = []
    for a in args:
        if not a: continue
        if type(a) is types.StringType:
            a = [a]
        r.append(a)
    args = r
    if not args: return []
    if len(args)==1:
        result = reduce(lambda a,b:a+b,map(glob,args[0]),[])
    elif len (args)==2:
        result = []
        for a0 in args[0]:
            for a1 in args[1]:
                result.extend(glob(os.path.join(a0,a1)))
    else:
        result = combine_paths(*(combine_paths(args[0],args[1])+args[2:]))
    verbosity = kws.get('verbosity',1)
    if verbosity>1 and result:
        print '(','paths:',','.join(result),')'
    return result

language_map = {'c':0,'c++':1,'f77':2,'f90':3}
inv_language_map = {0:'c',1:'c++',2:'f77',3:'f90'}
def dict_append(d,**kws):
    languages = []
    for k,v in kws.items():
        if k=='language':
            languages.append(v)
            continue
        if d.has_key(k):
            if k in ['library_dirs','include_dirs','define_macros']:
                [d[k].append(vv) for vv in v if vv not in d[k]]
            else:
                d[k].extend(v)
        else:
            d[k] = v
    if languages:
        l = inv_language_map[max([language_map.get(l,0) for l in languages])]
        d['language'] = l
    return

def show_all():
    import system_info
    import pprint
    match_info = re.compile(r'.*?_info').match
    show_only = []
    for n in sys.argv[1:]:
        if n[-5:] != '_info':
            n = n + '_info'
        show_only.append(n)
    show_all = not show_only
    for n in filter(match_info,dir(system_info)):
        if n in ['system_info','get_info']: continue
        if not show_all:
            if n not in show_only: continue
            del show_only[show_only.index(n)]
        c = getattr(system_info,n)()
        c.verbosity = 2
        r = c.get_info()
    if show_only:
        print 'Info classes not defined:',','.join(show_only)
if __name__ == "__main__":
    show_all()





