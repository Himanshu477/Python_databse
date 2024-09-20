from distutils.sysconfig import get_config_vars

if sys.platform == 'win32':
    default_lib_dirs = ['C:\\'] # probably not very helpful...
    default_include_dirs = []
    default_src_dirs = ['.']
    default_x11_lib_dirs = []
    default_x11_include_dirs = []
else:
    default_lib_dirs = ['/usr/local/lib', '/opt/lib', '/usr/lib',
                        '/sw/lib']
    default_include_dirs = ['/usr/local/include',
                            '/opt/include', '/usr/include',
                            '/sw/include']
    default_src_dirs = ['.','/usr/local/src', '/opt/src','/sw/src']
    default_x11_lib_dirs = ['/usr/X11R6/lib','/usr/X11/lib','/usr/lib']
    default_x11_include_dirs = ['/usr/X11R6/include','/usr/X11/include',
                                '/usr/include']

if os.path.join(sys.prefix, 'lib') not in default_lib_dirs:
    default_lib_dirs.insert(0,os.path.join(sys.prefix, 'lib'))
    default_include_dirs.append(os.path.join(sys.prefix, 'include'))
    default_src_dirs.append(os.path.join(sys.prefix, 'src'))

default_lib_dirs = filter(os.path.isdir, default_lib_dirs)
default_include_dirs = filter(os.path.isdir, default_include_dirs)
default_src_dirs = filter(os.path.isdir, default_src_dirs)

so_ext = get_config_vars('SO')[0] or ''

def get_info(name,notfound_action=0):
    """
    notfound_action:
      0 - do nothing
      1 - display warning message
      2 - raise error
    """
    cl = {'atlas':atlas_info,  # use lapack_opt or blas_opt instead
          'atlas_threads':atlas_threads_info,                # ditto
          'atlas_blas':atlas_blas_info,
          'atlas_blas_threads':atlas_blas_threads_info,
          'lapack_atlas':lapack_atlas_info,  # use lapack_opt instead
          'lapack_atlas_threads':lapack_atlas_threads_info,  # ditto
          'x11':x11_info,
          'fftw':fftw_info,
          'dfftw':dfftw_info,
          'sfftw':sfftw_info,
          'fftw_threads':fftw_threads_info,
          'dfftw_threads':dfftw_threads_info,
          'sfftw_threads':sfftw_threads_info,
          'djbfft':djbfft_info,
          'blas':blas_info,                  # use blas_opt instead
          'lapack':lapack_info,              # use lapack_opt instead
          'lapack_src':lapack_src_info,
          'blas_src':blas_src_info,
          'numpy':numpy_info,
          'numeric':numpy_info, # alias to numpy, for build_ext --backends support
          'numarray':numarray_info,
          'lapack_opt':lapack_opt_info,
          'blas_opt':blas_opt_info,
          'boost_python':boost_python_info,
          'agg2':agg2_info,
          'wx':wx_info,
          'gdk_pixbuf_xlib_2':gdk_pixbuf_xlib_2_info,
          'gdk-pixbuf-xlib-2.0':gdk_pixbuf_xlib_2_info,
          'gdk_pixbuf_2':gdk_pixbuf_2_info,
          'gdk-pixbuf-2.0':gdk_pixbuf_2_info,
          'gdk':gdk_info,
          'gdk_2':gdk_2_info,
          'gdk-2.0':gdk_2_info,
          'gdk_x11_2':gdk_x11_2_info,
          'gdk-x11-2.0':gdk_x11_2_info,
          'gtkp_x11_2':gtkp_x11_2_info,
          'gtk+-x11-2.0':gtkp_x11_2_info,
          'gtkp_2':gtkp_2_info,
          'gtk+-2.0':gtkp_2_info,
          'xft':xft_info,
          'freetype2':freetype2_info,
          }.get(name.lower(),system_info)
    return cl().get_info(notfound_action)

class NotFoundError(DistutilsError):
    """Some third-party program or library is not found."""

class AtlasNotFoundError(NotFoundError):
    """
    Atlas (http://math-atlas.sourceforge.net/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [atlas]) or by setting
    the ATLAS environment variable."""

class LapackNotFoundError(NotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [lapack]) or by setting
    the LAPACK environment variable."""

class LapackSrcNotFoundError(LapackNotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) sources not found.
    Directories to search for the sources can be specified in the
    scipy_distutils/site.cfg file (section [lapack_src]) or by setting
    the LAPACK_SRC environment variable."""

class BlasNotFoundError(NotFoundError):
    """
    Blas (http://www.netlib.org/blas/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [blas]) or by setting
    the BLAS environment variable."""

class BlasSrcNotFoundError(BlasNotFoundError):
    """
    Blas (http://www.netlib.org/blas/) sources not found.
    Directories to search for the sources can be specified in the
    scipy_distutils/site.cfg file (section [blas_src]) or by setting
    the BLAS_SRC environment variable."""

class FFTWNotFoundError(NotFoundError):
    """
    FFTW (http://www.fftw.org/) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [fftw]) or by setting
    the FFTW environment variable."""

class DJBFFTNotFoundError(NotFoundError):
    """
    DJBFFT (http://cr.yp.to/djbfft.html) libraries not found.
    Directories to search for the libraries can be specified in the
    scipy_distutils/site.cfg file (section [djbfft]) or by setting
    the DJBFFT environment variable."""

class F2pyNotFoundError(NotFoundError):
    """
    f2py2e (http://cens.ioc.ee/projects/f2py2e/) module not found.
    Get it from above location, install it, and retry setup.py."""

class NumericNotFoundError(NotFoundError):
    """
    Numeric (http://www.numpy.org/) module not found.
    Get it from above location, install it, and retry setup.py."""

class X11NotFoundError(NotFoundError):
    """X11 libraries not found."""

class system_info:

    """ get_info() is the only public method. Don't use others.
    """
    section = 'DEFAULT'
    dir_env_var = None
    search_static_first = 0 # XXX: disabled by default, may disappear in
                            # future unless it is proved to be useful.
    verbosity = 1
    saved_results = {}

    notfounderror = NotFoundError

    def __init__ (self,
                  default_lib_dirs=default_lib_dirs,
                  default_include_dirs=default_include_dirs,
                  verbosity = 1,
                  ):
        self.__class__.info = {}
        self.local_prefixes = []
        defaults = {}
        defaults['libraries'] = ''
        defaults['library_dirs'] = os.pathsep.join(default_lib_dirs)
        defaults['include_dirs'] = os.pathsep.join(default_include_dirs)
        defaults['src_dirs'] = os.pathsep.join(default_src_dirs)
        defaults['search_static_first'] = str(self.search_static_first)
        self.cp = ConfigParser.ConfigParser(defaults)
        try:
            f = __file__
        except NameError,msg:
            f = sys.argv[0]
        cf = os.path.join(os.path.split(os.path.abspath(f))[0],
                          'site.cfg')
        self.cp.read([cf])
        if not self.cp.has_section(self.section):
            self.cp.add_section(self.section)
        self.search_static_first = self.cp.getboolean(self.section,
                                                      'search_static_first')
        assert isinstance(self.search_static_first, type(0))

    def calc_libraries_info(self):
        libs = self.get_libraries()
        dirs = self.get_lib_dirs()
        info = {}
        for lib in libs:
            i = None
            for d in dirs:
                i = self.check_libs(d,[lib])        
                if i is not None:
                    break
            if i is not None:
                dict_append(info,**i)
            else:
                print 'Library %s was not found. Ignoring' % (lib)
        return info

    def set_info(self,**info):
        if info:       
            lib_info = self.calc_libraries_info()
            dict_append(info,**lib_info)
        self.saved_results[self.__class__.__name__] = info

    def has_info(self):
        return self.saved_results.has_key(self.__class__.__name__)

    def get_info(self,notfound_action=0):
        """ Return a dictonary with items that are compatible
            with scipy_distutils.setup keyword arguments.
        """
        flag = 0
        if not self.has_info():
            flag = 1
            if self.verbosity>0:
                print self.__class__.__name__ + ':'
            if hasattr(self, 'calc_info'):
                self.calc_info()
            if notfound_action:
                if not self.has_info():
                    if notfound_action==1:
                        warnings.warn(self.notfounderror.__doc__)
                    elif notfound_action==2:
                        raise self.notfounderror,self.notfounderror.__doc__
                    else:
                        raise ValueError,`notfound_action`

            if self.verbosity>0:
                if not self.has_info():
                    print '  NOT AVAILABLE'
                    self.set_info()
                else:
                    print '  FOUND:'
            
        res = self.saved_results.get(self.__class__.__name__)
        if self.verbosity>0 and flag:
            for k,v in res.items():
                v = str(v)
                if k=='sources' and len(v)>200: v = v[:60]+' ...\n... '+v[-60:]
                print '    %s = %s'%(k,v)
            print
        
        return res

    def get_paths(self, section, key):
        dirs = self.cp.get(section, key).split(os.pathsep)
        env_var = self.dir_env_var
        if env_var:
            if type(env_var) is type([]):
                e0 = env_var[-1]
                for e in env_var:
                    if os.environ.has_key(e):
                        e0 = e
                        break
                if not env_var[0]==e0:
                    print 'Setting %s=%s' % (env_var[0],e0)
                env_var = e0
        if env_var and os.environ.has_key(env_var):
            d = os.environ[env_var]
            if d=='None':
                print 'Disabled',self.__class__.__name__,'(%s is None)' \
                      % (self.dir_env_var)
                return []
            if os.path.isfile(d):
                dirs = [os.path.dirname(d)] + dirs
                l = getattr(self,'_lib_names',[])
                if len(l)==1:
                    b = os.path.basename(d)
                    b = os.path.splitext(b)[0]
                    if b[:3]=='lib':
                        print 'Replacing _lib_names[0]==%r with %r' \
                              % (self._lib_names[0], b[3:])
                        self._lib_names[0] = b[3:]
            else:
                ds = d.split(os.pathsep)
                ds2 = []
                for d in ds:
                    if os.path.isdir(d):
                        ds2.append(d)
                        for dd in ['include','lib']:
                            d1 = os.path.join(d,dd)
                            if os.path.isdir(d1):
                                ds2.append(d1)
                dirs = ds2 + dirs
        default_dirs = self.cp.get('DEFAULT', key).split(os.pathsep)
        dirs.extend(default_dirs)
        ret = []
        [ret.append(d) for d in dirs if os.path.isdir(d) and d not in ret]
        if self.verbosity>1:
            print '(',key,'=',':'.join(ret),')'
        return ret

    def get_lib_dirs(self, key='library_dirs'):
        return self.get_paths(self.section, key)

    def get_include_dirs(self, key='include_dirs'):
        return self.get_paths(self.section, key)

    def get_src_dirs(self, key='src_dirs'):
        return self.get_paths(self.section, key)

    def get_libs(self, key, default):
        try:
            libs = self.cp.get(self.section, key)
        except ConfigParser.NoOptionError:
            if not default:
                return []
            if type(default) is type(''):
                return [default]
            return default
        return [b for b in [a.strip() for a in libs.split(',')] if b]

    def get_libraries(self, key='libraries'):
        return self.get_libs(key,'')

    def check_libs(self,lib_dir,libs,opt_libs =[]):
        """ If static or shared libraries are available then return
            their info dictionary. """
        if self.search_static_first:
            exts = ['.a',so_ext]
        else:
            exts = [so_ext,'.a']
        if sys.platform=='cygwin':
            exts.append('.dll.a')
        for ext in exts:
            info = self._check_libs(lib_dir,libs,opt_libs,ext)
            if info is not None: return info
        return

    def _lib_list(self, lib_dir, libs, ext):
        assert type(lib_dir) is type('')
        liblist = []
        for l in libs:
            p = self.combine_paths(lib_dir, 'lib'+l+ext)
            if p:
                assert len(p)==1
                liblist.append(p[0])
        return liblist

    def _extract_lib_names(self,libs):
        return [os.path.splitext(os.path.basename(p))[0][3:] \
                for p in libs]

    def _check_libs(self,lib_dir,libs, opt_libs, ext):
        found_libs = self._lib_list(lib_dir, libs, ext)
        if len(found_libs) == len(libs):
            found_libs = self._extract_lib_names(found_libs)
            info = {'libraries' : found_libs, 'library_dirs' : [lib_dir]}
            opt_found_libs = self._lib_list(lib_dir, opt_libs, ext)
            if len(opt_found_libs) == len(opt_libs):
                opt_found_libs = self._extract_lib_names(opt_found_libs)
                info['libraries'].extend(opt_found_libs)
            return info

    def combine_paths(self,*args):
        return combine_paths(*args,**{'verbosity':self.verbosity})

class fftw_info(system_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['rfftw', 'fftw']
    includes = ['fftw.h','rfftw.h']
    macros = [('SCIPY_FFTW_H',None)]
    notfounderror = FFTWNotFoundError

    def __init__(self):
        system_info.__init__(self)

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        incl_dir = None
        libs = self.get_libs(self.section+'_libs', self.libs)
        info = None
        for d in lib_dirs:
            r = self.check_libs(d,libs)
            if r is not None:
                info = r
                break
        if info is not None:
            flag = 0
            for d in incl_dirs:
                if len(self.combine_paths(d,self.includes))==2:
                    dict_append(info,include_dirs=[d])
                    flag = 1
                    incl_dirs = [d]
                    incl_dir = d
                    break
            if flag:
                dict_append(info,define_macros=self.macros)
            else:
                info = None
        if info is not None:
            self.set_info(**info)

class dfftw_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['drfftw','dfftw']
    includes = ['dfftw.h','drfftw.h']
    macros = [('SCIPY_DFFTW_H',None)]

class sfftw_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['srfftw','sfftw']
    includes = ['sfftw.h','srfftw.h']
    macros = [('SCIPY_SFFTW_H',None)]

class fftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['rfftw_threads','fftw_threads']
    includes = ['fftw_threads.h','rfftw_threads.h']
    macros = [('SCIPY_FFTW_THREADS_H',None)]

class dfftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['drfftw_threads','dfftw_threads']
    includes = ['dfftw_threads.h','drfftw_threads.h']
    macros = [('SCIPY_DFFTW_THREADS_H',None)]

class sfftw_threads_info(fftw_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    libs = ['srfftw_threads','sfftw_threads']
    includes = ['sfftw_threads.h','srfftw_threads.h']
    macros = [('SCIPY_SFFTW_THREADS_H',None)]

class djbfft_info(system_info):
    section = 'djbfft'
    dir_env_var = 'DJBFFT'
    notfounderror = DJBFFTNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend(self.combine_paths(d,['djbfft'])+[d])
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        info = None
        for d in lib_dirs:
            p = self.combine_paths (d,['djbfft.a'])
            if p:
                info = {'extra_objects':p}
                break
            p = self.combine_paths (d,['libdjbfft.a'])
            if p:
                info = {'libraries':['djbfft'],'library_dirs':[d]}
                break
        if info is None:
            return
        for d in incl_dirs:
            if len(self.combine_paths(d,['fftc8.h','fftfreq.h']))==2:
                dict_append(info,include_dirs=[d],
                            define_macros=[('SCIPY_DJBFFT_H',None)])
                self.set_info(**info)
                return
        return

class atlas_info(system_info):
    section = 'atlas'
    dir_env_var = 'ATLAS'
    _lib_names = ['f77blas','cblas']
    if sys.platform[:7]=='freebsd':
        _lib_atlas = ['atlas_r']
        _lib_lapack = ['alapack_r']
    else:
        _lib_atlas = ['atlas']
        _lib_lapack = ['lapack']

    notfounderror = AtlasNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend(self.combine_paths(d,['atlas*','ATLAS*',
                                         'sse','3dnow','sse2'])+[d])
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        info = {}
        atlas_libs = self.get_libs('atlas_libs',
                                   self._lib_names + self._lib_atlas)
        lapack_libs = self.get_libs('lapack_libs',self._lib_lapack)
        atlas = None
        lapack = None
        atlas_1 = None
        for d in lib_dirs:
            atlas = self.check_libs(d,atlas_libs,[])
            lapack_atlas = self.check_libs(d,['lapack_atlas'],[])
            if atlas is not None:
                lib_dirs2 = self.combine_paths(d,['atlas*','ATLAS*'])+[d]
                for d2 in lib_dirs2:
                    lapack = self.check_libs(d2,lapack_libs,[])
                    if lapack is not None:
                        break
                else:
                    lapack = None
                if lapack is not None:
                    break
            if atlas:
                atlas_1 = atlas
        print self.__class__
        if atlas is None:
            atlas = atlas_1
        if atlas is None:
            return
        include_dirs = self.get_include_dirs()
        h = (self.combine_paths(lib_dirs+include_dirs,'cblas.h') or [None])[0]
        if h:
            h = os.path.dirname(h)
            dict_append(info,include_dirs=[h])
        info['language'] = 'c'
        if lapack is not None:
            dict_append(info,**lapack)
            dict_append(info,**atlas)
        elif 'lapack_atlas' in atlas['libraries']:
            dict_append(info,**atlas)
            dict_append(info,define_macros=[('ATLAS_WITH_LAPACK_ATLAS',None)])
            self.set_info(**info)
            return
        else:
            dict_append(info,**atlas)
            dict_append(info,define_macros=[('ATLAS_WITHOUT_LAPACK',None)])
            message = """
*********************************************************************
    Could not find lapack library within the ATLAS installation.
*********************************************************************
"""
            warnings.warn(message)
            self.set_info(**info)
            return
        
        # Check if lapack library is complete, only warn if it is not.
        lapack_dir = lapack['library_dirs'][0]
        lapack_name = lapack['libraries'][0]
        lapack_lib = None
        for e in ['.a',so_ext]:
            fn = os.path.join(lapack_dir,'lib'+lapack_name+e)
            if os.path.exists(fn):
                lapack_lib = fn
                break
        if lapack_lib is not None:
            sz = os.stat(lapack_lib)[6]
            if sz <= 4000*1024:
                message = """
*********************************************************************
    Lapack library (from ATLAS) is probably incomplete:
      size of %s is %sk (expected >4000k)

    Follow the instructions in the KNOWN PROBLEMS section of the file
    scipy/INSTALL.txt.
*********************************************************************
""" % (lapack_lib,sz/1024)
                warnings.warn(message)
            else:
                info['language'] = 'f77'

        self.set_info(**info)

class atlas_blas_info(atlas_info):
    _lib_names = ['f77blas','cblas']

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        info = {}
        atlas_libs = self.get_libs('atlas_libs',
                                   self._lib_names + self._lib_atlas)
        atlas = None
        for d in lib_dirs:
            atlas = self.check_libs(d,atlas_libs,[])
            if atlas is not None:
                break
        if atlas is None:
            return
        include_dirs = self.get_include_dirs()
        h = (self.combine_paths(lib_dirs+include_dirs,'cblas.h') or [None])[0]
        if h:
            h = os.path.dirname(h)
            dict_append(info,include_dirs=[h])
        info['language'] = 'c'

        dict_append(info,**atlas)

        self.set_info(**info)
        return


class atlas_threads_info(atlas_info):
    dir_env_var = ['PTATLAS','ATLAS']
    _lib_names = ['ptf77blas','ptcblas']

class atlas_blas_threads_info(atlas_blas_info):
    dir_env_var = ['PTATLAS','ATLAS']
    _lib_names = ['ptf77blas','ptcblas']

class lapack_atlas_info(atlas_info):
    _lib_names = ['lapack_atlas'] + atlas_info._lib_names

class lapack_atlas_threads_info(atlas_threads_info):
    _lib_names = ['lapack_atlas'] + atlas_threads_info._lib_names

class lapack_info(system_info):
    section = 'lapack'
    dir_env_var = 'LAPACK'
    _lib_names = ['lapack']
    notfounderror = LapackNotFoundError

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()

        lapack_libs = self.get_libs('lapack_libs', self._lib_names)
        for d in lib_dirs:
            lapack = self.check_libs(d,lapack_libs,[])
            if lapack is not None:
                info = lapack                
                break
        else:
            return
        info['language'] = 'f77'
        self.set_info(**info)

class lapack_src_info(system_info):
    section = 'lapack_src'
    dir_env_var = 'LAPACK_SRC'
    notfounderror = LapackSrcNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d,['LAPACK*/SRC','SRC']))
        return [ d for d in dirs if os.path.isdir(d) ]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d,'dgesv.f')):
                src_dir = d
                break
        if not src_dir:
            #XXX: Get sources from netlib. May be ask first.
            return
        # The following is extracted from LAPACK-3.0/SRC/Makefile
        allaux='''
        ilaenv ieeeck lsame lsamen xerbla
        ''' # *.f
        laux = '''
        bdsdc bdsqr disna labad lacpy ladiv lae2 laebz laed0 laed1
        laed2 laed3 laed4 laed5 laed6 laed7 laed8 laed9 laeda laev2
        lagtf lagts lamch lamrg lanst lapy2 lapy3 larnv larrb larre
        larrf lartg laruv las2 lascl lasd0 lasd1 lasd2 lasd3 lasd4
        lasd5 lasd6 lasd7 lasd8 lasd9 lasda lasdq lasdt laset lasq1
        lasq2 lasq3 lasq4 lasq5 lasq6 lasr lasrt lassq lasv2 pttrf
        stebz stedc steqr sterf
        ''' # [s|d]*.f
        lasrc = '''
        gbbrd gbcon gbequ gbrfs gbsv gbsvx gbtf2 gbtrf gbtrs gebak
        gebal gebd2 gebrd gecon geequ gees geesx geev geevx gegs gegv
        gehd2 gehrd gelq2 gelqf gels gelsd gelss gelsx gelsy geql2
        geqlf geqp3 geqpf geqr2 geqrf gerfs gerq2 gerqf gesc2 gesdd
        gesv gesvd gesvx getc2 getf2 getrf getri getrs ggbak ggbal
        gges ggesx ggev ggevx ggglm gghrd gglse ggqrf ggrqf ggsvd
        ggsvp gtcon gtrfs gtsv gtsvx gttrf gttrs gtts2 hgeqz hsein
        hseqr labrd lacon laein lags2 lagtm lahqr lahrd laic1 lals0
        lalsa lalsd langb lange langt lanhs lansb lansp lansy lantb
        lantp lantr lapll lapmt laqgb laqge laqp2 laqps laqsb laqsp
        laqsy lar1v lar2v larf larfb larfg larft larfx largv larrv
        lartv larz larzb larzt laswp lasyf latbs latdf latps latrd
        latrs latrz latzm lauu2 lauum pbcon pbequ pbrfs pbstf pbsv
        pbsvx pbtf2 pbtrf pbtrs pocon poequ porfs posv posvx potf2
        potrf potri potrs ppcon ppequ pprfs ppsv ppsvx pptrf pptri
        pptrs ptcon pteqr ptrfs ptsv ptsvx pttrs ptts2 spcon sprfs
        spsv spsvx sptrf sptri sptrs stegr stein sycon syrfs sysv
        sysvx sytf2 sytrf sytri sytrs tbcon tbrfs tbtrs tgevc tgex2
        tgexc tgsen tgsja tgsna tgsy2 tgsyl tpcon tprfs tptri tptrs
        trcon trevc trexc trrfs trsen trsna trsyl trti2 trtri trtrs
        tzrqf tzrzf
        ''' # [s|c|d|z]*.f
        sd_lasrc = '''
        laexc lag2 lagv2 laln2 lanv2 laqtr lasy2 opgtr opmtr org2l
        org2r orgbr orghr orgl2 orglq orgql orgqr orgr2 orgrq orgtr
        orm2l orm2r ormbr ormhr orml2 ormlq ormql ormqr ormr2 ormr3
        ormrq ormrz ormtr rscl sbev sbevd sbevx sbgst sbgv sbgvd sbgvx
        sbtrd spev spevd spevx spgst spgv spgvd spgvx sptrd stev stevd
        stevr stevx syev syevd syevr syevx sygs2 sygst sygv sygvd
        sygvx sytd2 sytrd
        ''' # [s|d]*.f
        cz_lasrc = '''
        bdsqr hbev hbevd hbevx hbgst hbgv hbgvd hbgvx hbtrd hecon heev
        heevd heevr heevx hegs2 hegst hegv hegvd hegvx herfs hesv
        hesvx hetd2 hetf2 hetrd hetrf hetri hetrs hpcon hpev hpevd
        hpevx hpgst hpgv hpgvd hpgvx hprfs hpsv hpsvx hptrd hptrf
        hptri hptrs lacgv lacp2 lacpy lacrm lacrt ladiv laed0 laed7
        laed8 laesy laev2 lahef lanhb lanhe lanhp lanht laqhb laqhe
        laqhp larcm larnv lartg lascl laset lasr lassq pttrf rot spmv
        spr stedc steqr symv syr ung2l ung2r ungbr unghr ungl2 unglq
        ungql ungqr ungr2 ungrq ungtr unm2l unm2r unmbr unmhr unml2
        unmlq unmql unmqr unmr2 unmr3 unmrq unmrz unmtr upgtr upmtr
        ''' # [c|z]*.f
        #######
        sclaux = laux + ' econd '                  # s*.f
        dzlaux = laux + ' secnd '                  # d*.f
        slasrc = lasrc + sd_lasrc                  # s*.f
        dlasrc = lasrc + sd_lasrc                  # d*.f
        clasrc = lasrc + cz_lasrc + ' srot srscl ' # c*.f
        zlasrc = lasrc + cz_lasrc + ' drot drscl ' # z*.f
        oclasrc = ' icmax1 scsum1 '                # *.f
        ozlasrc = ' izmax1 dzsum1 '                # *.f
        sources = ['s%s.f'%f for f in (sclaux+slasrc).split()] \
                  + ['d%s.f'%f for f in (dzlaux+dlasrc).split()] \
                  + ['c%s.f'%f for f in (clasrc).split()] \
                  + ['z%s.f'%f for f in (zlasrc).split()] \
                  + ['%s.f'%f for f in (allaux+oclasrc+ozlasrc).split()]
        sources = [os.path.join(src_dir,f) for f in sources]
        #XXX: should we check here actual existence of source files?
        info = {'sources':sources,'language':'f77'}
        self.set_info(**info)

atlas_version_c_text = r'''
/* This file is generated from scipy_distutils/system_info.py */
#ifdef __CPLUSPLUS__
extern "C" {
#endif
#include "Python.h"
static PyMethodDef module_methods[] = { {NULL,NULL} };
DL_EXPORT(void) initatlas_version(void) {
  void ATL_buildinfo(void);
  ATL_buildinfo();
  Py_InitModule("atlas_version", module_methods);
}
#ifdef __CPLUSCPLUS__
}
#endif
'''

def get_atlas_version(**config):
    from core import Extension, setup
    from misc_util import get_build_temp
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
    extra_args = ['--build-lib',get_build_temp()]
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

