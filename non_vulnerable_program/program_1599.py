from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.misc_util import msvc_runtime_library

# the same as cygwin plus some additional parameters
class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    """ A modified MingW32 compiler compatible with an MSVC built Python.

    """

    compiler_type = 'mingw32'

    def __init__ (self,
                  verbose=0,
                  dry_run=0,
                  force=0):

        distutils.cygwinccompiler.CygwinCCompiler.__init__ (self,
                                                       verbose,dry_run, force)

        # we need to support 3.2 which doesn't match the standard
        # get_versions methods regex
        if self.gcc_version is None:
            import re
            out = os.popen('gcc -dumpversion','r')
            out_string = out.read()
            out.close()
            result = re.search('(\d+\.\d+)',out_string)
            if result:
                self.gcc_version = StrictVersion(result.group(1))

        # A real mingw32 doesn't need to specify a different entry point,
        # but cygwin 2.91.57 in no-cygwin-mode needs it.
        if self.gcc_version <= "2.91.57":
            entry_point = '--entry _DllMain@12'
        else:
            entry_point = ''

        if self.linker_dll == 'dllwrap':
            # Commented out '--driver-name g++' part that fixes weird
            #   g++.exe: g++: No such file or directory
            # error (mingw 1.0 in Enthon24 tree, gcc-3.4.5).
            # If the --driver-name part is required for some environment
            # then make the inclusion of this part specific to that environment.
            self.linker = 'dllwrap' #  --driver-name g++'
        elif self.linker_dll == 'gcc':
            self.linker = 'g++'

        # **changes: eric jones 4/11/01
        # 1. Check for import library on Windows.  Build if it doesn't exist.

        build_import_library()

        # **changes: eric jones 4/11/01
        # 2. increased optimization and turned off all warnings
        # 3. also added --driver-name g++
        #self.set_executables(compiler='gcc -mno-cygwin -O2 -w',
        #                     compiler_so='gcc -mno-cygwin -mdll -O2 -w',
        #                     linker_exe='gcc -mno-cygwin',
        #                     linker_so='%s --driver-name g++ -mno-cygwin -mdll -static %s'
        #                                % (self.linker, entry_point))
        if self.gcc_version <= "3.0.0":
            self.set_executables(compiler='gcc -mno-cygwin -O2 -w',
                                 compiler_so='gcc -mno-cygwin -mdll -O2 -w -Wstrict-prototypes',
                                 linker_exe='g++ -mno-cygwin',
                                 linker_so='%s -mno-cygwin -mdll -static %s'
                                 % (self.linker, entry_point))
        else:
            self.set_executables(compiler='gcc -mno-cygwin -O2 -Wall',
                                 compiler_so='gcc -mno-cygwin -O2 -Wall -Wstrict-prototypes',
                                 linker_exe='g++ -mno-cygwin',
                                 linker_so='g++ -mno-cygwin -shared')
        # added for python2.3 support
        # we can't pass it through set_executables because pre 2.2 would fail
        self.compiler_cxx = ['g++']

        # Maybe we should also append -mthreads, but then the finished
        # dlls need another dll (mingwm10.dll see Mingw32 docs)
        # (-mthreads: Support thread-safe exception handling on `Mingw32')

        # no additional libraries needed
        #self.dll_libraries=[]
        return

    # __init__ ()

    def link(self,
             target_desc,
             objects,
             output_filename,
             output_dir,
             libraries,
             library_dirs,
             runtime_library_dirs,
             export_symbols = None,
             debug=0,
             extra_preargs=None,
             extra_postargs=None,
             build_temp=None,
             target_lang=None):
        # Include the appropiate MSVC runtime library if Python was built
        # with MSVC >= 7.0 (MinGW standard is msvcrt)
        runtime_library = msvc_runtime_library()
        if runtime_library:
            if not libraries:
                libraries = []
            libraries.append(runtime_library)
        args = (self,
                target_desc,
                objects,
                output_filename,
                output_dir,
                libraries,
                library_dirs,
                runtime_library_dirs,
                None, #export_symbols, we do this in our def-file
                debug,
                extra_preargs,
                extra_postargs,
                build_temp,
                target_lang)
        if self.gcc_version < "3.0.0":
            func = distutils.cygwinccompiler.CygwinCCompiler.link
        else:
            func = UnixCCompiler.link
        func(*args[:func.im_func.func_code.co_argcount])
        return

    def object_filenames (self,
                          source_filenames,
                          strip_dir=0,
                          output_dir=''):
        if output_dir is None: output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            # use normcase to make sure '.rc' is really '.rc' and not '.RC'
            (base, ext) = os.path.splitext (os.path.normcase(src_name))

            # added these lines to strip off windows drive letters
            # without it, .o files are placed next to .c files
            # instead of the build directory
            drv,base = os.path.splitdrive(base)
            if drv:
                base = base[1:]

            if ext not in (self.src_extensions + ['.rc','.res']):
                raise UnknownFileError, \
                      "unknown file type '%s' (from '%s')" % \
                      (ext, src_name)
            if strip_dir:
                base = os.path.basename (base)
            if ext == '.res' or ext == '.rc':
                # these need to be compiled to object files
                obj_names.append (os.path.join (output_dir,
                                                base + ext + self.obj_extension))
            else:
                obj_names.append (os.path.join (output_dir,
                                                base + self.obj_extension))
        return obj_names

    # object_filenames ()


def build_import_library():
    """ Build the import libraries for Mingw32-gcc on Windows
    """
    if os.name != 'nt':
        return
    lib_name = "python%d%d.lib" % tuple(sys.version_info[:2])
    lib_file = os.path.join(sys.prefix,'libs',lib_name)
    out_name = "libpython%d%d.a" % tuple(sys.version_info[:2])
    out_file = os.path.join(sys.prefix,'libs',out_name)
    if not os.path.isfile(lib_file):
        log.warn('Cannot build import library: "%s" not found' % (lib_file))
        return
    if os.path.isfile(out_file):
        log.debug('Skip building import library: "%s" exists' % (out_file))
        return
    log.info('Building import library: "%s"' % (out_file))

    from numpy.distutils import lib2def

    def_name = "python%d%d.def" % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix,'libs',def_name)
    nm_cmd = '%s %s' % (lib2def.DEFAULT_NM, lib_file)
    nm_output = lib2def.getnm(nm_cmd)
    dlist, flist = lib2def.parse_nm(nm_output)
    lib2def.output_def(dlist, flist, lib2def.DEF_HEADER, open(def_file, 'w'))

    dll_name = "python%d%d.dll" % tuple(sys.version_info[:2])
    args = (dll_name,def_file,out_file)
    cmd = 'dlltool --dllname %s --def %s --output-lib %s' % args
    status = os.system(cmd)
    # for now, fail silently
    if status:
        log.warn('Failed to build import library for gcc. Linking will fail.')
    #if not success:
    #    msg = "Couldn't find import library, and failed to build it."
    #    raise DistutilsPlatformError, msg
    return


#!/bin/env python
"""
This file defines a set of system_info classes for getting
information about various resources (libraries, library directories,
include directories, etc.) in the system. Currently, the following
classes are available:

  atlas_info
  atlas_threads_info
  atlas_blas_info
  atlas_blas_threads_info
  lapack_atlas_info
  blas_info
  lapack_info
  blas_opt_info       # usage recommended
  lapack_opt_info     # usage recommended
  fftw_info,dfftw_info,sfftw_info
  fftw_threads_info,dfftw_threads_info,sfftw_threads_info
  djbfft_info
  x11_info
  lapack_src_info
  blas_src_info
  numpy_info
  numarray_info
  numpy_info
  boost_python_info
  agg2_info
  wx_info
  gdk_pixbuf_xlib_2_info
  gdk_pixbuf_2_info
  gdk_x11_2_info
  gtkp_x11_2_info
  gtkp_2_info
  xft_info
  freetype2_info
  umfpack_info

Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw','lapack','blas',
  'lapack_src', 'blas_src', etc. For a complete list of allowed names,
  see the definition of get_info() function below.

  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (system_info could not find it).

  Several *_info classes specify an environment variable to specify
  the locations of software. When setting the corresponding environment
  variable to 'None' then the software will be ignored, even when it
  is available in system.

Global parameters:
  system_info.search_static_first - search static libraries (.a)
             in precedence to shared ones (.so, .sl) if enabled.
  system_info.verbosity - output the results to stdout if enabled.

The file 'site.cfg' is looked for in

1) Directory of main setup.py file being run.
2) Home directory of user running the setup.py file (Not implemented yet)
3) System wide directory (location of this file...)

The first one found is used to get system configuration options The
format is that used by ConfigParser (i.e., Windows .INI style). The
section DEFAULT has options that are the default for each section. The
available sections are fftw, atlas, and x11. Appropiate defaults are
used if nothing is specified.

The order of finding the locations of resources is the following:
 1. environment variable
 2. section in site.cfg
 3. DEFAULT section in site.cfg
Only the first complete match is returned.

Example:
----------
[DEFAULT]
library_dirs = /usr/lib:/usr/local/lib:/opt/lib
include_dirs = /usr/include:/usr/local/include:/opt/include
src_dirs = /usr/local/src:/opt/src
# search static libraries (.a) in preference to shared ones (.so)
search_static_first = 0

[fftw]
fftw_libs = rfftw, fftw
fftw_opt_libs = rfftw_threaded, fftw_threaded
# if the above aren't found, look for {s,d}fftw_libs and {s,d}fftw_opt_libs

[atlas]
library_dirs = /usr/lib/3dnow:/usr/lib/3dnow/atlas
# for overriding the names of the atlas libraries
atlas_libs = lapack, f77blas, cblas, atlas

[x11]
library_dirs = /usr/X11R6/lib
include_dirs = /usr/X11R6/include
----------

Authors:
  Pearu Peterson <pearu@cens.ioc.ee>, February 2002
  David M. Cooke <cookedm@physics.mcmaster.ca>, April 2002

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

