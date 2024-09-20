        from distutils.msvccompiler import MSVCCompiler
        m = MSVCCompiler()
        m.initialize()
        ar_exe = m.lib

    executables = {
        'version_cmd'  : ['<F90>', "/what"],
        'compiler_f77' : [fc_exe, "/f77rtl","/fixed"],
        'compiler_fix' : [fc_exe, "/fixed"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : ['<F90>'],
        'archiver'     : [ar_exe, "/OUT:"],
        'ranlib'       : None
        }

    def get_flags(self):
        return ['/nologo','/MD','/WX','/iface=(cref,nomixed_str_len_arg)',
                '/names:lowercase','/assume:underscore']
    def get_flags_opt(self):
        return ['/Ox','/fast','/optimize:5','/unroll:0','/math_library:fast']
    def get_flags_arch(self):
        return ['/threads']
    def get_flags_debug(self):
        return ['/debug']

if __name__ == '__main__':
