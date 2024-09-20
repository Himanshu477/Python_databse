        from distutils.msvccompiler import MSVCCompiler
        ar_exe = MSVCCompiler().lib

    executables = {
        'version_cmd'  : ['DF', "/what"],
        'compiler_f77' : ['DF', "/f77rtl","/fixed"],
        'compiler_fix' : ['DF', "/fixed"],
        'compiler_f90' : ['DF'],
        'linker_so'    : ['DF'],
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
