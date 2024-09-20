        from distutils.msvccompiler import MSVCCompiler
        ar_exe = MSVCCompiler().lib

    executables = {
        'version_cmd'  : [fc_exe, "-FI -V -c %(fname)s.f -o %(fname)s.o" \
                          % {'fname':dummy_fortran_file()}],
        'compiler_f77' : [fc_exe,"-FI","-w90","-w95"],
        'compiler_fix' : [fc_exe,"-FI","-4L72","-w"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : [fc_exe,"-shared"],
        'archiver'     : [ar_exe, "/verbose", "/OUT:"],
        'ranlib'       : None
        }

    compile_switch = '/c '
    object_switch = '/Fo' #No space after /Fo!

    def get_flags(self):
        opt = ['/nologo','/MD','/nbs','/Qlowercase','/us']
        return opt

    def get_flags_debug(self):
        return ['/4Yb','/d2']

    def get_flags_opt(self):
        return ['/O3','/Qip','/Qipo','/Qipo_obj']

    def get_flags_arch(self):
        opt = []
        if cpu.is_PentiumPro() or cpu.is_PentiumII():
            opt.extend(['/G6','/Qaxi'])
        elif cpu.is_PentiumIII():
            opt.extend(['/G6','/QaxK'])
        elif cpu.is_Pentium():
            opt.append('/G5')
        elif cpu.is_PentiumIV():
            opt.extend(['/G7','/QaxW'])
        if cpu.has_mmx():
            opt.append('/QaxM')
        return opt
    
if __name__ == '__main__':
