from cpuinfo import cpu
from fcompiler import FCompiler

class PGroupFCompiler(FCompiler):

    compiler_type = 'pg'
    version_pattern =  r'\s*pg(f77|f90|hpf) (?P<version>[\d.-]+).*'

    executables = {
        'version_cmd'  : ["pgf77", "-V 2>/dev/null"],
        'compiler_f77' : ["pgf77"],
        'compiler_fix' : ["pgf90", "-Mfixed"],
        'compiler_f90' : ["pgf90"],
        'linker_so'    : ["pgf90","-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_flags(self):
        opt = ['-Minform=inform','-Mnosecond_underscore']
        opt.append('-fpic')
        return opt
    def get_flags_opt(self):
        return ['-fast']
    def get_flags_debug(self):
        return ['-g']

if __name__ == '__main__':
