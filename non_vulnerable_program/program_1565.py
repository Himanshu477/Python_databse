import log
from exec_command import exec_command
from misc_util import cyg2win32, is_sequence, mingw32
from distutils.spawn import _nt_quote_args

# hack to set compiler optimizing options. Needs to integrated with something.
