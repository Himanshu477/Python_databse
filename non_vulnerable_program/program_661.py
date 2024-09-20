import os,sys,time,glob,string,traceback,unittest
import types
import imp

#
# Imports from scipy_base must be done at the end of this file.
#

DEBUG = 0

__all__.append('set_package_path')
def set_package_path(level=1):
    """ Prepend package directory to sys.path.

    set_package_path should be called from a test_file.py that
    satisfies the following tree structure:

      <somepath>/<somedir>/test_file.py

    Then the first existing path name from the following list

      <somepath>/build/lib.<platform>-<version>
      <somepath>/..

    is prepended to sys.path.
    The caller is responsible for removing this path by using

      restore_path()
    """
    from distutils.util import get_platform
    from scipy.distutils.misc_util import get_frame
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    d = os.path.dirname(os.path.dirname(os.path.abspath(testfile)))
    d1 = os.path.join(d,'build','lib.%s-%s'%(get_platform(),sys.version[:3]))
    if not os.path.isdir(d1):
        d1 = os.path.dirname(d)
    if DEBUG:
        print 'Inserting %r to sys.path' % (d1)
    sys.path.insert(0,d1)

__all__.append('set_local_path')
def set_local_path(reldir='', level=1):
    """ Prepend local directory to sys.path.

    The caller is responsible for removing this path by using

      restore_path()
    """
    from scipy.distutils.misc_util import get_frame
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    local_path = os.path.join(os.path.dirname(os.path.abspath(testfile)),reldir)
    if DEBUG:
        print 'Inserting %r to sys.path' % (local_path)
    sys.path.insert(0,local_path)

__all__.append('restore_path')
def restore_path():
    if DEBUG:
        print 'Removing %r from sys.path' % (sys.path[0])
    del sys.path[0]

__all__.extend(['jiffies','memusage'])
if sys.platform[:5]=='linux':
    def jiffies(_proc_pid_stat = '/proc/%s/stat'%(os.getpid()),
                _load_time=time.time()):
        """ Return number of jiffies (1/100ths of a second) that this
    process has been scheduled in user mode. See man 5 proc. """
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[13])
        except:
            return int(100*(time.time()-_load_time))

    def memusage(_proc_pid_stat = '/proc/%s/stat'%(os.getpid())):
        """ Return virtual memory size in bytes of the running python.
        """
        try:
            f=open(_proc_pid_stat,'r')
            l = f.readline().split(' ')
            f.close()
            return int(l[22])
        except:
            return
else:
    # os.getpid is not in all platforms available.
    # Using time is safe but inaccurate, especially when process
    # was suspended or sleeping.
    def jiffies(_load_time=time.time()):
        """ Return number of jiffies (1/100ths of a second) that this
    process has been scheduled in user mode. [Emulation with time.time]. """
        return int(100*(time.time()-_load_time))

    def memusage():
        """ Return memory usage of running python. [Not implemented]"""
        return

__all__.append('ScipyTestCase')
class ScipyTestCase (unittest.TestCase):

    def measure(self,code_str,times=1):
        """ Return elapsed time for executing code_str in the
        namespace of the caller for given times.
        """
        frame = sys._getframe(1)
        locs,globs = frame.f_locals,frame.f_globals
        code = compile(code_str,
                       'ScipyTestCase runner for '+self.__class__.__name__,
                       'exec')
        i = 0
        elapsed = jiffies()
        while i<times:
            i += 1
            exec code in globs,locs
        elapsed = jiffies() - elapsed
        return 0.01*elapsed

    def __call__(self, result=None):
        if result is None:
            return unittest.TestCase.__call__(self, result)

        nof_errors = len(result.errors)
        save_stream = result.stream
        result.stream = _dummy_stream(save_stream)
        unittest.TestCase.__call__(self, result)
        if nof_errors != len(result.errors):
            test, errstr = result.errors[-1]
            if type(errstr) is type(()):
                errstr = str(errstr[0])
            else:
                errstr = errstr.split('\n')[-2]
            l = len(result.stream.data)
            if errstr.startswith('IgnoreException:'):
                if l==1:
                    assert result.stream.data[-1]=='E',`result.stream.data`
                    result.stream.data[-1] = 'i'
                else:
                    assert result.stream.data[-1]=='ERROR\n',`result.stream.data`
                    result.stream.data[-1] = 'ignoring\n'
                del result.errors[-1]
        map(save_stream.write, result.stream.data)
        result.stream = save_stream

class _dummy_stream:
    def __init__(self,stream):
        self.data = []
        self.stream = stream
    def write(self,message):
        if not self.data and not message.startswith('E'):
            self.stream.write(message)
            self.stream.flush()
            message = ''
        self.data.append(message)
    def writeln(self,message):
        self.write(message+'\n')

__all__.append('IgnoreException')
class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"

#------------

def _get_all_method_names(cls):
    names = dir(cls)
    if sys.version[:3]<='2.1':
        for b in cls.__bases__:
            for n in dir(b)+_get_all_method_names(b):
                if n not in names:
                    names.append(n)
    return names

__all__.append('ScipyTest')
class ScipyTest:
    """ Scipy tests site manager.

    Usage:
      >>> ScipyTest(<package>).test(level=1,verbosity=2)

    <package> is package name or its module object.

    Package is supposed to contain a directory tests/
    with test_*.py files where * refers to the names of submodules.

    test_*.py files are supposed to define a classes, derived
