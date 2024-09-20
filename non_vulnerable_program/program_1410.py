from utils import jiffies


class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"


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
    return


def set_local_path(reldir='', level=1):
    """ Prepend local directory to sys.path.

    The caller is responsible for removing this path by using

      restore_path()
    """
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    local_path = os.path.join(os.path.dirname(os.path.abspath(testfile)),reldir)
    if DEBUG:
        print 'Inserting %r to sys.path' % (local_path)
    sys.path.insert(0,local_path)
    return


def restore_path():
    if DEBUG:
        print 'Removing %r from sys.path' % (sys.path[0])
    del sys.path[0]
    return


def output_exception():
    try:
        type, value, tb = sys.exc_info()
        info = traceback.extract_tb(tb)
        #this is more verbose
        #traceback.print_exc()
        filename, lineno, function, text = info[-1] # last line only
        print "%s:%d: %s: %s (in %s)" %\
              (filename, lineno, type.__name__, str(value), function)
    finally:
        type = value = tb = None # clean up
    return


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


class ScipyTestCase (unittest.TestCase):

    def measure(self,code_str,times=1):
        """ Return elapsed time for executing code_str in the
        namespace of the caller for given times.
        """
        frame = get_frame(1)
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

def _get_all_method_names(cls):
    names = dir(cls)
    if sys.version[:3]<='2.1':
        for b in cls.__bases__:
            for n in dir(b)+_get_all_method_names(b):
                if n not in names:
                    names.append(n)
    return names


# for debug build--check for memory leaks during the test.
class _SciPyTextTestResult(unittest._TextTestResult):
    def startTest(self, test):
        unittest._TextTestResult.startTest(self, test)
        if self.showAll:
            N = len(sys.getobjects(0))
            self._totnumobj = N
            self._totrefcnt = sys.gettotalrefcount()
        return

    def stopTest(self, test):
        if self.showAll:
            N = len(sys.getobjects(0))
            self.stream.write("objects: %d ===> %d;   " % (self._totnumobj, N))
            self.stream.write("refcnts: %d ===> %d\n" % (self._totrefcnt,
                              sys.gettotalrefcount()))
        return

class SciPyTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return _SciPyTextTestResult(self.stream, self.descriptions, self.verbosity)
    

class ScipyTest:
    """ Scipy tests site manager.

    Usage:
      >>> ScipyTest(<package>).test(level=1,verbosity=1)

    <package> is package name or its module object.

    Package is supposed to contain a directory tests/
    with test_*.py files where * refers to the names of submodules.

    test_*.py files are supposed to define a classes, derived
