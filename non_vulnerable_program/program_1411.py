    from ScipyTestCase or unittest.TestCase, with methods having
    names starting with test or bench or check.

    And that is it! No need to implement test or test_suite functions
    in each .py file.

    Also old styled test_suite(level=1) hooks are supported but
    soon to be removed.
    """
    def __init__(self, package=None):
        if package is None:
            from numpy.distutils.misc_util import get_frame
            f = get_frame(1)
            package = f.f_locals['__name__']
        self.package = package

    def _module_str(self, module):
        filename = module.__file__[-30:]
        if filename!=module.__file__:
            filename = '...'+filename
        return '<module %s from %s>' % (`module.__name__`, `filename`)

    def _get_method_names(self,clsobj,level):
        names = []
        for mthname in _get_all_method_names(clsobj):
            if mthname[:5] not in ['bench','check'] \
               and mthname[:4] not in ['test']:
                continue
            mth = getattr(clsobj, mthname)
            if type(mth) is not types.MethodType:
                continue
            d = mth.im_func.func_defaults
            if d is not None:
                mthlevel = d[0]
            else:
                mthlevel = 1
            if level>=mthlevel:
                if mthname not in names:
                    names.append(mthname)
            for base in clsobj.__bases__:
                for n in self._get_method_names(base,level):
                    if n not in names:
                        names.append(n)        
        return names

    def _get_module_tests(self,module,level,verbosity):
        mstr = self._module_str
        d,f = os.path.split(module.__file__)

        short_module_name = os.path.splitext(os.path.basename(f))[0]
        if short_module_name=='__init__':
            short_module_name = module.__name__.split('.')[-1]

        test_dir = os.path.join(d,'tests')
        test_file = os.path.join(test_dir,'test_'+short_module_name+'.py')

        local_test_dir = os.path.join(os.getcwd(),'tests')
        local_test_file = os.path.join(local_test_dir,
                                       'test_'+short_module_name+'.py')
        if os.path.basename(os.path.dirname(local_test_dir)) \
               == os.path.basename(os.path.dirname(test_dir)) \
           and os.path.isfile(local_test_file):
            test_file = local_test_file

        if not os.path.isfile(test_file):
            if short_module_name[:5]=='info_' \
               and short_module_name[5:]==module.__name__.split('.')[-2]:
                return []
            if short_module_name in ['__cvs_version__','__svn_version__']:
                return []
            if short_module_name[-8:]=='_version' \
               and short_module_name[:-8]==module.__name__.split('.')[-2]:
                return []
            if verbosity>1:
                print test_file
                print '   !! No test file %r found for %s' \
                      % (os.path.basename(test_file), mstr(module))
            return []

        try:
            if sys.version[:3]=='2.1':
                # Workaround for Python 2.1 .pyc file generator bug
                import random
                pref = '-nopyc'+`random.randint(1,100)`
            else:
                pref = ''
            f = open(test_file,'r')
            test_module = imp.load_module(\
                module.__name__+'.test_'+short_module_name+pref,
                f, test_file+pref,('.py', 'r', 1))
            f.close()
            if sys.version[:3]=='2.1' and os.path.isfile(test_file+pref+'c'):
                os.remove(test_file+pref+'c')
        except:
            print '   !! FAILURE importing tests for ', mstr(module)
            print '   ',
            output_exception()
            return []
        return self._get_suite_list(test_module, level, module.__name__)

    def _get_suite_list(self, test_module, level, module_name='__main__'):
        mstr = self._module_str
        if hasattr(test_module,'test_suite'):
            # Using old styled test suite
            try:
                total_suite = test_module.test_suite(level)
                return total_suite._tests
            except:
                print '   !! FAILURE building tests for ', mstr(test_module)
                print '   ',
                output_exception()
                return []
        suite_list = []
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if type(obj) is not type(unittest.TestCase) \
               or not issubclass(obj, unittest.TestCase) \
               or obj.__name__[:4] != 'test':
                continue
            for mthname in self._get_method_names(obj,level):
                suite = obj(mthname)
                if getattr(suite,'isrunnable',lambda mthname:1)(mthname):
                    suite_list.append(suite)
        print '  Found',len(suite_list),'tests for',module_name
        return suite_list

    def test(self,level=1,verbosity=1):
        """ Run Scipy module test suite with level and verbosity.
        """
        if type(self.package) is type(''):
            exec 'import %s as this_package' % (self.package)
        else:
            this_package = self.package

        package_name = this_package.__name__

        suites = []
        for name, module in sys.modules.items():
            if package_name != name[:len(package_name)] \
                   or module is None:
                continue
            if os.path.basename(os.path.dirname(module.__file__))=='tests':
                continue
            suites.extend(self._get_module_tests(module, level, verbosity))

        suites.extend(self._get_suite_list(sys.modules[package_name], level))

        all_tests = unittest.TestSuite(suites)
        #if hasattr(sys,'getobjects'):
        #    runner = SciPyTextTestRunner(verbosity=verbosity)
        #else:
        runner = unittest.TextTestRunner(verbosity=verbosity)
        # Use the builtin displayhook. If the tests are being run
        # under IPython (for instance), any doctest test suites will
        # fail otherwise.
        old_displayhook = sys.displayhook
        sys.displayhook = sys.__displayhook__
        try:
            runner.run(all_tests)
        finally:
            sys.displayhook = old_displayhook
        return runner

    def run(self):
        """ Run Scipy module test suite with level and verbosity
        taken from sys.argv. Requires optparse module.
        """
        try:
            from optparse import OptionParser
        except ImportError:
            print 'Failed to import optparse module, ignoring.'
            return self.test()
        usage = r'usage: %prog [-v <verbosity>] [-l <level>]'
        parser = OptionParser(usage)
        parser.add_option("-v", "--verbosity",
                          action="store",
                          dest="verbosity",
                          default=1,
                          type='int')
        parser.add_option("-l", "--level",
                          action="store",
                          dest="level",
                          default=1,
                          type='int')
        (options, args) = parser.parse_args()
        self.test(options.level,options.verbosity)
        return


