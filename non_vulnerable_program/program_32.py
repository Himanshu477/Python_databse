from inline_tools import inline
import ext_tools
from ext_tools import ext_module, ext_function

#---- testing ----#

def test():
    import unittest
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
    return runner

def test_suite():
    import scipy_test
    import compiler
    return scipy_test.harvest_test_suites(compiler)


