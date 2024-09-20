import ext_tools
standard_array_factories = [array_specification()] + ext_tools.default_type_factories

def test():
    from scipy_test import module_test
    module_test(__name__,__file__)

def test_suite():
    from scipy_test import module_test_suite
    return module_test_suite(__name__,__file__)    


