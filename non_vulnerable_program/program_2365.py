    import inspect
    def foo(x, y, z=None):
        return None

    print inspect.getargs(foo.func_code)
    print getargs(foo.func_code)

    print inspect.getargspec(foo)
    print getargspec(foo)

    print inspect.formatargspec(*inspect.getargspec(foo))
    print formatargspec(*getargspec(foo))


#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('compat',parent_package,top_path)
    return config

if __name__ == '__main__':
