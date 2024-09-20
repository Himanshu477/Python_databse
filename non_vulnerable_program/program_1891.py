import os
import sys
from numpy.distutils.exec_command import exec_command
from base import Component
from utils import FileSource

def write_files(container):
    s = ['creating files and directories:']
    for filename, i in container.label_map.items():
        content = container.list[i]
        d,f = os.path.split(filename)
        if d and not os.path.isdir(d):
            s.append('  %s/' % (d))
            if not Component._generate_dry_run:
                os.makedirs(d)
        s.append('  %s' % (filename))
        if not Component._generate_dry_run:
            f = file(filename,'w')
            f.write(content)
            f.close()
    return '\n'.join(s)


class SetupPy(Component):

    """
    >>> from __init__ import *
    >>> s = SetupPy('SetupPy_doctest')
    >>> s += PyCModule('foo')
    >>> s,o = s.execute('build_ext', '--inplace')
    >>> assert s==0,`s`
    >>> import SetupPy_doctest as mypackage
    >>> print mypackage.foo.__doc__ #doctest: +ELLIPSIS
    This module 'foo' is generated with ExtGen from NumPy version...
    
    """
    template_setup_py_start = '''\
def configuration(parent_package='', top_path = ''):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)'''
    template_setup_py_end = '''\
    return config
if __name__ == "__main__":
