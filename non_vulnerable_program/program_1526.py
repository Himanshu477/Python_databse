import numpy
import compat

__version__ = numpy.__version__

__all__ = ['__version__']
__all__ += numpy.__all__
__all__ += compat.__all__

del numpy
del compat



def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    return Configuration('oldnumeric',parent_package,top_path)

if __name__ == '__main__':
