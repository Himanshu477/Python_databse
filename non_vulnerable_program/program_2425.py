import os
import sys
sys.path = %(syspath)s

def configuration(parent_name='',top_path=None):
    global config
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_name, top_path)
    return config

