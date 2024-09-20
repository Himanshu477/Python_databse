import os
import sys
sys.path = %(syspath)s

def configuration(parent_name='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_name, top_path)
    %(config_code)s
    return config

if __name__ == "__main__":
