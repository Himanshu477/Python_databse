import sys,os
from distutils import util
p = os.path.abspath(os.path.join('.',
                                 'build',"lib.%s-%s" % \
                                 (util.get_platform(),
                                  sys.version[0:3])))
sys.path.insert(0,p)
