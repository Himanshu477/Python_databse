import os
svn_version_file = os.path.join(os.path.dirname(__file__),
                                'core','__svn_version__.py')
if os.path.isfile(svn_version_file):
