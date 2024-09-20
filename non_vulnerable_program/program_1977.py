    from numpy.distutils.core import setup
    #setup.update(nmasetup)
    config = configuration(top_path='').todict()
    setup(**config)


# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_core.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

