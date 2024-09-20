    import numpy as N
    from numpy.ma.testutils import assert_equal
    if 1:
        b = ones(5)
        m = [1,0,0,0,0]
        d = masked_array(b,mask=m)
        c = mr_[d,0,0,d]


"""
Generic statistics functions, with support to MA.

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_edu
:date: $Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $
:version: $Id: morestats.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'


