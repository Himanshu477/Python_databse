    from numpy.ma.testutils import assert_almost_equal
    if 1:
        a = numpy.ma.arange(1,101)
        a[1::2] = masked
        b = numpy.ma.resize(a, (100,100))
        assert_almost_equal(mquantiles(b), [25., 50., 75.])
        assert_almost_equal(mquantiles(b, axis=0), numpy.ma.resize(a,(3,100)))
        assert_almost_equal(mquantiles(b, axis=1),
                            numpy.ma.resize([24.9, 50., 75.1], (100,3)))


"""Miscellaneous functions for testing masked arrays and subclasses

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: testutils.py 3529 2007-11-13 08:01:14Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = "1.0"
__revision__ = "$Revision: 3529 $"
__date__ = "$Date: 2007-11-13 10:01:14 +0200 (Tue, 13 Nov 2007) $"


