    from maskedarray.testutils import assert_almost_equal
    if 1:
        a = maskedarray.arange(1,101)
        a[1::2] = masked
        b = maskedarray.resize(a, (100,100))
        assert_almost_equal(mquantiles(b), [25., 50., 75.])
        assert_almost_equal(mquantiles(b, axis=0), maskedarray.resize(a,(3,100)))
        assert_almost_equal(mquantiles(b, axis=1),
                            maskedarray.resize([24.9, 50., 75.1], (100,3)))


#!/usr/bin/env python
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

