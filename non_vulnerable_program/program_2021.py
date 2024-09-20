    import numpy as N
    from numpy.ma.testutils import assert_equal
    if 1:
        d = N.arange(5)
        m = numpy.ma.make_mask([1,0,0,1,1])
        base_d = N.r_[d,d[::-1]].reshape(2,-1).T
        base_m = N.r_[[m, m[::-1]]].T
        base = masked_array(base_d, mask=base_m).T
        mrecord = fromarrays(base,dtype=[('a',N.float_),('b',N.float_)])
        mrec = MaskedRecords(mrecord.copy())
        #
    if 1:
        mrec = mrec.copy()
        mrec.harden_mask()
        assert(mrec._hardmask)
        mrec._mask = nomask
        assert_equal(mrec._mask, N.r_[[m,m[::-1]]].all(0))
        mrec.soften_mask()
        assert(not mrec._hardmask)
        mrec.mask = nomask
        tmp = mrec['b']._mask
        assert(mrec['b']._mask is nomask)
        assert_equal(mrec['a']._mask,mrec['b']._mask)


"""
Generic statistics functions, with support to MA.

:author: Pierre GF Gerard-Marchant
:contact: pierregm_at_uga_edu
:date: $Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $
:version: $Id: mstats.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'


