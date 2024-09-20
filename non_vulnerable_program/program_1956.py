        import os
        from datetime import datetime
        fname = 'tmp%s' % datetime.now().strftime("%y%m%d%H%M%S%s")
        f = open(fname, 'w')
        f.write(fcontent)
        f.close()
        mrectxt = fromtextfile(fname,delimitor=',',varnames='ABCDEFG')
        os.unlink(fname)
        #
        assert(isinstance(mrectxt, MaskedRecords))
        assert_equal(mrectxt.F, [1,1,1,1])
        assert_equal(mrectxt.E._mask, [1,1,1,1])
        assert_equal(mrectxt.C, [1,2,3.e+5,-1e-10])

    def test_addfield(self):
        "Tests addfield"
        [d, m, mrec] = self.data
        mrec = addfield(mrec, masked_array(d+10, mask=m[::-1]))
        assert_equal(mrec.f2, d+10)
        assert_equal(mrec.f2._mask, m[::-1])

###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()


# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for maskedArray statistics.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_mstats.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

