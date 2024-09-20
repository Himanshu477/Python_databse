    from maskedarray.testutils import assert_equal, assert_almost_equal

    # Small arrays ..................................
    xs = numpy.random.uniform(-1,1,6).reshape(2,3)
    ys = numpy.random.uniform(-1,1,6).reshape(2,3)
    zs = xs + 1j * ys
    m1 = [[True, False, False], [False, False, True]]
    m2 = [[True, False, True], [False, False, True]]
    nmxs = numpy.ma.array(xs, mask=m1)
    nmys = numpy.ma.array(ys, mask=m2)
    nmzs = numpy.ma.array(zs, mask=m1)
    mmxs = array(xs, mask=m1)
    mmys = array(ys, mask=m2)
    mmzs = array(zs, mask=m1)
    # Big arrays ....................................
    xl = numpy.random.uniform(-1,1,100*100).reshape(100,100)
    yl = numpy.random.uniform(-1,1,100*100).reshape(100,100)
    zl = xl + 1j * yl
    maskx = xl > 0.8
    masky = yl < -0.8
    nmxl = numpy.ma.array(xl, mask=maskx)
    nmyl = numpy.ma.array(yl, mask=masky)
    nmzl = numpy.ma.array(zl, mask=maskx)
    mmxl = array(xl, mask=maskx, shrink=True)
    mmyl = array(yl, mask=masky, shrink=True)
    mmzl = array(zl, mask=maskx, shrink=True)
    #
    z = empty(3,)
    mmys.all(0, out=z)

    if 1:
        x = numpy.array([[ 0.13,  0.26,  0.90],
                     [ 0.28,  0.33,  0.63],
                     [ 0.31,  0.87,  0.70]])
        m = numpy.array([[ True, False, False],
                     [False, False, False],
                     [True,  True, False]], dtype=numpy.bool_)
        mx = masked_array(x, mask=m)
        xbig = numpy.array([[False, False,  True],
                        [False, False,  True],
                        [False,  True,  True]], dtype=numpy.bool_)
        mxbig = (mx > 0.5)
        mxsmall = (mx < 0.5)
        #
        assert (mxbig.all()==False)
        assert (mxbig.any()==True)
        assert_equal(mxbig.all(0),[False, False, True])
        assert_equal(mxbig.all(1), [False, False, True])
        assert_equal(mxbig.any(0),[False, False, True])
        assert_equal(mxbig.any(1), [True, True, True])

    if 1:
        xx = array([1+10j,20+2j], mask=[1,0])
        assert_equal(xx.imag,[10,2])
        assert_equal(xx.imag.filled(), [1e+20,2])
        assert_equal(xx.real,[1,20])
        assert_equal(xx.real.filled(), [1e+20,20])


"""Masked arrays add-ons.

A collection of utilities for maskedarray

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: extras.py 3473 2007-10-29 15:18:13Z jarrod.millman $
"""
__author__ = "Pierre GF Gerard-Marchant ($Author: jarrod.millman $)"
__version__ = '1.0'
__revision__ = "$Revision: 3473 $"
__date__     = '$Date: 2007-10-29 17:18:13 +0200 (Mon, 29 Oct 2007) $'

__all__ = [
'apply_along_axis', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average',
'vstack', 'hstack', 'dstack', 'row_stack', 'column_stack',
'compress_rowcols', 'compress_rows', 'compress_cols', 'count_masked',
'dot', 'hsplit',
'mask_rowcols','mask_rows','mask_cols','masked_all','masked_all_like',
'mediff1d', 'mr_',
'notmasked_edges','notmasked_contiguous',
'stdu', 'varu',
           ]

