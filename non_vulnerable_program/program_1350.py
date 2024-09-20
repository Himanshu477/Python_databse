    import numpy.linalg
except ImportError:
    pass
else:
    inv = numpy.linalg.inv
    svd = numpy.linalg.svd
    solve = numpy.linalg.solve
    det = numpy.linalg.det
    eig = numpy.linalg.eig
    eigvals = numpy.linalg.eigvals
    lstsq = numpy.linalg.lstsq
    pinv = numpy.linalg.pinv
    cholesky = numpy.linalg.cholesky
    


