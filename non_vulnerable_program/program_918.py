    from scipy.base import *

    def test(a, b):

        print "All numbers printed should be (almost) zero:"

        x = solve_linear_equations(a, b)
        check = b - matrixmultiply(a, x)
        print check


        a_inv = inverse(a)
        check = matrixmultiply(a, a_inv)-identity(a.shape[0])
        print check


        ev = eigenvalues(a)

        evalues, evectors = eigenvectors(a)
        check = ev-evalues
        print check

        evectors = transpose(evectors)
        check = matrixmultiply(a, evectors)-evectors*evalues
        print check


        u, s, vt = singular_value_decomposition(a)
        check = a - Numeric.matrixmultiply(u*s, vt)
        print check


        a_ginv = generalized_inverse(a)
        check = matrixmultiply(a, a_ginv)-identity(a.shape[0])
        print check


        det = determinant(a)
        check = det-multiply.reduce(evalues)
        print check

        x, residuals, rank, sv = linear_least_squares(a, b)
        check = b - matrixmultiply(a, x)
        print check
        print rank-a.shape[0]
        print sv-s

    a = array([[1.,2.], [3.,4.]])
    b = array([2., 1.])
    test(a, b)

    a = a+0j
    b = b+0j
    test(a, b)



