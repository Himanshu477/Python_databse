import timeit
# This is to show that NumPy is a poorer choice than nested Python lists
#   if you are writing nested for loops.
# This is slower than Numeric was but Numeric was slower than Python lists were
#   in the first place. 

N = 30
code2 = r"""
for k in xrange(%d):
    for l in xrange(%d):
        res = a[k,l].item() + a[l,k].item()
""" % (N,N)
code3 = r"""
for k in xrange(%d):
    for l in xrange(%d):
        res = a[k][l] + a[l][k]
""" % (N,N)
code = r"""
for k in xrange(%d):
    for l in xrange(%d):
        res = a[k,l] + a[l,k]
""" % (N,N)
setup3 = r"""
