        import increment_ext
    a = 1
    print 'a, a+1:', a, increment_ext.increment(a)
    print 'a, a+2:', a, increment_ext.increment_by_2(a)    


"""
Storing actual strings instead of their md5 value appears to 
be about 10 times faster.

>>> md5_speed.run(200,50000)
md5 build(len,sec): 50000 0.870999932289
md5 retrv(len,sec): 50000 0.680999994278
std build(len,sec): 50000 0.259999990463
std retrv(len,sec): 50000 0.0599999427795

This test actually takes several minutes to generate the random
keys used to populate the dictionaries.  Here is a smaller run,
but with longer keys.

>>> md5_speed.run(1000,4000)
md5 build(len,sec,per): 4000 0.129999995232 3.24999988079e-005
md5 retrv(len,sec,per): 4000 0.129999995232 3.24999988079e-005
std build(len,sec,per): 4000 0.0500000715256 1.25000178814e-005
std retrv(len,sec,per): 4000 0.00999999046326 2.49999761581e-006

Results are similar, though not statistically to good because of
the short times used and the available clock resolution.

Still, I think it is safe to say that, for speed, it is better 
to store entire strings instead of using md5 versions of 
their strings.  Yeah, the expected result, but it never hurts
to check...

"""
