    import time
    t1 = time.time()    
    dist = dummy_dist()    
    compiler_obj = create_compiler_instance(dist)
    exe_name = compiler_exe_name(compiler_obj)
    exe_path = compiler_exe_path(exe_name)
    chk_sum = check_sum(exe_path)    
    
    t2 = time.time()
    print 'compiler exe:', exe_path
    print 'check sum:', chk_sum
    print 'time (sec):', t2 - t1
    print
    """
    path = get_compiler_dir('gcc')
    print 'gcc path:', path
    print
    try: 
        path = get_compiler_dir('msvc')
        print 'gcc path:', path
    except ValueError:
        pass    


# TODO:  This needs to be
#        fleshed out
#  Adapted from Numarray by J. Todd Miller
"""
Large chararray test
>>> xx=array(None,itemsize=3,shape=220000)

>>> import cPickle
>>> c=cPickle.loads(cPickle.dumps(fromlist(["this","that","something else"])))
>>> c
CharArray(['this', 'that', 'something else'])
>>> c._type
CharArrayType(14)

>>> a=fromlist(["this"]*25); a.shape=(5,5); a[ range(2,4) ]
CharArray([['this', 'this', 'this', 'this', 'this'],
           ['this', 'this', 'this', 'this', 'this']])
>>> a[ range(2,4) ] = fromlist(["that"]); a
CharArray([['this', 'this', 'this', 'this', 'this'],
           ['this', 'this', 'this', 'this', 'this'],
           ['that', 'that', 'that', 'that', 'that'],
           ['that', 'that', 'that', 'that', 'that'],
           ['this', 'this', 'this', 'this', 'this']])

>>> array([], shape=(0,1,2))
CharArray([])

>>> a = _gen.concatenate([array(["1"]*3), array(["2"]*3)]); a
CharArray(['1', '1', '1', '2', '2', '2'])
>>> _gen.reshape(a, (2,3))
CharArray([['1', '1', '1'],
           ['2', '2', '2']])

>>> CharArray(buffer="thatthis", shape=(2,), itemsize=4,
...              bytestride=-4, byteoffset=4)
CharArray(['this', 'that'])
"""

