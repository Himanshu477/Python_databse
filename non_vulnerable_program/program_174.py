        import ramp_ext
    arr = array([0]*10000,Float64)
    for i in xrange(10000):
        ramp_ext.Ramp(arr, 10000, 0.0, 1.0)
    t2 = time.time()
    c_time = (t2 - t1)    
    print 'compiled numeric (seconds, speed up):', c_time, (py_time*10000/200.)/ c_time
    print 'arr[500]:', arr[500]
    
if __name__ == '__main__':
    main()

