    import numpy.f2py as f2py
    fid = open('add.f')
    source = fid.read()
    fid.close()
    f2py.compile(source, modulename='add')
