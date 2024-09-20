    import pygments
    if tuple([int(x) for x in pygments.__version__.split('.')]) < (0, 11):
        raise ImportError()
