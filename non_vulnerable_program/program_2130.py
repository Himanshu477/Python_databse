    import bz2
    _file_openers[".bz2"] = bz2.BZ2File
except ImportError:
    pass
try:
