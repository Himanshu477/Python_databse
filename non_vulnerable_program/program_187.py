import __builtin__

_open = __builtin__.open

_BLOCKSIZE = 512

error = IOError             # For anydbm

class _Database:

    def __init__(self, file):
        self._dirfile = file + '.dir'
        self._datfile = file + '.dat'
        self._bakfile = file + '.bak'
        # Mod by Jack: create data file if needed
        try:
            f = _open(self._datfile, 'r')
        except IOError:
            f = _open(self._datfile, 'w')
        f.close()
        self._update()
    
    def _update(self):
        import string   
        self._index = {}
        try:
            f = _open(self._dirfile)
        except IOError:
            pass
        else:
            while 1:
                line = string.rstrip(f.readline())
                if not line: break
                key, (pos, siz) = eval(line)
                self._index[key] = (pos, siz)
            f.close()

    def _commit(self):
        try: _os.unlink(self._bakfile)
        except _os.error: pass
        try: _os.rename(self._dirfile, self._bakfile)
        except _os.error: pass
        f = _open(self._dirfile, 'w')
        for key, (pos, siz) in self._index.items():
            f.write("%s, (%s, %s)\n" % (`key`, `pos`, `siz`))
        f.close()
    
    def __getitem__(self, key):
        pos, siz = self._index[key] # may raise KeyError
        f = _open(self._datfile, 'rb')
        f.seek(pos)
        dat = f.read(siz)
        f.close()
        return dat
    
    def _addval(self, val):
        f = _open(self._datfile, 'rb+')
        f.seek(0, 2)
        pos = f.tell()
## Does not work under MW compiler
##      pos = ((pos + _BLOCKSIZE - 1) / _BLOCKSIZE) * _BLOCKSIZE
##      f.seek(pos)
        npos = ((pos + _BLOCKSIZE - 1) / _BLOCKSIZE) * _BLOCKSIZE
        f.write('\0'*(npos-pos))
        pos = npos
        
        f.write(val)
        f.close()
        return (pos, len(val))
    
    def _setval(self, pos, val):
        f = _open(self._datfile, 'rb+')
        f.seek(pos)
        f.write(val)
        f.close()
        return (pos, len(val))
    
    def _addkey(self, key, (pos, siz)):
        self._index[key] = (pos, siz)
        f = _open(self._dirfile, 'a')
        f.write("%s, (%s, %s)\n" % (`key`, `pos`, `siz`))
        f.close()
    
    def __setitem__(self, key, val):
        if not type(key) == type('') == type(val):
            raise TypeError, "keys and values must be strings"
        if not self._index.has_key(key):
            (pos, siz) = self._addval(val)
            self._addkey(key, (pos, siz))
        else:
            pos, siz = self._index[key]
            oldblocks = (siz + _BLOCKSIZE - 1) / _BLOCKSIZE
            newblocks = (len(val) + _BLOCKSIZE - 1) / _BLOCKSIZE
            if newblocks <= oldblocks:
                pos, siz = self._setval(pos, val)
                self._index[key] = pos, siz
            else:
                pos, siz = self._addval(val)
                self._index[key] = pos, siz
            self._addkey(key, (pos, siz))
    
    def __delitem__(self, key):
        del self._index[key]
        self._commit()
    
    def keys(self):
        return self._index.keys()
    
    def has_key(self, key):
        return self._index.has_key(key)
    
    def __len__(self):
        return len(self._index)
    
    def close(self):
        self._index = None
        self._datfile = self._dirfile = self._bakfile = None


def open(file, flag = None, mode = None):
    # flag, mode arguments are currently ignored
    return _Database(file)


""" C/C++ code strings needed for converting most non-sequence
    Python variables:
        module_support_code -- several routines used by most other code 
                               conversion methods.  It holds the only
                               CXX dependent code in this file.  The CXX
                               stuff is used for exceptions
        file_convert_code
        instance_convert_code
        callable_convert_code
        module_convert_code
        
        scalar_convert_code
        non_template_scalar_support_code               
            Scalar conversion covers int, float, double, complex,
            and double complex.  While Python doesn't support all these,
            Numeric does and so all of them are made available.
            Python longs are currently converted to C ints.  Any
            better way to handle this?
"""

