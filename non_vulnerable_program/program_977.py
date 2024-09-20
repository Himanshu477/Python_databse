import sys
from __builtin__ import open as __open
import os
import mmap
import types
import operator
import copy_reg
import copy
from numeric import fromstring, frombuffer, empty

valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+","w+"]

mode_equivalents = {
    "readonly":"r",
    "copyonwrite":"c",
    "readwrite":"r+",
    "write":"w+"
    }

def _open(file, mode):
    return __open(file, mode+"b")

class memmap:
    def __init__(self, filename, mode="r+", len=None):
        """
        Valid "mode" values for a memmap are:
        "readonly"     or "r"
        "copyonwrite"  or "c"
        "readwrite"    or "r+"
        "write"        or "w+"

        Create a small test file with 100 zeroes:
        >>> _open("memmap.tst","w+").write(chr(0)*100)
        
        Map it as a readwrite mapping:
        >>> m = memmap("memmap.tst","r+")
        """
        self._filename = filename
        self._slices = []
        self._mmap = None

        if mode in mode_equivalents.keys():
            mode = mode_equivalents[mode]
        elif mode not in valid_filemodes:
            raise ValueError("mode must be one of %s" % \
                             (valid_filemodes + mode_equivalents.keys()))

        if mode == "w+" and len is None:
            raise ValueError("Must specify 'len' if mode is 'w+'")

        self._readonly  = (mode == "r")
        self._mode = mode

        file = _open(filename, (mode == "c" and "r" or mode))

        file.seek(0, 2)
        flen = file.tell()

        if len is None:
            len = flen

        if mode == "w+" or (mode == "r+" and flen < len):
            if len:
                file.seek(len-1, 0) # seek to the last byte and write it.
                file.write(chr(0))
                file.flush()
            flen = len

        if len:
            if mode == "c":
                acc = mmap.ACCESS_COPY
            elif mode == "r":
                acc = mmap.ACCESS_READ
            else:
                acc = mmap.ACCESS_WRITE
            self._mmap = mmap.mmap(file.fileno(), len, access=acc)
        else:
            self._mmap = None

        file.close()

    def __repr__(self):
        self._chkOverlaps(0,0)
        return "<memmap on file '%s' with mode='%s', length=%d, %d slices>" % \
            (self._filename, self._mode, len(self), len(self._slices))

    def close(self):
        """
        close(self) unites the memory map and any RAM based slices
        with its underlying file and removes the mapping and all
        references to its slices.  Don't call this till you're done
        using the memmap!
        """
        if self._mutablefile():
            self.flush()
        self._lose_map()

    def _buffer(self, begin=0, end=None):
        """_buffer(self) returns a buffer object for memmap 'self'.
        """
        if self._mmap is None:
            raise RuntimeError("memmap no longer valid.  (closed?)")
        if end is None:
            end = len(self)
        obj = frombuffer(self._mmap)
        if self._readonly:
            obj.flags.writeable=False
        return obj.data

    def _chkOverlaps(self, begin, end):
        """_chkOverlaps(self, begin, end) is called to raise an exception
        if the requested slice will overlap any slices which have already
        been taken.
        """
        for b,e,obj in self._slices:
            if (b < begin < e or 
                  b < end < e  or 
                  begin < b < end or 
                  begin < e < end or 
                  b == begin and e == end and b != e):
                raise IndexError("Slice overlaps prior slice of same file.")

    def __len__(self):
        """len(self) is the number of bytes in memmap 'self'.
        """
        if self._mmap:
            maplen = len(self._mmap)
        else:
            maplen = 0
        orig_len = reduce(operator.add, [ e-b for b,e,o in self._slices ], 0)
        obj_len = reduce(operator.add, [ len(o) for b,e,o in self._slices], 0)
        return int(maplen - orig_len + obj_len)

    def __str__(self):
        if self._mmap is not None:
            return self._mmap[:]
        else:
            return ""

    def _fix_slice(self, i):
        """_fix_slice(self, i) converts a __getitem__ 'key' into slice
        parameters, and returns a tuple (beg, end).
        """
        if isinstance(i, types.SliceType):
            if i.step is not None:
                raise IndexError("memmap does not support strides.")
            j, i = i.stop, i.start
        else:
            j = i+1
        i, j = self._chkIndex(i, 1), self._chkIndex(j, 1)
        return i, j

    def _chkIndex(self, i, isSlice=0):
        """_chkIndex(self, i) raises an IndexError if 'i' is not a valid
        index of 'self'
        """
        if i == int(2L**31-1):  # XXX Python indices are ints for now.  sys.maxint is a long.
            return len(self)
        if i < 0:
            i += len(self)
        if not(0 <= i < len(self)+isSlice):
            raise IndexError("Invalid memmap index: %d" % (i,))
        return i

    def __getitem__(self,i):
        """__getitem__(self,i) returns a memmapSlice corresponding to the
        index 'i' of the memmap 'self'.  The memmap keeps a record of the
        slice so that it can coordinate it with other slices from the same
        file.  Slices of a memmap are not permitted to overlap.
        """
        i, j = self._fix_slice(i)
        self._chkOverlaps(i, j)
        obj = memmapSlice(self._buffer(i, j), readonly=self._readonly)
        self._slices.append((i, j, obj))
        return obj

    def __delitem__(self,i):
        """__delitem__(self,i) deletes a slice from a memmap, removing the
        record of the "slice", but not the underlying data footprint.
        """
        i, j = self._fix_slice(i)
        for k in range(len(self._slices)):
            b,e,o = self._slices[k]
            if b==i and e==j:
                o._markDeleted()
                del self._slices[k]
                return
        else:
            raise ValueError("Can't find slice (%d,%d)" % (i,j))

    def _mutablefile(self):
        
        """_mutablefile returns 1 iff the file underlying the memory
        map can be modified.  Thus, it returns 0 for readonly and
        copyonwrite mappings."""
        
        return self._mode not in ["c","r"]

    def sync(self):
        """sync(self) ultimately calls msync, guaranteeing that updates
        to a MMap are already written to the underlying file system
        device when the call returns.
        """
        if self._mutablefile() and self._mmap is not None:
            self._mmap.flush()

    def _dirty(self):
        """_dirty(self) is 1 if any slice of self is "dirty".  A slice is
        dirty if it has been resized in any way, or was not part of the
        original memmap.  _dirty(self) specifically excludes in-place
        modification, since this can happen at the C-level and there's
        no way to know whether it has happened or not.
        """
        return reduce( operator.or_, [ o.dirty() for b,e,o in self._slices ],
                       0 )

    def _lose_map(self):
        """_lose_map(self) eliminates all references to the underlying mmap,
        so that it will be deleted.  This appeared necessary on Win-NT to be
        able to re-map the same file.
        """
        for b,e,o in self._slices:
            o._rebuffer(None)
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def _consolidate(self, new_map=None):
        
        """_consolidate(self) re-writes the memory map file,
        interspersing RAM based slices with the content of the mmap
        which has been updated in-place.  By default, the new memory
        mapped file is then mapped in place of the old one."""

        if (not self._mutablefile() and
            (new_map == self._filename or new_map is None)):
            raise RuntimeError("memmap trying to flush onto readonly file")

        temp_map = "memmap.tmp"            
        f = _open(temp_map, "w+")
        mlen = len(self)
        l = self._slices
        self._slices = []
        l.sort()
        m = 0
        for b, e, obj in l:
            if b > m:       # copy original mmap between slices
                f.write(self._buffer(m,b))
            ob = f.tell()
            f.write(obj.__buffer__())
            oe = f.tell()
            self._slices.append((ob,oe,obj))
            m = e
        if mlen != oe:
            f.write(self._buffer(m, mlen))
        f.close()
        self._lose_map()
        
        if new_map is not None:
            self._filename = new_map
        if os.path.exists(self._filename):
            os.remove(self._filename)
        os.rename(temp_map, self._filename)
            
        f = _open(self._filename,"r+")
        self._readonly = 0
        self._mode = "r+"
        self._mmap = mmap.mmap(f.fileno(), mlen)
        for b,e,o in self._slices:
            o._rebuffer(self._buffer(b, e))
        f.close()

    def flush(self, filename=None):
        
        """flush(self) first syncs the memory map with the underlying
        file, then consolidates it with any RAM based slices in a new
        file, then remaps the new file.  Both slice offsets and
        buffers change.  If there are no RAM based buffers, no
        consolidation is performed.

        It is possible to flush a map onto a new file by specifying
        its name with the filename parameter.
        """
        
        self.sync()
        if self._dirty() or (filename and self._filename != filename):
            self._consolidate(filename)

    def find(self, string, offset=0):
        """find(string, offset=0) returns the first index at which string
        is found, or -1 on failure.

        >>> _open("memmap.tst","w+").write("this is a test")
        >>> memmap("memmap.tst",len=14).find("is")
        2
        >>> memmap("memmap.tst",len=14).find("is", 3)
        5
        >>> _open("memmap.tst","w+").write("x")
        >>> memmap("memmap.tst",len=1).find("is")
        -1
        """
        if self._mmap is None:
            raise RuntimeError("_mmap is None;  memmap closed?")
        else:
            return self._mmap.find(string, offset)

    def move(self, dest, src, count):
        """move(dest, src, count) moves 'count' characters from 'src' to 'dest'
        within a memmap.
        """
        self._buffer()[dest:dest+count] = self._buffer()[src:src+count]

    def insert(self, offset, size=None, buffer=None):
        """
        insert(self, offset, size) inserts a new slice of 'size'
        bytes, possibly between two adjacent slices, at byte 'offset'.
        It is not legal to insert into the middle of another slice, but
        pretty much everything else is legal.  The resulting
        memmapSlice is returned.

        >>> m = open("memmap.tst",mode="w+",len=100)
        >>> n = m[0:50]
        >>> p = m[50:100]
        >>> q=m.insert(0, 200)
        >>> r=m.insert(50, 100)
        >>> s=m.insert(100, 300)
        >>> t=m.insert(45, 100)
        Traceback (most recent call last):
        ...
        IndexError: Slice overlaps prior slice of same file.
        >>> m.flush()
        >>> len(m)
        700
        >>> m.close()
        """
        self._chkIndex(offset, isSlice=1)
        self._chkOverlaps(offset, offset)
        if buffer is None:
            mem = empty((size,),'B').data
        elif size is None or len(buffer) == size:
            mem = buffer
            size = len(buffer)
        else:
            raise ValueError("buffer and size don't match.")
        obj = memmapSlice(mem, dirty=1)
        self._slices.append((offset, offset, obj))
        return obj

    def __del__(self):
        self.close()

class memmapSlice:
    def __init__(self, buffer, dirty=0, readonly=0):
        self._buffer = buffer
        self._dirty = dirty
        self._deleted = 0
        self._readonly = readonly

    def _markDeleted(self):
        self._deleted = 1

    def _checkDeleted(self):
        """_checkDeleted ensures that a deleted memmap region does not continue
        to be used.  For a map region to be re-used, a new slice must be taken.
        """
        if self._deleted:
            raise RuntimeError, "A deleted memmapSlice has been used."

    # Limited pickling support: memmapSlices are pickled uniquely, and are
    # restored as memmapSlices, but they are orphaned in the sense that
    # the memory mapped file from which they came is likely to be gone.  
    # Ideally, under these circumstances, the memmapSlice should just mutate
    # into a memory object on unpickling.  This seems hard to do...
    def __getstate__(self):
    """Returns the state of a memmapSlice for pickling."""
        self._checkDeleted()
    d = copy.copy(self.__dict__)
    d["_buffer"] = str(self._buffer)
    d["_dirty"] = 0
        d["_readonly"] = 0
    return d

    def __setstate__(self, state):
    """Restores the state of a memmapSlice after unpickling."""
        self.__dict__.update(state)
    self._buffer = fromstring(state["_buffer"]).data

    def __repr__(self):
        if self._readonly:
            s = "readonly"
        else:
            s = "writable"
        return "<memmapSlice of length:%d %s>" % (len(self), s)

    def __len__(self):
        if self._buffer:
            return len(self.__buffer__())
        else:
            return 0

    def dirty(self):
        """dirty(self, set=None) is 1 iff 'self' has changed its buffer since it
        was created.
        """
        return self._dirty

    def __getitem__(self,i):
        self._checkDeleted()
        if type(i) is types.IntType:
            return str(self.__buffer__()[i])
        elif type(i) is types.SliceType:
            return str(self.__buffer__()[i.start:i.stop])
        else:
            raise TypeError("Can't handle index type")

    def __setitem__(self, i, v):
        self._checkDeleted()
        if type(i) is types.IntType:
            self.__buffer__()[i] = v
        elif type(i) is types.SliceType:
            self.__buffer__()[i.start:i.stop] = v
        else:
            raise TypeError("Can't handle index type")

    def __buffer__(self):
        self._checkDeleted()
        if self._buffer is not None:
            return self._buffer
        else:
            raise RuntimeError("memmapSlice no longer valid...(memmap closed?)")

    def _rebuffer(self, b):
        self._buffer = b
        self._dirty = 0
        self._readonly = 0

    def _modify_buffer(self, offset, size):
        """_modify_buffer(self, offset, size) replaces the slice's mmap
        buffer with a resized RAM buffer, and copies the contents.
        """
        self._checkDeleted()
        self._dirty = 1
        olen = len(self)
        nlen = olen+size
        nm = empty((nlen,),'B').data
        nm[0:offset] = self._buffer[0:offset]
        if size > 0:
            nm[offset+size:] = self._buffer[offset:]
        else:
            nm[offset:] = self._buffer[offset-size:]
        self._buffer = nm

    def _insert(self, offset, size):
        """_insert(self, offset, size) expands the MMap at 'offset'
        by 'size' bytes.

        'offset' refers to a position between two existing characters, the
        beginning, or the end.
        """
        # Since insertion points aren't indices, tolerate the end
        self._checkDeleted()
        self._chkIndex(offset, 1)
        if offset + size > sys.maxint:
            raise ValueError("Insert makes file too big for integer offsets")
        self._modify_buffer(offset, size)

    def insert(self, offset, value, size=None, padc=0):
        """insert(self, offset, value, size=None) inserts string 'value' at
        'offset', possibly padding it with extra characters of value 'padc'.
        If size is None, the the size of the insert is len(value).
        """
        l = len(value)
        if size is None:
            size = l
        elif l < size:
            value += padc * (size-l)
        elif l > size:
            raise ValueError("'value' too long for 'size'")
        self._insert(offset, size)
        self.__buffer__()[offset:offset+size] = value

    def _append(self, size):
        """append(self, size) is similar to 'insert', but assumes the offset is
        the end of the current slice.
        """
        self._insert(len(self), size)

    def append(self, value):
        size = len(value)
        self._append(size)
        self.__buffer__()[-size:] = value

    def delete(self, offset, size):
        """delete(self, offset, size) removes 'size' bytes from the MMap,
        starting at 'offset'.

        'offset' refers to a position between two existing characters, the
        beginning, or the end.
        """
        self._chkIndex(offset+size, 1)
        self._insert(offset, -size)

    def truncate(self, size):
        """truncate(self, size) is similar to 'delete', but assumes the offset
        is the end of the current slice.
        """
        self.delete(len(self)-size, size)

    def resize(self, newsize):
        """resize(self, newsize) appends or truncates the memmapSlice to
        'newsize'.  Any newly added region is uninitialized.
        """
        self._checkDeleted()
        olen = len(self)
        if newsize > olen:
            self._append( newsize-olen )
        elif newsize < olen:
            if newsize < 0:
                raise ValueError("Negative resize value")
            self.truncate(olen - newsize)

    def flush(self):
        """flush(self)
        """
        raise TypeError("Only the 'root' memmap should be flushed.")

    def _chkIndex(self, i, End=0):
        """_chkIndex(self, i) raises an IndexError if 'i' is not a valid
        index of 'self'
        """
        olen = len(self)
        if i == sys.maxint:  # Assume i not maxint unless it's a slice stop
            return olen
        if i < 0:
            i += olen
        if not(0 <= i < olen+End):
            raise IndexError("Invalid memmap index: %d" % (i,))
        return i

    def __str__(self):
        return str(self.__buffer__())

def open(filename, mode="r+", len=None):
    """open(filename, mode="r+", len=None) creates a new memmap object.
    """
    return memmap(filename, mode, len)

def close(map):
    return map.close()

def test():
    """
    >>> import os
    >>> os.remove("memmap.tst")
    """
    import doctest, memmap
    return doctest.testmod(memmap)

def proveit(N, filename="memmap.dat", pagesize=1024, mode="r"):
    """proveit is a diagnostic function which creates a file of size 'N',
    memory maps it, and then reads one byte at 'pagesize' intervals."""
    
    import numeric as num
    import os
    
    f = _open(filename, "w+")
    f.seek(N-1)
    f.write("\0")
    f.close()

    m = memmap(filename, mode=mode)
    n = m[:]
    a = frombuffer(buffer=n, dtype='b', count=len(n))
    hits = num.arange(N//pagesize)*pagesize
    fetch = a[ hits ]  # force every page into RAM

    return a

if __name__ == "__main__":
    test()




