from fft_lite import *
from helper import *

ifft = inverse_fft
refft = real_fft
irefft = inverse_real_fft
hfft = hermite_fft
ihfft = inverse_hermite_fft

fftn = fftnd
ifftn = inverse_fftnd
refftn = real_fftnd
irefftn = inverse_real_fftnd

fft2 = fft2d
ifft2 = inverse_fft2d
refft2 = real_fft2d
irefft2 = inverse_real_fft2d



"""$Id: memmap.py,v 1.32 2005/10/21 20:42:04 jaytmiller Exp $

This module implements a class wrapper (memmap) around the
mmap type which adds some additional properties.

The intended use of the class(es) (memmap, memmapSlice) is to create a
single tier of slices from a file and to use these slices as if they
were buffers within the Numarray framework.  The slices have these
properties:

1. memmapSlices are non-overlapping.
2. memmapSlices are resizable.
3. memmapSlices from the same memmap remain "related" and affect one
anothers buffer offsets.
4. Changing the size of a memmapSlice changes the size of the parent memmap.

For example:

Open a memory map windowed on file 'memmap.tst'.  In practice, the
file might be (a) much bigger and (b) already existing.   Other file modes
"r" and "r+" handle existing files for readonly and read-write cases.

>>> m = open("memmap.tst","w+",len=48)
>>> m
<memmap on file 'memmap.tst' with mode='w+', length=48, 0 slices>

As in C stdio, when a file is opened for 'w+', it is truncated
(effectively deleted) if it already exists.  The initial contents of a
w+ memmap are undefined.

Slice m into the buffers "n" and "p" which will correspond to numarray:

>>> n = m[0:16]
>>> n
<memmapSlice of length:16 writable>

>>> p = m[24:48]
>>> p
<memmapSlice of length:24 writable>

NOTE: You *can not* make overlapping slices of a memmap:

>>> q = m[20:28]
Traceback (most recent call last):
...
IndexError: Slice overlaps prior slice of same file.

NOTE: you *can not* make an array directly from a memmap:

>>> import numarrayall as num
>>> c = num.NumArray(buffer=m, shape=(len(m)/4,), type=num.Int32)
Traceback (most recent call last):
...
error: NA_updateDataPtr: error getting read buffer data ptr

This fails because m, being the root memmap and not a memmapSlice, does not
define __buffer__() or resize().

Finally, the good part.  Make numarray from the memmapSlice "buffers":

>>> a = num.NumArray(buffer=n, shape=(len(n)/4,), type=num.Int32)
>>> a[:] = 0  # Since the initial contents of 'n' are undefined.
>>> a += 1
>>> num.explicit_type(a)
array([1, 1, 1, 1], type=Int32)

>>> b = num.NumArray(buffer=p, shape=(len(p)/8,), type=num.Float64)
>>> b[:] = 0 # Since the initial contents of 'p' are undefined.
>>> b += 2.0
>>> b
array([ 2.,  2.,  2.])

Here's the other good part about memmapSlices:  they're resizable.

>>> _junk = a.resize( 6 )
>>> num.explicit_type(a)
array([1, 1, 1, 1, 1, 1], type=Int32)
>>> b
array([ 2.,  2.,  2.])

What you should note is that "b" retains the correct contents (i.e., offset
within "m") even though "a" grew, effectively moving "b".  In reality, "b"
stayed where it always was and "a" has moved to a bigger RAM-based buffer.

Since we resized "a", "m" is now a different size as well:

>>> m
<memmap on file 'memmap.tst' with mode='w+', length=56, 2 slices>

After doing resizes, call m.flush() to synchronize the underlying file
of "m" with any RAM based slices.  This step is required to avoid
implicitly shuffling huge amounts of file space for every insert or
delete.  After calling m.flush(), all slices are once again memory
mapped rather than purely RAM based.

>>> m.flush()

NOTE: Since memory maps don't guarantee when the underlying file will
be updated with the values you have written to the map, call m.sync()
when you want to be sure your changes are on disk.  Note that sync()
does not consolidate the mapfile with any purely RAM based slices
which have been inserted into the map.

>>> m.sync()

Now "a" and "b" are both memory mapped on "memmap.tst" again.

It is also possible for "a" or "b" to shrink:

>>> _junk = a.resize(0)
>>> num.explicit_type(a)
array([], type=Int32)
>>> b
array([ 2.,  2.,  2.])
>>> m
<memmap on file 'memmap.tst' with mode='r+', length=32, 2 slices>

Arrays based on memmapSlices can be pickled:

>>> import cPickle
>>> c = cPickle.loads(cPickle.dumps(b))
>>> c
array([ 2.,  2.,  2.])

However, when the array is restored by the unpickler, the buffer is
restored as an "orphaned" memmapSlice.  There is currently no support
for pickling the memmap.

When you're done with the memory map and numarray, call m.close().  m.close()
calls m.flush() which will do consolidation if any is needed.

>>> m.close()

It is an error to use "m" (or slices of m) any further after closing
it.

>>> m._buffer()
Traceback (most recent call last):
...
RuntimeError: memmap no longer valid.  (closed?)


Slices of a memmap are memmapSlice objects.  Slices of a memmapSlice
are strings.

>>> m = memmap("memmap.tst",mode="w+",len=100)
>>> m1=m[:]
>>> m2=m1[:]
>>> m3=m1[:10]
>>> int(isinstance(m3, types.StringType))
1
>>> int(isinstance(m2, types.StringType))
1
>>> m.close()

Deletion of a slice of a memmap "un-registers" the slice, making that
region of the memmap available for reallocation:

>>> m = memmap("memmap.tst",mode="w+",len=100)
>>> m1 = m[0:50]

Delete directly from the memmap without referring to the memmapSlice:

>>> del m[0:50]
>>> m2 = m[0:70]

Note that since the region of m1 was deleted, there is no overlap when
m2 is created.  However, deleting the region of m1 has
invalidated it:

>>> m1
Traceback (most recent call last):
...
RuntimeError: A deleted memmapSlice has been used.

It is a bad idea to mix operations on a memmap which modify its data
or file structure with slice deletions.  DO NOT use a memmapSlice
where it's contents can be modified or resized and then delete the
region it refers to later.  In this case, the status of the
modifications is undefined; the underlying map may or may not reflect
the modifications after the deletion.

>>> m.close()

Copy-on-write memory maps can be opened using either mode="c" or
mode="copyonwrite".  Copy-on-write maps have writable slices,  but
cannot be resized, flushed, or synced to the original file.

>>> m = memmap("memmap.tst",mode="c",len=100)
>>> n = m[:]
>>> n
<memmapSlice of length:100 writable>
>>> a = num.NumArray(buffer=n, shape=(len(n),), type=num.Int8)
>>> a += 1
>>> # it worked!
>>> m.close()

Try a zero length memmap based on comments from Sebastian Hesse

>>> m = memmap("memmap.tst", mode='w+', len=0)
>>> n = m.insert(0,0)
>>> a = num.NumArray(buffer=n, type=num.UInt16, shape=(0,0,0))
>>> _junk = a.resize((100,100,100))
>>> m.flush()
>>> m.close()

Readonly memory maps can be opened using either mode="r" or
mode="readonly".  Readonly maps have readonly slices as well as all
of the restrictions of copy-on-write mappings.

>>> m = memmap("memmap.tst",mode="r",len=100)
>>> n = m[:]
>>> n
<memmapSlice of length:100 readonly>
>>> a = num.NumArray(buffer=n, shape=(len(n),), type=num.Int8)
>>> a += 1  # can't work...  buffer is readonly
Traceback (most recent call last):
...
error: add_11x1_vsxv: Problem with write buffer[2].
>>> b = a + 1  # this still works...
>>>

"""

