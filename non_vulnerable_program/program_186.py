from shelve import Shelf
import zlib
from cStringIO import  StringIO
import  cPickle  

class DbfilenameShelf(Shelf):
    """Shelf implementation using the "anydbm" generic dbm interface.

    This is initialized with the filename for the dbm database.
    See the module's __doc__ string for an overview of the interface.
    """
    
    def __init__(self, filename, flag='c'):
        import dumbdbm_patched
        Shelf.__init__(self, dumbdbm_patched.open(filename, flag))

    def __getitem__(self, key):
        compressed = self.dict[key]
        try:
            r = zlib.decompress(compressed)
        except zlib.error:
            r = compressed
        return cPickle.loads(r) 
        
    def __setitem__(self, key, value):
        s = cPickle.dumps(value,1)
        self.dict[key] = zlib.compress(s)

def open(filename, flag='c'):
    """Open a persistent dictionary for reading and writing.

    Argument is the filename for the dbm database.
    See the module's __doc__ string for an overview of the interface.
    """
    
    return DbfilenameShelf(filename, flag)


"""A dumb and slow but simple dbm clone.

For database spam, spam.dir contains the index (a text file),
spam.bak *may* contain a backup of the index (also a text file),
while spam.dat contains the data (a binary file).

XXX TO DO:

- seems to contain a bug when updating...

- reclaim free space (currently, space once occupied by deleted or expanded
items is never reused)

- support concurrent access (currently, if two processes take turns making
updates, they can mess up the index)

- support efficient access to large databases (currently, the whole index
is read when the database is opened, and some updates rewrite the whole index)

- support opening for read-only (flag = 'm')

"""

_os = __import__('os')
