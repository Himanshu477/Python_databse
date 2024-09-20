    from cStringIO import StringIO as BytesIO

_file = open
_string_like = _is_string_like

def seek_gzip_factory(f):
    """Use this factory to produce the class so that we can do a lazy
    import on gzip.

    """
    import gzip

    def seek(self, offset, whence=0):
        # figure out new position (we can only seek forwards)
        if whence == 1:
            offset = self.offset + offset

        if whence not in [0, 1]:
            raise IOError, "Illegal argument"

        if offset < self.offset:
            # for negative seek, rewind and do positive seek
            self.rewind()
            count = offset - self.offset
            for i in range(count // 1024):
                self.read(1024)
            self.read(count % 1024)

    def tell(self):
        return self.offset

    if isinstance(f, str):
        f = gzip.GzipFile(f)

    if sys.version_info[0] >= 3:
