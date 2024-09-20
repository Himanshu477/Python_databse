    import sys
    maxint = sys.maxint

    def __getitem__(self, item):
        if type(item) != type(()):
            return (item,)
        else:
            return item

    def __len__(self):
        return self.maxint

    def __getslice__(self, start, stop):
        if stop == self.maxint:
            stop = None
        return self[start:stop:None]

index_exp = _index_expression_class()

# End contribution from Konrad.


""" Basic functions for manipulating 2d arrays

"""

__all__ = ['diag','eye','fliplr','flipud','hankel','rot90','tri',
           'tril','triu','toeplitz','all_mat']
           
# These are from Numeric
