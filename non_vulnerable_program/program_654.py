import sys
class concatenator:
    """ Translates slice objects to concatenation along an axis.
    """
    def _retval(self, res):
        if not self.matrix:
            return res
        else:
            if self.axis == 0:
                return makemat(res)
            else:
                return makemat(res).T        
        
    def __init__(self, axis=0, matrix=0):
        self.axis = axis
        self.matrix = matrix
    def __getitem__(self,key):
        if isinstance(key,types.StringType):
            frame = sys._getframe().f_back
            mymat = matrix_base.bmat(key,frame.f_globals,frame.f_locals)
            if self.matrix:
                return mymat
            else:
                return asarray(mymat)
        if type(key) is not types.TupleType:
            key = (key,)
        objs = []
        for k in range(len(key)):
            if type(key[k]) is types.SliceType:
                typecode = _nx.Int
            step = key[k].step
                start = key[k].start
                stop = key[k].stop
                if start is None: start = 0
                if step is None:
                    step = 1
                if type(step) is type(1j):
                    size = int(abs(step))
                    typecode = _nx.Float
                    newobj = function_base.linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
            elif type(key[k]) in ScalarType:
                newobj = asarray([key[k]])
            else:
                newobj = key[k]
            objs.append(newobj)
        res = _nx.concatenate(tuple(objs),axis=self.axis)
        return self._retval(res)

    def __getslice__(self,i,j):
        res = _nx.arange(i,j)
        return self._retval(res)

    def __len__(self):
        return 0

r_=concatenator(0)
c_=concatenator(-1)
row = concatenator(0,1)
col = concatenator(-1,1)

# A nicer way to build up index tuples for arrays.
#
# You can do all this with slice() plus a few special objects,
# but there's a lot to remember. This version is simpler because
# it uses the standard array indexing syntax.
#
# Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-23
#
# Cosmetic changes by T. Oliphant 2001
#
#
# This module provides a convenient method for constructing
# array indices algorithmically. It provides one importable object,
# 'index_expression'.
#
# For any index combination, including slicing and axis insertion,
# 'a[indices]' is the same as 'a[index_expression[indices]]' for any
# array 'a'. However, 'index_expression[indices]' can be used anywhere
# in Python code and returns a tuple of slice objects that can be
# used in the construction of complex index expressions.

class _index_expression_class:
