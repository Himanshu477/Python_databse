from type_check import ScalarType
import function_base
import twodim_base as matrix_base
import matrix
makemat = matrix.matrix

class nd_grid:
    """ Construct a "meshgrid" in N-dimensions.

        grid = nd_grid() creates an instance which will return a mesh-grid
        when indexed.  The dimension and number of the output arrays are equal
        to the number of indexing dimensions.  If the step length is not a
        complex number, then the stop is not inclusive.
    
        However, if the step length is a COMPLEX NUMBER (e.g. 5j), then the
        integer part of it's magnitude is interpreted as specifying the
        number of points to create between the start and stop values, where
        the stop value IS INCLUSIVE.

        If instantiated with an argument of 1, the mesh-grid is open or not
        fleshed out so that only one-dimension of each returned argument is
        greater than 1
    
        Example:
    
           >>> mgrid = nd_grid()
           >>> mgrid[0:5,0:5]
           array([[[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4]],
                  [[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4]]])
           >>> mgrid[-1:1:5j]
           array([-1. , -0.5,  0. ,  0.5,  1. ])

           >>> ogrid = nd_grid(1)
           >>> ogrid[0:5,0:5]
           [array([[0],[1],[2],[3],[4]]), array([[0, 1, 2, 3, 4]])] 
    """
    def __init__(self, sparse=0):
        self.sparse = sparse
    def __getitem__(self,key):
        try:
        size = []
            typecode = _nx.Int
        for k in range(len(key)):
            step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if type(step) is type(1j):
                    size.append(int(abs(step)))
                    typecode = _nx.Float
                else:
                    size.append(int((key[k].stop - start)/(step*1.0)))
                if isinstance(step,types.FloatType) or \
                   isinstance(start, types.FloatType) or \
                   isinstance(key[k].stop, types.FloatType):
                       typecode = _nx.Float
            if self.sparse:
                nn = map(lambda x,t: _nx.arange(x,dtype=t),size,(typecode,)*len(size))
            else:
                nn = _nx.indices(size,typecode)
        for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if type(step) is type(1j):
                    step = int(abs(step))
                    step = (key[k].stop - start)/float(step-1)
                nn[k] = (nn[k]*step+start)
        if self.sparse:
                slobj = [_nx.NewAxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None,None)
                    nn[k] = nn[k][slobj]
                    slobj[k] = _nx.NewAxis
        return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None: start = 0
            if type(step) is type(1j):
                step = abs(step)
                length = int(step)
                step = (key.stop-start)/float(step-1)
                stop = key.stop+step
                return _nx.arange(0,length,1,_nx.Float)*step + start
            else:
                return _nx.arange(start, stop, step)
        
    def __getslice__(self,i,j):
        return _nx.arange(i,j)

    def __len__(self):
        return 0

mgrid = nd_grid()
ogrid = nd_grid(1)

