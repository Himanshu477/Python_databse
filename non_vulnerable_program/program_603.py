import log

# Note that UnixCCompiler._compile appeared in Python 2.3
def UnixCCompiler__compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    display = '%s: %s' % (os.path.basename(self.compiler_so[0]),src)
    try:
        self.spawn(self.compiler_so + cc_args + [src, '-o', obj] +
                   extra_postargs, display = display)
    except DistutilsExecError, msg:
        raise CompileError, msg
UnixCCompiler._compile = new.instancemethod(UnixCCompiler__compile,
                                            None,
                                            UnixCCompiler)


def UnixCCompile_create_static_lib(self, objects, output_libname,
                                   output_dir=None, debug=0, target_lang=None):
    objects, output_dir = self._fix_object_args(objects, output_dir)

    output_filename = \
                    self.library_filename(output_libname, output_dir=output_dir)
        
    if self._need_link(objects, output_filename):
        self.mkpath(os.path.dirname(output_filename))
        tmp_objects = objects + self.objects
        while tmp_objects:
            objects = tmp_objects[:50]
            tmp_objects = tmp_objects[50:]
            display = '%s: adding %d object files to %s' % (os.path.basename(self.archiver[0]),
                                               len(objects),output_filename)
            self.spawn(self.archiver + [output_filename] + objects,
                       display = display)

        # Not many Unices required ranlib anymore -- SunOS 4.x is, I
        # think the only major Unix that does.  Maybe we need some
        # platform intelligence here to skip ranlib if it's not
        # needed -- or maybe Python's configure script took care of
        # it for us, hence the check for leading colon.
        if self.ranlib:
            display = '%s:@ %s' % (os.path.basename(self.ranlib[0]),
                                   output_filename)
            try:
                self.spawn(self.ranlib + [output_filename],
                           display = display)
            except DistutilsExecError, msg:
                raise LibError, msg
    else:
        log.debug("skipping %s (up-to-date)", output_filename)
    return

UnixCCompiler.create_static_lib = \
  new.instancemethod(UnixCCompile_create_static_lib,
                     None,UnixCCompiler)


"""scipy.base defining a multi-dimensional array and useful procedures for
   Numerical computation.

Functions

-   array                      - NumPy Array construction
-   zeros                      - Return an array of all zeros
-   empty                      - Return an unitialized array
-   shape                      - Return shape of sequence or array
-   rank                       - Return number of dimensions
-   size                       - Return number of elements in entire array or a
                                 certain dimension
-   fromstring                 - Construct array from (byte) string
-   take                       - Select sub-arrays using sequence of indices
-   put                        - Set sub-arrays using sequence of 1-D indices
-   putmask                    - Set portion of arrays using a mask 
-   reshape                    - Return array with new shape
-   repeat                     - Repeat elements of array
-   choose                     - Construct new array from indexed array tuple
-   cross_correlate            - Correlate two 1-d arrays
-   searchsorted               - Search for element in 1-d array
-   sum                        - Total sum over a specified dimension
-   average                    - Average, possibly weighted, over axis or array.
-   cumsum                     - Cumulative sum over a specified dimension
-   product                    - Total product over a specified dimension
-   cumproduct                 - Cumulative product over a specified dimension
-   alltrue                    - Logical and over an entire axis
-   sometrue                   - Logical or over an entire axis
-   allclose               - Tests if sequences are essentially equal

More Functions:

-   arrayrange (arange)        - Return regularly spaced array
-   asarray                    - Guarantee NumPy array
-   sarray                     - Guarantee a NumPy array that keeps precision 
-   convolve                   - Convolve two 1-d arrays
-   swapaxes                   - Exchange axes
-   concatenate                - Join arrays together
-   transpose                  - Permute axes
-   sort                       - Sort elements of array
-   argsort                    - Indices of sorted array
-   argmax                     - Index of largest value                      
-   argmin                     - Index of smallest value
-   innerproduct               - Innerproduct of two arrays
-   dot                        - Dot product (matrix multiplication)
-   outerproduct               - Outerproduct of two arrays
-   resize                     - Return array with arbitrary new shape
-   indices                    - Tuple of indices
-   fromfunction               - Construct array from universal function
-   diagonal                   - Return diagonal array
-   trace                      - Trace of array
-   dump                       - Dump array to file object (pickle)
-   dumps                      - Return pickled string representing data
-   load                       - Return array stored in file object
-   loads                      - Return array from pickled string
-   ravel                      - Return array as 1-D 
-   nonzero                    - Indices of nonzero elements for 1-D array
-   shape                      - Shape of array
-   where                      - Construct array from binary result
-   compress                   - Elements of array where condition is true
-   clip                       - Clip array between two values
-   ones                       - Array of all ones
-   identity                   - 2-D identity array (matrix)

(Universal) Math Functions 

       add                    logical_or             exp        
       subtract               logical_xor            log        
       multiply               logical_not            log10      
       divide                 maximum                sin        
       divide_safe            minimum                sinh       
       conjugate              bitwise_and            sqrt       
       power                  bitwise_or             tan        
       absolute               bitwise_xor            tanh       
       negative               invert                 ceil       
       greater                left_shift             fabs       
       greater_equal          right_shift            floor      
       less                   arccos                 arctan2    
       less_equal             arcsin                 fmod       
       equal                  arctan                 hypot      
       not_equal              cos                    around     
       logical_and            cosh                   sign
       arccosh                arcsinh                arctanh

"""   

