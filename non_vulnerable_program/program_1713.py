    fromargs(sys.argv)


"""
This module converts code written for Numeric to run with numpy

Makes the following changes:
 * Changes import statements

   Stubs for
   convolve --> numarray.convolve
   image --> numarray.image
   nd_image --> numarray.nd_image

 * Makes search and replace changes to:
   - .imaginary --> .imag
   - .flat --> .ravel() (most of the time)
   - .byteswapped() --> .byteswap(False)
   - .byteswap() --> .byteswap(True)
   - .info() --> numarray.info(self)
   - .isaligned() --> .flags.aligned
   - .isbyteswapped() --> (not .dtype.isnative)
   - .typecode() --> .dtype.char
   - .iscontiguous() --> .flags.contiguous
   - .is_c_array() --> .flags.carray and .dtype.isnative
   - .is_fortran_contiguous() --> .flags.fortran
   - .is_f_array() --> .dtype.isnative and .flags.farray
   - .itemsize() --> .itemsize
   - .nelements() --> .size
   - self.new(None) --> emtpy_like(self)
   - self.new(type) --> empty(self.shape, type)
   - .repeat(r) --> .repeat(r, axis=0)
   - .size() --> .size
   - .type() -- numarray.type(self.dtype)
   - .typecode() --> .dtype.char
   - .stddev() --> .std()
   - .togglebyteorder() --> self.dtype=self.dtype.newbyteorder()
   - .getshape() --> .shape
   - .setshape(obj) --> .shape=obj
   - .getflat() --> .ravel()
   - .getreal() --> .real
   - .setreal() --> .real =
   - .getimag() --> .imag
   - .setimag() --> .imag =
   - .getimaginary() --> .imag
   - .setimaginary() --> .imag
   
"""
__all__ = ['fromfile', 'fromstr']

