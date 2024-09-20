from numpy.fft import fft
from numpy.fft import fft2 as fft2d
from numpy.fft import fftn as fftnd
from numpy.fft import hfft as hermite_fft
from numpy.fft import ifft as inverse_fft
from numpy.fft import ifft2 as inverse_fft2d
from numpy.fft import ifftn as inverse_fftnd
from numpy.fft import ihfft as inverse_hermite_fft
from numpy.fft import irfft as inverse_real_fft
from numpy.fft import irfft2 as inverse_real_fft2d
from numpy.fft import irfftn as inverse_real_fftnd
from numpy.fft import rfft as real_fft
from numpy.fft import rfft2 as real_fft2d
from numpy.fft import rfftn as real_fftnd


"""
This module converts code written for Numeric to run with numpy

Makes the following changes:
 * Changes import statements (warns of use of from Numeric import *)
 * Changes import statements (using numerix) ...
 * Makes search and replace changes to:
   - .typecode()
   - .iscontiguous()
   - .byteswapped()
   - .itemsize()
   - .toscalar()
 * Converts .flat to .ravel() except for .flat = xxx or .flat[xxx]
 * Replace xxx.spacesaver() with True
 * Convert xx.savespace(?) to pass + ## xx.savespace(?)

 * Converts uses of 'b' to 'B' in the typecode-position of
   functions:
   eye, tri (in position 4)
   ones, zeros, identity, empty, array, asarray, arange,
