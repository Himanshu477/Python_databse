    import numpy.fftpack
except ImportError:
    pass
else:
    fft = numpy.fftpack.fft
    ifft = numpy.fftpack.ifft
    fftn = numpy.fftpack.fftn
    ifftn = numpy.fftpack.ifftn
    fft2 = numpy.fftpack.fft2
    ifft2 = numpy.fftpack.ifft2


