from scipy.test.testing import ScipyTest 
test = ScipyTest().test


"""\
Basic tools
===========

linalg - lite version of scipy.linalg
fftpack - lite version of scipy.fftpack
helper - lite version of scipy.linalg.helper

"""

depends = ['base']
global_symbols = ['fft','ifft','rand','randn', 
                  'linalg','fftpack','random']


# To get sub-modules
