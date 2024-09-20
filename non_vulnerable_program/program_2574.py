import numpy

# We detect whether Cython is available, so that below, we can eventually ship
# pre-generated C for users to compile the extension without having Cython
# installed on their systems.
try:
