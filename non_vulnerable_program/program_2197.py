import scipy.signal as ss
b2 = ss.convolve(h,a,'same')


# This script takes a lyx file and runs the python code in it.
#  Then rewrites the lyx file again.
#
# Each section of code portion is assumed to be in the same namespace
# where a from numpy import * has been applied
#
#  If a PYNEW inside a Note is encountered, the name space is restarted
#
# The output (if any) is replaced in the file
#  by the output produced during the code run.
#
# Options:
#   -n name of code section  (default MyCode)
#   

