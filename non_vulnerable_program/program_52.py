import os, cxx_info
local_dir,junk = os.path.split(os.path.abspath(cxx_info.__file__))   
cxx_dir = os.path.join(local_dir,'CXX')

class cxx_info(base_info.base_info):
    _headers = ['"CXX/Objects.hxx"','"CXX/Extensions.hxx"','<algorithm>']
    _include_dirs = [local_dir]

    # should these be built to a library??
    _sources = [os.path.join(cxx_dir,'cxxsupport.cxx'),
                os.path.join(cxx_dir,'cxx_extensions.cxx'),
                os.path.join(cxx_dir,'IndirectPythonInterface.cxx'),
                os.path.join(cxx_dir,'cxxextensions.c')]
    _support_code = [string_support_code,list_support_code, dict_support_code,
                     tuple_support_code]


# Offers example of inline C for binary search algorithm.
# Borrowed from Kalle Svensson in the Python Cookbook.
# The results are nearly in the "not worth it" catagory.
#
# C:\home\ej\wrk\scipy\compiler\examples>python binary_search.py
# Binary search for 3000 items in 100000 length list of integers:
#  speed in python: 0.139999985695
#  speed in c: 0.0900000333786
#  speed up: 1.41
# search(a,3450) 3450 3450
# search(a,-1) -1 -1
# search(a,10001) 10001 10001
#
# Note -- really need to differentiate between conversion errors and
# run time errors.  This would reduce useless compiles and provide a
# more intelligent control of things.

