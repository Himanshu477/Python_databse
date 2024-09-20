from UserList import UserList
obj = UserList([1,[1,2],"hello"])
code = """
       int i;
       // find obj length and accesss each of its items
       std::cout << "UserList items: ";
       for(i = 0; i < obj.length(); i++)
           std::cout << obj[i] << " ";
       std::cout << std::endl;
       // assign new values to each of its items
       for(i = 0; i < obj.length(); i++)
           obj[i] = "goodbye";
       """
weave.inline(code,['obj'])       
print "obj with new values:", obj

#!/usr/bin/env python
"""
Bundle of SciPy core modules:
  scipy_test, scipy_distutils

Usage:
   python setup_scipy_core.py install
   python setup_scipy_core.py sdist -f -t MANIFEST_scipy_core.in
"""

