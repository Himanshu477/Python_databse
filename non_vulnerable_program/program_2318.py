from genapi import fullapi_hash

if __name__ == '__main__':
    curdir = dirname(__file__)
    files = [join(curdir, 'numpy_api_order.txt'),
             join(curdir, 'ufunc_api_order.txt')]
    print fullapi_hash(files)


"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.

Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceeded by a blank line.

"""
