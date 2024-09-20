    from Numeric import UfuncType
except ImportError:
    UfuncType = type(Numeric.sin)

__all__ = []
for k in globals().keys():
    if k[0] != "_":
        __all__.append(k)
__all__.append("_insert")
__all__.append("_unique")



"""numerix  imports either Numeric or numarray based on various selectors.

0.  If the value "--numarray" or "--Numeric" is specified on the
command line, then numerix imports the specified array package.

1. If the environment variable NUMERIX exists,  it's value is used to
choose Numeric or numarray.

2. The value of numerix in ~/.matplotlibrc: either Numeric or numarray
<currently not implemented for scipy>

3. If none of the above is done, the default array package is Numeric.
Because the .matplotlibrc always provides *some* value for numerix (it
has it's own system of default values), this default is most likely
never used.

To summarize: the  commandline is examined first, the  rc file second,
and the default array package is Numeric.  
"""

