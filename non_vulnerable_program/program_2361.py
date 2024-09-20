from genapi import API_FILES, find_functions

def order_dict(d):
    """Order dict by its values."""
    o = d.items()
    def cmp(x, y):
        return x[1] - y[1]
    return sorted(o, cmp=cmp)

def merge_api_dicts(dicts):
    ret = {}
    for d in dicts:
        for k, v in d.items():
            ret[k] = v

    return ret

def check_api_dict(d):
    """Check that an api dict is valid (does not use the same index twice)."""
    # We have if a same index is used twice: we 'revert' the dict so that index
    # become keys. If the length is different, it means one index has been used
    # at least twice
    revert_dict = dict([(v, k) for k, v in d.items()])
    if not len(revert_dict) == len(d):
        # We compute a dict index -> list of associated items
        doubled = {}
        for name, index in d.items():
            try:
                doubled[index].append(name)
            except KeyError:
                doubled[index] = [name]
        msg = """\
Same index has been used twice in api definition: %s
""" % ['index %d -> %s' % (index, names) for index, names in doubled.items() \
                                          if len(names) != 1]
        raise ValueError(msg)

def get_api_functions(tagname, api_dict):
    """Parse source files to get functions tagged by the given tag."""
    functions = []
    for f in API_FILES:
        functions.extend(find_functions(f, tagname))
    dfunctions = []
    for func in functions:
        o = api_dict[func.name]
        dfunctions.append( (o, func) )
    dfunctions.sort()
    return [a[1] for a in dfunctions]


# -*- coding: iso-8859-1 -*-
"""Get useful information from live Python objects.

This module encapsulates the interface provided by the internal special
attributes (func_*, co_*, im_*, tb_*, etc.) in a friendlier fashion.
It also provides some help for examining source code and class layout.

Here are some of the useful functions provided by this module:

    ismodule(), isclass(), ismethod(), isfunction(), istraceback(),
        isframe(), iscode(), isbuiltin(), isroutine() - check object types
    getmembers() - get members of an object that satisfy a given condition

    getfile(), getsourcefile(), getsource() - find an object's source code
    getdoc(), getcomments() - get documentation on an object
    getmodule() - determine the module that an object came from
    getclasstree() - arrange classes so as to represent their hierarchy

    getargspec(), getargvalues() - get info about function arguments
    formatargspec(), formatargvalues() - format an argument spec
    getouterframes(), getinnerframes() - get info about frames
    currentframe() - get the current stack frame
    stack(), trace() - get info about frames on the stack or in a traceback
"""

# This module is in the public domain.  No warranties.

__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__date__ = '1 Jan 2001'

