import re
import os

_has_f_extension = re.compile(r'.*[.](for|ftn|f77|f)\Z',re.I).match
_has_f_header = re.compile(r'-[*]-\s*fortran\s*-[*]-',re.I).search
_has_f90_header = re.compile(r'-[*]-\s*f90\s*-[*]-',re.I).search
_has_fix_header = re.compile(r'-[*]-\s*fix\s*-[*]-',re.I).search
_free_f90_start = re.compile(r'[^c*!]\s*[^\s\d\t]',re.I).match

def get_source_info(filename):
    """
    Determine if fortran file is
      - in fix format and contains Fortran 77 code    -> False, True
      - in fix format and contains Fortran 90 code    -> False, False
      - in free format and contains Fortran 90 code   -> True, False
      - in free format and contains signatures (.pyf) -> True, True
    """
    base,ext = os.path.splitext(filename)
    if ext=='.pyf':
        return True, True
    isfree = False
    isstrict = False
    f = open(filename,'r')
    firstline = f.readline()
    f.close()
    if _has_f_extension(filename) and \
       not (_has_f90_header(firstline) or _has_fix_header(firstline)):
        isstrict = True
    elif is_free_format(filename) and not _has_fix_header(firstline):
        isfree = True
    return isfree,isstrict

def is_free_format(file):
    """Check if file is in free format Fortran."""
    # f90 allows both fixed and free format, assuming fixed unless
    # signs of free format are detected.
    isfree = False
    f = open(file,'r')
    line = f.readline()
    n = 10000 # the number of non-comment lines to scan for hints
    if _has_f_header(line):
        n = 0
    elif _has_f90_header(line):
        n = 0
        isfree = True
    while n>0 and line:
        if line[0]!='!' and line.strip():
            n -= 1
            if (line[0]!='\t' and _free_f90_start(line[:5])) or line[-2:-1]=='&':
                isfree = True
                break
        line = f.readline()
    f.close()
    return isfree


#!/usr/bin/env python
"""
Defines LineSplitter.

Copyright 2006 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision:$
$Date: 2000/07/31 07:04:03 $
Pearu Peterson
"""

__all__ = ['LineSplitter','String']

class String(str): pass

class LineSplitter:
    """ Splits a line into non strings and strings. E.g.
    abc=\"123\" -> ['abc=','\"123\"']
    Handles splitting lines with incomplete string blocks.
    """
    def __init__(self, line, quotechar = None, lower=False):
        self.fifo_line = [c for c in line]
        self.fifo_line.reverse()
        self.quotechar = quotechar
        self.lower = lower

    def __iter__(self):
        return self

    def next(self):
        item = ''
        while not item:
            item = self.get_item() # get_item raises StopIteration
        return item

    def get_item(self):
        fifo_pop = self.fifo_line.pop
        try:
            char = fifo_pop()
        except IndexError:
            raise StopIteration
        fifo_append = self.fifo_line.append
        quotechar = self.quotechar
        l = []
        l_append = l.append
        
        nofslashes = 0
        if quotechar is None:
            # search for string start
            while 1:
                if char in '"\'' and not nofslashes % 2:
                    self.quotechar = char
                    fifo_append(char)
                    break
                if char=='\\':
                    nofslashes += 1
                else:
                    nofslashes = 0
                l_append(char)
                try:
                    char = fifo_pop()
                except IndexError:
                    break
            item = ''.join(l)
            if self.lower: item = item.lower()
            return item

        if char==quotechar:
            # string starts with quotechar
            l_append(char)
            try:
                char = fifo_pop()
            except IndexError:
                return String(''.join(l))
        # else continued string
        while 1:
            if char==quotechar and not nofslashes % 2:
                l_append(char)
                self.quotechar = None
                break
            if char=='\\':
                nofslashes += 1
            else:
                nofslashes = 0
            l_append(char)
            try:
                char = fifo_pop()
            except IndexError:
                break
        return String(''.join(l))
                
def test():
    splitter = LineSplitter('abc\\\' def"12\\"3""56"dfad\'a d\'')
    l = [item for item in splitter]
    assert l==['abc\\\' def','"12\\"3"','"56"','dfad','\'a d\''],`l`
    assert splitter.quotechar is None

    splitter = LineSplitter('"abc123&')
    l = [item for item in splitter]
    assert l==['"abc123&'],`l`
    assert splitter.quotechar=='"'

    splitter = LineSplitter(' &abc"123','"')
    l = [item for item in splitter]
    assert l==[' &abc"','123']
    assert splitter.quotechar is None

if __name__ == '__main__':
    test()



#!/usr/bin/env python
"""
Defines Block classes.

Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006

"""

__all__ = ['Block','Module','PythonModule','Interface',
           'Subroutine','Function','Type']

