import sys
import os
import re
import glob

# To convert typecharacters we need to 
# Not very safe.  Disabled for now..
def replacetypechars(astr):
    astr = astr.replace("'s'","'h'")
    astr = astr.replace("'b'","'B'")
    astr = astr.replace("'1'","'b'")
    astr = astr.replace("'w'","'H'")
    astr = astr.replace("'u'","'I'")
    return astr

def changeimports(fstr, name, newname):
    importstr = 'import %s' % name
    importasstr = 'import %s as ' % name
    fromstr = 'from %s import ' % name
    fromall=0

    fstr = fstr.replace(importasstr, 'import %s as ' % newname)
    fstr = fstr.replace(importstr, 'import %s as %s' % (newname,name))

    ind = 0
    Nlen = len(fromstr)
    Nlen2 = len("from %s import " % newname)
    while 1:
        found = fstr.find(fromstr,ind)
        if (found < 0):
            break
        ind = found + Nlen
        if fstr[ind] == '*':
            continue
        fstr = "%sfrom %s import %s" % (fstr[:found], newname, fstr[ind:])
        ind += Nlen2 - Nlen
    return fstr, fromall

def replaceattr(astr):
     astr = astr.replace("matrixmultiply","dot")
    return astr

def replaceother(astr):
    astr = re.sub(r'typecode\s*=', 'dtype=', astr)
    astr = astr.replace('ArrayType', 'ndarray')
    astr = astr.replace('NewAxis', 'newaxis')
    return astr

