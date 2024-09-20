import sys
import os
import re
import glob

flatindex_re = re.compile('([.]flat(\s*?[[=]))')

def replacetypechars(astr):
#    astr = astr.replace("'s'","'h'")
#    astr = astr.replace("'c'","'S1'")
    astr = astr.replace("'b'","'B'")
    astr = astr.replace("'1'","'b'")
#    astr = astr.replace("'w'","'H'")
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
    astr = astr.replace(".typecode()",".dtypechar")
    astr = astr.replace(".iscontiguous()",".flags.contiguous")
    astr = astr.replace(".byteswapped()",".byteswap()")
    astr = astr.replace(".toscalar()", ".item()")
    astr = astr.replace(".itemsize()",".itemsize")
    # preserve uses of flat that should be o.k.
    tmpstr = flatindex_re.sub("@@@@\\2",astr)
    # replace other uses of flat
    tmpstr = tmpstr.replace(".flat",".ravel()")
    # put back .flat where it was valid
    astr = tmpstr.replace("@@@@", ".flat")
    return astr

svspc = re.compile(r'(\S+\s*[(].+),\s*savespace\s*=.+\s*[)]')
svspc2 = re.compile(r'([^,(\s]+[.]spacesaver[(][)])')
svspc3 = re.compile(r'(\S+[.]savespace[(].*[)])')
#shpe = re.compile(r'(\S+\s*)[.]shape\s*=[^=]\s*(.+)')
def replaceother(astr):
    astr = astr.replace("typecode=","dtype=")
    astr = astr.replace("UserArray","ndarray")
    astr = svspc.sub('\\1)',astr)
    astr = svspc2.sub('True',astr)
    astr = svspc3.sub('pass  ## \\1', astr)
    #astr = shpe.sub('\\1=\\1.reshape(\\2)', astr)
    return astr

