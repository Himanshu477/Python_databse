import sys
import os
import re
import glob


_func4 = ['eye', 'tri']
_meth1 = ['astype']
_func2 = ['ones', 'zeros', 'identity', 'fromstring', 'indices',
         'empty', 'array', 'asarray', 'arange', 'array_constructor']

func_re = {}

for name in _func2:
    _astr = r"""(%s\s*[(][^,]*?[,][^'"]*?['"])b(['"][^)]*?[)])"""%name
    func_re[name] = re.compile(_astr, re.DOTALL)

for name in _func4:
    _astr = r"""(%s\s*[(][^,]*?[,][^,]*?[,][^,]*?[,][^'"]*?['"])b(['"][^)]*?[)])"""%name
    func_re[name] = re.compile(_astr, re.DOTALL)    

for name in _meth1:
    _astr = r"""(.%s\s*[(][^'"]*?['"])b(['"][^)]*?[)])"""%name
    func_re[name] = re.compile(_astr, re.DOTALL)

def fixtypechars(fstr):
    for name in _func2 + _func4 + _meth1:
        fstr = func2_re[name].sub('\\1B\\2',fstr)
    return fstr

flatindex_re = re.compile('([.]flat(\s*?[[=]))')

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
    astr = astr.replace(".typecode()",".dtype.char")
    astr = astr.replace(".iscontiguous()",".flags.contiguous")
    astr = astr.replace(".byteswapped()",".byteswap()")
    astr = astr.replace(".toscalar()", ".item()")
    astr = astr.replace(".itemsize()",".itemsize")
    # preserve uses of flat that should be o.k.
    tmpstr = flatindex_re.sub(r"@@@@\2",astr)
    # replace other uses of flat
    tmpstr = tmpstr.replace(".flat",".ravel()")
    # put back .flat where it was valid
    astr = tmpstr.replace("@@@@", ".flat")
    return astr

svspc2 = re.compile(r'([^,(\s]+[.]spacesaver[(][)])')
svspc3 = re.compile(r'(\S+[.]savespace[(].*[)])')
#shpe = re.compile(r'(\S+\s*)[.]shape\s*=[^=]\s*(.+)')
def replaceother(astr):
    astr = svspc2.sub('True',astr)
    astr = svspc3.sub(r'pass  ## \1', astr)
    #astr = shpe.sub('\\1=\\1.reshape(\\2)', astr)
    return astr

