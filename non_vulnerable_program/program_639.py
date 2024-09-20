importre = re.compile('import\s+?(\S+?[^,]+?,\s*)+?')
def expand_import(astr):
    
    return astr


def changeimports(fstr, name, newname):
    fstr = expand_import(fstr)    
    importstr = 'import %s' % name
    importasstr = 'import %s as ' % name
    fromstr = 'from %s import ' % name
    fromallstr = 'from %s import *' % name
    fromall=0

    fstr = fstr.replace(importasstr, 'import %s as ' % newname)
    fstr = fstr.replace(importstr, 'import %s as %s' % (newname,name))
    if (fstr.find(fromallstr) >= 0):
        warnings.warn('Usage of %s found.' % fromallstr)
        fstr = fstr.replace(fromallstr, 'from %s import *' % newname)
        fromall=1

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
    astr = astr.replace(".iscontiguous()",".flags['CONTIGUOUS']")
    astr = astr.replace(".byteswapped()",".byteswap()")
    astr = astr.replace(".itemsize()",".itemsize")

    # preserve uses of flat that should be o.k.
    tmpstr = flatindex_re.sub("@@@@\\2",astr)
    # replace other uses of flat
    tmpstr = tmpstr.replace(".flat",".ravel()")
    # put back .flat where it was valid
    astr = tmpstr.replace("@@@@", ".flat")

    return astr

def replaceother(astr):
    astr = astr.replace("typecode=","dtype=")
    astr = astr.replace("UserArray","ndarray")
    return astr

def warnofnewtypes(filestr):
    if int_re.search(filestr) or \
       float_re.search(filestr) or \
       complex_re.search(filestr) or \
       unicode_re.search(filestr) or \
       bool_re.search(filestr):
        warnings.warn("Use of builtin bool, int, float, complex, or unicode\n" \
                      "found when import * used -- these will be handled by\n" \
                      "new array scalars under scipy.base")
        
    return
    

def process(filestr):
    filestr = replacetypechars(filestr)
    filestr, fromall1 = changeimports(filestr, 'Numeric', 'scipy.base')
    filestr, fromall1 = changeimports(filestr, 'multiarray',
                                      'scipy.base.multiarray')
    filestr, fromall1 = changeimports(filestr, 'umath',
                                          'scipy.base.umath')
    filestr, fromall2 = changeimports(filestr, 'numerix', 'scipy.base')
    filestr, fromall3 = changeimports(filestr, 'scipy_base', 'scipy.base')
    filestr, fromall3 = changeimports(filestr, 'MLab', 'scipy.linalg')
    filestr, fromall3 = changeimports(filestr, 'LinearAlgebra', 'scipy.linalg')
    filestr, fromall3 = changeimports(filestr, 'RNG', 'scipy.stats')
    filestr, fromall3 = changeimports(filestr, 'RandomArray', 'scipy.stats')
    filestr, fromall3 = changeimports(filestr, 'FFT', 'scipy.fftpack')
    fromall = fromall1 or fromall2 or fromall3
    filestr = replaceattr(filestr)
    filestr = replaceother(filestr)
    if fromall:
        warnofnewtypes(filestr)
    return filestr

def makenewfile(name, filestr):
    fid = file(name, 'w')
    fid.write(filestr)
    fid.close()

def getandcopy(name):
    fid = file(name)
    filestr = fid.read()
    fid.close()
    base, ext = os.path.splitext(name)
    makenewfile(base+'.orig', filestr)
    return filestr
       
def main(args):
    filename = args[1]
    filestr = getandcopy(filename)
    filestr = process(filestr)
    makenewfile(filename, filestr)

if __name__ == '__main__':
    main(sys.argv)
    
             



# Compatibility module containing deprecated names

