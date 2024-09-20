import datetime
def fromstr(filestr):
    filestr = replacetypechars(filestr)
    filestr, fromall1 = changeimports(filestr, 'Numeric', 'numpy')
    filestr, fromall1 = changeimports(filestr, 'multiarray',
                                      'numpy.core.multiarray')
    filestr, fromall1 = changeimports(filestr, 'umath',
                                          'numpy.core.umath')
    filestr, fromall1 = changeimports(filestr, 'Precision', 'numpy.core')
    filestr, fromall2 = changeimports(filestr, 'numerix', 'numpy.core')
    filestr, fromall3 = changeimports(filestr, 'scipy_base', 'numpy.core')
    filestr, fromall3 = changeimports(filestr, 'MLab', 'numpy.core.mlab')
    filestr, fromall3 = changeimports(filestr, 'LinearAlgebra', 'numpy.linalg')
    filestr, fromall3 = changeimports(filestr, 'RNG', 'numpy.random')
    filestr, fromall3 = changeimports(filestr, 'RandomArray', 'numpy.random')
    filestr, fromall3 = changeimports(filestr, 'FFT', 'numpy.dft')
    filestr, fromall3 = changeimports(filestr, 'MA', 'numpy.core.ma')
    fromall = fromall1 or fromall2 or fromall3
    filestr = replaceattr(filestr)
    filestr = replaceother(filestr)
    today = datetime.date.today().strftime('%b %d, %Y')
    name = os.path.split(sys.argv[0])[-1]
    filestr = '## Automatically adapted for '\
              'scipy %s by %s\n\n%s' % (today, name, filestr)
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

def fromfile(filename):
    filestr = getandcopy(filename)
    filestr = fromstr(filestr)
    makenewfile(filename, filestr)
           
def fromargs(args):
    filename = args[1]
    fromfile(filename)

def convertall(direc=''):
    files = glob.glob(os.path.join(direc,'*.py'))
    for afile in files:
        fromfile(afile)

if __name__ == '__main__':
