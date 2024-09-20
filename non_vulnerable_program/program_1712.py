import datetime
def fromstr(filestr):
    #filestr = replacetypechars(filestr)
    filestr, fromall1 = changeimports(filestr, 'numpy.oldnumeric', 'numpy')
    filestr, fromall1 = changeimports(filestr, 'numpy.core.multiarray', 'numpy')
    filestr, fromall1 = changeimports(filestr, 'numpy.core.umath', 'numpy')
    filestr, fromall3 = changeimports(filestr, 'LinearAlgebra',
                                      'numpy.linalg.old')
    filestr, fromall3 = changeimports(filestr, 'RNG', 'numpy.random.oldrng')
    filestr, fromall3 = changeimports(filestr, 'RNG.Statistics', 'numpy.random.oldrngstats')
    filestr, fromall3 = changeimports(filestr, 'RandomArray', 'numpy.random.oldrandomarray')
    filestr, fromall3 = changeimports(filestr, 'FFT', 'numpy.fft.old')
    filestr, fromall3 = changeimports(filestr, 'MA', 'numpy.core.ma')
    fromall = fromall1 or fromall2 or fromall3
    filestr = replaceattr(filestr)
    filestr = replaceother(filestr)
    today = datetime.date.today().strftime('%b %d, %Y')
    name = os.path.split(sys.argv[0])[-1]
    filestr = '## Automatically adapted for '\
              'numpy %s by %s\n\n%s' % (today, name, filestr)
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

def convertfile(filename):
    """Convert the filename given from using Numeric to using NumPy

    Copies the file to filename.orig and then over-writes the file
    with the updated code
    """
    filestr = getandcopy(filename)
    filestr = fromstr(filestr)
    makenewfile(filename, filestr)

def fromargs(args):
    filename = args[1]
    convertfile(filename)

def convertall(direc=os.path.curdir):
    """Convert all .py files to use NumPy (from Numeric) in the directory given

    For each file, a backup of <usesnumeric>.py is made as
    <usesnumeric>.py.orig.  A new file named <usesnumeric>.py
    is then written with the updated code.
    """
    files = glob.glob(os.path.join(direc,'*.py'))
    for afile in files:
        convertfile(afile)

if __name__ == '__main__':
