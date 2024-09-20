import sys
import optparse
import cStringIO
import re
import os

newre = re.compile(r"\\begin_inset Note.*PYNEW\s+\\end_inset", re.DOTALL)

def getoutput(tstr, dic):
    print "\n\nRunning..."
    print tstr,    
    tempstr = cStringIO.StringIO()
    sys.stdout = tempstr
    code = compile(tstr, '<input>', 'exec')
    try:
        res = eval(tstr, dic)
        sys.stdout = sys.__stdout__        
    except SyntaxError:
        try:
            res = None
            exec code in dic
        finally:
            sys.stdout = sys.__stdout__
    if res is None:
        res = tempstr.getvalue()
    else:
        res = tempstr.getvalue() + '\n' + repr(res)
    if res != '':
        print "\nOutput is"
        print res,   
    return res

# now find the code in the code segment
def getnewcodestr(substr, dic):
    end = substr.find('\\layout ')
    lines = substr[:end].split('\\newline')
    outlines = []
    first = 1
    cmd = ''
    lines.append('dummy')
    for line in lines:
        line = line.strip()
        if (line[:3]=='>>>') or (line == 'dummy'):
            # we have a new output
            pyoutstr = getoutput(cmd, dic).strip()
            if pyoutstr != '':
                pyout = pyoutstr.split('\n')
                outlines.extend(pyout)
            cmd = line[4:]
        elif (line[:3]=='...'):
            # continuation output
            cmd += "\n%s" % line[4:]
        else:
            # first line or output
            if first:
                first = 0
                cmd = line
            else:
                continue
        if line != 'dummy':
            outlines.append(line)
    return "\n\\newline \n".join(outlines), end
            

def runpycode(lyxstr, name='MyCode'):
    schobj = re.compile(r"\\layout %s\s+>>> " % name)
    outstr = cStringIO.StringIO()
    num = 0
    indx = []
    for it in schobj.finditer(lyxstr):
        indx.extend([it.start(), it.end()])
        num += 1
        
    if num == 0:
        print "Nothing found for %s" % name
        return lyxstr

    start = 0
    del indx[0]
    indx.append(len(lyxstr))
    edic = {}
    exec 'from numpy import *' in edic
    exec 'set_printoptions(linewidth=65)' in edic
    # indx now contains [st0,en0, ..., stN,enN]
    #  where stX is the start of code segment X
    #  and enX is the start of \layout MyCode for
    #  the X+1 code section (or string length if X=N)
    for k in range(num):
        # first write everything up to the start of the code segment
        substr = lyxstr[start:indx[2*k]]
        outstr.write(substr)        
        if start > 0:
            mat = newre.search(substr)
            # if PYNEW found, then start a new namespace
            if mat:
                edic = {}
                exec 'from numpy import *' in edic
                exec 'set_printoptions(linewidth=65)' in edic               
        # now find the code in the code segment
        # endoutput will contain the index just past any output
        #  already present in the lyx string.
        substr = lyxstr[indx[2*k]:indx[2*k+1]]
        lyxcodestr, endcode = getnewcodestr(substr, edic)
        # write the lyx for the input + new output
        outstr.write(lyxcodestr)
        outstr.write('\n')
        start = endcode + indx[2*k]

    outstr.write(lyxstr[start:])
    return outstr.getvalue()


def main(args):
    usage = "%prog {options} filename"
    parser = optparse.OptionParser(usage)
    parser.add_option('-n','--name', default='MyCode')

    options, args = parser.parse_args(args)
    if len(args) < 1:
        parser.error("incorrect number of arguments")

    os.system('cp -f %s %s.bak' % (args[0], args[0]))
    fid = file(args[0])
    str = fid.read()
    fid.close()
    print "Processing %s" % options.name
    newstr = runpycode(str, options.name)
    fid = file(args[0],'w')
    fid.write(newstr)
    fid.close()

if __name__ == "__main__":
    main(sys.argv[1:])


"""
=============================
Subclassing ndarray in python
=============================

Credits
-------

This page is based with thanks on the wiki page on subclassing by Pierre
Gerard-Marchant - http://www.scipy.org/Subclasses. 

Introduction
------------
Subclassing ndarray is relatively simple, but you will need to
understand some behavior of ndarrays to understand some minor
complications to subclassing.  There are examples at the bottom of the
page, but you will probably want to read the background to understand
why subclassing works as it does.

ndarrays and object creation
============================
The creation of ndarrays is complicated by the need to return views of
ndarrays, that are also ndarrays.  For example::

  >>> import numpy as np
  >>> arr = np.zeros((3,))
  >>> type(arr)
  <type 'numpy.ndarray'>
  >>> v = arr[1:]
  >>> type(v)
  <type 'numpy.ndarray'>
  >>> v is arr
  False

So, when we take a view (here a slice) from the ndarray, we return a
new ndarray, that points to the data in the original.  When we
subclass ndarray, taking a view (such as a slice) needs to return an
object of our own class.  There is machinery to do this, but it is
this machinery that makes subclassing slightly non-standard.

To allow subclassing, and views of subclasses, ndarray uses the
ndarray ``__new__`` method for the main work of object initialization,
rather then the more usual ``__init__`` method.  

``__new__`` and ``__init__``
============================

``__new__`` is a standard python method, and, if present, is called
before ``__init__`` when we create a class instance. Consider the
following::  

  class C(object):
      def __new__(cls, *args):
      print 'Args in __new__:', args
      return object.__new__(cls, *args)
      def __init__(self, *args):
      print 'Args in __init__:', args

  C('hello')

The code gives the following output::

  cls is: <class '__main__.C'>
  Args in __new__: ('hello',)
  self is : <__main__.C object at 0xb7dc720c>
  Args in __init__: ('hello',)

When we call ``C('hello')``, the ``__new__`` method gets its own class
as first argument, and the passed argument, which is the string
``'hello'``.  After python calls ``__new__``, it usually (see below)
calls our ``__init__`` method, with the output of ``__new__`` as the
first argument (now a class instance), and the passed arguments
following.

As you can see, the object can be initialized in the ``__new__``
method or the ``__init__`` method, or both, and in fact ndarray does
not have an ``__init__`` method, because all the initialization is
done in the ``__new__`` method. 

Why use ``__new__`` rather than just the usual ``__init__``?  Because
in some cases, as for ndarray, we want to be able to return an object
of some other class.  Consider the following::

  class C(object):
      def __new__(cls, *args):
      print 'cls is:', cls
      print 'Args in __new__:', args
      return object.__new__(cls, *args)
      def __init__(self, *args):
      print 'self is :', self
      print 'Args in __init__:', args

  class D(C):
      def __new__(cls, *args):
      print 'D cls is:', cls
      print 'D args in __new__:', args
      return C.__new__(C, *args)
      def __init__(self, *args):
      print 'D self is :', self
      print 'D args in __init__:', args

  D('hello')

which gives::

  D cls is: <class '__main__.D'>
  D args in __new__: ('hello',)
  cls is: <class '__main__.C'>
  Args in __new__: ('hello',)

The definition of ``C`` is the same as before, but for ``D``, the
``__new__`` method returns an instance of class ``C`` rather than
``D``.  Note that the ``__init__`` method of ``D`` does not get
called.  In general, when the ``__new__`` method returns an object of
class other than the class in which it is defined, the ``__init__``
method of that class is not called.

This is how subclasses of the ndarray class are able to return views
that preserve the class type.  When taking a view, the standard
ndarray machinery creates the new ndarray object with something
like::

  obj = ndarray.__new__(subtype, shape, ...

where ``subdtype`` is the subclass.  Thus the returned view is of the
same class as the subclass, rather than being of class ``ndarray``.

That solves the problem of returning views of the same type, but now
we have a new problem.  The machinery of ndarray can set the class
this way, in its standard methods for taking views, but the ndarray
``__new__`` method knows nothing of what we have done in our own
``__new__`` method in order to set attributes, and so on.  (Aside -
why not call ``obj = subdtype.__new__(...`` then?  Because we may not
have a ``__new__`` method with the same call signature).  

So, when creating a new view object of our subclass, we need to be
able to set any extra attributes from the original object of our
class. This is the role of the ``__array_finalize__`` method of
ndarray.  ``__array_finalize__`` is called from within the
ndarray machinery, each time we create an ndarray of our own class,
and passes in the new view object, created as above, as well as the
old object from which the view has been taken.  In it we can take any
attributes from the old object and put then into the new view object,
or do any other related processing.  Now we are ready for a simple
example.

Simple example - adding an extra attribute to ndarray
-----------------------------------------------------

::
