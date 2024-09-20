import iotest
#except: put('unsuccesful\\n');sys.exit()
#put('successful\\n')
""")
# Body
i=0
for k in mmap.keys():
    i=i+1
    ft=k
    p=`i`
    if string.find(k,'logical')>=0:
        ff.write("""
      function f%sio(f%sin,f%sout,f%sinout)
        %s f%sin,f%sout,f%sinout,f%sio
        f%sout = f%sin
        f%sinout = .not.f%sinout
        f%sio = .TRUE.
      end"""%(p,p,p,p,ft,p,p,p,p,p,p,p,p,p))
    else:
        ff.write("""
      function f%sio(f%sin,f%sout,f%sinout)
        %s f%sin,f%sout,f%sinout,f%sio
        f%sout = f%sin
        f%sinout = f%sinout + f%sinout
        f%sio = f%sinout + f%sin
      end"""%(p,p,p,p,ft,p,p,p,p,p,p,p,p,p,p,p,p))
    hf.write("""\
    interface
        function f%sio(f%sin,f%sout,f%sinout)
             %s f%sio
             %s intent(in) :: f%sin
             %s intent(in,out):: f%sout
             %s intent(inout):: f%sinout
        end
    end
"""%(p,p,p,p,ft,p,ft,p,ft,p,ft,p))
    py.write("""
i = %s
o = 0.0
io = array(%s,'%s')
print '\\n(%s)',iotest.f%sio.__doc__
if 1:
\tr,o=iotest.f%sio(i,o,io)
\tif r != %s:
\t\tprint 'FAILURE',
\telse:
\t\tprint 'SUCCESS',
\tprint '(%s:out)',`r`,'==',`%s`,'(expected)'
\tif o != %s:
\t\tprint 'FAILURE',
\telse:
\t\tprint 'SUCCESS',
\tprint '(%s:in,out)',`o`,'==',`%s`,'(expected)'
\tif io != %s:
\t\tprint 'FAILURE',
\telse:
\t\tprint 'SUCCESS',
\tprint '(%s:inout)',`io`,'==',`%s`,'(expected)'
print 'ok'
""" % (mmap[k]['in'],mmap[k]['inout'],mmap[k]['pref'],k,p,p,
       mmap[k]['ret'],k,mmap[k]['ret'],
       mmap[k]['out_res'],k,mmap[k]['out_res'],
       mmap[k]['inout_res'],k,mmap[k]['inout_res'],
       ))
# Close up
hf.write('end pythonmodule iotest')
ff.close()
hf.close()
py.close()



















#!/usr/bin/env python
"""
Usage:
  runme.py <scipy_distutils commands/options and --no-wrap-functions>

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision: 1.4 $
$Date: 2002/01/09 21:56:31 $
Pearu Peterson
"""

__version__ = "$Id: runme.py,v 1.4 2002/01/09 21:56:31 pearu Exp $"

