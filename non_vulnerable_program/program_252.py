import commands
exec_command = commands.getstatusoutput

def copy_file(src,dst,logger=None):
    if not logger:
        logger = logging
    logger.info("Copying %s->%s" % (src,dst))        
    try:
        file_util.copy_file(src,dst)
    except Exception, e:     
        logger.exception("Copy Failed")        
        raise

def copy_tree(src,dst,logger=None):
    if not logger:
        logger = logging
    logger.info("Copying directory tree %s->%s" % (src,dst))        
    try:
        dir_util.copy_tree(src,dst)
    except Exception, e:     
        logger.exception("Copy Failed")        
        raise

def remove_file(file,logger=None):
    if not logger:
        logger = logging
    logger.info("Remove file %s" % file)        
    try:
        os.remove(file)
    except Exception, e:     
        logger.exception("Remove failed")        
        raise

def write_file(file,contents,logger=None,mode='wb'):
    if not logger:
        logger = logging
    logger.info('Write file: %s' % file)
    try:
        new_file = open(file,mode)
        new_file.write(contents)
        new_file.close()
    except Exception, e:     
        logger.exception("Write failed")        
        raise

# I know, I know...
old_dir = []

def change_dir(d, logger = None):
    if not logger:
        logger = logging
    global old_dir 
    cwd = os.getcwd()   
    old_dir.append(cwd)
    d = os.path.abspath(d)
    if d != old_dir[-1]:
        logger.info("Change directory: %s" % d)            
        try:
            os.chdir(d)
        except Exception, e:     
            logger.exception("Change directory failed")
            raise        
        #if d == '.':
        #    import sys,traceback
        #    f = sys._getframe()
        #    traceback.print_stack(f)

def unchange_dir(logger=None):
    if not logger:
        logger = logging            
    global old_dir
    try:
        cwd = os.getcwd()
        d = old_dir.pop(-1)            
        try:
            if d != cwd:
                logger.info("Change directory : %s" % d)
                os.chdir(d)
        except Exception, e:     
            logger.exception("Change directory failed")
            raise                    
    except IndexError:
        logger.exception("Change directory failed")
        
def decompress_file(src,dst,logger = None):
    if not logger:
        logger = logging
    logger.info("Upacking %s->%s" % (src,dst))
    try:
        f = gzip.open(src,'rb')
        contents = f.read(-1)
        f = open(dst, 'wb')
        f.write(contents)
    except Exception, e:     
        logger.exception("Unpack failed")
        raise        

    
def untar_file(file,dst_dir='.',logger = None,silent_failure = 0):    
    if not logger:
        logger = logging
    logger.info("Untarring file: %s" % (file))
    try:
        run_command('tar -xf ' + file,directory = dst_dir,
                    logger=logger, silent_failure = silent_failure)
    except Exception, e:
        if not silent_failure:     
            logger.exception("Untar failed")
        raise        

def unpack_file(file,logger = None):
    """ equivalent to 'tar -xzvf file'
    """
    dst = 'tmp.tar'
    decompress_file(file,dst,logger)                
    untar_file(dst.logger)
    remove_file(dst,logger)        

def run_command(cmd,directory='.',logger=None,silent_failure = 0):
    if not logger:
        logger = logging
    change_dir(directory,logger)    
    try:        
        msg = 'Command: %s' % cmd
        status,text = exec_command(cmd)
        if status:
            msg = msg + ' (failed)'
        logger.info(msg)    
        if status and not silent_failure:
            logger.error('command failed with status: %d' % status)    
        #if text:
        #    logger.info('Command Output:\n'+text)
    finally:
        unchange_dir(logger)
    if status:
        raise ValueError,'Command had non-zero exit status'
    return text            

def full_scipy_build(build_dir = '.',
                     python_version  = '2.2.1',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot'):
    
    # for now the atlas version is ignored.  Only the 
    # binaries for RH are supported at the moment.
    
    dst_dir = os.path.join(build_dir,sys.platform)

    logger = logging.Logger("SciPy Test")
    fmt = logging.Formatter(logging.BASIC_FORMAT)
    log_stream = cStringIO.StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    # also write to stderr
    stderr = logging.StreamHandler()
    stderr.setFormatter(fmt)
    logger.addHandler(stderr)
    
    python = python_installation(version=python_version,
                                 logger = logger,
                                 dst_dir = dst_dir)
    python.install()
    
    python_name = python.get_exe_name()

    numeric = numeric_installation(version=numeric_version,
                                   dst_dir = dst_dir,
                                   logger = logger,
                                   python_exe=python_name)
    numeric.install()
    
    f2py =  f2py_installation(version=f2py_version,
                              logger = logger,
                              dst_dir = dst_dir,
                              python_exe=python_name)
    f2py.install()                                

    # download files don't have a version specified    
    #lapack =  lapack_installation(version='',
    #                              dst_dir = dst_dir
    #                              python_exe=python_name)
    #lapack.install()                                

    # download files don't have a version specified    
    #blas =  blas_installation(version='',
    #                          logger = logger,
    #                          dst_dir = dst_dir,
    #                          python_exe=python_name)
    #blas.install()                                
    
    # ATLAS
    atlas =  atlas_installation(version=atlas_version,
                                logger = logger,
                                dst_dir = dst_dir,
                                python_exe=python_name)
    atlas.install()
    
    # version not currently used -- need to fix this.
    scipy =  scipy_installation(version=scipy_version,
                                logger = logger,
                                dst_dir = dst_dir,
                                python_exe=python_name)
    scipy.install()                                

    # The change to tmp makes sure there isn't a scipy directory in 
    # the local scope.
    # All tests are run.
    lvl = 1
    cmd = python_name + ' -c "import sys,scipy;suite=scipy.test(%d);"' % lvl
    test_results = run_command(cmd, logger=logger,
                               directory = tempfile.gettempdir())
    
    vitals = '%s,py%s,num%s,scipy%s' % (sys.platform,python_version,
                                        numeric_version,scipy_version) 
                                 
    msg =       'From: scipy-test@enthought.com\n'
    msg = msg + 'To: knucklehead@enthought.com\n'
    msg = msg + 'Subject: %s\n'                     % vitals
    msg = msg + '\r\n\r\n'
    msg = msg + 'platform:                  %s,%s\n' % (sys.platform,os.name)
    msg = msg + 'python_version:            %s\n' % python_version
    msg = msg + 'numeric_version:           %s\n' % numeric_version
    msg = msg + 'f2py_version:              %s\n' % f2py_version
    msg = msg + 'atlas_version(hard coded): %s\n' % atlas_version
    msg = msg + 'scipy_version:             %s\n' % scipy_version
    msg = msg + test_results    
    msg = msg + '-----------------------------\n' 
    msg = msg + '--------  BUILD LOG   -------\n' 
    msg = msg + '-----------------------------\n' 
    msg = msg + log_stream.getvalue()
    # mail results
    import smtplib 
    toaddrs = "eric@enthought.com"
    fromaddr = "eric@enthought.com"
    server = smtplib.SMTP('enthought.com')    
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()
    print msg

    
if __name__ == '__main__':
    build_dir = '/tmp/scipy_test'

    full_scipy_build(build_dir = build_dir,
                     python_version  = '2.2.1',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # an older python
    full_scipy_build(build_dir = build_dir,
                     python_version  = '2.1.3',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # an older numeric
    full_scipy_build(build_dir = build_dir,
                     python_version  = '2.1.3',
                     numeric_version = '20.3',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    full_scipy_build(build_dir = build_dir,
                     python_version  = '2.1.3',
                     numeric_version = '20.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    full_scipy_build(build_dir = build_dir,
                     python_version  = '2.1.3',
                     numeric_version = '19.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    full_scipy_build(build_dir = build_dir,
                     python_version  = '2.1.3',
                     numeric_version = '18.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')


#! /usr/bin/env python
#
# Copyright 2001-2002 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# For the change history, see README.txt in the distribution.
#
# This file is part of the Python logging distribution. See
# http://www.red-dove.com/python_logging.html
#

"""
Logging module for Python. Based on PEP 282 and comments thereto in
comp.lang.python, and influenced by Apache's log4j system.

Should work under Python versions >= 1.5.2, except that source line
information is not available unless 'inspect' is.

Copyright (C) 2001-2002 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging' and log away!
"""

