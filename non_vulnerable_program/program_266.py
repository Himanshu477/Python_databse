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

def remove_tree(directory,logger=None):
    if not logger:
        logger = logging
    logger.info("Removing directory tree %s" % directory)        
    try:
        dir_util.remove_tree(directory)
    except Exception, e:     
        logger.exception("Remove failed: %s" % e)        
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

def make_dir(name,logger=None):
    if not logger:
        logger = logging
    logger.info('Make directory: %s' % name)
    try:        
        dir_util.mkpath(os.path.abspath(name))
    except Exception, e:     
        logger.exception("Make Directory failed")        
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
        msg = 'Running: %s' % cmd
        logger.info(msg)    
        status,text = exec_command(cmd)
        if status and silent_failure:
            msg = '(failed silently)'
            logger.info(msg)    
        if status and text and not silent_failure:
            logger.error('Command Failed (status=%d)\n'% status +text)
    finally:
        unchange_dir(logger)
    if status:
        raise ValueError, (status,text)
    return text            

def mail_report(from_addr,to_addr,subject,mail_server,
                build_log, test_results,info):
    
    msg = ''
    msg = msg + 'To: %s\n'   % to_addr
    msg = msg + 'Subject: %s\n' % subject
    msg = msg + '\r\n\r\n'

    for k,v in info.items():   
        msg = msg + '%s: %s\n' % (k,v)
    msg = msg + test_results + '\n'
    msg = msg + '-----------------------------\n' 
    msg = msg + '--------  BUILD LOG   -------\n' 
    msg = msg + '-----------------------------\n' 
    msg = msg + build_log
    print msg
    
    # mail results
    import smtplib 
    server = smtplib.SMTP(mail_server)    
    server.sendmail(from_addr, to_addr, msg)
    server.quit()
    

def full_scipy_build(build_dir = '.',
                     test_level = 10,
                     python_version  = '2.2.1',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot'):
    
    # for now the atlas version is ignored.  Only the 
    # binaries for RH are supported at the moment.

    build_info = {'python_version' : python_version,
                  'test_level'     : test_level,
                  'numeric_version': numeric_version,
                  'f2py_version'   : f2py_version,
                  'atlas_version'  : atlas_version,
                  'scipy_version'  : scipy_version}
                    
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

    try:
        try:    
        
            # before doing anything, we need to wipe the 
            # /bin, /lib, /man, and /include directories
            # in dst_dir.  Don't run as root.            
            make_dir(dst_dir,logger=logger)            
            change_dir(dst_dir   , logger)
            for d in ['bin','lib','man','include']:
                try:            remove_tree(d, logger)
                except OSError: pass                
            unchange_dir(logger)
            
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
            logger.info('Beginning Test')
            cmd = python_name +' -c "import sys,scipy;suite=scipy.test(%d);"'\
                                % test_level
            test_results = run_command(cmd, logger=logger,
                                       directory = tempfile.gettempdir())
            build_info['results'] = 'test completed (check below for pass/fail)'
        except Exception, msg:
            test_results = ''
            build_info['results'] = 'build failed: %s' % msg
            logger.exception('Build failed')
    finally:    
        to_addr = "scipy-testlog@scipy.org"
        from_addr = "scipy-test@enthought.com"
        subject = '%s,py%s,num%s,scipy%s' % (sys.platform,python_version,
                                            numeric_version,scipy_version) 
        build_log = log_stream.getvalue()
        mail_report(from_addr,to_addr,subject,local_mail_server,
                    build_log,test_results,build_info)

if __name__ == '__main__':
    build_dir = '/tmp/scipy_test'
    level = 10

    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.2.1',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # an older python
    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '21.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # an older numeric
    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '20.3',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    # This fails because multiarray doesn't have 
    # arange defined.
    """
    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '20.0.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '19.0.0',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')

    full_scipy_build(build_dir = build_dir,
                     test_level = level,
                     python_version  = '2.1.3',
                     numeric_version = '18.4.1',
                     f2py_version    = '2.13.175-1250',
                     atlas_version   = '3.3.14',
                     scipy_version   = 'snapshot')
    """


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

