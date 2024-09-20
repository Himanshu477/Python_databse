import paver
import paver.doctools
import paver.path
from paver.easy import options, Bunch, task, needs, dry, sh, call_task
from paver.setuputils import setup

# NOTES/Changelog stuff
RELEASE = 'doc/release/1.3.0-notes.rst'
LOG_START = 'tags/1.2.0'
LOG_END = 'master'

def compute_md5():
    released = paver.path.path('installers').listdir()
    checksums = []
    for f in released:
        m = md5.md5(open(f, 'r').read())
        checksums.append('%s  %s' % (m.hexdigest(), f))

    return checksums

def write_release_task(filename='NOTES.txt'):
    source = paver.path.path(RELEASE)
    target = paver.path.path(filename)
    if target.exists():
        target.remove()
    source.copy(target)
    ftarget = open(str(target), 'a')
    ftarget.writelines("""
Checksums
=========

""")
    ftarget.writelines(['%s\n' % c for c in compute_md5()])

def write_log_task(filename='Changelog'):
    st = subprocess.Popen(
            ['git', 'svn', 'log',  '%s..%s' % (LOG_START, LOG_END)],
            stdout=subprocess.PIPE)

    out = st.communicate()[0]
    a = open(filename, 'w')
    a.writelines(out)
    a.close()

@task
def write_release():
    write_release_task()

@task
def write_log():
    write_log_task()


#! /usr/bin/env python
# Last Change: Sat Mar 28 02:00 AM 2009 J

# Try to identify instruction set used in binary (x86 only). This works by
# checking the assembly for instructions specific to sse, etc... Obviously,
# this won't work all the times (for example, if some instructions are used
# only after proper detection of the running CPU, this will give false alarm).

