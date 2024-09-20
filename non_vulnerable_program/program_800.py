import string,commands,sys,os

sys.stderr=open('runtests.log','w')
f2py=sys.executable+' '+os.path.abspath(os.path.join('..','..','f2py2e.py'))
devnull='> /dev/null'
devnull=''

tests = []

